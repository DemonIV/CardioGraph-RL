# AGENT: demo_agent
"""
Tek EKG kaydı için tam inference pipeline.
wfdb kayıt stem yolu → tanı + güven + attention + sembolik kurallar
"""
import sys
import tempfile
import numpy as np
import torch
import wfdb
from pathlib import Path
from torch_geometric.data import Data, Batch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from preprocessing.pipeline import process_record
from graph.builder import build_clinical_graph, compute_dtw_weights
from models.gat import CardioGAT
from models.temperature_scaling import TemperatureScaler
from symbolic.classifier import SymbolicClassifier, extract_clinical_features
from symbolic.fusion import NeuralSymbolicFusion

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES_TR = {
    "NORM": "Normal Ritim",
    "MI":   "Miyokard Enfarktüsü",
    "STTC": "ST-T Değişikliği",
    "CD":   "İletim Bozukluğu",
    "HYP":  "Hipertrofi",
}

CHECKPOINT = BASE / "models" / "checkpoints" / "best_model.pt"
TEMP_PATH  = BASE / "models" / "checkpoints" / "temperature.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA      = 0.35

_model  = None
_scaler = None
_x_mean = None
_x_std  = None


def _load_model():
    global _model, _scaler, _x_mean, _x_std
    if _model is not None:
        return
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {CHECKPOINT}\n"
            "model_agent eğitimini tamamlayın."
        )
    ckpt = torch.load(str(CHECKPOINT), weights_only=False)
    _x_mean = ckpt["x_mean"].to(DEVICE)
    _x_std  = ckpt["x_std"].to(DEVICE)
    _model = CardioGAT().to(DEVICE)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    t = 1.0256
    if TEMP_PATH.exists():
        t_ckpt = torch.load(str(TEMP_PATH), weights_only=False)
        t = float(t_ckpt.get("temperature", 1.0256))
    _scaler = TemperatureScaler(initial_temperature=t)
    _scaler.eval()


def run_inference(record_path: str) -> dict:
    """
    Parameters
    ----------
    record_path : str
        wfdb kayıt stem yolu (.hea ve .dat aynı dizinde).

    Returns
    -------
    dict
        signal, lead_names, fs, n_beats, pred_class, pred_class_tr,
        gat_probs (5,), fused_probs (5,), beat_attention (N,),
        symbolic_rules, clinical_features
    """
    _load_model()

    # 1. Preprocessing → beats (N, 96)
    with tempfile.TemporaryDirectory() as tmp:
        res = process_record(record_path, tmp)
        if res is None:
            raise ValueError("Kayıt işlenemedi: yeterli R-tepe bulunamadı.")
        beats = np.load(res["output_path"])

    # 2. Ham sinyal (görselleştirme)
    record     = wfdb.rdrecord(str(record_path))
    signal     = np.nan_to_num(record.p_signal, nan=0.0)
    lead_names = record.sig_name
    fs         = record.fs

    # 3. Graf
    edge_index, node_features = build_clinical_graph(beats)
    edge_attr = compute_dtw_weights(beats, edge_index)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=edge_attr,
        y=torch.tensor([0], dtype=torch.long),
    )
    batch = Batch.from_data_list([data]).to(DEVICE)
    batch.x = (batch.x - _x_mean) / _x_std

    # 4. GAT forward
    with torch.no_grad():
        logits, attn = _model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

    # 5. Kalibre GAT probs (T=1.0256, ECE=0.017)
    cal_logits = _scaler.scale(logits.cpu()).detach()
    gat_probs  = torch.softmax(cal_logits, dim=-1).numpy()[0]   # (5,)

    # 6. Sembolik
    sc             = SymbolicClassifier()
    sym_probs      = sc.predict(node_features)                   # (5,)
    clin_feats     = extract_clinical_features(node_features)
    symbolic_rules = sc.engine.query(clin_feats)

    # 7. Nöro-sembolik füzyon (alpha=0.35)
    fused_probs = NeuralSymbolicFusion(alpha=ALPHA).fuse(gat_probs, sym_probs)  # (5,)

    # 8. Beat attention → (N,) normalize edilmiş
    attn_np    = attn.cpu().numpy()                             # (E', heads)
    ei         = batch.edge_index.cpu().numpy()                 # (2, E)
    n_nodes    = node_features.shape[0]
    n_use      = min(ei.shape[1], attn_np.shape[0])
    attn_1d    = attn_np[:n_use].mean(axis=1)

    beat_attn  = np.zeros(n_nodes)
    beat_count = np.zeros(n_nodes)
    for k in range(n_use):
        dst = int(ei[1, k])
        if dst < n_nodes:
            beat_attn[dst]  += attn_1d[k]
            beat_count[dst] += 1
    beat_count = np.where(beat_count == 0, 1, beat_count)
    beat_attn  = beat_attn / beat_count
    if beat_attn.max() > beat_attn.min():
        beat_attn = (beat_attn - beat_attn.min()) / (beat_attn.max() - beat_attn.min())

    pred_idx = int(fused_probs.argmax())
    return {
        "signal":            signal,
        "lead_names":        lead_names,
        "fs":                fs,
        "n_beats":           n_nodes,
        "pred_class":        CLASS_NAMES[pred_idx],
        "pred_class_tr":     CLASS_NAMES_TR[CLASS_NAMES[pred_idx]],
        "gat_probs":         gat_probs,
        "fused_probs":       fused_probs,
        "beat_attention":    beat_attn,
        "symbolic_rules":    symbolic_rules,
        "clinical_features": clin_feats,
    }
