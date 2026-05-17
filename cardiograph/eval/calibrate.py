# AGENT: eval_agent
"""
Temperature Scaling kalibrasyonu.
T validation setinde optimize edilir, ECE test setinde raporlanır.
Kullanım: python eval/calibrate.py
"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from models.gat import CardioGAT
from models.temperature_scaling import TemperatureScaler
from symbolic.classifier import SymbolicClassifier
from symbolic.fusion import NeuralSymbolicFusion
from eval.metrics import expected_calibration_error

BASE       = Path(__file__).parent.parent
GRAPHS_DIR = BASE / "data" / "processed" / "graphs"
CSV_PATH   = BASE / "data" / "raw" / "ptbxl" / "ptbxl_database.csv"
CHECKPOINT = BASE / "models" / "checkpoints" / "best_model.pt"
TEMP_PATH  = BASE / "models" / "checkpoints" / "temperature.pt"
MEM_PATH   = BASE / "memory" / "eval_memory.json"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
ALPHA       = 0.35


def load_splits():
    df = pd.read_csv(CSV_PATH)
    df["record_id"] = df["filename_hr"].apply(lambda x: Path(str(x)).stem)
    fold_map = dict(zip(df["record_id"], df["strat_fold"]))
    val_data, test_data = [], []
    for pt in sorted(GRAPHS_DIR.glob("graph_*.pt")):
        rid  = pt.stem.replace("graph_", "")
        fold = fold_map.get(rid)
        if fold == 9:
            val_data.append(torch.load(str(pt), weights_only=False))
        elif fold == 10:
            test_data.append(torch.load(str(pt), weights_only=False))
    return val_data, test_data


@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    logits_list, labels_list = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        logits_list.append(logits.cpu())
        labels_list.append(batch.y.squeeze().cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def ece_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    conf    = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    return expected_calibration_error(conf, correct)


def main():
    print(f"Device: {DEVICE}")
    print("Veri yükleniyor...")
    val_orig, test_orig = load_splits()
    print(f"  Val: {len(val_orig)} | Test: {len(test_orig)}")

    ckpt   = torch.load(str(CHECKPOINT), weights_only=False)
    x_mean = ckpt["x_mean"].to(DEVICE)
    x_std  = ckpt["x_std"].to(DEVICE)

    model = CardioGAT().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def normalize(data):
        out = [d.clone() for d in data]
        for d in out:
            d.x = (d.x.to(DEVICE) - x_mean) / x_std
        return out

    val_norm  = normalize(val_orig)
    test_norm = normalize(test_orig)

    val_loader  = DataLoader(val_norm,  batch_size=64, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)

    print("Test logitleri toplanıyor...")
    test_logits, test_labels = collect_logits(model, test_loader)
    test_labels_np = test_labels.numpy().astype(int)

    # Sembolik olasılıklar (orijinal ölçek, bir kez hesapla)
    print("Sembolik olasılıklar hesaplanıyor...")
    sc    = SymbolicClassifier()
    sym_p = np.stack([sc.predict(d.x.cpu().numpy()) for d in test_orig])

    fusion = NeuralSymbolicFusion(alpha=ALPHA)

    # ── Kalibrasyon ÖNCESİ ────────────────────────────────────────────────────
    gat_probs_before = torch.softmax(test_logits, dim=-1).numpy()
    fused_before     = fusion.fuse_batch(gat_probs_before, sym_p)

    ece_gat_before   = ece_from_probs(gat_probs_before, test_labels_np)
    ece_fused_before = ece_from_probs(fused_before,     test_labels_np)
    f1_before        = f1_score(test_labels_np, fused_before.argmax(1),
                                average="macro", zero_division=0)

    print(f"\n  [Öncesi] ECE GAT only : {ece_gat_before:.4f}")
    print(f"  [Öncesi] ECE Füzyon   : {ece_fused_before:.4f}")
    print(f"  [Öncesi] F1 macro     : {f1_before:.4f}")

    # ── Temperature optimize et (VAL seti) ────────────────────────────────────
    print("\nTemperature optimize ediliyor (LBFGS, max_iter=50, val seti)...")
    scaler = TemperatureScaler().to("cpu")
    T = scaler.calibrate(model, val_loader, DEVICE, max_iter=50)
    print(f"  Optimal T = {T:.4f}")

    # ── Kalibrasyon SONRASI ───────────────────────────────────────────────────
    gat_probs_after = torch.softmax(scaler.scale(test_logits).detach(), dim=-1).numpy()
    fused_after     = fusion.fuse_batch(gat_probs_after, sym_p)

    ece_gat_after   = ece_from_probs(gat_probs_after, test_labels_np)
    ece_fused_after = ece_from_probs(fused_after,     test_labels_np)
    f1_after        = f1_score(test_labels_np, fused_after.argmax(1),
                               average="macro", zero_division=0)

    print(f"\n  [Sonrası] ECE GAT only : {ece_gat_after:.4f}")
    print(f"  [Sonrası] ECE Füzyon   : {ece_fused_after:.4f}")
    print(f"  [Sonrası] F1 macro     : {f1_after:.4f}  (değişmemeli)")

    # ── Özet ──────────────────────────────────────────────────────────────────
    ece_imp = (ece_fused_before - ece_fused_after) / ece_fused_before * 100
    print("\n" + "=" * 50)
    print("KALİBRASYON ÖZETI (test seti)")
    print("=" * 50)
    print(f"  Optimal T              : {T:.4f}")
    print(f"  ECE öncesi (Füzyon)    : {ece_fused_before:.4f}")
    print(f"  ECE sonrası (Füzyon)   : {ece_fused_after:.4f}")
    print(f"  ECE iyileşme           : {ece_imp:.1f}%")
    print(f"  F1 macro (değişim)     : {f1_before:.4f} → {f1_after:.4f}")

    # ── Kaydet ────────────────────────────────────────────────────────────────
    torch.save({"temperature": T}, str(TEMP_PATH))
    print(f"\n  temperature.pt kaydedildi: {TEMP_PATH}")

    # eval_memory.json güncelle
    if MEM_PATH.exists():
        with open(MEM_PATH, encoding="utf-8") as f:
            mem = json.load(f)
    else:
        mem = {}
    mem["temperature"]       = round(T, 4)
    mem["ece_before_cal"]    = round(ece_fused_before, 4)
    mem["ece_after_cal"]     = round(ece_fused_after,  4)
    mem["ece_improvement_pct"] = round(ece_imp, 1)
    mem["f1_after_cal"]      = round(f1_after, 4)
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MEM_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)
    print("  eval_memory.json güncellendi.")


if __name__ == "__main__":
    main()
