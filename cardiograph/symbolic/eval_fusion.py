# AGENT: symbolic_agent
"""
Füzyon değerlendirmesi: D11-A GAT + SymbolicClassifier → val/test F1.

Kullanım: python symbolic/eval_fusion.py
"""
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report

from models.gat import CardioGAT
from symbolic.classifier import SymbolicClassifier, CLASS_ORDER
from symbolic.fusion import NeuralSymbolicFusion

BASE          = Path(__file__).parent.parent
GRAPHS_DIR    = BASE / "data" / "processed" / "graphs"
CSV_PATH      = BASE / "data" / "raw" / "ptbxl" / "ptbxl_database.csv"
CHECKPOINT    = BASE / "models" / "checkpoints" / "best_model.pt"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES   = ["NORM", "MI", "STTC", "CD", "HYP"]   # train.py sırası
# SymbolicClassifier CLASS_ORDER: norm, mi, sttc, cd, hyp (lowercase)
# → ikisi aynı sıra, sadece büyük/küçük harf farkı

ALPHA_GRID = [0.0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50]


def load_split():
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


def normalize(data, x_mean, x_std):
    for d in data:
        d.x = (d.x - x_mean) / x_std
    return data


@torch.no_grad()
def gat_probs_labels(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(batch.y.squeeze().cpu().tolist())
    return np.vstack(all_probs), np.array(all_labels)


def sym_probs_for_loader(sc, data_list):
    """Her grafın x tensörünü SymbolicClassifier'a ver."""
    probs = []
    for d in data_list:
        x = d.x.numpy()          # (N_beats, 96) — normalleştirilmiş
        p = sc.predict(x)         # (5,) — norm,mi,sttc,cd,hyp sırası
        probs.append(p)
    return np.stack(probs)


def evaluate(alpha, gat_p, sym_p, labels, split_name):
    fusion = NeuralSymbolicFusion(alpha=alpha)
    fused  = fusion.fuse_batch(gat_p, sym_p)
    preds  = fused.argmax(axis=1)
    f1     = f1_score(labels, preds, average="macro", zero_division=0)
    mi_f1  = f1_score(labels, preds, labels=[1], average="macro", zero_division=0)
    return f1, mi_f1


def main():
    print(f"Device: {DEVICE}")
    print("Graflar yükleniyor...")
    val_data, test_data = load_split()
    print(f"  Val: {len(val_data)} | Test: {len(test_data)}")

    # Checkpoint yükle
    ckpt   = torch.load(str(CHECKPOINT), weights_only=False)
    x_mean = ckpt["x_mean"].to(DEVICE)
    x_std  = ckpt["x_std"].to(DEVICE)

    model = CardioGAT().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Normalizasyon (veriler üzerinde in-place)
    val_norm  = [d.clone() for d in val_data]
    test_norm = [d.clone() for d in test_data]
    for split in [val_norm, test_norm]:
        for d in split:
            d.x = (d.x.to(DEVICE) - x_mean) / x_std

    val_loader  = DataLoader(val_norm,  batch_size=64, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=64, shuffle=False)

    print("GAT olasılıkları hesaplanıyor...")
    gat_val_p,  val_labels  = gat_probs_labels(model, val_loader)
    gat_test_p, test_labels = gat_probs_labels(model, test_loader)

    # GAT baseline (alpha=0)
    val_f1_gat  = f1_score(val_labels,  gat_val_p.argmax(1),  average="macro", zero_division=0)
    test_f1_gat = f1_score(test_labels, gat_test_p.argmax(1), average="macro", zero_division=0)
    val_mi_gat  = f1_score(val_labels,  gat_val_p.argmax(1),  labels=[1], average="macro", zero_division=0)
    print(f"\n[GAT baseline] val_f1={val_f1_gat:.4f}  MI_f1={val_mi_gat:.4f}  test_f1={test_f1_gat:.4f}")

    print("\nSembolik olasılıklar hesaplanıyor...")
    sc = SymbolicClassifier()
    # Sembolik classifier normalize edilmemiş özellikler bekliyor — orijinal veriyi kullan
    sym_val_p  = sym_probs_for_loader(sc, val_data)
    sym_test_p = sym_probs_for_loader(sc, test_data)

    print("\nalpha  | val_f1 | MI_f1  | test_f1 | hedef?")
    print("-" * 52)
    best = {"alpha": 0.0, "val_f1": 0.0, "mi_f1": 0.0}
    for alpha in ALPHA_GRID:
        vf1, vmi = evaluate(alpha, gat_val_p, sym_val_p, val_labels,  "val")
        tf1, _   = evaluate(alpha, gat_test_p, sym_test_p, test_labels, "test")
        met = "✓" if vf1 > 0.65 and vmi > 0.55 else " "
        print(f" {alpha:.2f}  | {vf1:.4f} | {vmi:.4f} | {tf1:.4f}  | {met}")
        if vf1 > best["val_f1"]:
            best = {"alpha": alpha, "val_f1": vf1, "mi_f1": vmi, "test_f1": tf1}

    print(f"\nEn iyi alpha: {best['alpha']} → val_f1={best['val_f1']:.4f}, MI_f1={best['mi_f1']:.4f}")

    # En iyi alpha için detaylı rapor
    print(f"\n{'='*55}")
    print(f"Detaylı rapor (alpha={best['alpha']}):")
    fusion = NeuralSymbolicFusion(alpha=best["alpha"])
    fused  = fusion.fuse_batch(gat_val_p, sym_val_p)
    preds  = fused.argmax(axis=1)
    print(classification_report(val_labels, preds, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
