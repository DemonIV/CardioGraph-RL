# AGENT: model_agent
"""
CardioGAT eğitim döngüsü.
Çıktı: models/checkpoints/best_model.pt
Kullanım: python models/train.py
"""
import sys
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gat import CardioGAT

# ── Hiperparametreler ──────────────────────────────────────────
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 300
PATIENCE     = 50
GRAD_CLIP    = 1.0
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES  = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
# ──────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal Loss — azınlık sınıflarına (HYP) daha fazla ağırlık verir."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_sampler(train_data, num_classes=5):
    """WeightedRandomSampler — her batch'te azınlık sınıfları daha çok görülür."""
    labels  = [d.y.item() for d in train_data]
    counts  = torch.zeros(num_classes)
    for l in labels:
        counts[l] += 1
    class_w  = 1.0 / counts.clamp(min=1)
    sample_w = torch.tensor([class_w[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def load_split(graphs_dir, csv_path):
    """PTB-XL strat_fold ile train/val/test bölme.
    fold 1-8 → train, fold 9 → val, fold 10 → test
    """
    df = pd.read_csv(csv_path)
    df['record_id'] = df['filename_hr'].apply(lambda x: Path(str(x)).stem)
    fold_map = dict(zip(df['record_id'], df['strat_fold']))

    train_data, val_data, test_data = [], [], []

    for pt_path in sorted(Path(graphs_dir).glob('graph_*.pt')):
        record_id = pt_path.stem.replace('graph_', '')
        fold = fold_map.get(record_id)
        if fold is None:
            continue
        data = torch.load(str(pt_path), weights_only=False)
        if fold <= 8:
            train_data.append(data)
        elif fold == 9:
            val_data.append(data)
        else:
            test_data.append(data)

    return train_data, val_data, test_data


def normalize_features(train_data, val_data, test_data):
    """Train setinden hesaplanan istatistiklerle global feature normalizasyonu.
    Sabit özellikler (std=0) sıfıra dönüşür — zararsız ama bilgi taşımaz.
    """
    all_x = torch.cat([d.x for d in train_data], dim=0)
    x_mean = all_x.mean(dim=0)
    x_std  = all_x.std(dim=0).clamp(min=1e-6)

    for split in [train_data, val_data, test_data]:
        for d in split:
            d.x = (d.x - x_mean) / x_std

    return x_mean, x_std


def compute_class_weights(train_data, num_classes=5):
    """Ters frekans ağırlıkları — sınıf dengesizliğini dengele."""
    counts = torch.zeros(num_classes)
    for d in train_data:
        counts[d.y.item()] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(batch.y.squeeze().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1, all_preds, all_labels


def train_model(graphs_dir, csv_path, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Graflar yükleniyor...")
    train_data, val_data, test_data = load_split(graphs_dir, csv_path)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    print("  Feature normalizasyonu hesaplanıyor...")
    x_mean, x_std = normalize_features(train_data, val_data, test_data)
    n_const = (x_std < 1e-5).sum().item()
    n_feat = x_mean.shape[0]
    print(f"  Sabit ozellik sayisi (std~0): {n_const}/{n_feat} -> sifira donusturuldu")

    weights = compute_class_weights(train_data).to(DEVICE)
    print("  Sınıf ağırlıkları:", {n: f"{w:.3f}" for n, w in zip(CLASS_NAMES, weights.cpu())})

    pin = (DEVICE == 'cuda')
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    model     = CardioGAT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # CosineAnnealingWarmRestarts: T_0=50 ilk döngü, T_mult=2 sonraki döngüleri ikiye katlar
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Eğitilebilir parametre: {n_params:,}")

    best_val_f1 = 0.0
    best_epoch  = 0
    no_improve  = 0
    history     = []

    print("\nEpoch | train_loss | val_loss | val_f1 | lr")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'train_loss': round(train_loss, 5),
                        'val_loss': round(val_loss, 5), 'val_f1': round(val_f1, 5)})

        marker = " ***" if val_f1 > best_val_f1 else ""
        print(f"  {epoch:3d} | {train_loss:.4f}     | {val_loss:.4f}   | "
              f"{val_f1:.4f} | {current_lr:.2e}{marker}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            no_improve  = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_f1': val_f1,
                'history': history,
                'x_mean': x_mean,
                'x_std': x_std,
            }, str(checkpoint_dir / 'best_model.pt'))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nErken durdurma: {PATIENCE} epoch iyileşme yok.")
                break

    # Test değerlendirmesi — en iyi checkpoint
    ckpt = torch.load(str(checkpoint_dir / 'best_model.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE)

    print(f"\n{'='*55}")
    print(f"En iyi epoch : {best_epoch}")
    print(f"Val F1 macro : {best_val_f1:.4f}")
    print(f"Test F1 macro: {test_f1:.4f}")
    print(f"\nSınıf bazlı test raporu:")
    print(classification_report(test_labels, test_preds,
                                 target_names=CLASS_NAMES, zero_division=0))

    return {
        'best_epoch': best_epoch,
        'val_f1': round(best_val_f1, 5),
        'test_f1': round(test_f1, 5),
        'history': history,
    }


def evaluate_model(model, loader):
    """Döndürür: {f1_macro, attention_scores}"""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    _, f1, preds, labels = evaluate(model, loader, criterion, device)
    return {'f1_macro': f1, 'preds': preds, 'labels': labels}


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    base = Path(__file__).parent.parent
    train_model(
        graphs_dir=str(base / 'data' / 'processed' / 'graphs'),
        csv_path=str(base / 'data' / 'raw' / 'ptbxl' / 'ptbxl_database.csv'),
        checkpoint_dir=str(base / 'models' / 'checkpoints'),
    )
