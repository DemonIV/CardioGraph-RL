# AGENT: graph_agent
"""
EKG beat dizisinden PyG graf nesnesi oluşturma.
Giriş: data/processed/beats/beats_{id}.npy  — shape (N, 48)
Çıkış: data/processed/graphs/graph_{id}.pt

v2: Klinik lead korelasyon grafı.
NaturalVG tek-lead yerine 3 klinik grup (inferior, lateral, anterior)
üzerinden çalışır; kenarlar union alınır.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from ts2vg import NaturalVG
from dtaidistance import dtw
from pathlib import Path

# 12-lead sırası: I(0) II(1) III(2) aVR(3) aVL(4) aVF(5) V1(6)..V6(11)
# Her lead 4 özellik kaplar: [lead*4 : lead*4+4]
LEAD_GROUPS = {
    'inferior': [1, 2, 5],       # II, III, aVF
    'lateral':  [0, 4, 10, 11],  # I, aVL, V5, V6
    'anterior': [6, 7, 8, 9],    # V1, V2, V3, V4
}


def _beats_to_node_features(beats_array):
    """Her beat → düğüm öznitelik vektörü."""
    if beats_array.ndim == 3:          # (N, T, L) ham beat → ortalama havuzlama
        return beats_array.mean(axis=1).astype(np.float32)
    else:                               # (N, F) zaten özellik
        return beats_array.astype(np.float32)


def _group_scalar(node_features, lead_indices):
    """(N, 48) → (N,) skaler: gruptaki leadlerin mean özelliği (ilk özellik = mean amplitüd)."""
    cols = [li * 4 for li in lead_indices]
    return node_features[:, cols].mean(axis=1).astype(np.float64)


def build_clinical_graph(beats_array):
    """
    Klinik lead korelasyon grafı.
    Her lead grubu için ayrı NaturalVG kurulur, kenarlar union alınır.
    Döndürür: (edge_index, node_features)
    """
    node_features = _beats_to_node_features(beats_array)  # (N, F)

    # F < 12 lead sayısı × 4 özellik durumunda fallback (test senaryosu)
    n_features = node_features.shape[1]
    n_leads = n_features // 4

    feat_per_lead = n_features // 12  # 4 (48-özellik) veya 8 (96-özellik)

    all_edges = set()
    for group_name, lead_indices in LEAD_GROUPS.items():
        valid_leads = [li for li in lead_indices if li < n_leads]
        if not valid_leads:
            continue
        scalar = node_features[:, [li * feat_per_lead for li in valid_leads]].mean(axis=1).astype(np.float64)
        g = NaturalVG()
        g.build(scalar)
        for i, j in g.edges:
            all_edges.add((min(i, j), max(i, j)))

    if all_edges:
        edges = sorted(all_edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        src_bi = src + dst
        dst_bi = dst + src
        edge_index = np.array([src_bi, dst_bi], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return edge_index, node_features


def _beat_signal_1d(beats_array, idx):
    """DTW için tekil beatten 1D sinyal."""
    beat = beats_array[idx]
    if beat.ndim == 2:
        return beat[:, 0].astype(np.float64)
    return beat.astype(np.float64)


def compute_dtw_weights(beats_array, edge_index):
    """Kenar ağırlığı = 1 / (1 + DTW). Döndürür: edge_attr tensor."""
    ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    n_edges = ei.shape[1]
    weights = np.zeros(n_edges, dtype=np.float32)

    cache = {}
    for k in range(n_edges):
        i, j = int(ei[0, k]), int(ei[1, k])
        key = (min(i, j), max(i, j))
        if key not in cache:
            a = _beat_signal_1d(beats_array, i)
            b = _beat_signal_1d(beats_array, j)
            cache[key] = dtw.distance(a, b)
        weights[k] = 1.0 / (1.0 + cache[key])

    return torch.tensor(weights, dtype=torch.float32)


def build_graph(record_id, beats_path, output_dir, label):
    """Tam pipeline. Döndürür: PyG Data nesnesi."""
    beats = np.load(str(beats_path))

    edge_index, node_features = build_clinical_graph(beats)
    edge_attr = compute_dtw_weights(beats, edge_index)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
    )
    data.record_id = record_id

    out_path = Path(output_dir) / f"graph_{record_id}.pt"
    torch.save(data, str(out_path))
    return data


def build_graph_batch(processed_dir, output_dir, labels_df):
    """Toplu graf oluşturma."""
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failed = 0
    for beats_path in sorted(processed_dir.glob("beats_*.npy")):
        record_id = beats_path.stem.replace("beats_", "")
        out_path = output_dir / f"graph_{record_id}.pt"
        if out_path.exists():
            results.append({"record_id": record_id, "n_nodes": 0})
            continue
        rows = labels_df[labels_df["record_id"] == record_id]
        if rows.empty:
            failed += 1
            continue
        try:
            data = build_graph(record_id, beats_path, str(output_dir), int(rows.iloc[0]["label"]))
            results.append({"record_id": record_id, "n_nodes": data.num_nodes})
        except Exception:
            failed += 1

    return {"built": len(results), "failed": failed, "results": results}
