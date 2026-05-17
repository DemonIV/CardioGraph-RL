# AGENT: graph_agent
"""
PTB-XL beats → PyG graf dönüştürme script.
Kullanım: python tools/build_graphs.py [--max N]
"""
import sys
import ast
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Proje kökü olarak cardiograph/ dizinini ekle
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.builder import build_graph_batch

# PTB-XL SCP kodu → 5-sınıf superdiagnostic eşlemesi
SUPERCLASS = {
    # 0 — NORM
    'NORM': 0, 'SR': 0, 'SARRH': 0,
    # 1 — MI (Myocardial Infarction)
    'AMI': 1, 'IMI': 1, 'ILMI': 1, 'ALMI': 1, 'ASMI': 1,
    'INJAS': 1, 'LMI': 1, 'INJAL': 1, 'IPLMI': 1, 'IPMI': 1,
    'INJIN': 1, 'INJLA': 1, 'PMI': 1, 'INJIL': 1, 'QWAVE': 1,
    # 2 — STTC (ST/T Change)
    'STD_': 2, 'STE_': 2, 'NST_': 2, 'NDT': 2, 'NT_': 2,
    'INVT': 2, 'LOWT': 2, 'TAB_': 2, 'ANEUR': 2, 'EL': 2,
    'ISC_': 2, 'ISCA': 2, 'ISCI': 2, 'ISCAL': 2, 'ISCAN': 2,
    'ISCAS': 2, 'ISCIL': 2, 'ISCIN': 2, 'ISCLA': 2,
    'DIG': 2, 'LNGQT': 2,
    # 3 — CD (Conduction Disturbance)
    'LAFB': 3, 'LPFB': 3, 'LBBB': 3, 'CLBBB': 3, 'ILBBB': 3,
    'RBBB': 3, 'CRBBB': 3, 'IRBBB': 3, 'IVCD': 3, 'ABQRS': 3,
    'WPW': 3, '1AVB': 3, '2AVB': 3, '3AVB': 3, 'LPR': 3, 'PRC(S)': 3,
    'AFIB': 3, 'AFLT': 3, 'STACH': 3, 'SBRAD': 3,
    'SVTAC': 3, 'PSVT': 3, 'SVARR': 3,
    'BIGU': 3, 'TRIGU': 3, 'PAC': 3, 'PVC': 3, 'PACE': 3,
    # 4 — HYP (Hypertrophy)
    'LVH': 4, 'RVH': 4, 'VCLVH': 4, 'SEHYP': 4,
    'LVOLT': 4, 'HVOLT': 4, 'LAO/LAE': 4, 'RAO/RAE': 4,
}


def assign_label(scp_str):
    """scp_codes string → int sınıf (0-4). Bilinmeyen → -1."""
    try:
        codes = ast.literal_eval(str(scp_str))
    except Exception:
        return -1

    scores = {}
    for code, likelihood in codes.items():
        sc = SUPERCLASS.get(code)
        if sc is not None:
            scores[sc] = scores.get(sc, 0.0) + float(likelihood)

    if not scores:
        return -1
    return max(scores, key=scores.get)


def build_labels_df(csv_path):
    """CSV'den record_id + label DataFrame oluştur."""
    df = pd.read_csv(csv_path)
    df['record_id'] = df['filename_hr'].apply(lambda x: Path(str(x)).stem)
    df['label'] = df['scp_codes'].apply(assign_label)
    valid = df[df['label'] >= 0][['record_id', 'label']].copy()
    print(f"Toplam kayıt: {len(df)} | Etiketlenebilen: {len(valid)} | Atılan: {len(df)-len(valid)}")
    print("Sınıf dağılımı:")
    names = {0: 'NORM', 1: 'MI', 2: 'STTC', 3: 'CD', 4: 'HYP'}
    for cls, cnt in valid['label'].value_counts().sort_index().items():
        print(f"  {cls} ({names[cls]}): {cnt}")
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=None, help='Max kayıt sayısı (test için)')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    beats_dir  = base / 'data' / 'processed' / 'beats'
    graphs_dir = base / 'data' / 'processed' / 'graphs'
    csv_path   = base / 'data' / 'raw' / 'ptbxl' / 'ptbxl_database.csv'

    labels_df = build_labels_df(csv_path)

    if args.max:
        # Mevcut beats dosyalarıyla kesişen record_id'leri sırala, ilk N al
        existing = {p.stem.replace('beats_', '') for p in beats_dir.glob('beats_*.npy')}
        available = labels_df[labels_df['record_id'].isin(existing)]
        labels_df = available.head(args.max).copy()
        print(f"\n[TEST MODU] İlk {args.max} kayıtla çalışılıyor.")

    print(f"\nToplamda {len(labels_df)} kayıt işlenecek.")
    print(f"Çıktı dizini: {graphs_dir}")

    t0 = time.time()
    result = build_graph_batch(
        processed_dir=str(beats_dir),
        output_dir=str(graphs_dir),
        labels_df=labels_df,
    )
    elapsed = time.time() - t0

    print(f"\nTamamlandı: {result['built']} başarılı, {result['failed']} başarısız")
    print(f"Süre: {elapsed:.1f}s  ({elapsed/max(result['built'],1):.2f}s/kayıt)")

    # Kayıt sayısını doğrula
    saved = list(graphs_dir.glob('graph_*.pt'))
    print(f"Diske kaydedilen .pt dosyası: {len(saved)}")
    return result


if __name__ == '__main__':
    main()
