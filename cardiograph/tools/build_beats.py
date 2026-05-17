# AGENT: graph_agent
"""
PTB-XL kayıtlarından beats_{id}.npy dosyalarını (yeniden) üretir.
Kullanım: python tools/build_beats.py [--max N]
"""
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.pipeline import process_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=None, help='Max kayıt (test için)')
    args = parser.parse_args()

    base      = Path(__file__).parent.parent
    data_dir  = base / 'data' / 'raw' / 'ptbxl' / 'records500'
    beats_dir = base / 'data' / 'processed' / 'beats'
    beats_dir.mkdir(parents=True, exist_ok=True)

    record_paths = sorted(data_dir.rglob('*.hea'))
    record_stems = [str(p.with_suffix('')) for p in record_paths]
    print(f"Bulunan kayıt: {len(record_stems)}")

    if args.max:
        record_stems = record_stems[:args.max]
        print(f"[TEST MODU] İlk {args.max} kayıt işleniyor.")

    print(f"Çıktı: {beats_dir}\n")
    t0 = time.time()
    result = process_batch(record_stems, str(beats_dir))
    elapsed = time.time() - t0

    print(f"Tamamlandı: {result['processed']} başarılı | {result['failed']} başarısız")
    print(f"Süre: {elapsed:.1f}s  ({elapsed/max(result['processed'],1):.2f}s/kayıt)")


if __name__ == '__main__':
    main()
