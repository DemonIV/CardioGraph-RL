# preprocessing_agent

## Oturum Başı (3 adım, sırayla)
1. memory/preprocessing_memory.json oku
2. open_issues listesinin ilk maddesini al
3. pytest tests/test_preprocessing.py -q --tb=short

## Kapsam
YALNIZCA: preprocessing/ ve tests/test_preprocessing.py

## Çıktı Sözleşmesi
data/processed/beats/beats_{id}.npy → shape (N_beats, 12)
Öznitelik sırası memory dosyasındaki output_contract'ta

## Kurallar
- Ham veri okuma → wfdb.rdrecord() kullan
- Büyük çıktı üretme → pytest -q --tb=short
- Veri özeti → python tools/summarize_preprocessing.py

## Oturum Sonu (bu format, değiştirme)
memory/preprocessing_memory.json güncelle, sonra yaz:
TAMAMLANDI: ...
KARAR: ...
METRİK: ...
SONRAKI ADIM: ...
BAĞIMLILIK: evet/hayır
Dur.
