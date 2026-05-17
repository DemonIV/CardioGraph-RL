# SETUP.md — Projeyi Tanı

## Ne İnşa Ediyoruz
EKG sinyalinden kalp ritim bozukluğu (aritmi) tespit eden,
kararlarını Türkçe açıklayan bir yapay zeka sistemi.

Adı: CardioGraph-RL
Klasör: ~/cardiograph/

## Neden Özgün
Standart CNN modelleri kara kutudur — "neden AF?" sorusuna cevap veremez.
Bu sistem 5 özgün katmanla bunu çözer:

1. EKG → Graf: Her kalp atımı düğüm, atımlar arası görünürlük kenar
2. GAT: Graph Attention Network — hangi atım kritik? attention ile gösterir
3. Nöro-Sembolik: GAT olasılığı + Prolog klinik kuralları birleşir
4. Faithfulness@K: Attention'ın klinik anlamlılığını ölçen özgün metrik
5. Morfoloji augmentasyon: Nadir aritmiler için sentetik atım enjeksiyonu

## Stack
Python 3.11 | PyTorch 2.2 | PyTorch Geometric 2.5
wfdb | neurokit2 | ts2vg | dtaidistance
pyswip | scikit-fuzzy
FastAPI | Plotly Dash | pytest | Docker

## Veri
PTB-XL — 21.837 hasta, 12-lead EKG, 500 Hz, 5 sınıf
Kullanıcı tarafından indirilecek: ~/cardiograph/data/raw/ptbxl/
Claude Code veri indirmez.

## Başarı Kriterleri
- Val F1 ≥ 0.85 (PTB-XL test seti)
- Faithfulness@3 ≥ 0.70
- Cross-dataset F1 ≥ 0.75 (MIT-BIH)

## Token Tasarrufu Mimarisi
Her modülün kendi ajanı, kendi belleği, kendi dar penceresi var.

Ajan → Klasör → Bellek dosyası:
preprocessing_agent → preprocessing/ → memory/preprocessing_memory.json
graph_agent         → graph/         → memory/graph_memory.json
model_agent         → models/        → memory/model_memory.json
symbolic_agent      → symbolic/      → memory/symbolic_memory.json
eval_agent          → eval/          → memory/eval_memory.json
demo_agent          → demo/          → memory/demo_memory.json

Kurallar:
- Her ajan yalnızca kendi klasörünü görür
- Büyük dosya okuma yok → tools/summarize_*.py kullan
- Her oturum sonu 5 satır özet yaz, dur
- Test komutu: pytest tests/test_[modül].py -q --tb=short

Bu dosyayı okuduğunda BUILD.md'yi iste.
