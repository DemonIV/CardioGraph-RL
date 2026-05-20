# CardioGraph-RL

EKG sinyalinden kalp ritim bozukluğu (aritmi) tespit eden, kararlarını Türkçe açıklayan açıklanabilir yapay zeka sistemi.

## Hızlı Başlangıç

Tüm komutlar `cardiograph/` dizininden çalıştırılır:

```bash
cd cardiograph
```

| Amaç | Komut |
|---|---|
| Streamlit demo arayüzü | `streamlit run demo/app.py --server.port 8050` |
| FastAPI REST servisi | `uvicorn demo.api:app --port 8000 --reload` |
| Tüm testleri çalıştır | `pytest tests/ -q --tb=short` |
| Veri indirme | `python data/raw/ptbxl/example_physionet.py` |
| Beat segmentasyonu | `python tools/build_beats.py` |
| Graf oluşturma | `python tools/build_graphs.py` |
| Model eğitimi | `python models/train.py` |
| Kalibrasyon | `python eval/calibrate.py` |
| Değerlendirme raporu | `python eval/eval_report.py` |

- Streamlit: `http://localhost:8050`
- FastAPI: `http://localhost:8000` — API dokümantasyonu: `http://localhost:8000/docs`

---

## Genel Bakış

Standart derin öğrenme modelleri "kara kutu"dur — neden belirli bir tanı koyduğunu açıklayamazlar. CardioGraph-RL, 5 özgün katmanı birleştirerek hem yüksek doğruluk hem de klinik açıklanabilirlik sağlar.

```
Ham EKG → Beat Segmentasyonu → Görünürlük Grafı → GAT → Nöro-Sembolik Füzyon → Açıklamalı Tanı
```

## Mimari

### 5 Özgün Katman

| # | Katman | Açıklama |
|---|---|---|
| 1 | EKG → Graf | Her kalp atımı düğüm, atımlar arası NaturalVG görünürlük kenarları |
| 2 | GAT | Graph Attention Network — hangi atım kritik? attention ile gösterir |
| 3 | Nöro-Sembolik | GAT olasılığı + Prolog klinik kuralları füzyonu (alpha=0.35) |
| 4 | Faithfulness@K | Attention'ın klinik anlamlılığını ölçen özgün metrik |
| 5 | Temperature Scaling | Post-hoc kalibrasyon (T=1.0256, GAT ECE=0.017) |

### 6 Ajan Pipeline

```
preprocessing_agent → graph_agent → model_agent → symbolic_agent → eval_agent → demo_agent
```

Her ajan yalnızca kendi klasörünü görür; orkestratör ajanlar arası geçişi yönetir.

## Sonuçlar (PTB-XL Test Seti)

| Metrik | Değer |
|---|---|
| AUC-ROC | 0.8702 |
| F1 macro | 0.5974 |
| F1 weighted | 0.6635 |
| GAT ECE | 0.0171 (mükemmel) |
| Füzyon ECE | 0.1573 |
| Faithfulness@3 | 0.3829 |

**Sınıf bazlı F1:**

| NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|
| 0.80 | 0.49 | 0.59 | 0.65 | 0.46 |

## Veri

**PTB-XL** — 21.837 hasta, 12-lead EKG, 500 Hz, 5 sınıf (NORM, MI, STTC, CD, HYP)

Veri setini indirmek için:
```bash
cd cardiograph
python data/raw/ptbxl/example_physionet.py
```

Kayıtlar `cardiograph/data/raw/ptbxl/` dizinine indirilir.

## Kurulum

**Gereksinimler:** Python 3.10+, CUDA (opsiyonel)

```bash
pip install -r cardiograph/requirements.txt
```

SWI-Prolog (opsiyonel — yoksa Python fallback kullanılır):
- Windows: https://www.swi-prolog.org/download/stable
- Linux: `sudo apt install swi-prolog`

## Kullanım

Tüm komutlar `cardiograph/` dizininden çalıştırılır.

### Demo Arayüzü (Streamlit)

```bash
streamlit run demo/app.py --server.port 8050
```

Tarayıcıda `http://localhost:8050` adresini açın.

1. **EKG Yükle:** `.hea` ve `.dat` dosyalarını yükleyin, "Analiz Et"e tıklayın
2. **Analiz:** Tanı, GAT güven skoru, beat attention ısı haritası, sınıf olasılıkları
3. **Gerekçe:** Tetiklenen Prolog kuralları Türkçe açıklama, klinik proxy değerleri

### REST API (FastAPI)

```bash
uvicorn demo.api:app --port 8000 --reload
```

```bash
# Sağlık kontrolü
curl http://localhost:8000/health

# Tahmin (wfdb dosyalarıyla)
curl -X POST http://localhost:8000/predict \
  -F "hea_file=@record.hea" \
  -F "dat_file=@record.dat"
```

API dokümantasyonu: `http://localhost:8000/docs`

### Pipeline Adım Adım

```bash
# 1. Beat segmentasyonu
python tools/build_beats.py

# 2. Graf oluşturma
python tools/build_graphs.py

# 3. Model eğitimi
python models/train.py

# 4. Sembolik füzyon değerlendirmesi
python symbolic/eval_fusion.py

# 5. Kalibrasyon (temperature scaling)
python eval/calibrate.py

# 6. Tam değerlendirme raporu
python eval/eval_report.py
```

### Testler

```bash
pytest tests/ -q --tb=short
```

| Test Dosyası | Kapsam | Test Sayısı |
|---|---|---|
| test_preprocessing.py | Beat segmentasyonu, özellik çıkarımı | 12 |
| test_graph.py | NaturalVG, DTW, PyG Data | 7 |
| test_model.py | CardioGAT forward, FocalLoss | — |
| test_symbolic.py | Prolog kuralları, füzyon | — |
| test_eval.py | Faithfulness, ECE, metrikler | 15 |
| test_demo.py | Pipeline, API, tam inference | 31 |

## Proje Yapısı

```
CardioGraph-RL/
└── cardiograph/
    ├── preprocessing/
    │   └── pipeline.py          # Beat segmentasyonu, 96-dim özellik çıkarımı
    ├── graph/
    │   └── builder.py           # NaturalVG + DTW → PyG Data
    ├── models/
    │   ├── gat.py               # CardioGAT (3 katman, 8 head, hidden=128)
    │   ├── train.py             # FocalLoss + WeightedRandomSampler
    │   ├── temperature_scaling.py
    │   └── checkpoints/
    │       ├── best_model.pt    # Model + x_mean/x_std
    │       └── temperature.pt   # {"temperature": 1.0256}
    ├── symbolic/
    │   ├── rules.py             # PrologEngine + Python fallback
    │   ├── rules.pl             # SWI-Prolog klinik kuralları
    │   ├── classifier.py        # SymbolicClassifier (96-dim → 5-sınıf)
    │   └── fusion.py            # NeuralSymbolicFusion (alpha=0.35)
    ├── eval/
    │   ├── metrics.py           # Faithfulness@K, ECE, compute_all_metrics
    │   ├── eval_report.py       # Tam değerlendirme raporu
    │   └── calibrate.py         # Temperature Scaling
    ├── demo/
    │   ├── pipeline.py          # run_inference() — tek kayıt inference
    │   ├── app.py               # Streamlit arayüzü (port 8050)
    │   └── api.py               # FastAPI REST API (port 8000)
    ├── tests/                   # pytest test suite
    ├── tools/                   # Toplu işleme scriptleri
    └── memory/                  # Ajan bellek dosyaları (JSON)
```

## Teknik Kararlar

**Güven skoru neden GAT'tan alınıyor, füzyondan değil?**
Temperature Scaling sonrası GAT ECE=0.017 (mükemmel kalibre). Sembolik füzyon deterministik kural çıktısı kullandığından Füzyon ECE=0.157 daha yüksek. Demo'da güven için GAT, tanı kararı için füzyon kullanılır.

**FocalLoss + WeightedRandomSampler birlikte kullanılmamalı.**
İkisi birlikte NORM sınıfını baskılar. Yalnızca biri kullanılmalı (bkz. `feedback_focal_sampler.md`).

**Sembolik MI tespiti zayıf (F1=0.49).**
Beat-level ST elevasyonu / Q dalgası proxy'si yüzeysel; gerçek MI tespiti için sequence-level analiz gerekir.

## Stack

| Kategori | Kütüphane |
|---|---|
| Sinyal işleme | `wfdb`, `neurokit2`, `scipy` |
| Graf öğrenmesi | `torch-geometric`, `ts2vg`, `dtaidistance` |
| Sembolik AI | `pyswip` (SWI-Prolog), `scikit-fuzzy` |
| Web | `streamlit`, `fastapi`, `uvicorn` |
| Görselleştirme | `plotly` |
| Test | `pytest` |
