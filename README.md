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
| 3 | Nöro-Sembolik | GAT olasılığı + FuzzyRuleEngine klinik kuralları füzyonu (alpha=0.10) |
| 4 | Faithfulness@K | Attention'ın klinik anlamlılığını ölçen özgün metrik |
| 5 | Temperature Scaling | Post-hoc kalibrasyon (T=1.0256, GAT ECE=0.017) |

### 6 Ajan Pipeline

```
preprocessing_agent → graph_agent → model_agent → symbolic_agent → eval_agent → demo_agent
```

Her ajan yalnızca kendi klasörünü görür; orkestratör ajanlar arası geçişi yönetir.

## Sonuçlar (PTB-XL Test Seti, D12 Checkpoint)

| Metrik | Değer |
|---|---|
| AUC-ROC | **0.8786** |
| F1 macro | 0.5928 |
| F1 weighted | 0.6675 |
| GAT ECE | 0.0171 (mükemmel kalibre) |
| Füzyon ECE | **0.0427** (iyi kalibre) |
| Faithfulness@3 | **0.8479** |

**Sınıf bazlı F1:**

| NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|
| 0.81 | 0.50 | 0.58 | 0.65 | 0.42 |

**İyileştirme geçmişi (iyileştirme fazı boyunca):**

| Metrik | Başlangıç | Final | Fark |
|---|---|---|---|
| Faithfulness@3 | 0.383 | 0.848 | **+%121** |
| Füzyon ECE | 0.153 | 0.043 | **-72%** |
| AUC-ROC | 0.870 | 0.879 | +%1 |
| val F1 macro | 0.513 | 0.627 | **+%22** |

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

SWI-Prolog (opsiyonel — yoksa FuzzyRuleEngine Python fallback kullanılır):
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
3. **Gerekçe:** Tetiklenen fuzzy klinik kurallar, Türkçe açıklama, klinik proxy değerleri

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

# 4. Sembolik füzyon değerlendirmesi (alpha grid search + adaptive fusion)
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
| test_preprocessing.py | Beat segmentasyonu, 108-dim özellik çıkarımı | 12 |
| test_graph.py | NaturalVG, DTW, PyG Data | 7 |
| test_model.py | CardioGAT forward, FocalLoss, faithful_loss | 4 |
| test_symbolic.py | FuzzyRuleEngine, AdaptiveFusion, füzyon | 40 |
| test_eval.py | Faithfulness, ECE, metrikler | 15 |
| test_demo.py | Pipeline, API, tam inference | 31 |
| **Toplam** | | **71** |

## Proje Yapısı

```
CardioGraph-RL/
└── cardiograph/
    ├── preprocessing/
    │   └── pipeline.py          # Beat segmentasyonu, 108-dim hibrit özellik
    ├── graph/
    │   └── builder.py           # NaturalVG + DTW → PyG Data
    ├── models/
    │   ├── gat.py               # CardioGAT — (logits, (attn_ei, attn)) döndürür
    │   ├── train.py             # FocalLoss + faithful_loss + lead masking aug.
    │   ├── temperature_scaling.py
    │   └── checkpoints/
    │       ├── best_model.pt        # D12: val_f1=0.6273, 108-dim
    │       ├── temperature.pt       # {"temperature": 1.0256}
    │       └── adaptive_alphas.npy  # per-class alpha (CD=0.10, diğerleri≈0)
    ├── symbolic/
    │   ├── rules.py             # PrologEngine + FuzzyRuleEngine (sigmoid tabanlı)
    │   ├── classifier.py        # SymbolicClassifier (108-dim → 5-sınıf)
    │   └── fusion.py            # NeuralSymbolicFusion + AdaptiveFusion + learn_alphas
    ├── eval/
    │   ├── metrics.py           # Faithfulness@K, ECE, compute_all_metrics
    │   ├── eval_report.py       # Tam değerlendirme raporu (alpha=0.10)
    │   └── calibrate.py         # Temperature Scaling
    ├── demo/
    │   ├── pipeline.py          # run_inference() — tek kayıt inference
    │   ├── app.py               # Streamlit arayüzü (port 8050)
    │   └── api.py               # FastAPI REST API (port 8000)
    ├── tests/                   # pytest — 71 test
    ├── tools/                   # Toplu işleme scriptleri
    └── memory/                  # Ajan bellek dosyaları (JSON)
```

## Teknik Kararlar

**Neden alpha=0.10?**
eval_fusion grid search (alpha ∈ [0, 0.50]) ile doğrulandı. D12 modelinin 108-dim hibrit özellikleri (spectral_centroid dahil) GAT'ı yeterince güçlendirdiğinden sembolik füzyon çok küçük bir katkı sağlıyor. Per-class AdaptiveFusion ile de doğrulandı: LBFGS optimizasyonu CD hariç tüm sınıflarda alpha≈0 buldu.

**Güven skoru neden GAT'tan alınıyor, füzyondan değil?**
Temperature Scaling sonrası GAT ECE=0.017 (mükemmel kalibre). Füzyon ECE=0.043 (alpha=0.10'da iyi). Demo'da her ikisi de güvenilir, ancak GAT tek başına daha temiz bir kalibrasyon sunar.

**FocalLoss + WeightedRandomSampler birlikte kullanılmamalı.**
İkisi birlikte NORM sınıfını baskılar. Yalnızca biri kullanılmalı.

**Faithfulness@3 neden 0.38'den 0.85'e çıktı?**
`faithful_loss` eklendi: ST>0.05 veya Q<-0.05 olan düğümlere gelen attention maximize ediliyor. `L_total = L_CE + 0.1 × L_faithful`.

**108-dim özellik neden 96-dim'den iyi?**
`spectral_centroid` eklemesi kritik: energy yerine sc → val_f1 0.51→0.63 (+0.12). Spektral ağırlık merkezi MI morfoljisini daha iyi kodluyor.

## Stack

| Kategori | Kütüphane |
|---|---|
| Sinyal işleme | `wfdb`, `neurokit2`, `scipy` |
| Graf öğrenmesi | `torch-geometric`, `ts2vg`, `dtaidistance` |
| Sembolik AI | `pyswip` (SWI-Prolog), Python FuzzyRuleEngine |
| Web | `streamlit`, `fastapi`, `uvicorn` |
| Görselleştirme | `plotly` |
| Test | `pytest` |
