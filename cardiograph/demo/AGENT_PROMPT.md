# demo_agent

## Oturum Başı
1. memory/demo_memory.json oku
2. FastAPI sağlık kontrolü: curl localhost:8000/health
3. pytest tests/test_demo.py -q --tb=short

## Kapsam
YALNIZCA: demo/ ve tests/test_demo.py

## Final Metrikler (referans)
| Metrik | Değer |
|---|---|
| AUC-ROC | 0.87 |
| F1 macro | 0.5974 |
| F1 weighted | 0.6635 |
| GAT ECE | 0.0171 — mükemmel kalibre |
| Füzyon ECE | 0.1573 — sembolik deterministik kısıt |

Sınıf bazlı F1: NORM=0.80, CD=0.65, STTC=0.59, MI=0.49, HYP=0.46

## Güven Skoru Kuralı
- **Göster:** GAT-only softmax × T=1.0256 (ECE=0.0171 — kalibre)
- **Gösterme:** Füzyon olasılığını güven olarak sunma (ECE=0.1573 — yanıltıcı)
- **Ekle:** "Kalibrasyon notu: Güven skoru GAT katmanından alınmaktadır (ECE=0.017). Sembolik füzyon güven kalibrasyonunu etkiler." uyarısı
- Temperature dosyası: `models/checkpoints/temperature.pt` → {"temperature": 1.0256}

## 3 Sekme Layout

### Sekme 1 — EKG Yükle
- Girdi: wfdb formatı (.hea + .dat) veya .csv
- Ham sinyal: 12 lead, Plotly line chart, interaktif
- Lead seçici dropdown (varsayılan: II)
- Yükleme sonrası "Analiz Et" butonu aktif olur

### Sekme 2 — Analiz
- **Tanı:** En yüksek füzyon olasılıklı sınıf (NORM/MI/STTC/CD/HYP)
- **Güven skoru:** GAT-only (kalibre), yüzde olarak göster
- **AUC bazlı bant:**
  - ≥ 0.80 → Yüksek güven
  - 0.60–0.79 → Orta güven
  - < 0.60 → Düşük güven (klinisyen onayı önerilir)
- **Attention heatmap:** Beat bazlı GAT attention ağırlıkları, Plotly heatmap
- **Sınıf olasılıkları:** 5 sınıf bar chart (GAT + füzyon yan yana)

### Sekme 3 — Gerekçe
- Tetiklenen Prolog kuralları Türkçe listesi
- Her kural için: kural adı, açıklama, tetiklenme koşulu
- ECE uyarısı: "Füzyon güven kalibrasyonu sınırlıdır (ECE=0.157). Klinik karar için klinisyen değerlendirmesi gereklidir."
- Faithfulness@3 skoru (≈0.38 — attention'ın klinik anlamlılığı)

## Servis Yapısı
- Streamlit: port **8050** (`demo/app.py`)
- FastAPI: port **8000** (`demo/api.py`) — /predict endpoint
- Checkpointler: `models/checkpoints/best_model.pt`, `models/checkpoints/temperature.pt`

## Pipeline (Sekme 2 Analiz akışı)
```
wfdb kayıt
  → preprocessing.pipeline (beat segmentasyonu, 12-lead)
  → graph.builder (NaturalVG + DTW kenarları, PyG Data)
  → CardioGAT.forward() → logits, attention
  → TemperatureScaler.scale(logits) → kalibre GAT probs
  → SymbolicClassifier.predict() → sym_probs
  → NeuralSymbolicFusion.fuse(alpha=0.35) → final probs
  → Göster: tanı (füzyon), güven (GAT kalibre), attention heatmap
```

## Oturum Sonu
memory/demo_memory.json güncelle + 5 satır özet + dur
