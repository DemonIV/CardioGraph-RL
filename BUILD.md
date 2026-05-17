# BUILD.md — Dosyaları Oluştur

Aşağıdaki her dosyayı sırayla oluştur.
Her dosyadan sonra bir sonrakine geç, onay bekleme.

---

## GRUP A — Konfigürasyon (4 dosya)

### A1: .gitignore
```
data/raw/
data/processed/
models/checkpoints/
__pycache__/
*.pyc
.env
```

### A2: requirements.txt
```
wfdb>=4.1.0
neurokit2>=0.2.9
scipy>=1.12.0
numpy>=1.26.0
torch>=2.2.0
torch-geometric>=2.5.0
ts2vg>=1.2.2
dtaidistance>=2.3.10
pyswip>=0.2.10
scikit-fuzzy>=0.4.2
scikit-learn>=1.4.0
fastapi>=0.110.0
uvicorn>=0.29.0
dash>=2.16.0
plotly>=5.20.0
pytest>=8.0.0
```

### A3: CLAUDE.md (proje anayasası — değiştirme)
```markdown
# CardioGraph-RL Proje Anayasası

## Mutlak Kurallar
1. Test: pytest tests/test_[modül].py -q --tb=short
2. Yeni dosya → ilk satır: # AGENT: <ajan_adı>
3. Kendi klasörün dışına çıkma
4. Büyük dosya okuma → tools/summarize_*.py kullan
5. Sıra: preprocessing→graph→model→symbolic→eval→demo
6. Oturum sonu: memory JSON güncelle + 5 satır özet + dur

## Ajan-Klasör Eşlemesi
preprocessing_agent → preprocessing/
graph_agent         → graph/
model_agent         → models/
symbolic_agent      → symbolic/
eval_agent          → eval/
demo_agent          → demo/
```

### A4: .mcp.json
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "TAM_YOL"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```
TAM_YOL'u `pwd` çıktısıyla değiştir.

---

## GRUP B — Memory dosyaları (7 dosya)

Her birini memory/ altına oluştur:

### B1: memory/orchestrator_memory.json
```json
{
  "current_phase": "preprocessing",
  "agent_status": {
    "preprocessing_agent": "active",
    "graph_agent": "waiting",
    "model_agent": "waiting",
    "symbolic_agent": "waiting",
    "eval_agent": "waiting",
    "demo_agent": "waiting"
  },
  "completed_tasks": [],
  "global_decisions": []
}
```

### B2: memory/preprocessing_memory.json
```json
{
  "current_task": "Pipeline kurulumu",
  "decisions": [],
  "current_metrics": {"records_processed": 0},
  "open_issues": [
    "wfdb ile PTB-XL okuma",
    "Bandpass filtre (0.5-40 Hz)",
    "Pan-Tompkins R-peak tespiti",
    "Beat segmentasyonu (R ±200 sample)",
    "12 boyutlu morfoloji çıkarımı",
    "data/processed/beats/ altına kaydetme"
  ],
  "completed": [],
  "output_contract": {
    "file": "data/processed/beats/beats_{id}.npy",
    "shape": "[N_beats, 12]",
    "features": ["P_amp","P_dur","QRS_amp","QRS_dur","T_amp","T_dur",
                 "RR_interval","ST_slope","PR_interval","QT_interval",
                 "heart_rate","RR_variability"]
  }
}
```

### B3: memory/graph_memory.json
```json
{
  "current_task": "Beklemede — preprocessing tamamlanınca başlar",
  "decisions": [],
  "current_metrics": {"graphs_built": 0},
  "open_issues": [
    "ts2vg visibility graph",
    "DTW kenar ağırlıkları",
    "PyG Data nesnesi",
    "data/processed/graphs/ kaydetme"
  ],
  "completed": []
}
```

### B4: memory/model_memory.json
```json
{
  "current_task": "Beklemede",
  "decisions": [],
  "current_metrics": {"val_f1": null, "best_epoch": null},
  "open_issues": [
    "GAT mimarisi (3 katman, 8 head, hidden=128)",
    "PTB-XL 5 sınıf etiket eşlemesi",
    "Eğitim döngüsü + erken durdurma",
    "Attention skorlarını checkpoint'e kaydetme"
  ],
  "completed": []
}
```

### B5: memory/symbolic_memory.json
```json
{
  "current_task": "Beklemede",
  "decisions": [],
  "active_rules": 0,
  "open_issues": [
    "50 AHA/ESC Prolog kuralı",
    "pyswip entegrasyonu",
    "scikit-fuzzy füzyon (alpha=0.35)"
  ],
  "completed": []
}
```

### B6: memory/eval_memory.json
```json
{
  "current_task": "Beklemede",
  "decisions": [],
  "current_metrics": {},
  "open_issues": [
    "F1 macro/weighted, AUC-ROC",
    "Faithfulness@K (K=1,3,5)",
    "ECE kalibrasyon skoru",
    "MIT-BIH cross-dataset testi"
  ],
  "completed": []
}
```

### B7: memory/demo_memory.json
```json
{
  "current_task": "Beklemede",
  "decisions": [],
  "open_issues": [
    "Plotly Dash 3 sekme layout",
    "EKG sinyal görselleştirme",
    "Attention heatmap",
    "Prolog gerekçe paneli"
  ],
  "completed": []
}
```

---

## GRUP C — Ajan promptları (7 dosya)

### C1: ORCHESTRATOR_PROMPT.md (kök klasörde)
```markdown
# Orkestratör — Her Ajan Geçişinde Kullan

1. memory/orchestrator_memory.json oku
2. current_phase'e bak → o ajanın AGENT_PROMPT.md'sini ver
3. Ajan "BAĞIMLILIK: evet" dönünce → orchestrator_memory güncelle → sıradaki ajana geç
4. Bir seferde tek ajan. Alt ajan çıktısını asla ham okuma, sadece 5 satır özet al.
```

### C2: preprocessing/AGENT_PROMPT.md
```markdown
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
```

### C3: graph/AGENT_PROMPT.md
```markdown
# graph_agent

## Oturum Başı
1. memory/graph_memory.json oku
2. python tools/summarize_preprocessing.py (giriş verisini anla)
3. pytest tests/test_graph.py -q --tb=short

## Kapsam
YALNIZCA: graph/ ve tests/test_graph.py

## Algoritma (değiştirme)
beats_{id}.npy → ts2vg.NaturalVisibilityGraph → kenar ağırlığı: 1/(1+DTW) → PyG Data → graph_{id}.pt

## Oturum Sonu
memory/graph_memory.json güncelle + 5 satır özet + dur
```

### C4: models/AGENT_PROMPT.md
```markdown
# model_agent

## Oturum Başı
1. memory/model_memory.json oku
2. python tools/summarize_graph.py
3. pytest tests/test_model.py -q --tb=short

## Kapsam
YALNIZCA: models/ ve tests/test_model.py

## GAT Mimarisi (değiştirme)
3x GATConv | 8 head | hidden=128 | dropout=0.3 | global mean pool | 5 sınıf
forward() → (logits, attention_weights) döndür  ← faithfulness için şart

## Hedef
Val F1 ≥ 0.85 — geçemezsen open_issues'a yaz, durma

## Oturum Sonu
memory/model_memory.json güncelle + 5 satır özet + dur
```

### C5: symbolic/AGENT_PROMPT.md
```markdown
# symbolic_agent

## Oturum Başı
1. memory/symbolic_memory.json oku
2. python tools/summarize_model.py
3. pytest tests/test_symbolic.py -q --tb=short

## Kapsam
YALNIZCA: symbolic/ ve tests/test_symbolic.py

## Prolog Kural Formatı
diagnosis(sinif, [gerekce]) :- klinik_ozellik Operator Esik.

## Füzyon
final = gat_prob*(1-alpha) + symbolic_prob*alpha  (alpha=0.35 başlangıç)

## Oturum Sonu
memory/symbolic_memory.json güncelle + 5 satır özet + dur
```

### C6: eval/AGENT_PROMPT.md
```markdown
# eval_agent

## Oturum Başı
1. memory/eval_memory.json oku
2. python tools/summarize_model.py
3. pytest tests/test_eval.py -q --tb=short

## Kapsam
YALNIZCA: eval/ ve tests/test_eval.py

## Metrikler
Standart: F1 macro, AUC-ROC, confusion matrix
Özgün: Faithfulness@K = |top_k_attention ∩ patolojik| / K
Kalibrasyon: ECE (n_bins=10)

## Oturum Sonu
memory/eval_memory.json güncelle + 5 satır özet + dur
```

### C7: demo/AGENT_PROMPT.md
```markdown
# demo_agent

## Oturum Başı
1. memory/demo_memory.json oku
2. FastAPI sağlık kontrolü: curl localhost:8000/health
3. pytest tests/test_demo.py -q --tb=short

## Kapsam
YALNIZCA: demo/ ve tests/test_demo.py

## 3 Sekme
1. EKG Yükle → ham sinyal çizimi (12 lead, Plotly)
2. Analiz → tanı + güven skoru + attention heatmap
3. Gerekçe → Prolog kuralları Türkçe açıklama

## Oturum Sonu
memory/demo_memory.json güncelle + 5 satır özet + dur
```

---

## GRUP D — Kaynak kod iskeletleri (10 dosya)

Her dosyayı oluştur. Fonksiyon imzaları ve docstring'ler tam olsun,
implementasyon ilgili ajan tarafından doldurulacak.

### D1: preprocessing/__init__.py
```python
# AGENT: preprocessing_agent
from .pipeline import process_record, process_batch
```

### D2: preprocessing/pipeline.py
```python
# AGENT: preprocessing_agent
"""
EKG sinyal önişleme pipeline.
Giriş: wfdb kayıt yolu
Çıkış: data/processed/beats/beats_{id}.npy — shape (N_beats, 12)
"""
import numpy as np
import wfdb
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from pathlib import Path

SAMPLING_RATE = 500
BEAT_WINDOW = 200
N_FEATURES = 12

def bandpass_filter(signal, fs=SAMPLING_RATE):
    """0.5-40 Hz Butterworth bandpass, 4. derece."""
    raise NotImplementedError

def detect_r_peaks(signal_1d, fs=SAMPLING_RATE):
    """neurokit2 Pan-Tompkins. Döndürür: sample index dizisi."""
    raise NotImplementedError

def segment_beats(signal, r_peaks):
    """R ±BEAT_WINDOW sample. Döndürür: (N_beats, 2*BEAT_WINDOW, 12)."""
    raise NotImplementedError

def extract_morphology(beat, rr_intervals, beat_idx, fs=SAMPLING_RATE):
    """12 boyutlu öznitelik vektörü. NaN/Inf içermemeli."""
    raise NotImplementedError

def process_record(record_path, output_dir):
    """Tek kayıt işle, .npy kaydet. Döndürür: dict veya None."""
    raise NotImplementedError

def process_batch(record_list, output_dir, max_records=None):
    """Toplu işle. Döndürür: {processed, failed, results}."""
    raise NotImplementedError
```

### D3: graph/__init__.py
```python
# AGENT: graph_agent
from .builder import build_graph, build_graph_batch
```

### D4: graph/builder.py
```python
# AGENT: graph_agent
"""
EKG beat dizisinden PyG graf nesnesi oluşturma.
Giriş: data/processed/beats/beats_{id}.npy
Çıkış: data/processed/graphs/graph_{id}.pt
"""
import numpy as np
import torch
from torch_geometric.data import Data
from ts2vg import NaturalVisibilityGraph
from pathlib import Path

def build_visibility_graph(beats_array):
    """ts2vg ile düğüm/kenar. Döndürür: (edge_index, node_features)."""
    raise NotImplementedError

def compute_dtw_weights(beats_array, edge_index):
    """Kenar ağırlığı = 1 / (1 + DTW). Döndürür: edge_attr tensor."""
    raise NotImplementedError

def build_graph(record_id, beats_path, output_dir, label):
    """Tam pipeline. Döndürür: PyG Data nesnesi."""
    raise NotImplementedError

def build_graph_batch(processed_dir, output_dir, labels_df):
    """Toplu graf oluşturma."""
    raise NotImplementedError
```

### D5: models/__init__.py
```python
# AGENT: model_agent
from .gat import CardioGAT
from .train import train_model, evaluate_model
```

### D6: models/gat.py
```python
# AGENT: model_agent
"""
Graph Attention Network — aritmi sınıflandırma.
3 katman | 8 head | hidden=128 | dropout=0.3 | 5 sınıf
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class CardioGAT(nn.Module):
    def __init__(self, in_channels=12, hidden=128, heads=8,
                 num_classes=5, dropout=0.3):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Döndürür: (logits, attention_weights)
        attention_weights faithfulness hesabı için şart — silme.
        """
        raise NotImplementedError
```

### D7: models/train.py
```python
# AGENT: model_agent
"""Eğitim döngüsü ve değerlendirme."""

def train_model(model, train_loader, val_loader, config):
    """Checkpoint: {epoch, model_state_dict, val_f1, attention_scores, model_config}"""
    raise NotImplementedError

def evaluate_model(model, loader):
    """Döndürür: {f1_macro, auc_roc, attention_scores}"""
    raise NotImplementedError
```

### D8: symbolic/rules.py
```python
# AGENT: symbolic_agent
"""Prolog klinik kural motoru."""
from pyswip import Prolog

class PrologEngine:
    def __init__(self, rules_path="symbolic/rules.pl"):
        raise NotImplementedError

    def query(self, clinical_features: dict):
        """Döndürür: [{diagnosis, reasons, confidence}]"""
        raise NotImplementedError
```

### D9: eval/metrics.py
```python
# AGENT: eval_agent
"""Standart + özgün metrikler."""

def compute_all_metrics(y_true, y_pred, y_prob):
    """F1 macro/weighted, AUC-ROC, confusion matrix."""
    raise NotImplementedError

def faithfulness_at_k(attention_scores, pathological_labels, k=3):
    """
    Özgün metrik: top-K attention atımlarının patolojik olma oranı.
    faithfulness@K = |top_k ∩ patolojik| / K
    """
    raise NotImplementedError

def expected_calibration_error(y_prob, y_true, n_bins=10):
    """ECE kalibrasyon skoru."""
    raise NotImplementedError
```

### D10: symbolic/rules.pl
```prolog
% AGENT: symbolic_agent
% AHA/ESC kılavuzundan temel klinik kurallar
% symbolic_agent daha fazlasını ekleyecek

diagnosis(atrial_fibrillation, [irregular_rr, absent_p_wave]) :-
    rr_variability > 0.15,
    p_wave_amplitude < 0.05.

diagnosis(left_bundle_branch_block, [wide_qrs, notched_r]) :-
    qrs_duration_ms > 120,
    qrs_amplitude_v6 > 0.5.

diagnosis(normal_sinus_rhythm, [regular_rr, normal_qrs]) :-
    rr_variability < 0.05,
    qrs_duration_ms < 100,
    p_wave_amplitude > 0.1.
```

---

## GRUP E — Test iskeletleri (6 dosya)

Her test dosyasını oluştur. Sentetik EKG kullan, gerçek veri gerekmez.

### E1: tests/__init__.py (boş)

### E2: tests/test_preprocessing.py
Şu sınıfları içersin:
- TestBandpassFilter: shape (1D/2D), frekans bastırma
- TestRPeakDetection: sentetik EKG, peak sayısı ≥8, sınır kontrolü
- TestBeatSegmentation: shape, sınır dışı erişim yok
- TestMorphologyExtraction: shape==(12,), NaN/Inf yok, HR 40-120
- Yardımcı: make_synthetic_ecg(duration_s, fs, n_leads, hr_bpm)

### E3: tests/test_graph.py
Şu sınıfları içersin:
- TestVisibilityGraph: düğüm sayısı, kenar bağlantılılığı
- TestDTWWeights: ağırlık aralığı [0,1], simetri kontrolü
- TestPyGData: x/edge_index/edge_attr shape doğrulama

### E4: tests/test_model.py
Şu sınıfları içersin:
- TestCardioGAT: forward pass shape, attention çıktısı varlığı
- TestTrainingStep: tek batch kaybı pozitif ve sonlu

### E5: tests/test_symbolic.py
Şu sınıfları içersin:
- TestPrologEngine: bilinen özelliklerle AF tespiti
- TestFusion: çıktı toplamı ≈ 1.0, alpha aralığı

### E6: tests/test_eval.py
Şu sınıfları içersin:
- TestFaithfulness: bilinen attention ile @K hesabı
- TestCalibration: mükemmel model ECE ≈ 0

---

## GRUP F — Araç betikleri (3 dosya)

### F1: tools/summarize_preprocessing.py
data/processed/beats/ özetle → JSON çıktı:
{total_files, avg_beats, feature_dim, open_issues, last_decision}

### F2: tools/summarize_graph.py
data/processed/graphs/ özetle → JSON çıktı:
{total_graphs, avg_nodes, avg_edges, class_distribution}

### F3: tools/summarize_model.py
models/checkpoints/ en iyi checkpoint özetle → JSON çıktı:
{val_f1, epoch, architecture, open_issues}

---

Tüm gruplar tamamlandığında VERIFY.md'yi iste.
