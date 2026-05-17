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
