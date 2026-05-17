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
