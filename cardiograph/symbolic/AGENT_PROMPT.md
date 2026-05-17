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
