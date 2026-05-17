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
