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
