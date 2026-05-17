# Orkestratör — Her Ajan Geçişinde Kullan

1. memory/orchestrator_memory.json oku
2. current_phase'e bak → o ajanın AGENT_PROMPT.md'sini ver
3. Ajan "BAĞIMLILIK: evet" dönünce → orchestrator_memory güncelle → sıradaki ajana geç
4. Bir seferde tek ajan. Alt ajan çıktısını asla ham okuma, sadece 5 satır özet al.
