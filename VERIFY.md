# VERIFY.md — Doğrula ve Rapor Ver

## Adım 1 — Dosya sayısını kontrol et
```bash
find ~/cardiograph -type f | grep -v __pycache__ | sort
```

Şunların hepsi var mı?
- [ ] CLAUDE.md
- [ ] .mcp.json
- [ ] requirements.txt
- [ ] ORCHESTRATOR_PROMPT.md
- [ ] memory/ altında 7 JSON dosyası
- [ ] preprocessing/, graph/, models/, symbolic/, eval/, demo/ altında AGENT_PROMPT.md
- [ ] preprocessing/pipeline.py, graph/builder.py, models/gat.py, models/train.py
- [ ] symbolic/rules.py, symbolic/rules.pl, eval/metrics.py
- [ ] tests/ altında 6 dosya
- [ ] tools/ altında 3 dosya

## Adım 2 — Syntax kontrolü
```bash
cd ~/cardiograph
python -m py_compile preprocessing/pipeline.py && echo "preprocessing OK"
python -m py_compile graph/builder.py && echo "graph OK"
python -m py_compile models/gat.py && echo "model OK"
python -m py_compile symbolic/rules.py && echo "symbolic OK"
python -m py_compile eval/metrics.py && echo "eval OK"
```

## Adım 3 — Test iskeletlerini çalıştır
```bash
pytest tests/ -q --tb=short --ignore-glob="*test_model*" 2>&1 | tail -15
```
(torch-geometric kurulu değilse model testleri skip olabilir — normal)

## Adım 4 — .mcp.json yolunu düzelt
```bash
GERCEK_YOL=$(pwd)
sed -i "s|TAM_YOL|$GERCEK_YOL|g" .mcp.json
cat .mcp.json | python -m json.tool > /dev/null && echo "MCP JSON geçerli"
```

## Adım 5 — Raporu yaz

Tam olarak şu formatı kullan:

```
╔══════════════════════════════════════════════════╗
║         CardioGraph-RL Kurulum Raporu            ║
╠══════════════════════════════════════════════════╣
║ Toplam dosya     : __                            ║
║ Memory dosyaları : __ / 7                        ║
║ Ajan promptları  : __ / 7                        ║
║ Kaynak dosyaları : __ / 10                       ║
║ Test dosyaları   : __ / 6                        ║
║ Syntax hatası    : yok / var (hangi dosya)       ║
║ MCP config       : geçerli / hatalı              ║
╠══════════════════════════════════════════════════╣
║ SONRAKI 3 ADIM                                   ║
║ 1. pip install -r requirements.txt               ║
║ 2. PTB-XL indir → data/raw/ptbxl/ altına koy    ║
║ 3. Yeni oturumda preprocessing/AGENT_PROMPT.md   ║
║    içeriğini yapıştır → preprocessing başlar     ║
╚══════════════════════════════════════════════════╝
```

Kurulum tamamlandı. Bu master prompt serisini bir daha kullanma.
Bundan sonra her oturum için yalnızca ilgili AGENT_PROMPT.md yeterli.
