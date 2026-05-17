# AGENT: preprocessing_agent
"""
data/processed/beats/ içeriğini özetle.
Büyük dosyaları Claude Code'a okutmadan JSON çıktı verir.
"""
import json
import numpy as np
from pathlib import Path


def summarize():
    beats_dir = Path("data/processed/beats")
    memory_file = Path("memory/preprocessing_memory.json")

    files = list(beats_dir.glob("beats_*.npy"))
    result = {
        "total_files": len(files),
        "avg_beats": None,
        "feature_dim": None,
        "open_issues": [],
        "last_decision": None,
    }

    if files:
        beat_counts = []
        for f in files[:20]:  # İlk 20'yi örnekle
            try:
                arr = np.load(f, mmap_mode="r")
                beat_counts.append(arr.shape[0])
                result["feature_dim"] = arr.shape[1] if arr.ndim > 1 else None
            except Exception:
                pass
        result["avg_beats"] = float(np.mean(beat_counts)) if beat_counts else None

    if memory_file.exists():
        with open(memory_file) as f:
            mem = json.load(f)
        result["open_issues"] = mem.get("open_issues", [])
        decisions = mem.get("decisions", [])
        result["last_decision"] = decisions[-1] if decisions else None

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize()
