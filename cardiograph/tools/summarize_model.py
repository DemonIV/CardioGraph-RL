# AGENT: model_agent
"""
models/checkpoints/ içindeki en iyi checkpoint'i özetle.
"""
import json
from pathlib import Path


def summarize():
    ckpt_dir = Path("models/checkpoints")
    memory_file = Path("memory/model_memory.json")

    result = {
        "val_f1": None,
        "epoch": None,
        "architecture": None,
        "open_issues": [],
        "checkpoints_found": 0,
    }

    ckpt_files = list(ckpt_dir.glob("*.pt"))
    result["checkpoints_found"] = len(ckpt_files)

    if ckpt_files:
        try:
            import torch
            best_f1, best_ckpt = -1, None
            for f in ckpt_files:
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
                f1 = ckpt.get("val_f1", -1)
                if f1 > best_f1:
                    best_f1, best_ckpt = f1, ckpt
            if best_ckpt:
                result["val_f1"] = best_ckpt.get("val_f1")
                result["epoch"] = best_ckpt.get("epoch")
                result["architecture"] = best_ckpt.get("model_config")
        except Exception as e:
            result["error"] = str(e)

    if memory_file.exists():
        with open(memory_file) as f:
            mem = json.load(f)
        result["open_issues"] = mem.get("open_issues", [])

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize()
