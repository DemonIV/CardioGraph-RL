# AGENT: graph_agent
"""
data/processed/graphs/ içeriğini özetle.
"""
import json
from pathlib import Path
from collections import Counter


def summarize():
    graphs_dir = Path("data/processed/graphs")
    memory_file = Path("memory/graph_memory.json")

    files = list(graphs_dir.glob("graph_*.pt"))
    result = {
        "total_graphs": len(files),
        "avg_nodes": None,
        "avg_edges": None,
        "class_distribution": {},
        "open_issues": [],
    }

    if files:
        try:
            import torch
            node_counts, edge_counts, labels = [], [], []
            for f in files[:50]:
                data = torch.load(f, weights_only=True)
                node_counts.append(data.x.shape[0])
                edge_counts.append(data.edge_index.shape[1])
                if hasattr(data, "y"):
                    labels.append(int(data.y))
            result["avg_nodes"] = float(sum(node_counts) / len(node_counts))
            result["avg_edges"] = float(sum(edge_counts) / len(edge_counts))
            result["class_distribution"] = dict(Counter(labels))
        except Exception as e:
            result["error"] = str(e)

    if memory_file.exists():
        with open(memory_file) as f:
            mem = json.load(f)
        result["open_issues"] = mem.get("open_issues", [])

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    summarize()
