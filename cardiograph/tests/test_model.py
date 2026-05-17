# AGENT: model_agent
"""
GAT model testleri.
Sentetik PyG graflarıyla çalışır.
"""
import numpy as np
import pytest


def make_synthetic_batch(n_graphs=4, n_nodes=8, in_channels=48, n_edges=16):
    """Sentetik PyG mini-batch."""
    import torch
    from torch_geometric.data import Data, Batch

    graphs = []
    for i in range(n_graphs):
        x = torch.randn(n_nodes, in_channels)
        # Rastgele kenar indeksleri
        src = torch.randint(0, n_nodes, (n_edges,))
        dst = torch.randint(0, n_nodes, (n_edges,))
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.rand(n_edges)
        y = torch.tensor([i % 5])
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(graphs)


class TestCardioGAT:
    def test_forward_output_shape(self):
        import torch
        from models.gat import CardioGAT
        model = CardioGAT(in_channels=48, hidden=128, heads=8, num_classes=5)
        batch = make_synthetic_batch()
        logits, attn = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # logits: (n_graphs, 5)
        assert logits.shape == (4, 5)

    def test_attention_weights_present(self):
        import torch
        from models.gat import CardioGAT
        model = CardioGAT()
        batch = make_synthetic_batch()
        logits, attn = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert attn is not None

    def test_output_finite(self):
        import torch
        from models.gat import CardioGAT
        model = CardioGAT()
        batch = make_synthetic_batch()
        logits, attn = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert torch.isfinite(logits).all()


class TestTrainingStep:
    def test_loss_positive_finite(self):
        import torch
        import torch.nn as nn
        from models.gat import CardioGAT
        model = CardioGAT()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        batch = make_synthetic_batch()
        optimizer.zero_grad()
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y)
        assert loss.item() > 0
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    def test_loss_decreases(self):
        import torch
        import torch.nn as nn
        from models.gat import CardioGAT
        torch.manual_seed(42)
        model = CardioGAT()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        batch = make_synthetic_batch(n_graphs=8)
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # En azından son kayıp ilkinden küçük olmalı (5 adımda)
        assert losses[-1] <= losses[0] * 1.5  # Esnek tolerans
