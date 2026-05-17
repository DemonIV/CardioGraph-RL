# AGENT: model_agent
"""
Graph Attention Network — aritmi sınıflandırma.
3 katman | 8 head | hidden=128 | dropout=0.3 | 5 sınıf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class CardioGAT(nn.Module):
    def __init__(self, in_channels=96, hidden=128, heads=8,
                 num_classes=5, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        per_head = hidden // heads  # 16 — concat=True → heads*16=128 çıkış

        self.conv1 = GATConv(in_channels, per_head, heads=heads,
                             dropout=dropout, edge_dim=1, concat=True)
        self.conv2 = GATConv(hidden, per_head, heads=heads,
                             dropout=dropout, edge_dim=1, concat=True)
        # Son katman: concat=False → heads ortalaması = hidden boyut
        self.conv3 = GATConv(hidden, hidden, heads=heads,
                             dropout=dropout, edge_dim=1, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        self.lin = nn.Linear(hidden, num_classes)

        # Giriş normalizasyonu
        self.input_bn = nn.BatchNorm1d(in_channels)

        # Residual projeksiyon: in_channels → hidden (boyut eşleştirme)
        self.res_proj = nn.Linear(in_channels, hidden, bias=False)  # 96 → 128

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Döndürür: (logits, attention_weights)
        attention_weights faithfulness hesabı için şart — silme.
        """
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # (E,) → (E, 1)

        x = self.input_bn(x)
        res = self.res_proj(x)                   # 12 → 128, residual için sakla

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(self.bn1(x)) + res             # residual: giriş + conv1 çıkışı

        res = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(self.bn2(x)) + res             # residual: conv1 + conv2 çıkışı

        res = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (_, attn) = self.conv3(x, edge_index, edge_attr=edge_attr,
                                   return_attention_weights=True)
        x = F.elu(self.bn3(x)) + res             # residual: conv2 + conv3 çıkışı

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lin(x)

        return logits, attn
