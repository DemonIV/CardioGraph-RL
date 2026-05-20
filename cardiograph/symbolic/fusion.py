# AGENT: symbolic_agent
"""
Nöro-sembolik füzyon katmanı.
final = gat_prob * (1 - alpha) + sym_prob * alpha

NeuralSymbolicFusion  — tek global alpha (geriye dönük uyumlu)
AdaptiveFusion        — sınıf başına farklı alpha (öğrenilebilir)
"""
import math
import numpy as np


class NeuralSymbolicFusion:
    """
    GAT softmax çıkışı ile sembolik kural olasılığını birleştirir.

    Parameters
    ----------
    alpha : float
        Sembolik ağırlık. 0 → tamamen GAT, 1 → tamamen sembolik.
        Başlangıç: 0.35 (proje kararı).
    """

    def __init__(self, alpha: float = 0.35):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha [0,1] aralığında olmalı, verildi: {alpha}")
        self.alpha = alpha

    def fuse(self, gat_probs: np.ndarray, sym_probs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        gat_probs : ndarray (N_classes,)  GAT softmax çıkışı
        sym_probs : ndarray (N_classes,)  Sembolik sınıf olasılıkları

        Returns
        -------
        ndarray (N_classes,)  Normalize edilmiş füzyon olasılığı
        """
        gat_probs = np.asarray(gat_probs, dtype=np.float64)
        sym_probs = np.asarray(sym_probs, dtype=np.float64)

        fused = gat_probs * (1.0 - self.alpha) + sym_probs * self.alpha
        total = fused.sum()
        if total < 1e-12:
            return np.ones_like(fused) / len(fused)
        return fused / total

    def fuse_batch(self, gat_batch: np.ndarray,
                   sym_batch: np.ndarray) -> np.ndarray:
        """Toplu füzyon. Her satır bir örnek."""
        return np.stack([
            self.fuse(g, s) for g, s in zip(gat_batch, sym_batch)
        ])


class AdaptiveFusion:
    """
    Sınıf başına farklı alpha ile nöro-sembolik füzyon.

    Her sınıf için bağımsız alpha — örn. MI için alpha≈0 (GAT baskın),
    HYP için alpha>0.35 (sembolik kural yardımcı).

    Parameters
    ----------
    alphas : array-like (5,) veya None
        Her sınıf için sembolik ağırlık [0,1].
        None → [0.35] * 5 (global baseline ile aynı başlangıç)
    class_names : list[str], opsiyonel
        Sınıf isimleri (repr/kayıt için).
    """
    N_CLASSES = 5

    def __init__(self, alphas=None, class_names=None):
        if alphas is None:
            self.alphas = np.full(self.N_CLASSES, 0.35)
        else:
            self.alphas = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 1.0)
            if len(self.alphas) != self.N_CLASSES:
                raise ValueError(
                    f"alphas uzunluğu {self.N_CLASSES} olmalı, verildi: {len(self.alphas)}"
                )
        self.class_names = class_names or ["NORM", "MI", "STTC", "CD", "HYP"]

    def fuse(self, gat_probs: np.ndarray, sym_probs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        gat_probs : (N_classes,)
        sym_probs : (N_classes,)

        Returns
        -------
        fused : (N_classes,)  normalize edilmiş
        """
        gat_probs = np.asarray(gat_probs, dtype=np.float64)
        sym_probs = np.asarray(sym_probs, dtype=np.float64)

        fused = gat_probs * (1.0 - self.alphas) + sym_probs * self.alphas
        total = fused.sum()
        if total < 1e-12:
            return np.ones_like(fused) / len(fused)
        return fused / total

    def fuse_batch(self, gat_batch: np.ndarray, sym_batch: np.ndarray) -> np.ndarray:
        """Toplu füzyon. Her satır bir örnek."""
        return np.stack([self.fuse(g, s) for g, s in zip(gat_batch, sym_batch)])

    def save(self, path) -> None:
        """Learned alphas'ı .npy olarak kaydet."""
        np.save(str(path), self.alphas)

    @classmethod
    def load(cls, path, class_names=None):
        """Kaydedilmiş alphas'ı yükle."""
        alphas = np.load(str(path))
        return cls(alphas=alphas, class_names=class_names)

    def __repr__(self):
        pairs = ", ".join(f"{n}={a:.3f}" for n, a in zip(self.class_names, self.alphas))
        return f"AdaptiveFusion({pairs})"


def learn_alphas(
    gat_probs: np.ndarray,
    sym_probs: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.5,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Validation setinde per-class alpha'yı NLL minimize ederek öğren.

    sigmoid parametrizasyonu ile [0,1] kısıtını garantiler.
    Başlangıç: sigmoid(logit(0.35)) = 0.35 (global baseline).

    Parameters
    ----------
    gat_probs : (N, 5)  GAT softmax olasılıkları
    sym_probs : (N, 5)  Sembolik sınıf olasılıkları
    labels    : (N,)    Gerçek sınıf indeksleri
    lr        : LBFGS öğrenme hızı
    max_iter  : LBFGS maksimum iterasyon

    Returns
    -------
    alphas : (5,)  — her sınıf için optimal alpha
    """
    import torch
    import torch.nn.functional as F

    gat_t    = torch.tensor(gat_probs, dtype=torch.float32)
    sym_t    = torch.tensor(sym_probs, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    # sigmoid(logit(0.35)) = 0.35 → global baseline'dan başla
    init_raw = math.log(0.35 / 0.65)
    raw = torch.tensor([init_raw] * 5, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.LBFGS([raw], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        alphas = torch.sigmoid(raw)                                    # (5,)
        fused  = gat_t * (1.0 - alphas) + sym_t * alphas              # (N, 5)
        fused  = fused / fused.sum(dim=1, keepdim=True).clamp(min=1e-12)
        loss   = F.nll_loss(torch.log(fused.clamp(min=1e-10)), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.sigmoid(raw).detach().numpy()
