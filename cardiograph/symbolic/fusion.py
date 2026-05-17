# AGENT: symbolic_agent
"""
Nöro-sembolik füzyon katmanı.
final = gat_prob * (1 - alpha) + sym_prob * alpha
"""
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
