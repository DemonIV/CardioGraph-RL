# AGENT: symbolic_agent
"""
SymbolicClassifier: PyG graf → 5-sınıf sembolik olasılık vektörü.

Feature eşlemesi (preprocessing/pipeline.py ile senkron):
  Lead sırası : I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
  İstatistik  : mean std skewness p2p kurtosis rms zcr spectral_centroid
  Toplam      : 12 × 8 = 96 özellik

Klinik proxy'ler:
  ST elevasyonu  → max(mean_V2, mean_V3, mean_V4)
  Q dalgası      → min(mean_II, mean_III, mean_aVF)
  T inversiyonu  → min(skewness_V3..V5)
  Sokolow-Lyon   → p2p_V1 + p2p_V5
  aVL voltajı    → rms_aVL
  ST çökmesi     → min(mean_II, mean_V4..V6)
  T değişikliği  → std(skewness_V1..V6)
  QRS genişliği  → 200 - zcr_V1 * 600  (ms, [60, 200])
"""
import numpy as np
from symbolic.rules import PrologEngine

# PTB-XL sınıf sırası (train.py CLASS_NAMES ile aynı)
CLASS_ORDER = ["norm", "mi", "sttc", "cd", "hyp"]

_LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_STAT = ["mean", "std", "skewness", "p2p", "kurtosis", "rms", "zcr", "sc"]


def _f(x: np.ndarray, lead: str, stat: str) -> float:
    return float(x[_LEAD.index(lead) * 8 + _STAT.index(stat)])


def extract_clinical_features(x: np.ndarray) -> dict:
    """
    96-dim beat özellik vektörü → klinik proxy özellik sözlüğü.
    x: (96,) veya (N_beats, 96) — ikincisi beat ortalaması alınır.
    """
    if x.ndim == 2:
        x = x.mean(axis=0)

    # ST elevasyonu proxy
    st_elev = max(_f(x, "V2", "mean"), _f(x, "V3", "mean"), _f(x, "V4", "mean"))

    # Patolojik Q dalgası proxy (inferior derivasyonlarda negatif mean)
    q_wave = min(_f(x, "II", "mean"), _f(x, "III", "mean"), _f(x, "aVF", "mean"))

    # T inversiyonu proxy (prekordiyal negatif skewness)
    t_inv = min(_f(x, "V3", "skewness"), _f(x, "V4", "skewness"), _f(x, "V5", "skewness"))

    # Sokolow-Lyon: S_V1 + R_V5 proxy (peak-to-peak toplamı)
    sokolow = _f(x, "V1", "p2p") + _f(x, "V5", "p2p")

    # Cornell voltaj proxy
    avl_r = _f(x, "aVL", "rms")

    # ST çökmesi proxy
    st_dep = min(_f(x, "II", "mean"), _f(x, "V4", "mean"),
                 _f(x, "V5", "mean"), _f(x, "V6", "mean"))

    # T dalgası morfoloji değişikliği (prekordiyal skewness yayılımı)
    skews = [_f(x, f"V{i}", "skewness") for i in range(1, 7)]
    t_change = float(np.std(skews))

    # QRS süre proxy: düşük ZCR → geniş QRS
    zcr_v1 = _f(x, "V1", "zcr")
    qrs_ms = float(np.clip(200.0 - zcr_v1 * 600.0, 60.0, 200.0))

    return {
        "st_elevation_mv":   st_elev,
        "q_wave_mv":         q_wave,
        "t_inversion_score": t_inv,
        "sokolow_lyon_mv":   sokolow,
        "avl_r_mv":          avl_r,
        "st_depression_mv":  st_dep,
        "t_amplitude_change": t_change,
        "qrs_duration_ms":   qrs_ms,
    }


class SymbolicClassifier:
    """
    96-dim beat özelliğini PTB-XL 5-sınıf sembolik olasılığa dönüştürür.

    Kullanım::

        sc = SymbolicClassifier()
        probs = sc.predict(graph.x.numpy())   # (5,)
    """

    # Kaç kuralın eşleştiği → olasılık tablosu
    _MATCH_SCORE   = 0.70   # en az 1 kural eşleşen sınıf için taban skor
    _BASELINE      = 0.05   # eşleşmeyen sınıf baseline'ı

    def __init__(self, engine: PrologEngine | None = None):
        self.engine = engine or PrologEngine()

    def _sym_probs(self, features: dict) -> np.ndarray:
        results = self.engine.query(features)

        # Her sınıf için eşleşen kural sayısı
        counts = {cls: 0 for cls in CLASS_ORDER}
        for r in results:
            d = r["diagnosis"]
            if d in counts:
                counts[d] += 1

        scores = np.array([counts[c] for c in CLASS_ORDER], dtype=float)
        total  = scores.sum()

        if total == 0:
            return np.ones(5) / 5.0

        # Eşleşen sınıflar: _MATCH_SCORE * (count/total), eşleşmeyenler: _BASELINE
        probs = np.where(
            scores > 0,
            self._MATCH_SCORE * (scores / total) + self._BASELINE,
            self._BASELINE,
        )
        return probs / probs.sum()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : (N_beats, 96) veya (96,)

        Returns
        -------
        probs : (5,)  — CLASS_ORDER sırasında
        """
        features = extract_clinical_features(x)
        return self._sym_probs(features)

    def predict_batch(self, xs) -> np.ndarray:
        """xs: liste/dizi, her eleman (N_beats, 96) veya (96,). Döndürür: (N, 5)."""
        return np.stack([self.predict(x) for x in xs])
