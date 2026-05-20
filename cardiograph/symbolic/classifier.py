# AGENT: symbolic_agent
"""
SymbolicClassifier: PyG graf → 5-sınıf sembolik olasılık vektörü.

Feature eşlemesi (preprocessing/pipeline.py ile senkron — hibrit layout):
  Lead sırası : I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
  Özellikler  : mean std skewness kurtosis peak_to_peak energy zero_crossings
                ST_elevation Q_wave
  Toplam      : 12 × 9 = 108 özellik

Klinik proxy'ler (R-peak tabanlı basit hesaplama):
  ST elevasyonu  → max(ST_elevation_V2, ST_elevation_V3, ST_elevation_V4)
  Q dalgası      → min(Q_wave_II, Q_wave_III, Q_wave_aVF)
  T inversiyonu  → min(ST_elevation_V3..V5)  [R+120ms negatif → T inversion]
  Sokolow-Lyon   → peak_to_peak_V1 + peak_to_peak_V5
  aVL voltajı    → peak_to_peak_aVL
  ST çökmesi     → min(ST_elevation_II, ST_elevation_V4..V6)
  T değişikliği  → std(ST_elevation_V1..V6)  [prekordiyal morfoloji varyansı]
  QRS genişliği  → sabit 100ms (yeni layout'ta QRS timing özelliği yok)
"""
import numpy as np
from symbolic.rules import PrologEngine, FuzzyRuleEngine

# PTB-XL sınıf sırası (train.py CLASS_NAMES ile aynı)
CLASS_ORDER = ["norm", "mi", "sttc", "cd", "hyp"]

_LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_STAT = ["mean", "std", "skewness", "kurtosis", "peak_to_peak", "energy",
         "zero_crossings", "ST_elevation", "Q_wave"]


def _f(x: np.ndarray, lead: str, stat: str) -> float:
    return float(x[_LEAD.index(lead) * 9 + _STAT.index(stat)])


def extract_clinical_features(x: np.ndarray) -> dict:
    """
    108-dim beat özellik vektörü → klinik proxy özellik sözlüğü.
    x: (108,) veya (N_beats, 108) — ikincisi beat ortalaması alınır.
    """
    if x.ndim == 2:
        x = x.mean(axis=0)

    # ST elevasyonu: artık J+80ms'deki gerçek amplitüd (mV)
    st_elev = max(_f(x, "V2", "ST_elevation"), _f(x, "V3", "ST_elevation"),
                  _f(x, "V4", "ST_elevation"))

    # Patolojik Q dalgası: artık QRS_onset→R arası gerçek minimum (mV)
    q_wave = min(_f(x, "II", "Q_wave"), _f(x, "III", "Q_wave"), _f(x, "aVF", "Q_wave"))

    # T inversiyonu: J+80ms negatif amplitüd → T inversion (prekordiyal)
    t_inv = min(_f(x, "V3", "ST_elevation"), _f(x, "V4", "ST_elevation"),
                _f(x, "V5", "ST_elevation"))

    # Sokolow-Lyon: V1 + V5 peak-to-peak toplamı (voltaj proxy)
    sokolow = _f(x, "V1", "peak_to_peak") + _f(x, "V5", "peak_to_peak")

    # Cornell voltaj proxy: aVL peak-to-peak
    avl_r = _f(x, "aVL", "peak_to_peak")

    # ST çökmesi: inferior + lateral ST amplitüd minimumu
    st_dep = min(_f(x, "II", "ST_elevation"), _f(x, "V4", "ST_elevation"),
                 _f(x, "V5", "ST_elevation"), _f(x, "V6", "ST_elevation"))

    # T dalgası morfoloji değişikliği: prekordiyal ST_elevation varyansı
    st_prekordiyal = [_f(x, f"V{i}", "ST_elevation") for i in range(1, 7)]
    t_change = float(np.std(st_prekordiyal))

    # QRS süresi: yeni layout'ta QRS timing özelliği yok → sabit normal değer
    qrs_ms = 100.0

    return {
        "st_elevation_mv":    st_elev,
        "q_wave_mv":          q_wave,
        "t_inversion_score":  t_inv,
        "sokolow_lyon_mv":    sokolow,
        "avl_r_mv":           avl_r,
        "st_depression_mv":   st_dep,
        "t_amplitude_change": t_change,
        "qrs_duration_ms":    qrs_ms,
    }


class SymbolicClassifier:
    """
    108-dim beat özelliğini PTB-XL 5-sınıf sembolik olasılığa dönüştürür.

    Varsayılan engine: FuzzyRuleEngine (sigmoid tabanlı, sürekli confidence).
    ECE hedefi: 0.15 → 0.07.

    Kullanım::

        sc = SymbolicClassifier()
        probs = sc.predict(graph.x.numpy())   # (5,)
    """

    _BASELINE = 0.02   # her sınıf için minimum olasılık tabanı

    def __init__(self, engine=None):
        self.engine = engine if engine is not None else FuzzyRuleEngine()

    def _sym_probs(self, features: dict) -> np.ndarray:
        results = self.engine.query(features)

        # Her sınıf için maksimum confidence (birden fazla kural varsa en güçlüsü)
        class_conf = {cls: 0.0 for cls in CLASS_ORDER}
        for r in results:
            d = r["diagnosis"]
            if d in class_conf:
                class_conf[d] = max(class_conf[d], r["confidence"])

        scores = np.array([class_conf[c] for c in CLASS_ORDER], dtype=float)
        scores += self._BASELINE
        return scores / scores.sum()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : (N_beats, 108) veya (108,)

        Returns
        -------
        probs : (5,)  — CLASS_ORDER sırasında
        """
        features = extract_clinical_features(x)
        return self._sym_probs(features)

    def predict_batch(self, xs) -> np.ndarray:
        """xs: liste/dizi, her eleman (N_beats, 108) veya (108,). Döndürür: (N, 5)."""
        return np.stack([self.predict(x) for x in xs])


# Alias — açık isim tercih edildiğinde
FuzzySymbolicClassifier = SymbolicClassifier
