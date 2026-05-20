# AGENT: symbolic_agent
"""
Prolog klinik kural motoru.
pyswip + SWI-Prolog varsa kullanır; yoksa saf Python RuleEngine'e düşer.
"""
from pathlib import Path
import numpy as np

_RULES_PL = Path(__file__).parent / "rules.pl"

# ── Saf Python kural tanımları (rules.pl ile senkron tutulmalı) ──────────────
# Her kural: (tanı, gerekçe_listesi, koşul_listesi)
# Koşul: (özellik_adı, operatör, eşik)
_PY_RULES = [
    # MI
    ("mi",   ["st_elevation"],   [("st_elevation_mv",   ">",  0.10)]),
    ("mi",   ["pathological_q"], [("q_wave_mv",         "<", -0.10)]),
    ("mi",   ["t_inversion"],    [("t_inversion_score", "<", -0.50)]),
    # HYP
    ("hyp",  ["sokolow_lyon"],   [("sokolow_lyon_mv",   ">",  3.50)]),
    ("hyp",  ["avl_voltage"],    [("avl_r_mv",          ">",  1.10)]),
    # STTC
    ("sttc", ["st_depression"],  [("st_depression_mv",  "<", -0.05)]),
    ("sttc", ["t_wave_change"],  [("t_amplitude_change",">",  0.50)]),
    # CD
    ("cd",   ["wide_qrs"],       [("qrs_duration_ms",   ">", 120.0)]),
    # AF / NSR (geriye dönük uyumluluk)
    ("atrial_fibrillation", ["irregular_rr", "absent_p_wave"],
     [("rr_variability", ">", 0.15), ("p_wave_amplitude", "<", 0.05)]),
    ("normal_sinus_rhythm", ["regular_rr", "normal_qrs"],
     [("rr_variability", "<", 0.05),
      ("qrs_duration_ms", "<", 100.0),
      ("p_wave_amplitude", ">", 0.10)]),
]

# NORM için kontrol edilecek patoloji özellikleri ve eşikleri
_NORM_CHECKS = [
    ("st_elevation_mv",   ">",  0.10),
    ("q_wave_mv",         "<", -0.10),
    ("sokolow_lyon_mv",   ">",  3.50),
    ("st_depression_mv",  "<", -0.05),
    ("qrs_duration_ms",   ">", 120.0),
]


# ── Fuzzy parametreleri ───────────────────────────────────────────────────────
# sigmoid 5%→95% geçiş aralığı = ±3 × sharpness (eşik etrafında)
_FUZZY_SHARPNESS: dict[tuple, float] = {
    ("st_elevation_mv",    ">",  0.10): 0.025,   # ±0.075 mV geçiş
    ("q_wave_mv",          "<", -0.10): 0.025,
    ("t_inversion_score",  "<", -0.50): 0.10,
    ("sokolow_lyon_mv",    ">",  3.50): 0.30,    # ±0.9 mV geçiş
    ("avl_r_mv",           ">",  1.10): 0.10,
    ("st_depression_mv",   "<", -0.05): 0.015,   # ±0.045 mV geçiş
    ("t_amplitude_change", ">",  0.50): 0.10,
    ("qrs_duration_ms",    ">", 120.0): 10.0,    # ±30 ms geçiş
    ("rr_variability",     ">",  0.15): 0.02,
    ("p_wave_amplitude",   "<",  0.05): 0.01,
    ("rr_variability",     "<",  0.05): 0.01,
    ("p_wave_amplitude",   ">",  0.10): 0.02,
    ("qrs_duration_ms",    "<", 100.0): 10.0,
}

# NORM yalnızca bu eşiğin üzerinde dahil edilir
_NORM_INCLUDE_THRESHOLD = 0.30


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))))


class FuzzyRuleEngine:
    """
    Sigmoid tabanlı fuzzy kural motoru.

    ">" kuralı: confidence = sigmoid((val − thresh) / sharpness)
    "<" kuralı: confidence = sigmoid((thresh − val) / sharpness)
    Çoklu koşul → min (soft AND).
    sharpness parametreleri öğrenilebilir: constructor'a dict geçilebilir.
    """

    def __init__(self, sharpness: dict | None = None):
        self.sharpness: dict[tuple, float] = dict(_FUZZY_SHARPNESS)
        if sharpness:
            self.sharpness.update(sharpness)

    def _cond_conf(self, features: dict, feat: str, op: str, thresh: float) -> float:
        val = features.get(feat)
        if val is None:
            return 0.0
        k = self.sharpness.get((feat, op, thresh), 0.10)
        if op == ">":
            return _sigmoid((val - thresh) / k)
        return _sigmoid((thresh - val) / k)

    def _rule_conf(self, features: dict, conditions: list) -> float:
        confs = [self._cond_conf(features, *c) for c in conditions]
        return min(confs) if confs else 0.0

    def query(self, clinical_features: dict) -> list:
        results = []
        for diag, reasons, conditions in _PY_RULES:
            conf = self._rule_conf(clinical_features, conditions)
            if conf > 1e-4:
                results.append({
                    "diagnosis":  diag,
                    "reasons":    list(reasons),
                    "confidence": float(conf),
                })

        # NORM: 1 − max(pathology_confidences)
        ptbxl_keys = {"st_elevation_mv", "q_wave_mv", "sokolow_lyon_mv",
                      "st_depression_mv", "qrs_duration_ms"}
        if ptbxl_keys & clinical_features.keys():
            path_confs = [
                self._cond_conf(clinical_features, feat, op, thresh)
                for feat, op, thresh in _NORM_CHECKS
                if clinical_features.get(feat) is not None
            ]
            if path_confs:
                norm_conf = float(1.0 - max(path_confs))
                if norm_conf > _NORM_INCLUDE_THRESHOLD:
                    results.append({
                        "diagnosis":  "norm",
                        "reasons":    ["normal_findings"],
                        "confidence": norm_conf,
                    })
        return results


class RuleEngine:
    """SWI-Prolog olmadan çalışan saf Python kural motoru."""

    def _check(self, features: dict, conditions: list) -> bool:
        for feat, op, thresh in conditions:
            val = features.get(feat)
            if val is None:
                return False
            if op == ">" and not (val > thresh):
                return False
            if op == "<" and not (val < thresh):
                return False
        return True

    def query(self, clinical_features: dict) -> list:
        results = []
        for diag, reasons, conditions in _PY_RULES:
            if self._check(clinical_features, conditions):
                results.append({
                    "diagnosis": diag,
                    "reasons":   list(reasons),
                    "confidence": 1.0,
                })

        # NORM: patoloji bulgusu yoksa ve ilgili özellikler mevcutsa
        ptbxl_keys = {"st_elevation_mv", "q_wave_mv", "sokolow_lyon_mv",
                      "st_depression_mv", "qrs_duration_ms"}
        has_ptbxl = ptbxl_keys & clinical_features.keys()
        if has_ptbxl:
            pathology = any(
                self._check(clinical_features, [c])
                for c in _NORM_CHECKS
                if clinical_features.get(c[0]) is not None
            )
            if not pathology:
                results.append({
                    "diagnosis":  "norm",
                    "reasons":    ["normal_findings"],
                    "confidence": 1.0,
                })
        return results


class PrologEngine:
    """
    Klinik kural motoru.
    SWI-Prolog kurulu → pyswip kullanır.
    Kurulu değil → RuleEngine fallback.
    """

    def __init__(self, rules_path: str | None = None):
        self._fallback = None
        rp = Path(rules_path) if rules_path else _RULES_PL
        try:
            from pyswip import Prolog
            self._prolog = Prolog()
            self._prolog.consult(str(rp.resolve()))
        except Exception:
            self._prolog = None
            self._fallback = RuleEngine()

    def query(self, clinical_features: dict) -> list:
        if self._fallback is not None:
            return self._fallback.query(clinical_features)
        return self._prolog_query(clinical_features)

    def _prolog_query(self, clinical_features: dict) -> list:
        list(self._prolog.query("retractall(feature(_, _))"))
        for name, val in clinical_features.items():
            if isinstance(val, bool):
                atom = "true" if val else "false"
                self._prolog.assertz(f"feature({name}, {atom})")
            else:
                self._prolog.assertz(f"feature({name}, {float(val)})")

        results, seen = [], set()
        for sol in self._prolog.query("diagnosis(Class, Reasons)"):
            diag    = str(sol["Class"])
            reasons = [str(r) for r in sol["Reasons"]]
            key     = (diag, tuple(reasons))
            if key not in seen:
                seen.add(key)
                results.append({
                    "diagnosis":  diag,
                    "reasons":    reasons,
                    "confidence": 1.0,
                })
        return results
