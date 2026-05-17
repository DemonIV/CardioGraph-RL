# AGENT: symbolic_agent
"""
Prolog klinik kural motoru.
pyswip + SWI-Prolog varsa kullanır; yoksa saf Python RuleEngine'e düşer.
"""
from pathlib import Path

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
