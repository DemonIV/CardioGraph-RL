# AGENT: symbolic_agent
"""
Sembolik kural motoru ve füzyon testleri.
"""
import numpy as np
import pytest


# ── PrologEngine / RuleEngine testleri ──────────────────────────────────────

class TestPrologEngine:
    def test_af_detection(self):
        """Bilinen AF özellikleriyle atriyal fibrilasyon tespiti."""
        from symbolic.rules import PrologEngine
        engine = PrologEngine()
        features = {
            "rr_variability":   0.20,   # > 0.15 → AF
            "p_wave_amplitude": 0.02,   # < 0.05 → AF
            "qrs_duration_ms":  90,
        }
        results = engine.query(features)
        diagnoses = [r["diagnosis"] for r in results]
        assert "atrial_fibrillation" in diagnoses

    def test_normal_detection(self):
        """Normal sinüs ritmi özellikleri."""
        from symbolic.rules import PrologEngine
        engine = PrologEngine()
        features = {
            "rr_variability":    0.03,
            "p_wave_amplitude":  0.15,
            "qrs_duration_ms":   85,
            "qrs_amplitude_v6":  0.3,
        }
        results = engine.query(features)
        diagnoses = [r["diagnosis"] for r in results]
        assert "normal_sinus_rhythm" in diagnoses

    def test_returns_reasons(self):
        """Her sonuç reasons listesi içermeli."""
        from symbolic.rules import PrologEngine
        engine = PrologEngine()
        features = {"rr_variability": 0.20, "p_wave_amplitude": 0.02}
        results = engine.query(features)
        assert len(results) > 0
        assert "reasons" in results[0]
        assert isinstance(results[0]["reasons"], list)


# ── PTB-XL 5-sınıf kural testleri ──────────────────────────────────────────

class TestPTBXLRules:
    """MI, HYP, STTC, CD, NORM kuralları için birim testler."""

    def _engine(self):
        from symbolic.rules import PrologEngine
        return PrologEngine()

    def test_mi_st_elevation(self):
        """ST elevasyonu > 0.1 mV → MI."""
        results = self._engine().query({"st_elevation_mv": 0.15})
        assert any(r["diagnosis"] == "mi" for r in results)
        mi = next(r for r in results if r["diagnosis"] == "mi")
        assert "st_elevation" in mi["reasons"]

    def test_mi_pathological_q(self):
        """Patolojik Q dalgası < -0.1 mV → MI."""
        results = self._engine().query({"q_wave_mv": -0.15})
        assert any(r["diagnosis"] == "mi" for r in results)

    def test_mi_t_inversion(self):
        """T inversiyonu skoru < -0.5 → MI."""
        results = self._engine().query({"t_inversion_score": -0.7})
        assert any(r["diagnosis"] == "mi" for r in results)

    def test_hyp_sokolow_lyon(self):
        """Sokolow-Lyon > 3.5 mV → HYP."""
        results = self._engine().query({"sokolow_lyon_mv": 4.0})
        assert any(r["diagnosis"] == "hyp" for r in results)
        hyp = next(r for r in results if r["diagnosis"] == "hyp")
        assert "sokolow_lyon" in hyp["reasons"]

    def test_hyp_avl_voltage(self):
        """aVL voltajı > 1.1 mV → HYP."""
        results = self._engine().query({"avl_r_mv": 1.5})
        assert any(r["diagnosis"] == "hyp" for r in results)

    def test_cd_wide_qrs(self):
        """QRS > 120 ms → CD."""
        results = self._engine().query({"qrs_duration_ms": 130})
        assert any(r["diagnosis"] == "cd" for r in results)
        cd = next(r for r in results if r["diagnosis"] == "cd")
        assert "wide_qrs" in cd["reasons"]

    def test_sttc_st_depression(self):
        """ST çökmesi < -0.05 mV → STTC."""
        results = self._engine().query({"st_depression_mv": -0.1})
        assert any(r["diagnosis"] == "sttc" for r in results)

    def test_sttc_t_wave_change(self):
        """T dalgası değişikliği > 0.5 → STTC."""
        results = self._engine().query({"t_amplitude_change": 0.8})
        assert any(r["diagnosis"] == "sttc" for r in results)

    def test_norm_no_abnormalities(self):
        """Tüm değerler normal aralıkta → NORM."""
        results = self._engine().query({
            "st_elevation_mv":   0.02,
            "q_wave_mv":        -0.02,
            "sokolow_lyon_mv":   2.0,
            "st_depression_mv": -0.01,
            "qrs_duration_ms":   90.0,
        })
        assert any(r["diagnosis"] == "norm" for r in results)

    def test_norm_not_fired_with_mi(self):
        """MI bulgusu varken NORM tetiklenmemeli."""
        results = self._engine().query({
            "st_elevation_mv":   0.15,   # MI → tetiklenir
            "q_wave_mv":        -0.02,
            "sokolow_lyon_mv":   2.0,
            "st_depression_mv": -0.01,
            "qrs_duration_ms":   90.0,
        })
        diagnoses = [r["diagnosis"] for r in results]
        assert "mi" in diagnoses
        assert "norm" not in diagnoses

    def test_multiple_diagnoses(self):
        """Birden fazla kural aynı anda tetiklenebilir."""
        results = self._engine().query({
            "sokolow_lyon_mv":  4.0,   # HYP
            "qrs_duration_ms": 130.0,  # CD
        })
        diagnoses = [r["diagnosis"] for r in results]
        assert "hyp" in diagnoses
        assert "cd" in diagnoses


# ── NeuralSymbolicFusion testleri ───────────────────────────────────────────

class TestFusion:
    def test_output_sums_to_one(self):
        """Füzyon çıktısı olasılık dağılımı — toplam ≈ 1."""
        from symbolic.fusion import NeuralSymbolicFusion
        fusion    = NeuralSymbolicFusion(alpha=0.35)
        gat_probs = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        sym_probs = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        result    = fusion.fuse(gat_probs, sym_probs)
        assert abs(result.sum() - 1.0) < 1e-5

    def test_alpha_range(self):
        """alpha [0,1] dışında hata vermeli."""
        from symbolic.fusion import NeuralSymbolicFusion
        with pytest.raises((ValueError, AssertionError)):
            NeuralSymbolicFusion(alpha=1.5)

    def test_alpha_zero_equals_gat(self):
        """alpha=0 → sonuç tamamen GAT olasılığı."""
        from symbolic.fusion import NeuralSymbolicFusion
        fusion    = NeuralSymbolicFusion(alpha=0.0)
        gat_probs = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        sym_probs = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
        result    = fusion.fuse(gat_probs, sym_probs)
        np.testing.assert_allclose(result, gat_probs, atol=1e-6)

    def test_alpha_one_equals_symbolic(self):
        """alpha=1 → sonuç tamamen sembolik olasılık."""
        from symbolic.fusion import NeuralSymbolicFusion
        fusion    = NeuralSymbolicFusion(alpha=1.0)
        gat_probs = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        sym_probs = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
        result    = fusion.fuse(gat_probs, sym_probs)
        np.testing.assert_allclose(result, sym_probs, atol=1e-6)

    def test_batch_fuse_shape(self):
        """Toplu füzyon doğru şekil döndürmeli."""
        from symbolic.fusion import NeuralSymbolicFusion
        fusion = NeuralSymbolicFusion(alpha=0.35)
        gat    = np.random.dirichlet([1]*5, size=10)
        sym    = np.random.dirichlet([1]*5, size=10)
        result = fusion.fuse_batch(gat, sym)
        assert result.shape == (10, 5)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(10), atol=1e-5)


# ── SymbolicClassifier testleri ─────────────────────────────────────────────

class TestSymbolicClassifier:
    def test_output_shape_and_sums(self):
        """96-dim girişten (5,) olasılık, toplam ≈ 1."""
        from symbolic.classifier import SymbolicClassifier
        sc = SymbolicClassifier()
        x  = np.random.randn(9, 96).astype(np.float32)
        p  = sc.predict(x)
        assert p.shape == (5,)
        assert abs(p.sum() - 1.0) < 1e-5
        assert (p >= 0).all()

    def test_hyp_signal_raises_hyp_prob(self):
        """HYP sinyali → hyp sınıfının sembolik olasılığı yükselir."""
        from symbolic.classifier import SymbolicClassifier, CLASS_ORDER
        sc = SymbolicClassifier()
        x  = np.zeros((1, 96), dtype=np.float32)
        # V1 p2p (idx 6*8+3=51) + V5 p2p (idx 10*8+3=83) → Sokolow-Lyon > 3.5
        x[0, 51] = 2.0   # p2p_V1
        x[0, 83] = 2.0   # p2p_V5
        p = sc.predict(x)
        hyp_idx = CLASS_ORDER.index("hyp")
        assert p[hyp_idx] > 0.15, f"HYP olasılığı düşük: {p}"
