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
        """108-dim girişten (5,) olasılık, toplam ≈ 1."""
        from symbolic.classifier import SymbolicClassifier
        sc = SymbolicClassifier()
        x  = np.random.randn(9, 108).astype(np.float32)
        p  = sc.predict(x)
        assert p.shape == (5,)
        assert abs(p.sum() - 1.0) < 1e-5
        assert (p >= 0).all()

    def test_hyp_signal_raises_hyp_prob(self):
        """HYP sinyali → hyp sınıfının sembolik olasılığı yükselir."""
        from symbolic.classifier import SymbolicClassifier, CLASS_ORDER
        sc = SymbolicClassifier()
        x  = np.zeros((1, 108), dtype=np.float32)
        # V1 peak_to_peak (idx 6*9+4=58) + V5 peak_to_peak (idx 10*9+4=94) → Sokolow-Lyon > 3.5
        x[0, 58] = 2.0   # peak_to_peak_V1
        x[0, 94] = 2.0   # peak_to_peak_V5
        p = sc.predict(x)
        hyp_idx = CLASS_ORDER.index("hyp")
        assert p[hyp_idx] > 0.15, f"HYP olasılığı düşük: {p}"


# ── FuzzyRuleEngine testleri ─────────────────────────────────────────────────

class TestFuzzyRuleEngine:
    def _engine(self):
        from symbolic.rules import FuzzyRuleEngine
        return FuzzyRuleEngine()

    def test_confidence_in_unit_interval(self):
        """Tüm confidence değerleri [0, 1] aralığında."""
        eng = self._engine()
        results = eng.query({"st_elevation_mv": 0.15, "sokolow_lyon_mv": 4.0,
                             "qrs_duration_ms": 90.0})
        for r in results:
            assert 0.0 <= r["confidence"] <= 1.0, f"Aralık dışı: {r}"

    def test_threshold_at_midpoint_gives_half(self):
        """Eşik değerinde confidence ≈ 0.5 (sigmoid(0) = 0.5)."""
        eng = self._engine()
        results = eng.query({"st_elevation_mv": 0.10})   # tam eşikte
        mi = next((r for r in results if r["diagnosis"] == "mi"), None)
        assert mi is not None, "Eşik değerinde MI bulunamadı"
        assert 0.45 <= mi["confidence"] <= 0.55, f"confidence={mi['confidence']}"

    def test_mi_confidence_increases_with_elevation(self):
        """ST elevasyonu yükseldikçe MI confidence monoton artar."""
        eng = self._engine()
        vals = [0.05, 0.10, 0.15, 0.25]
        confs = []
        for v in vals:
            res = eng.query({"st_elevation_mv": v})
            mi = next((r for r in res if r["diagnosis"] == "mi"), None)
            confs.append(mi["confidence"] if mi else 0.0)
        assert confs == sorted(confs), f"Monotonluk bozuldu: {confs}"

    def test_norm_conf_decreases_with_pathology(self):
        """Patoloji arttıkça NORM confidence düşer."""
        eng = self._engine()
        base = {"st_elevation_mv": 0.02, "q_wave_mv": -0.02,
                "sokolow_lyon_mv": 2.0, "st_depression_mv": -0.01,
                "qrs_duration_ms": 90.0}

        def _norm_conf(feats):
            res = eng.query(feats)
            n = next((r for r in res if r["diagnosis"] == "norm"), None)
            return n["confidence"] if n else 0.0

        c_normal = _norm_conf(base)
        c_patho  = _norm_conf(dict(base, st_elevation_mv=0.25))
        assert c_normal > c_patho, f"norm_normal={c_normal} norm_patho={c_patho}"

    def test_sharpness_is_learnable(self):
        """Küçük sharpness → çok keskin geçiş (eşik+0.01'de bile ~1.0)."""
        from symbolic.rules import FuzzyRuleEngine
        eng = FuzzyRuleEngine(sharpness={("st_elevation_mv", ">", 0.10): 0.001})
        results = eng.query({"st_elevation_mv": 0.11})
        mi = next((r for r in results if r["diagnosis"] == "mi"), None)
        assert mi is not None
        assert mi["confidence"] > 0.99, f"Beklenen keskin geçiş: {mi['confidence']}"

    def test_norm_suppressed_by_clear_pathology(self):
        """Belirgin patoloji varken NORM tetiklenmemeli (confidence < threshold)."""
        eng = self._engine()
        results = eng.query({
            "st_elevation_mv":  0.25,   # açıkça yüksek → norm_conf ≈ 0.01
            "q_wave_mv":       -0.02,
            "sokolow_lyon_mv":  2.0,
            "st_depression_mv": -0.01,
            "qrs_duration_ms":  90.0,
        })
        diagnoses = [r["diagnosis"] for r in results]
        assert "norm" not in diagnoses, "Belirgin MI'da NORM tetiklendi"


# ── AdaptiveFusion testleri ─────────────────────────────────────────────────

class TestAdaptiveFusion:
    def test_output_sums_to_one(self):
        """Per-class alpha füzyon çıktısı toplam ≈ 1."""
        from symbolic.fusion import AdaptiveFusion
        af = AdaptiveFusion(alphas=[0.1, 0.0, 0.4, 0.3, 0.5])
        gat = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        sym = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        result = af.fuse(gat, sym)
        assert abs(result.sum() - 1.0) < 1e-5

    def test_all_zero_alphas_equals_gat(self):
        """Tüm alpha=0 → sonuç tamamen GAT."""
        from symbolic.fusion import AdaptiveFusion
        af = AdaptiveFusion(alphas=[0.0] * 5)
        gat = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        sym = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
        result = af.fuse(gat, sym)
        np.testing.assert_allclose(result, gat, atol=1e-6)

    def test_all_one_alphas_equals_symbolic(self):
        """Tüm alpha=1 → sonuç tamamen sembolik."""
        from symbolic.fusion import AdaptiveFusion
        af = AdaptiveFusion(alphas=[1.0] * 5)
        gat = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        sym = np.array([0.2, 0.4, 0.2, 0.1, 0.1])
        result = af.fuse(gat, sym)
        np.testing.assert_allclose(result, sym, atol=1e-6)

    def test_per_class_differs_from_global(self):
        """Per-class alpha, global alpha'dan farklı sonuç verir."""
        from symbolic.fusion import AdaptiveFusion, NeuralSymbolicFusion
        af     = AdaptiveFusion(alphas=[0.0, 0.0, 0.5, 0.5, 0.7])
        global_f = NeuralSymbolicFusion(alpha=0.35)
        gat = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        sym = np.array([0.1, 0.3, 0.2, 0.2, 0.2])
        assert not np.allclose(af.fuse(gat, sym), global_f.fuse(gat, sym))

    def test_batch_shape_and_sum(self):
        """Toplu füzyon doğru şekil ve toplam=1."""
        from symbolic.fusion import AdaptiveFusion
        af  = AdaptiveFusion()
        gat = np.random.dirichlet([1] * 5, size=10)
        sym = np.random.dirichlet([1] * 5, size=10)
        result = af.fuse_batch(gat, sym)
        assert result.shape == (10, 5)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(10), atol=1e-5)

    def test_invalid_alphas_length(self):
        """Yanlış uzunlukta alphas → ValueError."""
        from symbolic.fusion import AdaptiveFusion
        with pytest.raises(ValueError):
            AdaptiveFusion(alphas=[0.3, 0.3])

    def test_save_load_roundtrip(self, tmp_path):
        """Save/load ile alphas korunur."""
        from symbolic.fusion import AdaptiveFusion
        alphas = [0.1, 0.0, 0.4, 0.6, 0.3]
        af = AdaptiveFusion(alphas=alphas)
        path = tmp_path / "test_alphas.npy"
        af.save(path)
        loaded = AdaptiveFusion.load(path)
        np.testing.assert_allclose(loaded.alphas, alphas, atol=1e-6)

    def test_learn_alphas_shape_and_range(self):
        """learn_alphas (5,) array döndürür, tüm değerler [0,1]."""
        from symbolic.fusion import learn_alphas
        np.random.seed(42)
        N = 100
        gat    = np.random.dirichlet([2, 1, 1, 1, 1], size=N).astype(np.float32)
        sym    = np.random.dirichlet([1] * 5, size=N).astype(np.float32)
        labels = np.random.randint(0, 5, size=N)
        alphas = learn_alphas(gat, sym, labels)
        assert alphas.shape == (5,)
        assert np.all(alphas >= 0.0) and np.all(alphas <= 1.0)

    def test_learn_alphas_improves_nll(self):
        """Öğrenilen alpha'lar baseline (0.35) NLL'ini düşürür veya eşit kalır."""
        from symbolic.fusion import learn_alphas, AdaptiveFusion, NeuralSymbolicFusion
        import torch, torch.nn.functional as F
        np.random.seed(0)
        N = 200
        gat    = np.random.dirichlet([3, 1, 1, 1, 1], size=N).astype(np.float32)
        sym    = np.random.dirichlet([1] * 5, size=N).astype(np.float32)
        labels = np.random.randint(0, 5, size=N)

        learned = learn_alphas(gat, sym, labels)
        af      = AdaptiveFusion(alphas=learned)
        base_f  = NeuralSymbolicFusion(alpha=0.35)

        af_probs   = af.fuse_batch(gat, sym).astype(np.float32)
        base_probs = np.stack([base_f.fuse(g, s) for g, s in zip(gat, sym)]).astype(np.float32)

        labels_t = torch.tensor(labels, dtype=torch.long)
        nll_learned  = F.nll_loss(torch.log(torch.tensor(af_probs).clamp(min=1e-10)),
                                  labels_t).item()
        nll_baseline = F.nll_loss(torch.log(torch.tensor(base_probs).clamp(min=1e-10)),
                                  labels_t).item()
        assert nll_learned <= nll_baseline + 1e-4, (
            f"Adaptive NLL={nll_learned:.4f} > baseline NLL={nll_baseline:.4f}"
        )


# ── FuzzySymbolicClassifier testleri ────────────────────────────────────────

class TestFuzzySymbolicClassifier:
    def test_alias_same_as_symbolic_classifier(self):
        """FuzzySymbolicClassifier, SymbolicClassifier ile aynı sınıf olmalı."""
        from symbolic.classifier import SymbolicClassifier, FuzzySymbolicClassifier
        assert FuzzySymbolicClassifier is SymbolicClassifier

    def test_output_shape_and_sums(self):
        """108-dim girişten (5,) olasılık, toplam ≈ 1, tümü ≥ 0."""
        from symbolic.classifier import FuzzySymbolicClassifier
        sc = FuzzySymbolicClassifier()
        x  = np.random.randn(9, 108).astype(np.float32)
        p  = sc.predict(x)
        assert p.shape == (5,)
        assert abs(p.sum() - 1.0) < 1e-5
        assert (p >= 0).all()

    def test_no_bimodal_spike(self):
        """Çıkış {0.70} sabit değerini içermemeli (sürekli dağılım)."""
        from symbolic.classifier import FuzzySymbolicClassifier
        sc = FuzzySymbolicClassifier()
        x  = np.zeros((1, 108), dtype=np.float32)
        p  = sc.predict(x)
        assert not np.any(np.abs(p - 0.70) < 1e-4), f"Bimodal spike: {p}"

    def test_mi_graded_response(self):
        """Daha güçlü ST elevasyonu → daha yüksek MI olasılığı."""
        from symbolic.classifier import FuzzySymbolicClassifier, CLASS_ORDER
        sc = FuzzySymbolicClassifier()
        mi_idx = CLASS_ORDER.index("mi")
        # V2 ST_elevation = lead_idx(V2)*9 + stat_idx(ST_elevation) = 7*9+7 = 70
        V2_ST_ELEVATION = 70

        x_weak = np.zeros((1, 108), dtype=np.float32)
        x_weak[0, V2_ST_ELEVATION] = 0.12   # eşiğin biraz üstünde

        x_strong = np.zeros((1, 108), dtype=np.float32)
        x_strong[0, V2_ST_ELEVATION] = 0.40  # açıkça yüksek

        p_weak   = sc.predict(x_weak)
        p_strong = sc.predict(x_strong)
        assert p_strong[mi_idx] > p_weak[mi_idx], (
            f"MI graded bozuldu: weak={p_weak[mi_idx]:.3f} strong={p_strong[mi_idx]:.3f}"
        )
