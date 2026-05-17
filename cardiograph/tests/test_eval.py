# AGENT: eval_agent
"""
Metrik ve değerlendirme testleri.
"""
import numpy as np
import pytest


class TestFaithfulness:
    def test_perfect_faithfulness(self):
        """Top-K tamamen patolojik → Faithfulness@K = 1.0"""
        from eval.metrics import faithfulness_at_k
        # 10 atım, attention skorları azalan sırada
        attention = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        # İlk 3 atım patolojik
        pathological = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        score = faithfulness_at_k(attention, pathological, k=3)
        assert abs(score - 1.0) < 1e-6

    def test_zero_faithfulness(self):
        """Top-K hiç patolojik değil → Faithfulness@K = 0.0"""
        from eval.metrics import faithfulness_at_k
        attention = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        pathological = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        score = faithfulness_at_k(attention, pathological, k=3)
        assert abs(score - 0.0) < 1e-6

    def test_partial_faithfulness(self):
        """Top-3'te 2 patolojik → Faithfulness@3 = 2/3"""
        from eval.metrics import faithfulness_at_k
        attention = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        pathological = np.array([1, 1, 0, 0, 0])
        score = faithfulness_at_k(attention, pathological, k=3)
        assert abs(score - 2.0 / 3.0) < 1e-5

    def test_k_values(self):
        """K=1,3,5 hepsinde çalışmalı."""
        from eval.metrics import faithfulness_at_k
        attention = np.random.rand(10)
        pathological = np.random.randint(0, 2, 10)
        for k in [1, 3, 5]:
            score = faithfulness_at_k(attention, pathological, k=k)
            assert 0.0 <= score <= 1.0

    def test_k_larger_than_n(self):
        """k > N ise clamp uygulanır, hata verilmez."""
        from eval.metrics import faithfulness_at_k
        attention = np.array([0.9, 0.5])
        pathological = np.array([1, 0])
        result = faithfulness_at_k(attention, pathological, k=100)
        assert 0.0 <= result <= 1.0

    def test_empty_input(self):
        """Boş girdi → 0.0 döner, istisna yok."""
        from eval.metrics import faithfulness_at_k
        assert faithfulness_at_k([], [], k=3) == 0.0

    def test_k1_top_node(self):
        """K=1 → sadece en yüksek attention düğümü."""
        from eval.metrics import faithfulness_at_k
        attention = np.array([0.1, 0.9, 0.5])
        pathological = np.array([0, 1, 0])  # en yüksek attention → patolojik
        assert faithfulness_at_k(attention, pathological, k=1) == 1.0


class TestCalibration:
    def test_perfect_calibration_ece_zero(self):
        """Mükemmel kalibre model → ECE ≈ 0."""
        from eval.metrics import expected_calibration_error
        # Güven = gerçek doğruluk
        n = 1000
        y_prob = np.random.uniform(0, 1, n)
        # Her örnek için etiketi güven oranıyla belirle
        y_true = (np.random.rand(n) < y_prob).astype(int)
        ece = expected_calibration_error(y_prob, y_true, n_bins=10)
        # Mükemmel kalibrasyon beklenmez ama çok düşük olmalı
        assert ece < 0.15

    def test_overconfident_high_ece(self):
        """Aşırı güvenli model → yüksek ECE."""
        from eval.metrics import expected_calibration_error
        n = 500
        # Tüm tahminler 0.95 güvende ama doğruluk %50
        y_prob = np.full(n, 0.95)
        y_true = np.random.randint(0, 2, n)
        ece = expected_calibration_error(y_prob, y_true, n_bins=10)
        assert ece > 0.3

    def test_ece_range(self):
        """ECE her zaman [0, 1] aralığında."""
        from eval.metrics import expected_calibration_error
        y_prob = np.random.rand(100)
        y_true = np.random.randint(0, 2, 100)
        ece = expected_calibration_error(y_prob, y_true)
        assert 0.0 <= ece <= 1.0

    def test_ece_bins_parameter(self):
        """n_bins farklı değerlerle çalışmalı."""
        from eval.metrics import expected_calibration_error
        y_prob = np.random.rand(200)
        y_true = np.random.randint(0, 2, 200)
        for bins in [5, 10, 15, 20]:
            ece = expected_calibration_error(y_prob, y_true, n_bins=bins)
            assert 0.0 <= ece <= 1.0


class TestComputeAllMetrics:
    def test_perfect_prediction(self):
        """Mükemmel tahmin → F1=1.0, AUC=1.0, diagonal confusion matrix."""
        from eval.metrics import compute_all_metrics
        y_true = np.tile(np.arange(5), 20)          # 100 örnek, dengeli
        y_pred = y_true.copy()
        y_prob = np.zeros((100, 5))
        y_prob[np.arange(100), y_true] = 1.0

        m = compute_all_metrics(y_true, y_pred, y_prob)
        assert abs(m["f1_macro"]    - 1.0) < 1e-6
        assert abs(m["f1_weighted"] - 1.0) < 1e-6
        assert abs(m["auc_roc"]     - 1.0) < 1e-6
        assert m["confusion_matrix"].trace() == 100

    def test_output_keys(self):
        """Döndürülen dict gerekli anahtarları içermeli."""
        from eval.metrics import compute_all_metrics
        y_true = np.tile(np.arange(5), 4)
        y_pred = y_true.copy()
        y_prob = np.zeros((20, 5))
        y_prob[np.arange(20), y_true] = 1.0
        m = compute_all_metrics(y_true, y_pred, y_prob)
        for key in ("f1_macro", "f1_weighted", "auc_roc", "confusion_matrix"):
            assert key in m

    def test_confusion_matrix_shape(self):
        """confusion_matrix 5×5 boyutunda olmalı."""
        from eval.metrics import compute_all_metrics
        y_true = np.tile(np.arange(5), 10)
        y_pred = y_true.copy()
        y_prob = np.zeros((50, 5))
        y_prob[np.arange(50), y_true] = 1.0
        m = compute_all_metrics(y_true, y_pred, y_prob)
        assert m["confusion_matrix"].shape == (5, 5)

    def test_f1_range(self):
        """F1 değerleri [0, 1] aralığında olmalı."""
        from eval.metrics import compute_all_metrics
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        y_prob = np.random.dirichlet(np.ones(5), 100)
        m = compute_all_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= m["f1_macro"]    <= 1.0
        assert 0.0 <= m["f1_weighted"] <= 1.0
