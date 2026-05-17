# AGENT: demo_agent
"""
demo_agent testleri: sabitler, attention normalizasyonu,
graf/sembolik katman izole testler, API endpoint'leri,
checkpoint varsa tam inference doğrulaması.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

CHECKPOINT = BASE / "models" / "checkpoints" / "best_model.pt"
checkpoint_exists = CHECKPOINT.exists()


# ── Sabitler ──────────────────────────────────────────────────────────────────

class TestDemoConstants:
    def test_class_names_order(self):
        """CLASS_NAMES PTB-XL 5-sınıf sırasında olmalı."""
        from demo.pipeline import CLASS_NAMES
        assert CLASS_NAMES == ["NORM", "MI", "STTC", "CD", "HYP"]

    def test_class_names_tr_coverage(self):
        """CLASS_NAMES_TR tüm sınıfları kapsamalı."""
        from demo.pipeline import CLASS_NAMES, CLASS_NAMES_TR
        for c in CLASS_NAMES:
            assert c in CLASS_NAMES_TR, f"{c} için Türkçe karşılık eksik"

    def test_alpha_value(self):
        """Sembolik füzyon alpha=0.35 (proje kararı)."""
        from demo.pipeline import ALPHA
        assert ALPHA == 0.35

    def test_temperature_default(self):
        """Varsayılan T=1.0256 (temperature.pt yoksa fallback)."""
        from demo.pipeline import TemperatureScaler
        scaler = TemperatureScaler(initial_temperature=1.0256)
        assert abs(float(scaler.temperature.item()) - 1.0256) < 1e-4


# ── Attention normalizasyonu (izole, model gerektirmez) ───────────────────────

class TestAttentionNormalization:
    """pipeline.run_inference içindeki beat attention mantığını izole test eder."""

    @staticmethod
    def _norm(attn_np, ei, n_nodes):
        n_use   = min(ei.shape[1], attn_np.shape[0])
        attn_1d = attn_np[:n_use].mean(axis=1)
        beat_attn  = np.zeros(n_nodes)
        beat_count = np.zeros(n_nodes)
        for k in range(n_use):
            dst = int(ei[1, k])
            if dst < n_nodes:
                beat_attn[dst]  += attn_1d[k]
                beat_count[dst] += 1
        beat_count = np.where(beat_count == 0, 1, beat_count)
        beat_attn  = beat_attn / beat_count
        if beat_attn.max() > beat_attn.min():
            beat_attn = (beat_attn - beat_attn.min()) / (beat_attn.max() - beat_attn.min())
        return beat_attn

    def test_output_range_zero_to_one(self):
        """Normalize attention [0, 1] aralığında olmalı."""
        attn = np.random.rand(10, 8)
        ei   = np.array([list(range(10)), [i % 10 for i in range(1, 11)]])
        res  = self._norm(attn, ei, n_nodes=10)
        assert res.min() >= -1e-6
        assert res.max() <= 1.0 + 1e-6

    def test_output_length_equals_n_nodes(self):
        """Çıktı n_nodes uzunluğunda olmalı."""
        attn = np.random.rand(5, 4)
        ei   = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        res  = self._norm(attn, ei, n_nodes=8)
        assert len(res) == 8

    def test_uniform_attention_no_crash(self):
        """Tüm dikkat eşit olduğunda (min==max) normalize uygulanmaz, hata yok."""
        attn = np.ones((6, 4))
        ei   = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
        res  = self._norm(attn, ei, n_nodes=6)
        assert len(res) == 6
        assert np.all(res >= 0)

    def test_edge_attn_count_mismatch_clipped(self):
        """attn satırı < kenar sayısı olduğunda güvenle çalışmalı."""
        attn = np.random.rand(3, 4)    # 3 satır
        ei   = np.array([[0,1,2,3,4,5], [1,2,3,4,5,0]])  # 6 kenar
        res  = self._norm(attn, ei, n_nodes=6)
        assert len(res) == 6

    def test_max_node_has_highest_attention(self):
        """Tek yüksek kenar ağırlığı doğru node'a atanmalı."""
        attn = np.zeros((3, 1))
        attn[1, 0] = 1.0           # 2. kenar → dst=node 2
        ei   = np.array([[0, 1, 2], [0, 2, 0]])
        res  = self._norm(attn, ei, n_nodes=3)
        assert res[2] == pytest.approx(1.0, abs=1e-5)


# ── Graf oluşturma (checkpoint gerektirmez) ───────────────────────────────────

class TestGraphBuilding:
    def test_node_features_shape(self):
        """96-dim beats → düğüm özellik matrisi (N, 96)."""
        from graph.builder import build_clinical_graph
        beats = np.random.randn(10, 96).astype(np.float32)
        edge_index, node_features = build_clinical_graph(beats)
        assert node_features.shape == (10, 96)

    def test_edge_index_shape(self):
        """edge_index (2, E) boyutunda, en az 1 kenar."""
        from graph.builder import build_clinical_graph
        beats = np.random.randn(10, 96).astype(np.float32)
        edge_index, _ = build_clinical_graph(beats)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

    def test_dtw_weights_count_matches_edges(self):
        """DTW ağırlık sayısı kenar sayısıyla eşleşmeli."""
        from graph.builder import build_clinical_graph, compute_dtw_weights
        beats = np.random.randn(6, 96).astype(np.float32)
        edge_index, _ = build_clinical_graph(beats)
        edge_attr = compute_dtw_weights(beats, edge_index)
        assert edge_attr.shape[0] == edge_index.shape[1]

    def test_dtw_weights_in_valid_range(self):
        """DTW ağırlıkları 1/(1+d) ∈ (0, 1]."""
        from graph.builder import build_clinical_graph, compute_dtw_weights
        beats = np.random.randn(5, 96).astype(np.float32)
        edge_index, _ = build_clinical_graph(beats)
        edge_attr = compute_dtw_weights(beats, edge_index)
        assert float(edge_attr.min()) > 0.0
        assert float(edge_attr.max()) <= 1.0 + 1e-5


# ── Sembolik katman — sentetik beats (checkpoint gerektirmez) ─────────────────

class TestSymbolicOnSyntheticBeats:
    def test_predict_output_shape_and_sum(self):
        """SymbolicClassifier 96-dim girişten (5,) olasılık üretmeli, toplam ≈ 1."""
        from symbolic.classifier import SymbolicClassifier
        sc   = SymbolicClassifier()
        x    = np.random.randn(8, 96).astype(np.float32)
        prob = sc.predict(x)
        assert prob.shape == (5,)
        assert abs(prob.sum() - 1.0) < 1e-5
        assert (prob >= 0).all()

    def test_extract_clinical_features_keys(self):
        """extract_clinical_features 8 gerekli anahtarı döndürmeli."""
        from symbolic.classifier import extract_clinical_features
        x    = np.random.randn(5, 96).astype(np.float32)
        feat = extract_clinical_features(x)
        required = {
            "st_elevation_mv", "q_wave_mv", "t_inversion_score",
            "sokolow_lyon_mv", "avl_r_mv", "st_depression_mv",
            "t_amplitude_change", "qrs_duration_ms",
        }
        assert required <= set(feat.keys())


# ── API — health ve geçersiz girdi (model gerektirmez) ────────────────────────

class TestAPIHealth:
    def test_health_ok(self):
        """/health 200, status='ok', model='CardioGAT' döndürmeli."""
        from fastapi.testclient import TestClient
        from demo.api import app
        resp = TestClient(app).get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "CardioGAT"

    def test_openapi_schema_title(self):
        """/openapi.json ulaşılabilir ve doğru başlık içermeli."""
        from fastapi.testclient import TestClient
        from demo.api import app
        resp = TestClient(app).get("/openapi.json")
        assert resp.status_code == 200
        assert "CardioGraph-RL" in resp.json()["info"]["title"]


class TestAPIInvalidInput:
    def test_predict_empty_files_returns_error(self):
        """/predict boş dosyayla 4xx veya 5xx döndürmeli."""
        from fastapi.testclient import TestClient
        from demo.api import app
        resp = TestClient(app).post(
            "/predict",
            files={
                "hea_file": ("test.hea", b"", "application/octet-stream"),
                "dat_file": ("test.dat", b"", "application/octet-stream"),
            },
        )
        assert resp.status_code >= 400

    def test_predict_missing_dat_returns_422(self):
        """/predict .dat olmadan FastAPI doğrulama hatası (422) vermeli."""
        from fastapi.testclient import TestClient
        from demo.api import app
        resp = TestClient(app).post(
            "/predict",
            files={"hea_file": ("test.hea", b"dummy", "application/octet-stream")},
        )
        assert resp.status_code == 422

    def test_predict_garbage_content_returns_error(self):
        """/predict anlamsız içerikle 4xx veya 5xx döndürmeli."""
        from fastapi.testclient import TestClient
        from demo.api import app
        resp = TestClient(app).post(
            "/predict",
            files={
                "hea_file": ("rec.hea", b"NOT A VALID HEADER\n", "application/octet-stream"),
                "dat_file": ("rec.dat", b"\x00\x01\x02\x03", "application/octet-stream"),
            },
        )
        assert resp.status_code >= 400


# ── Tam inference (checkpoint zorunlu) ───────────────────────────────────────

@pytest.mark.skipif(
    not checkpoint_exists,
    reason="Checkpoint yok — model_agent eğitimi tamamlanmamış",
)
class TestFullInference:
    """Sentetik wfdb kaydıyla tam pipeline'ı doğrular."""

    @pytest.fixture(scope="class")
    def synthetic_record(self, tmp_path_factory):
        """500 Hz, 12-lead, 5 sn sentetik EKG kaydı oluşturur."""
        import wfdb
        tmp      = tmp_path_factory.mktemp("wfdb")
        fs       = 500
        n        = fs * 5
        t        = np.arange(n) / fs
        r_times  = np.arange(0.5, 5.0, 0.80)   # ~75 bpm

        signal = np.zeros((n, 12), dtype=np.float64)
        for lead in range(12):
            noise = np.random.randn(n) * 0.02
            qrs   = noise.copy()
            for rt in r_times:
                center = int(rt * fs)
                idx    = np.arange(max(0, center - 20), min(n, center + 20))
                qrs[idx] += 1.2 * np.exp(-((idx - center) ** 2) / (2 * 5 ** 2))
            signal[:, lead] = qrs

        wfdb.wrsamp(
            record_name="test_rec",
            write_dir=str(tmp),
            fs=fs,
            units=["mV"] * 12,
            sig_name=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
            p_signal=signal,
            fmt=["16"] * 12,
        )
        return str(tmp / "test_rec")

    def test_inference_completes(self, synthetic_record):
        """run_inference istisna olmadan tamamlanmalı."""
        from demo.pipeline import run_inference
        assert run_inference(synthetic_record) is not None

    def test_output_keys_present(self, synthetic_record):
        """Çıktı dict gerekli anahtarları içermeli."""
        from demo.pipeline import run_inference
        result = run_inference(synthetic_record)
        required = {
            "signal", "lead_names", "fs", "n_beats",
            "pred_class", "pred_class_tr",
            "gat_probs", "fused_probs", "beat_attention",
            "symbolic_rules", "clinical_features",
        }
        assert required <= set(result.keys())

    def test_pred_class_is_valid(self, synthetic_record):
        """pred_class geçerli bir PTB-XL sınıfı olmalı."""
        from demo.pipeline import CLASS_NAMES, run_inference
        result = run_inference(synthetic_record)
        assert result["pred_class"] in CLASS_NAMES

    def test_pred_class_tr_matches_pred_class(self, synthetic_record):
        """pred_class_tr pred_class ile tutarlı olmalı."""
        from demo.pipeline import CLASS_NAMES_TR, run_inference
        result = run_inference(synthetic_record)
        assert result["pred_class_tr"] == CLASS_NAMES_TR[result["pred_class"]]

    def test_gat_probs_valid_distribution(self, synthetic_record):
        """GAT olasılıkları (5,), toplam ≈ 1, tüm değerler ≥ 0."""
        from demo.pipeline import run_inference
        p = run_inference(synthetic_record)["gat_probs"]
        assert p.shape == (5,)
        assert abs(p.sum() - 1.0) < 1e-5
        assert (p >= 0).all()

    def test_fused_probs_valid_distribution(self, synthetic_record):
        """Füzyon olasılıkları (5,), toplam ≈ 1, tüm değerler ≥ 0."""
        from demo.pipeline import run_inference
        p = run_inference(synthetic_record)["fused_probs"]
        assert p.shape == (5,)
        assert abs(p.sum() - 1.0) < 1e-5
        assert (p >= 0).all()

    def test_beat_attention_range(self, synthetic_record):
        """Beat attention [0, 1] aralığında olmalı."""
        from demo.pipeline import run_inference
        a = run_inference(synthetic_record)["beat_attention"]
        assert a.min() >= -1e-6
        assert a.max() <= 1.0 + 1e-6

    def test_beat_attention_length_matches_n_beats(self, synthetic_record):
        """beat_attention uzunluğu n_beats ile eşleşmeli."""
        from demo.pipeline import run_inference
        result = run_inference(synthetic_record)
        assert len(result["beat_attention"]) == result["n_beats"]

    def test_signal_is_12_lead(self, synthetic_record):
        """Ham sinyal (T, 12) boyutunda döndürülmeli."""
        from demo.pipeline import run_inference
        sig = run_inference(synthetic_record)["signal"]
        assert sig.ndim == 2
        assert sig.shape[1] == 12

    def test_gat_conf_above_random_baseline(self, synthetic_record):
        """GAT güveni rastgele sınırın (0.20) üzerinde olmalı."""
        from demo.pipeline import CLASS_NAMES, run_inference
        result = run_inference(synthetic_record)
        conf   = float(result["gat_probs"][CLASS_NAMES.index(result["pred_class"])])
        assert conf > 0.20, f"GAT güveni çok düşük: {conf:.3f}"

    def test_idempotent_same_record(self, synthetic_record):
        """Aynı kayıt iki kez çalıştırılınca aynı tanıyı vermeli."""
        from demo.pipeline import run_inference
        r1 = run_inference(synthetic_record)
        r2 = run_inference(synthetic_record)
        assert r1["pred_class"] == r2["pred_class"]
        np.testing.assert_allclose(r1["gat_probs"], r2["gat_probs"], atol=1e-4)
