# AGENT: graph_agent
"""
Graf oluşturma testleri.
Sentetik 48-özellik beat dizisiyle çalışır, gerçek veri gerekmez.
"""
import numpy as np
import pytest


def make_synthetic_beats(n_beats=10, n_features=48):
    """Test için sentetik beat özellik dizisi (48 boyut = 12 lead × 4 özellik)."""
    return np.random.randn(n_beats, n_features).astype(np.float32)


class TestClinicalGraph:
    def test_node_count(self):
        from graph.builder import build_clinical_graph
        beats = make_synthetic_beats(n_beats=8)
        edge_index, node_features = build_clinical_graph(beats)
        assert node_features.shape[0] == 8

    def test_edge_index_shape(self):
        from graph.builder import build_clinical_graph
        beats = make_synthetic_beats(n_beats=6)
        edge_index, node_features = build_clinical_graph(beats)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

    def test_connectivity(self):
        from graph.builder import build_clinical_graph
        beats = make_synthetic_beats(n_beats=5)
        edge_index, _ = build_clinical_graph(beats)
        # En az n-1 kenar (bağlı graf için)
        assert edge_index.shape[1] >= 4

    def test_more_edges_than_single_group(self):
        """Union üç gruptan tek grup VG'den daha fazla ya da eşit kenar üretmeli."""
        from graph.builder import build_clinical_graph, LEAD_GROUPS
        from ts2vg import NaturalVG
        beats = make_synthetic_beats(n_beats=10)
        edge_index, node_features = build_clinical_graph(beats)
        # Tek grup kenar sayısı
        lead_indices = list(LEAD_GROUPS.values())[0]
        scalar = node_features[:, [li * 4 for li in lead_indices]].mean(axis=1).astype(np.float64)
        g = NaturalVG()
        g.build(scalar)
        single_edges = len(g.edges)
        union_edges = edge_index.shape[1] // 2  # bidirectional → undirected
        assert union_edges >= single_edges


class TestDTWWeights:
    def test_weight_range(self):
        from graph.builder import build_clinical_graph, compute_dtw_weights
        beats = make_synthetic_beats(n_beats=5)
        edge_index, _ = build_clinical_graph(beats)
        weights = compute_dtw_weights(beats, edge_index)
        w = weights if isinstance(weights, np.ndarray) else weights.numpy()
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_weight_symmetry(self):
        from graph.builder import build_clinical_graph, compute_dtw_weights
        beats = make_synthetic_beats(n_beats=4)
        edge_index, _ = build_clinical_graph(beats)
        weights = compute_dtw_weights(beats, edge_index)
        ei = edge_index if isinstance(edge_index, np.ndarray) else edge_index.numpy()
        w = weights if isinstance(weights, np.ndarray) else weights.numpy()
        edge_dict = {(ei[0, k], ei[1, k]): w[k] for k in range(ei.shape[1])}
        for (i, j), wij in edge_dict.items():
            if (j, i) in edge_dict:
                assert abs(edge_dict[(j, i)] - wij) < 1e-5


class TestPyGData:
    def test_data_fields(self):
        from graph.builder import build_graph
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            beats = make_synthetic_beats(n_beats=8)
            beats_path = os.path.join(tmp, "beats_test.npy")
            np.save(beats_path, beats)
            data = build_graph("test", beats_path, tmp, label=0)
            assert hasattr(data, "x")
            assert hasattr(data, "edge_index")
            assert hasattr(data, "edge_attr")
            assert hasattr(data, "y")

    def test_data_shapes(self):
        from graph.builder import build_graph
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            beats = make_synthetic_beats(n_beats=8)
            beats_path = os.path.join(tmp, "beats_test.npy")
            np.save(beats_path, beats)
            data = build_graph("test", beats_path, tmp, label=2)
            assert data.x.shape[0] == 8
            assert data.x.shape[1] == 48
            assert data.edge_index.shape[0] == 2
            assert data.edge_attr.shape[0] == data.edge_index.shape[1]
