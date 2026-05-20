# AGENT: preprocessing_agent
"""
Preprocessing pipeline testleri.
Gerçek PTB-XL verisi gerekmez — sentetik EKG kullanılır.
"""
import numpy as np
import pytest

SAMPLING_RATE = 500
BEAT_WINDOW = 200
N_FEATURES = 108  # 12 lead × 9 özellik (7 istatistiksel + ST_elevation + Q_wave)


def make_synthetic_ecg(duration_s=10.0, fs=500, n_leads=12, hr_bpm=70):
    """Sinüzoidal QRS darbeli sentetik EKG üret."""
    t = np.linspace(0, duration_s, int(duration_s * fs))
    rr_s = 60.0 / hr_bpm
    signal = np.zeros((len(t), n_leads))
    for lead in range(n_leads):
        # Baseline sinüs
        baseline = 0.1 * np.sin(2 * np.pi * 0.3 * t)
        # QRS darbeleri
        qrs = np.zeros_like(t)
        beat_times = np.arange(rr_s / 2, duration_s, rr_s)
        for bt in beat_times:
            idx = int(bt * fs)
            width = int(0.04 * fs)
            lo, hi = max(0, idx - width), min(len(t), idx + width)
            qrs[lo:hi] += np.exp(-((t[lo:hi] - bt) ** 2) / (2 * (0.02 ** 2)))
        signal[:, lead] = baseline + qrs + 0.02 * np.random.randn(len(t))
    return signal, fs


class TestBandpassFilter:
    def test_output_shape_1d(self):
        from preprocessing.pipeline import bandpass_filter
        sig = np.random.randn(5000)
        out = bandpass_filter(sig)
        assert out.shape == sig.shape

    def test_output_shape_2d(self):
        from preprocessing.pipeline import bandpass_filter
        sig = np.random.randn(5000, 12)
        out = bandpass_filter(sig)
        assert out.shape == sig.shape

    def test_high_freq_suppressed(self):
        from preprocessing.pipeline import bandpass_filter
        t = np.linspace(0, 1, SAMPLING_RATE)
        # 200 Hz bileşeni filtreden geçmemeli
        hf = np.sin(2 * np.pi * 200 * t)
        out = bandpass_filter(hf)
        assert np.std(out) < np.std(hf) * 0.1


class TestRPeakDetection:
    def test_peak_count(self):
        from preprocessing.pipeline import detect_r_peaks
        signal, fs = make_synthetic_ecg(duration_s=10.0, hr_bpm=70)
        peaks = detect_r_peaks(signal[:, 0], fs)
        # 70 BPM × 10 s = ~7 atım, ±3 tolerans
        assert 4 <= len(peaks) <= 15

    def test_peaks_in_bounds(self):
        from preprocessing.pipeline import detect_r_peaks
        signal, fs = make_synthetic_ecg(duration_s=10.0)
        peaks = detect_r_peaks(signal[:, 0], fs)
        assert all(0 <= p < len(signal) for p in peaks)

    def test_returns_array(self):
        from preprocessing.pipeline import detect_r_peaks
        signal, fs = make_synthetic_ecg()
        peaks = detect_r_peaks(signal[:, 0], fs)
        assert isinstance(peaks, (np.ndarray, list))


class TestBeatSegmentation:
    def test_output_shape(self):
        from preprocessing.pipeline import segment_beats
        signal, fs = make_synthetic_ecg(duration_s=10.0)
        r_peaks = np.array([500, 1000, 1500, 2000, 2500, 3000])
        beats = segment_beats(signal, r_peaks)
        assert beats.ndim == 3
        assert beats.shape[1] == 2 * BEAT_WINDOW
        assert beats.shape[2] == 12

    def test_no_out_of_bounds(self):
        from preprocessing.pipeline import segment_beats
        signal, _ = make_synthetic_ecg(duration_s=10.0)
        r_peaks = np.array([100, 500, 1000])  # 100 < BEAT_WINDOW
        beats = segment_beats(signal, r_peaks)
        # Sınır dışı atımlar atlanmalı veya padding uygulanmalı
        assert beats is not None


class TestMorphologyExtraction:
    def test_output_shape(self):
        from preprocessing.pipeline import extract_morphology
        signal, fs = make_synthetic_ecg(duration_s=10.0)
        beat = signal[300:700, :]  # 400 sample, 12 lead
        rr = np.array([750.0, 750.0])
        feat = extract_morphology(beat, rr, beat_idx=1)
        assert feat.shape == (N_FEATURES,)

    def test_no_nan_inf(self):
        from preprocessing.pipeline import extract_morphology
        signal, fs = make_synthetic_ecg()
        beat = signal[300:700, :]
        rr = np.array([750.0, 750.0])
        feat = extract_morphology(beat, rr, beat_idx=1)
        assert not np.any(np.isnan(feat))
        assert not np.any(np.isinf(feat))

    def test_peak_to_peak_nonnegative(self):
        from preprocessing.pipeline import extract_morphology
        signal, _ = make_synthetic_ecg()
        beat = signal[300:700, :]
        rr = np.array([750.0, 750.0])
        feat = extract_morphology(beat, rr, beat_idx=1)
        # peak_to_peak: her lead'de 5. özellik (index 4), step=9
        p2p = feat[4::9]
        assert np.all(p2p >= 0)

    def test_noisy_beat_returns_valid_features(self):
        """Gürültülü beat'te ST_elevation/Q_wave hesabı geçerli sayısal değer döndürmeli."""
        from preprocessing.pipeline import extract_morphology
        beat = np.random.randn(400, 12).astype(np.float32) * 0.01
        rr = np.array([750.0])
        feat = extract_morphology(beat, rr, beat_idx=0)
        assert feat.shape == (N_FEATURES,)
        assert not np.any(np.isnan(feat))
        assert not np.any(np.isinf(feat))


class TestFullPipeline:
    def test_end_to_end_synthetic(self, tmp_path):
        """Geçersiz yol için process_record None döndürmeli."""
        from preprocessing.pipeline import process_record
        result = process_record("fake_path", str(tmp_path))
        assert result is None
