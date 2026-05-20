# AGENT: preprocessing_agent
"""
EKG sinyal önişleme pipeline.
Giriş: wfdb kayıt yolu
Çıkış: data/processed/beats/beats_{id}.npy — shape (N_beats, 108)
"""
import numpy as np
import wfdb
import neurokit2 as nk
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
from pathlib import Path

SAMPLING_RATE = 500
BEAT_WINDOW = 200
N_FEATURES = 108  # 12 lead × 9 özellik

# Özellik sırası per lead (9):
#   0: mean  1: std  2: skewness  3: kurtosis  4: peak_to_peak
#   5: spectral_centroid  6: zero_crossings  7: ST_elevation  8: Q_wave

# ST ve Q için R-peak'e göreli pencere sabit hesaplama (ecg_delineate YOK)
_ST_OFFSET_MS  = 120   # R + 120ms → J noktası tahmini (~R+40ms) + 80ms
_QW_ONSET_MS   = 30    # R - 30ms : R → Q dalgası arama penceresi


def bandpass_filter(signal, fs=SAMPLING_RATE):
    """0.5-40 Hz Butterworth bandpass, 4. derece."""
    sos = butter(4, [0.5, 40.0], btype='bandpass', fs=fs, output='sos')

    def _filt(x):
        n = len(x)
        pad = min(n - 1, int(2 * fs))
        p = np.pad(x, pad, mode='wrap')
        return sosfiltfilt(sos, p)[pad:pad + n]

    if signal.ndim == 1:
        return _filt(signal)
    return np.column_stack([_filt(signal[:, i]) for i in range(signal.shape[1])])


def detect_r_peaks(signal_1d, fs=SAMPLING_RATE):
    """neurokit2 Pan-Tompkins. Döndürür: sample index dizisi."""
    sig = bandpass_filter(signal_1d, fs)
    _, info = nk.ecg_peaks(sig, sampling_rate=fs, method='pantompkins1985')
    return info['ECG_R_Peaks']


def segment_beats(signal, r_peaks):
    """R ±BEAT_WINDOW sample. Döndürür: (N_beats, 2*BEAT_WINDOW, 12)."""
    n_samples = signal.shape[0]
    beats = []
    for r in r_peaks:
        lo, hi = r - BEAT_WINDOW, r + BEAT_WINDOW
        if lo >= 0 and hi <= n_samples:
            beats.append(signal[lo:hi, :])
    if not beats:
        return np.empty((0, 2 * BEAT_WINDOW, signal.shape[1]))
    return np.stack(beats, axis=0)


def extract_morphology(beat, rr_intervals, beat_idx, fs=SAMPLING_RATE):
    """
    108-dim özellik: 12 lead × 9 özellik.

    Per lead sırası:
      [0] mean          [1] std           [2] skewness     [3] kurtosis
      [4] peak_to_peak  [5] energy        [6] zero_crossings
      [7] ST_elevation  [8] Q_wave

    ST ve Q, R-peak'e (beat merkezi) göreli sabit offset ile hesaplanır.
    ecg_delineate kullanılmaz — her beat'te deterministik çalışır.
    """
    n_samples = beat.shape[0]
    n_leads   = beat.shape[1]
    r_idx     = n_samples // 2                      # R-peak beat penceresinin tam merkezinde

    st_sample = r_idx + int(_ST_OFFSET_MS * fs / 1000)   # R + 120ms
    qw_lo     = r_idx - int(_QW_ONSET_MS  * fs / 1000)   # R - 30ms
    qw_hi     = r_idx                                      # R

    feats = []
    for lead in range(n_leads):
        seg = beat[:, lead].astype(np.float64)

        # ── 7 istatistiksel özellik ──────────────────────────────────────
        mean_v = float(np.mean(seg))
        std_v  = float(np.std(seg))
        skew_v = float(scipy_skew(seg))     if len(seg) > 2 else 0.0
        kurt_v = float(scipy_kurtosis(seg)) if len(seg) > 3 else 0.0
        p2p_v  = float(np.max(seg) - np.min(seg))
        freqs  = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        power  = np.abs(np.fft.rfft(seg)) ** 2
        sc_v   = float(np.sum(freqs * power) / (np.sum(power) + 1e-8))
        zcr_v  = float(np.sum(np.diff(np.sign(seg)) != 0) / max(len(seg) - 1, 1))

        # ── 2 klinik özellik (R-peak'e göreli, ecg_delineate YOK) ───────
        st_elev = float(seg[st_sample]) if 0 <= st_sample < n_samples else 0.0

        lo = max(0, qw_lo)
        if lo < qw_hi:
            min_val = float(np.min(seg[lo:qw_hi]))
            q_wave  = min_val if min_val < 0.0 else 0.0
        else:
            q_wave = 0.0

        feats.extend([mean_v, std_v, skew_v, kurt_v, p2p_v, sc_v, zcr_v,
                      st_elev, q_wave])

    feat = np.array(feats, dtype=np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def process_record(record_path, output_dir):
    """Tek kayıt işle, .npy kaydet. Döndürür: dict veya None."""
    try:
        record = wfdb.rdrecord(str(record_path))
    except Exception:
        return None

    signal = record.p_signal
    if signal is None or signal.ndim < 2:
        return None

    fs = record.fs
    record_id = Path(record_path).stem

    signal_f  = bandpass_filter(signal, fs)
    lead_idx  = min(1, signal_f.shape[1] - 1)
    r_peaks   = detect_r_peaks(signal_f[:, lead_idx], fs)

    if len(r_peaks) < 2:
        return None

    rr_intervals = np.diff(r_peaks).astype(float)
    beats        = segment_beats(signal_f, r_peaks)

    if beats.shape[0] == 0:
        return None

    n_beats  = beats.shape[0]
    features = np.zeros((n_beats, N_FEATURES), dtype=np.float32)
    for i in range(n_beats):
        rr_ctx      = rr_intervals[max(0, i - 1):i + 1] if i > 0 else rr_intervals[:1]
        features[i] = extract_morphology(beats[i], rr_ctx, beat_idx=i, fs=fs)

    out_path = Path(output_dir) / f"beats_{record_id}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, features)

    return {"record_id": record_id, "n_beats": n_beats, "output_path": str(out_path)}


def process_batch(record_list, output_dir, max_records=None):
    """Toplu işle. Döndürür: {processed, failed, results}."""
    if max_records is not None:
        record_list = record_list[:max_records]
    results   = []
    processed = 0
    failed    = 0
    for rec in record_list:
        r = process_record(rec, output_dir)
        if r is not None:
            results.append(r)
            processed += 1
        else:
            failed += 1
    return {"processed": processed, "failed": failed, "results": results}
