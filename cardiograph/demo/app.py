# AGENT: demo_agent
"""
CardioGraph-RL Streamlit arayüzü — 3 sekme.
Çalıştırmak için (cardiograph/ dizininden):
    streamlit run demo/app.py --server.port 8050
"""
import sys
import shutil
import tempfile
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from demo.pipeline import CLASS_NAMES, CLASS_NAMES_TR, run_inference

# ── Sayfa Yapılandırması ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioGraph-RL",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("CardioGraph-RL — EKG Aritmi Tespiti")
st.caption(
    "Graf Dikkat Ağı (GAT) + Nöro-Sembolik Füzyon  |  PTB-XL  |  "
    "AUC-ROC=0.87  |  F1 macro=0.5974  |  GAT ECE=0.017"
)

# ── Sözlükler ─────────────────────────────────────────────────────────────
RULE_TR = {
    "st_elevation":      ("ST Elevasyonu",          "Prekordiyal derivasyonlarda ST yükselmesi >0.10 mV"),
    "pathological_q":    ("Patolojik Q Dalgası",    "İnferior derivasyonlarda Q dalgası <−0.10 mV"),
    "t_inversion":       ("T İnversiyonu",           "V3-V5'te negatif T dalgası skewness"),
    "sokolow_lyon":      ("Sokolow-Lyon Kriteri",    "S_V1 + R_V5 >3.50 mV — Sol ventrikül hipertrofisi"),
    "avl_voltage":       ("aVL Voltaj Artışı",       "aVL RMS >1.10 mV — Hipertrofi göstergesi"),
    "st_depression":     ("ST Çökmesi",              "İnferior/lateral derivasyonlarda ST <−0.05 mV"),
    "t_wave_change":     ("T Dalgası Değişikliği",   "Prekordiyal skewness yayılımı >0.50"),
    "wide_qrs":          ("Geniş QRS Kompleksi",     "QRS süresi >120 ms — Dal bloğu / iletim bozukluğu"),
    "normal_findings":   ("Normal Bulgular",          "Patolojik eşik aşılmadı"),
    "irregular_rr":      ("Düzensiz RR Aralığı",     "RR değişkenliği >0.15 — Olası AF"),
    "absent_p_wave":     ("P Dalgası Yokluğu",       "P amplitüdü <0.05 mV"),
    "regular_rr":        ("Düzenli RR Aralığı",      "RR değişkenliği <0.05"),
    "normal_qrs":        ("Normal QRS",              "QRS süresi <100 ms"),
}

DIAG_TR = {
    "mi":                  "Miyokard Enfarktüsü (MI)",
    "hyp":                 "Hipertrofi (HYP)",
    "sttc":                "ST-T Değişikliği (STTC)",
    "cd":                  "İletim Bozukluğu (CD)",
    "norm":                "Normal Ritim (NORM)",
    "atrial_fibrillation": "Atriyal Fibrilasyon",
    "normal_sinus_rhythm": "Normal Sinüs Ritmi",
}

CLASS_COLORS = {
    "NORM": "#27ae60",
    "MI":   "#e74c3c",
    "STTC": "#f39c12",
    "CD":   "#2980b9",
    "HYP":  "#8e44ad",
}

FEAT_TR = {
    "st_elevation_mv":    ("ST Elevasyonu",       "mV"),
    "q_wave_mv":          ("Q Dalgası",           "mV"),
    "t_inversion_score":  ("T İnversiyon Skoru",  ""),
    "sokolow_lyon_mv":    ("Sokolow-Lyon",        "mV"),
    "avl_r_mv":           ("aVL RMS",             "mV"),
    "st_depression_mv":   ("ST Çökmesi",          "mV"),
    "t_amplitude_change": ("T Amplitüd Değişimi", ""),
    "qrs_duration_ms":    ("QRS Süresi",          "ms"),
}

# ── Session State ──────────────────────────────────────────────────────────
for k, v in [("result", None), ("tmp_dir", None), ("error", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── 3 Sekme ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂  EKG Yükle", "🔬  Analiz", "📋  Gerekçe"])

# ══════════════════════════════════════════════════════════════════════════
# SEKME 1 — EKG Yükle
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("wfdb Kayıt Yükle")
    st.markdown(
        "PTB-XL formatı: **`.hea`** (başlık) ve **`.dat`** (sinyal) "
        "dosyalarını birlikte yükleyin."
    )

    col_hea, col_dat = st.columns(2)
    with col_hea:
        hea_file = st.file_uploader("Header dosyası (.hea)", type=["hea"])
    with col_dat:
        dat_file = st.file_uploader("Sinyal dosyası (.dat)", type=["dat"])

    ready = hea_file is not None and dat_file is not None
    analyze_btn = st.button(
        "🔍 Analiz Et", disabled=not ready, type="primary", use_container_width=False
    )

    if analyze_btn and ready:
        if st.session_state.tmp_dir:
            shutil.rmtree(st.session_state.tmp_dir, ignore_errors=True)

        tmp = tempfile.mkdtemp()
        st.session_state.tmp_dir = tmp
        stem = Path(hea_file.name).stem

        (Path(tmp) / f"{stem}.hea").write_bytes(hea_file.read())
        (Path(tmp) / f"{stem}.dat").write_bytes(dat_file.read())

        with st.spinner("Model yükleniyor ve analiz ediliyor..."):
            try:
                st.session_state.result = run_inference(str(Path(tmp) / stem))
                st.session_state.error  = None
            except Exception as exc:
                st.session_state.result = None
                st.session_state.error  = str(exc)

    if st.session_state.error:
        st.error(f"Hata: {st.session_state.error}")

    if st.session_state.result:
        r          = st.session_state.result
        signal     = r["signal"]
        lead_names = r["lead_names"]
        fs         = r["fs"]
        T          = signal.shape[0]
        t_axis     = np.arange(T) / fs
        n_leads    = signal.shape[1]

        st.success(
            f"Kayıt yüklendi — {T/fs:.1f} sn  |  {r['n_beats']} beat  |  "
            f"{n_leads} lead  |  {fs} Hz"
        )

        fig = make_subplots(
            rows=n_leads, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.004,
        )
        for i in range(n_leads):
            name = lead_names[i] if i < len(lead_names) else f"L{i+1}"
            fig.add_trace(
                go.Scatter(
                    x=t_axis, y=signal[:, i],
                    name=name,
                    line=dict(width=0.7, color="#2c3e50"),
                    hovertemplate=f"<b>{name}</b>  t=%{{x:.3f}}s  mV=%{{y:.4f}}<extra></extra>",
                ),
                row=i + 1, col=1,
            )
            fig.update_yaxes(
                title_text=name, title_standoff=2,
                tickfont=dict(size=8), row=i + 1, col=1,
            )
        fig.update_xaxes(title_text="Zaman (s)", row=n_leads, col=1)
        fig.update_layout(
            height=55 * n_leads + 50,
            showlegend=False,
            margin=dict(l=55, r=10, t=30, b=40),
            title=dict(text="12-Lead EKG", font=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif not ready:
        st.info("Analiz için .hea ve .dat dosyalarını yükleyin, ardından 'Analiz Et'e tıklayın.")

# ══════════════════════════════════════════════════════════════════════════
# SEKME 2 — Analiz
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.result is None:
        st.info("Önce **EKG Yükle** sekmesinde bir kayıt yükleyin ve 'Analiz Et'e tıklayın.")
    else:
        r           = st.session_state.result
        pred        = r["pred_class"]
        pred_tr     = r["pred_class_tr"]
        gat_probs   = r["gat_probs"]
        fused_probs = r["fused_probs"]
        beat_attn   = r["beat_attention"]
        conf        = float(gat_probs[CLASS_NAMES.index(pred)])

        # ── Tanı + Güven + Bant ───────────────────────────────────────────
        st.subheader("Tanı")
        col_d, col_c, col_b = st.columns([2, 1, 2])

        with col_d:
            color = CLASS_COLORS.get(pred, "#7f8c8d")
            st.markdown(
                f"<div style='background:{color};padding:18px;border-radius:10px;"
                f"color:white;font-size:22px;font-weight:bold;text-align:center'>"
                f"{pred_tr}</div>",
                unsafe_allow_html=True,
            )

        with col_c:
            st.metric(
                label="GAT Güven (T=1.026, ECE=0.017)",
                value=f"{conf * 100:.1f}%",
            )

        with col_b:
            if conf >= 0.80:
                band, bc = "Yüksek Güven", "#27ae60"
            elif conf >= 0.60:
                band, bc = "Orta Güven", "#f39c12"
            else:
                band, bc = "Düşük Güven — Klinisyen Onayı Önerilir", "#e74c3c"
            st.markdown(
                f"<div style='background:{bc};padding:11px;border-radius:8px;"
                f"color:white;font-size:13px;text-align:center;margin-top:8px'>"
                f"{band}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Beat Attention Heatmap ────────────────────────────────────────
        st.subheader("Beat Attention Isı Haritası")
        st.caption(
            "Her hücre bir kalp atımı. Renk yoğunluğu (kırmızı = yüksek) "
            "GAT 3. katman dikkat ağırlığını gösterir."
        )
        n_beats = len(beat_attn)
        fig_h = go.Figure(go.Heatmap(
            z=[beat_attn.tolist()],
            x=[f"Beat {i + 1}" for i in range(n_beats)],
            y=["Dikkat"],
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            colorbar=dict(title="Dikkat", thickness=14, len=0.7),
            hovertemplate="<b>%{x}</b><br>Dikkat: %{z:.3f}<extra></extra>",
        ))
        fig_h.update_layout(
            height=150,
            margin=dict(l=10, r=10, t=10, b=35),
            xaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        st.divider()

        # ── Sınıf Olasılıkları (GAT vs Füzyon) ───────────────────────────
        st.subheader("Sınıf Olasılıkları")
        st.caption(
            "GAT kalibre (ECE=0.017, güven için kullan) ile "
            "Nöro-Sembolik Füzyon (alpha=0.35, tanı kararı için kullan)."
        )
        bar_colors = [CLASS_COLORS.get(c, "#7f8c8d") for c in CLASS_NAMES]
        fig_b = go.Figure([
            go.Bar(
                name="GAT Kalibre (ECE=0.017)",
                x=CLASS_NAMES,
                y=(gat_probs * 100).tolist(),
                marker_color=bar_colors,
                opacity=0.95,
                text=[f"{v*100:.1f}%" for v in gat_probs],
                textposition="outside",
            ),
            go.Bar(
                name="Nöro-Sembolik Füzyon (ECE=0.157)",
                x=CLASS_NAMES,
                y=(fused_probs * 100).tolist(),
                marker_color=bar_colors,
                opacity=0.45,
                text=[f"{v*100:.1f}%" for v in fused_probs],
                textposition="outside",
            ),
        ])
        fig_b.update_layout(
            barmode="group",
            yaxis=dict(title="Olasılık (%)", range=[0, 112]),
            xaxis=dict(title="Sınıf"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            height=370,
            margin=dict(l=45, r=10, t=50, b=40),
        )
        st.plotly_chart(fig_b, use_container_width=True)

        st.info(
            "**Kalibrasyon Notu:** Güven skoru GAT katmanından alınmaktadır "
            "(ECE=0.017, T=1.026 — iyi kalibre). Sembolik füzyon deterministik kural "
            "çıktısı kullandığından Füzyon ECE=0.157 daha yüksektir; tanı kararı için "
            "füzyon, güven gösterimi için GAT kullanılır."
        )

# ══════════════════════════════════════════════════════════════════════════
# SEKME 3 — Gerekçe
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.result is None:
        st.info("Önce **EKG Yükle** sekmesinde analiz yapın.")
    else:
        r     = st.session_state.result
        rules = r["symbolic_rules"]
        clin  = r["clinical_features"]

        # ── Prolog Kuralları ──────────────────────────────────────────────
        st.subheader("Tetiklenen Prolog Kuralları")
        if not rules:
            st.warning("Hiçbir sembolik kural tetiklenmedi.")
        else:
            for rule in rules:
                diag         = rule["diagnosis"]
                reasons      = rule["reasons"]
                diag_display = DIAG_TR.get(diag, diag.upper())
                with st.expander(f"**{diag_display}**", expanded=True):
                    for reason in reasons:
                        name_tr, desc = RULE_TR.get(reason, (reason, "—"))
                        st.markdown(f"- **{name_tr}:** {desc}")

        st.divider()

        # ── Klinik Proxy Değerleri ────────────────────────────────────────
        st.subheader("Klinik Proxy Değerleri")
        st.caption("Prolog motoruna giren ham ölçümler (preprocessing özelliklerinden türetildi)")

        cols = st.columns(4)
        for i, (key, (label, unit)) in enumerate(FEAT_TR.items()):
            val = clin.get(key, 0.0)
            cols[i % 4].metric(
                label=label,
                value=f"{val:.3f} {unit}".strip(),
            )

        st.divider()

        # ── Uyarılar ─────────────────────────────────────────────────────
        st.warning(
            "**Klinik Uyarı:** Füzyon güven kalibrasyonu sınırlıdır (ECE=0.157). "
            "Bu sistem bir karar destek aracıdır; nihai klinik değerlendirme "
            "klinisyen tarafından yapılmalıdır."
        )
        st.info(
            "**Faithfulness@3 = 0.383** — Attention ağırlıklarının yaklaşık %38'i "
            "klinik bulguyla örtüşmektedir. Rastgele seçimden (%20) yüksek, "
            "ancak klinik yorumda dikkatli kullanılmalıdır."
        )
