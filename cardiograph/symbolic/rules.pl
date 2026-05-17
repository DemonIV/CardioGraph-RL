% AGENT: symbolic_agent
% AHA/ESC kılavuzu — PTB-XL 5-sınıf + AF/NSR klinik kurallar
% Özellikler Python tarafından assertz ile eklenir: feature(ad, deger)
%
% PTB-XL sınıfları: norm(0), mi(1), sttc(2), cd(3), hyp(4)
% Kullanım: feature/2 ile assert et, diagnosis/2 ile sorgula.

:- dynamic feature/2.

% ── MI (Miyokard Enfarktüsü) ─────────────────────────────────────
% ST elevasyonu > 0.1 mV (AHA kriterleri)
diagnosis(mi, [st_elevation]) :-
    feature(st_elevation_mv, X), X > 0.1.

% Patolojik Q dalgası — derinliği > 0.1 mV
diagnosis(mi, [pathological_q]) :-
    feature(q_wave_mv, X), X < -0.1.

% T inversiyonu — prekordiyal derivasyonlarda negatif skewness
diagnosis(mi, [t_inversion]) :-
    feature(t_inversion_score, X), X < -0.5.

% ── HYP (Hipertrofi) ─────────────────────────────────────────────
% Sokolow-Lyon: S_V1 + R_V5 > 3.5 mV
diagnosis(hyp, [sokolow_lyon]) :-
    feature(sokolow_lyon_mv, X), X > 3.5.

% Cornell kriterleri: R_aVL > 1.1 mV
diagnosis(hyp, [avl_voltage]) :-
    feature(avl_r_mv, X), X > 1.1.

% ── STTC (ST/T Değişikliği) ──────────────────────────────────────
% ST çökmesi > 0.05 mV
diagnosis(sttc, [st_depression]) :-
    feature(st_depression_mv, X), X < -0.05.

% T dalga morfoloji değişikliği
diagnosis(sttc, [t_wave_change]) :-
    feature(t_amplitude_change, X), X > 0.5.

% ── CD (İletim Bozukluğu) ────────────────────────────────────────
% Geniş QRS > 120 ms (dal bloğu)
diagnosis(cd, [wide_qrs]) :-
    feature(qrs_duration_ms, X), X > 120.

% ── NORM (Normal) ────────────────────────────────────────────────
% Patoloji bulgusu yoksa normal
diagnosis(norm, [normal_findings]) :-
    \+ (feature(st_elevation_mv, X1), X1 > 0.1),
    \+ (feature(q_wave_mv, X2), X2 < -0.1),
    \+ (feature(sokolow_lyon_mv, X3), X3 > 3.5),
    \+ (feature(st_depression_mv, X4), X4 < -0.05),
    \+ (feature(qrs_duration_ms, X5), X5 > 120).

% ── Geriye Dönük Uyumluluk: AF / NSR ────────────────────────────
diagnosis(atrial_fibrillation, [irregular_rr, absent_p_wave]) :-
    feature(rr_variability, X1), X1 > 0.15,
    feature(p_wave_amplitude, X2), X2 < 0.05.

diagnosis(normal_sinus_rhythm, [regular_rr, normal_qrs]) :-
    feature(rr_variability, X1), X1 < 0.05,
    feature(qrs_duration_ms, X2), X2 < 100,
    feature(p_wave_amplitude, X3), X3 > 0.1.
