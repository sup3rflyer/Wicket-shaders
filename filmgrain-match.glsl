// Small-template (TPL) architecture since 2026-07-10: grain is generated into
// a 960x540 toroidal vocabulary and assembled in normalized active-picture
// space from per-block randomized template windows, AV1-FGS style. The current
// 2160-sample picture-height lattice is an implementation bandwidth/calibration
// choice, not a film format or a "grain resolution": source-matched correlation
// length stays picture-relative at every source/output resolution. This file is
// CANONICAL and hand-maintained;
// the pre-TPL full-field A/B build is archived outside the public release repo.
// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE
//
// Film Grain — MATCH+ — complementary grain remastering.
// ============================================================================
// Learns a persistent title grain model from the best stochastic evidence that
// survives delivery, commits a sensitivity-aware presentation at shot cuts, and
// restores per-luma power that compression or finishing erased. Film, sensors and
// animation masters all carry acquisition/finishing noise: weak measurement shifts
// weight toward the title prior, never toward a fictitious noiseless master.
//
// The observer is hidden scratch. It measures eight exposure-weighted luma zones,
// three picture-relative spatial bands, temporal authenticity, coverage, motion,
// cut distance and delivery rolloff. It updates the title posterior slowly. The
// visible shot model may adapt quickly only in the short perceptual window after a
// real cut; inside a shot all upward change is deliberately slow and motion can
// only freeze learning. Independent source and synthetic powers combine in
// quadrature over the untouched source.
//
// Architecture (LUMA measure + source-locked template gen, OUTPUT composite):
//   PASS 1 - Compute 32x32 at LUMA, measurement only: saves to a 1x1 dummy
//            (WIDTH/HEIGHT 1 -> a single workgroup; the LUMA plane itself is
//            untouched -- the old full-plane passthrough copy was the shader's
//            entire measurable GPU cost at 4K). Samples HOOKED.r across the
//            source raster with a fixed cut/matte probe, remaps the grain
//            observer into the committed active picture, and writes only
//            GRAIN_STATE.
//   PASS 2 - Compute 32x32 at LUMA: conditionally regenerates the persistent
//            960x540 grain vocabulary; skipped ticks retain it while PASS 3
//            still chooses a fresh arrangement.
//   PASS 3 - Compute 32x32 at OUTPUT: assembles the picture-space field from
//            randomized template windows and composites it only
//            inside the committed active picture.
//
// Runtime params. Only match_grain / debug_match / value_warp are on keys (F3 / Ctrl+F3 /
// Alt+F3 via shader-toggle.lua); the rest are tuned by editing the DEFAULT value under each
// param block below and reloading (the lua no longer overrides them). The param blocks are
// ordered to match these groups. (Comments cannot sit between the param blocks -- the parser
// rejects it -- so every param is documented HERE.)
//
//  == CONTROL ===============================================================
//   match_grain      0 = no synthetic output, 1 = Match Grain+; observation
//                    continues so live A/B does not cold-start.                [F3]
//   debug_match      compact 52-row machine-readable posterior/geometry overlay. [Ctrl+F3]
//   state_epoch      harness reset token; bump once per source file.
//   grain_pause      machine-owned pause input; freezes temporal state; baked
//                    look edits may rebuild the standing field in place.
//
//  == GRAIN LOOK (the dials to tune by eye) =================================
//   grain_gain       overall grain AMOUNT/strength (1 = calibrated; up to 12).
//   grain_size       grain SIZE: <1 finer, >1 coarser (1.2 = default).
//   grain_contrast   spectral hardness: 0 = soft/lowpass, 1 = sandpaper bandpass, up to 2 =
//                    more DC removed / peppery (difference-of-Gaussians).
//   value_warp       VALUE-domain contrast: 0 = Gaussian (bit-identical), ~2 hard, ~3 extreme
//                    = bimodal/high-per-grain-contrast (CyberCity "harsh"). Amplitude-
//                    preserving; the value-domain cousin of grain_contrast.    [Alt+F3]
//   grain_sharpness  measured-size influence: 0.3 = default, 1 = literal read.
//   grain_rate       visible temporal cadence: fraction of SOURCE frames that
//                    choose a fresh on-screen arrangement (1 = on ones; 0.5 =
//                    on twos). Source-locked and display-refresh independent.
//   grain_gen_rate   template-vocabulary regeneration rate as a fraction of
//                    visible grain ticks. 1 = every tick; 0.25 = every fourth.
//                    Arrangement still rehashes on every visible tick, so this
//                    is a small performance option, not a boil-speed control.
//   grain_base_sat   explicit per-channel generator prior. 0.25 = calibrated
//                    look; 0 = true mono. Off-default values drift per-channel
//                    grain RMS by a few percent.
//
//  == RESTORATION (how much grain to rebuild on degraded sources) ===========
//   restore_gain     missing-power lane only: 0 = character complement C,
//                    1 = inferred target, >1 = explicit amplitude override;
//                    values near 5 are aggressive shot-specific overrides,
//                    not a preset; 6 is the expert ceiling.
//
//  == PIPELINE / SIZING =====================================================
//   density_combine  0 = additive, 1 = multiplicative density.
//
//  == OUTPUT CHAIN (what the OUTPUT hook is handed) =========================
//   grain_hdr        the player's OUTPUT transfer, NOT a look knob. libplacebo
//                    runs OUTPUT hooks AFTER the conversion to the target
//                    colorspace, so this pass receives target-space codes: set
//                    1 whenever mpv's target-trc resolves to pq (an explicit
//                    target-trc=pq, or an HDR-signalled display under
//                    target-colorspace-hint), else 0. It is independent of
//                    which shaders precede us. Wrong value = the model is
//                    measured on gamma SDR source codes at LUMA but keyed and
//                    applied onto PQ codes here: the highlight fade and black
//                    gate both go inert and grain lands off its tone bell.
//   grain_headroom   whether the CONTENT extends above reference white, as
//                    opposed to grain_hdr which describes the container. Only
//                    read when grain_hdr = 1. 1 = an upstream SDR->HDR stage
//                    (CelFlare) expands highlights past ref white: work-domain
//                    1.0 is paper white with real detail above, so grain fades
//                    just ABOVE it. 0 = plain SDR carried in a PQ container:
//                    work-domain 1.0 IS the source clip ceiling, so the SDR
//                    clip fade and the near-clip channel clamp apply exactly
//                    as on an SDR target. Wrong value = grain in clipped
//                    whites (1 on SDR content) or grain shaved off real
//                    expanded highlights (0 under CelFlare).
//   grain_fade       where grain fades to white: the work-domain luma at
//                    which grain reaches zero. ONE knob for every chain --
//                    below 0.5 it also widens the rise out of black; between
//                    that toe and the upper fade, amount follows the title's
//                    own rendition curve (no aesthetic shoulder). Clip-limited
//                    chains (grain_hdr = 0, or grain_headroom = 0) cap the
//                    effective top at 0.95 (the near-clip dead zone), so
//                    the 1.10 default is stock on every chain; with
//                    headroom it is the above-ref-white reach -- we cannot
//                    know the upstream expansion's tuning, so match it by
//                    eye, per source if needed. Below ref white the knob
//                    moves the LUMA fade only: the per-channel grain bound
//                    stays floored at ref white (overshoot physics), so
//                    bright saturated channels keep their grain. Values near
//                    0.2-0.3 aggressively confine grain to low luminance.
//   grain_ref_white  the nit level the chain anchors SDR white to; the bridge
//                    divides by it, so it is DESCRIPTIVE -- it must equal what
//                    the chain actually did, not what we would prefer. Plain
//                    mpv PQ output anchors at hdr-reference-white, which is
//                    "auto" by default = libplacebo's BT.2408 203 nits (hence
//                    the 203 default; measured 202.4 on this chain). shampv
//                    syncs this from hdr-reference-white ONLY while that is
//                    pinned numeric -- so PIN IT, and any upstream SDR->PQ
//                    encoder's own reference white gets pinned to the same
//                    number and the whole chain stays coherent. If an upstream
//                    shader owns the SDR->PQ encode and you leave
//                    hdr-reference-white on auto, nothing syncs: match that
//                    shader's reference white here by hand or grain keys off
//                    the wrong point on the bell. Measured invariant: a wrong
//                    ref white is a pure KEYING error, never a colour error --
//                    the bridge is a self-consistent inverse pair for any
//                    value, round-tripping within 1 LSB.
//                    Three standing limits of the bridge, all verified by
//                    measurement and all fine under the author's pinned config
//                    (target-prim=bt.2020, bt709 sources, target-peak 1000):
//                    (a) the 2020->709 matrix is HARDCODED, so it assumes
//                    target-prim=bt.2020 -- under target-prim=display-p3 (and
//                    target-prim defaults to auto) the matrix is simply wrong;
//                    (b) the 2.4 inverse assumes a bt709/bt1886-tagged source
//                    (exact there; an sRGB-tagged source mis-keys ~+5.7% at
//                    code 0.5, ~+25% at 0.06); (c) the anchor holds only while
//                    frame peak stays under target-peak -- above it libplacebo's
//                    spline tone map engages and shifts what we are keying off.
// ============================================================================

// shampv shader API (plain comments to libplacebo). All params are DYNAMIC:
// glsl-shader-opts changes apply next frame, no recompile; bump state_epoch
// to invalidate the persisted GRAIN_STATE live.
//@shampv input sdr
//@shampv ref-white-param grain_ref_white
//@shampv target-trc-param grain_hdr
//@shampv pause-param grain_pause
//@shampv epoch-param state_epoch
//@shampv toggle match_grain grain_hdr grain_headroom debug_match density_combine
//@shampv measures LUMA

//!PARAM match_grain
//!DESC Match Grain+ mix (shampv A/B toggle). 1 = complementary remastering · 0 = no synthetic output while observation remains live · intermediate mix values stay valid via opts.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM debug_match
//!DESC Debug overlay (toggle). 1 = compact 52-row Match Grain+ posterior/geometry readout · 0 = normal output.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM state_epoch
//!DESC Persisted-state reset token — bump by at least 1 to wipe saved grain state live (once per source file). The magnitude itself is meaningless.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 65535.0
0.0

//!PARAM grain_pause
//!DESC Machine-owned pause state. 1 freezes observation, cadence and arrangement so redraws of a paused frame are bit-stable; baked look edits may rebuild the standing field once. shampv mirrors mpv pause automatically.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM grain_gain
//!DESC Overall grain amount. ↑ stronger / more visible · ↓ fainter, 0 = none. Default 1 = calibrated restoration; >1 is an artistic/evaluation override.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 12.0
1.0

//!PARAM grain_size
//!DESC Grain cell size. ↓ finer · ↑ coarser. Default 1.2 = restoration scan character; 1 = neutral scale.
//!TYPE DYNAMIC float
//!MINIMUM 0.3
//!MAXIMUM 2.5
1.2

//!PARAM grain_contrast
//!DESC Spectral hardness (difference-of-Gaussians). ↓ toward 0 = soft/lowpass · ↑ toward 2 = crisper grain edges / more DC stripped. Default 2.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 2.0
2.0

//!PARAM value_warp
//!DESC Value-domain contrast, amplitude-preserving. 0 = Gaussian (bit-identical) · ↑ ~2 = hard, ~3 = bimodal/harsh. The amplitude-domain cousin of grain_contrast.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 4.0
0.0

//!PARAM grain_sharpness
//!DESC Measured-size influence. 1 = literal compact-observer estimate · 0 = neutral generator. Default 0.3 keeps source character without inheriting delivery softness.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.3

//!PARAM grain_rate
//!DESC Visible arrangement cadence in SOURCE frames. 1 = fresh arrangement every frame / on ones · 0.5 = on twos. ↓ slows the boil. Display-refresh independent.
//!TYPE DYNAMIC float
//!MINIMUM 0.1
//!MAXIMUM 1.0
1.0

//!PARAM grain_gen_rate
//!DESC Template-vocabulary regeneration rate as a fraction of visible grain ticks. 1 = every tick (default) · 0.25 = every fourth tick. Block arrangement still rehashes every visible tick, but finite-vocabulary reuse can change higher-order temporal correlation; lower values are an optional small performance trade, not an identical fast path.
//!TYPE DYNAMIC float
//!MINIMUM 0.1
//!MAXIMUM 1.0
1.0

//!PARAM grain_base_sat
//!DESC Generator chroma prior. 0 = mono · ↑ = more channel independence. This is explicit prior character, not source-measured chroma.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.25

//!PARAM restore_gain
//!DESC Missing-power authority. 0 = complement only (C) · 1 = inferred missing-power target · >1 = manual amplitude override for known-damaged material · near 5 is aggressive and shot-specific, not a preset · 6 is the expert ceiling · above 2 restored power rises roughly with the square and can severely overgrain intact material.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 6.0
1.0

//!PARAM density_combine
//!DESC Grain combine mode (toggle). 1 = multiplicative density, rides brightness like film (shipped) · 0 = additive.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM grain_hdr
//!DESC Output transfer — match mpv's target-trc. 1 = PQ BT.2020 out: grain keyed/applied in SDR via a PQ bridge, fades above ref white · 0 = plain SDR (bit-identical).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM grain_headroom
//!DESC Content headroom above ref white — only used when grain_hdr = 1. 1 = upstream SDR→HDR expansion: highlights extend past ref white, grain fades just above it · 0 = plain SDR in a PQ container: ref white is the clip ceiling, SDR clip fade + near-clip channel clamp apply.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM grain_fade
//!DESC Where grain fades to white — work-domain luma of full fade-out, one knob for every chain. Default 1.10 = stock top end (clip-limited chains cap the effective top at 0.95) · 0.2-0.3 confines grain to low luminance and widens its rise out of black · with grain_headroom = 1 it is the above-ref-white reach — tune to the upstream expansion by eye · below ref white the knob moves the luma envelope only (the per-channel grain bound stays floored at ref white) · between the shadow toe and upper fade, amount follows the title's rendition curve.
//!TYPE DYNAMIC float
//!MINIMUM 0.2
//!MAXIMUM 2.5
1.10

//!PARAM grain_ref_white
//!DESC SDR reference white (nits) the output chain anchors to — match hdr-reference-white (203 = its auto anchor). Only used when grain_hdr = 1.
//!TYPE DYNAMIC float
//!MINIMUM 80.0
//!MAXIMUM 480.0
203.0

//!BUFFER GRAIN_STATE
//!VAR float m_observed
//!VAR float m_structure_ratio
//!VAR float m_measured
//!VAR float m_prev_ready
//!VAR float m_state_magic
//!VAR float m_state_epoch
//!VAR float m_coverage
//!VAR float m_motion
//!VAR float m_cut_score
//!VAR float m_gen_frame
//!VAR float m_eff_render
//!VAR float m_title_conf
//!VAR float m_title_power
//!VAR float m_temporal_support
//!VAR float m_est_missing
//!VAR float m_loss_conf
//!VAR float m_loss_mix
//!VAR float m_tone_conf
//!VAR float m_title_size
//!VAR float m_title_hardness
//!VAR float m_shot_size
//!VAR float m_shot_hardness
//!VAR float m_shot_age
//!VAR float m_shot_conf
//!VAR float m_shot_gain
//!VAR float m_shot_restore_boost
//!VAR float m_master_p[8]
//!VAR float m_master_w[8]
//!VAR float m_shot_obs_p[8]
//!VAR float m_shot_obs_w[8]
//!VAR float m_char_p[8]
//!VAR float m_restore_p[8]
//!VAR float m_arr_seed
//!VAR float m_field_seed
//!VAR float m_regen
//!VAR float m_regen_pending
//!VAR float m_field_valid
//!VAR float m_field_cov_rg
//!VAR float m_field_cov_rb
//!VAR float m_field_cov_gb
//!VAR float m_field_var_r
//!VAR float m_field_var_g
//!VAR float m_field_var_b
//!VAR float m_baked_grain_size
//!VAR float m_baked_grain_contrast
//!VAR float m_baked_value_warp
//!VAR float m_baked_grain_base_sat
//!VAR float m_baked_grain_sharpness
//!VAR float m_baked_match_grain
//!VAR float prev_grid[4096]
//!VAR float prev_grid_off[4096]
//!VAR float m_source_aspect
//!VAR float m_active_inset_x
//!VAR float m_active_inset_y
//!VAR float m_pending_inset_x
//!VAR float m_pending_inset_y
//!VAR float m_geom_streak
//!VAR float m_geom_streak_y
//!VAR float m_geom_known
//!VAR float m_geom_known_y
//!VAR float m_geom_blackout
//!VAR float m_geom_blackout_y
//!VAR float m_geom_changed
//!VAR float prev_probe[4096]
//!VAR float m_pan_px
//!STORAGE

//!TEXTURE GRAIN_FIELD
//!SIZE 960 540
//!FORMAT rgba16f
//!STORAGE

//!HOOK LUMA
//!BIND HOOKED
//!BIND GRAIN_STATE
//!SAVE GRAIN_STATS
//!WIDTH 1
//!HEIGHT 1
//!COMPUTE 32 32
//!DESC Film Grain Match: LUMA measure

#define AMP_BINS               32
#define MP_GRID_W              64
#define MP_GRID_H              64
#define MP_GRID_N              (MP_GRID_W * MP_GRID_H)
#define MP_TONE_BINS           8
#define MP_BANDS               3
#define MP_AMP_MAX             0.008
#define MP_TEMP_MAX            0.002
#define MP_HP_TO_SOURCE        1.7888544
#define MP_MEDABS_TO_STD       1.4826022
#define MP_COMPLEMENT_POWER    0.150
// Normalized picture-space calibration density. Distances below are physical
// fractions of active picture height expressed on this finite synthesis
// lattice. Raising output resolution reveals the same grain more faithfully;
// it does not make the source's grains smaller or redefine their identity.
#define MP_PICTURE_DENSITY     2160.0
// MUST equal PICTURE_DENSITY / PICTURE_DENSITY_OUT in PASS 2 / PASS 3.
// These are separate translation units and have no compile-time cross-check.
// Surviving delivered grain is DAMAGED evidence — clumped, blocked,
// DCT-mushed — so it counts against the restoration deficit at its
// fidelity, never at its raw energy (author spec 2026-07-17: the shader
// lays an even bed of restoration-grade grain across the picture; a
// JND-level addition on grainy titles is a failure, because even inside
// surviving grain we are repairing compressed patches and clumping).
// 0.50 = provisional: delivered grain is at best half restoration-grade.
// Future: key this to measured delivery health (m_structure_ratio) and
// the ProRes master-vs-degraded erasure calibration.
#define MP_SURVIVOR_FIDELITY   0.50
#define MP_PRIOR_SIGMA         0.00055
// Effective midtone output RMS of a unit control after density application and
// the measured tone basis.
#define MP_FIELD_STD           0.0185
#define MP_STATE_MAGIC         0.949451
#define MP_MIN_BIN_SAMPLES     24u
// Film-plausible evidence band. Per-frame sigma above this band is not
// photographic grain (fireworks, confetti, dense near-field rain, damage):
// soft-reject it from BOTH the per-bin master update and title presence,
// upstream of every lane, so implausible evidence cannot enter the
// persistent posterior at all. Calibrated ABOVE the heaviest catalogued
// legitimate grain — Golden Spurtle Super35 reads sigma 0.0026-0.0035 in
// this domain (harness, 2026-07-17) and must pass at full weight; in-band
// grain twins are bounded by the rate/breadth/reversibility layers instead,
// a level test cannot catch them. The absolute master ceiling backstops
// sustained band-edge evidence.
#define MP_SIGMA_PLAUS_LO      0.0040
#define MP_SIGMA_PLAUS_HI      0.0065
#define MP_MASTER_P_MAX        2.0e-5
#define MP_BAR_DARK_MAX        0.085
#define MP_BAR_RANGE_MAX       0.025
#define MP_BAR_DARK_SAMPLES    36u
#define MP_BAR_MIN_CELLS       2
#define MP_BAR_SIGNAL_MIN      32u
#define MP_BAR_PICTURE_RANGE   0.05
#define MP_BAR_EDGE_PICTURE    48u
#define MP_BAR_LEVEL_MAX       0.008
#define MP_ACTIVE_INSET_MAX    0.24
#define MP_BLACKOUT_CODE_MAX   0.075
#define MP_BLACKOUT_SIGNAL_MAX 8u
#define MP_BLACKOUT_LATCH_MAX  4.0
#define MP_GEOM_X_BOOTSTRAP    24.0
#define MP_GEOM_Y_BOOTSTRAP    3.0
// S4 evidence veto -- global-translation (pan/shake) gate. A translating
// texture is a per-frame grain twin: it decorrelates band0 while q_still's
// absolute per-point threshold stays blind on dark content and q_random's
// innovation/spatial ratio lands in the grain band (measured: Odyssey
// 160-330 s, m_motion 0.0 throughout the shake, eff 2.64x staircase). A
// grid-level single-step Lucas-Kanade translation estimate is the tell:
// grain decorrelates across grid neighbours so its g*d products average
// to ~0 over the lattice, while a coherent camera translation correlates
// (proven lineage: the old build's pan-freeze gate). Magnitude is EMA'd,
// so alternating-direction SHAKE holds it elevated like a sustained pan.
// The veto only REDUCES learning authority (rise lanes); down-reads stay
// live because translation decorrelation can only INFLATE sigma -- a
// sub-master read under motion remains a valid one-sided bound. PAN_LO/HI
// are in picture-lattice samples/frame and are calibrated on the PC harness
// decode at MP_PICTURE_DENSITY
// (row 51): they must sit above the static-grain floor read on ground
// truth (Golden Spurtle) so legitimate grain never freezes.
#define MP_LK_SCALE            2.0e5
#define MP_LK_CLAMP            1.0e6
#define MP_PAN_LO              0.35
#define MP_PAN_HI              1.00

// Flattened for conservative SPIRV-Cross/D3D11 lowering.
shared uint s_hist[MP_TONE_BINS * MP_BANDS * AMP_BINS];
shared uint s_content_hist[AMP_BINS];
shared uint s_luma_now[MP_TONE_BINS];
shared uint s_luma_prev[MP_TONE_BINS];
shared float s_probe[MP_GRID_N];
shared uint s_row_dark[MP_GRID_H];
shared uint s_col_dark[MP_GRID_W];
shared float s_row_range[MP_GRID_H];
shared float s_col_range[MP_GRID_W];
shared float s_row_level[MP_GRID_H];
shared float s_col_level[MP_GRID_W];
shared float s_refine_probe[512];
shared uint s_state_ok;
shared uint s_prev_ready;
shared uint s_raster_changed;
shared uint s_picture_signal;
shared uint s_raster_signal;
shared uint s_probe_count;
shared uint s_probe_changed;
shared float s_probe_hist_l1;
shared uint s_probe_hard_cut;
shared uint s_history_ready;
shared float s_active_inset_x;
shared float s_active_inset_y;
shared float s_candidate_inset_x;
shared float s_candidate_inset_y;
shared uint s_candidate_valid_x;
shared uint s_candidate_valid_y;
shared uint s_candidate_immediate_x;
shared uint s_candidate_immediate_y;
shared uint s_scan_x0;
shared uint s_scan_x1;
shared uint s_scan_y0;
shared uint s_scan_y1;
shared float s_scan_inset_x;
shared float s_scan_inset_y;
shared uint s_valid_count;
shared uint s_changed_count;
shared uint s_fine_energy;
shared uint s_mid_energy;
shared uint s_coarse_energy;
shared uint s_structure_fine;
shared uint s_structure_broad;
shared uint s_lk_gxx;
shared uint s_lk_gyy;
shared uint s_lk_gxy_p;
shared uint s_lk_gxy_n;
shared uint s_lk_bx_p;
shared uint s_lk_bx_n;
shared uint s_lk_by_p;
shared uint s_lk_by_n;

float measure_luma(vec2 uv) {
    return HOOKED_tex(uv).r;
}

int tone_bin(float y) {
    return clamp(int(sqrt(clamp(y, 0.0, 1.0)) * float(MP_TONE_BINS)),
                 0, MP_TONE_BINS - 1);
}

float prior_tone_shape(int b) {
    float q = (float(b) + 0.5) / float(MP_TONE_BINS);
    float y = q * q;
    float shadow = mix(0.78, 1.0, smoothstep(0.0, 0.22, y));
    float highlight = 1.0 - 0.55 * smoothstep(0.72, 1.0, y);
    return shadow * highlight;
}

float prior_missing_fraction(int b) {
    float q = (float(b) + 0.5) / float(MP_TONE_BINS);
    float y = q * q;
    float d = mix(0.92, 0.85, smoothstep(0.06, 0.45, y));
    return mix(d, 0.80, smoothstep(0.58, 1.0, y));
}

// Compact picture-relative observer. Its measurements update hidden title/shot
// candidates; no instantaneous statistic is allowed to modulate visible grain.
void lean_observe() {
    uint lid = gl_LocalInvocationIndex;
    uint nthreads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    if (lid == 0u) {
        bool ok = abs(m_state_magic - MP_STATE_MAGIC) < 0.0001
               && abs(m_state_epoch - state_epoch) < 0.5;
        float raster_aspect = HOOKED_size.x / max(HOOKED_size.y, 1.0);
        bool raster_changed = ok && abs(m_source_aspect - raster_aspect) > 0.001;
        s_state_ok = ok ? 1u : 0u;
        s_raster_changed = raster_changed ? 1u : 0u;
        s_prev_ready = (ok && m_prev_ready > 0.5 && !raster_changed) ? 1u : 0u;
    }
    barrier();
    bool state_ok = s_state_ok != 0u;
    bool prev_ready = s_prev_ready != 0u;

    if (!state_ok) {
        for (uint k = lid; k < uint(MP_GRID_N); k += nthreads) {
            prev_grid[k] = -1.0;
            prev_grid_off[k] = 0.0;
            prev_probe[k] = 0.0;
        }
        if (lid == 0u) {
            m_state_magic = MP_STATE_MAGIC;
            m_state_epoch = state_epoch;
            m_prev_ready = 0.0;
            m_measured = 0.0;
            m_gen_frame = 0.0;
            m_arr_seed = 0.0;
            m_field_seed = 0.0;
            m_regen = 1.0;
            m_regen_pending = 1.0;
            m_field_valid = 0.0;
            m_field_cov_rg = 0.0;
            m_field_cov_rb = 0.0;
            m_field_cov_gb = 0.0;
            m_field_var_r = 0.0;
            m_field_var_g = 0.0;
            m_field_var_b = 0.0;
            m_baked_grain_size = -1.0;
            m_baked_grain_contrast = -1.0;
            m_baked_value_warp = -1.0;
            m_baked_grain_base_sat = -1.0;
            m_baked_grain_sharpness = -1.0;
            m_baked_match_grain = -1.0;
            m_source_aspect = HOOKED_size.x / max(HOOKED_size.y, 1.0);
            m_active_inset_x = 0.0;
            m_active_inset_y = 0.0;
            m_pending_inset_x = 0.0;
            m_pending_inset_y = 0.0;
            m_geom_streak = 0.0;
            m_geom_streak_y = 0.0;
            m_geom_known = 0.0;
            m_geom_known_y = 0.0;
            m_geom_blackout = 0.0;
            m_geom_blackout_y = 0.0;
            m_geom_changed = 0.0;
            m_observed = 0.0;
            m_structure_ratio = 1.0;
            m_coverage = 0.0;
            m_motion = 0.0;
            m_cut_score = 0.0;
            m_pan_px = 0.0;
            m_temporal_support = 0.0;
            m_title_conf = 0.0;
            m_title_power = MP_PRIOR_SIGMA * MP_PRIOR_SIGMA;
            m_title_size = 0.50;
            m_title_hardness = 0.50;
            m_shot_size = 0.50;
            m_shot_hardness = 0.50;
            m_shot_age = 0.0;
            m_shot_conf = 0.0;
            m_shot_gain = 1.0;
            m_shot_restore_boost = 1.0;
            m_est_missing = 0.0;
            m_loss_conf = 0.35;
            m_loss_mix = 0.0;
            m_tone_conf = 0.0;
            float eff_sum = 0.0;
            for (int b = 0; b < MP_TONE_BINS; b++) {
                float s = MP_PRIOR_SIGMA * prior_tone_shape(b);
                float p = s * s;
                m_master_p[b] = p;
                m_master_w[b] = 0.0;
                m_shot_obs_p[b] = 0.0;
                m_shot_obs_w[b] = 0.0;
                m_char_p[b] = MP_COMPLEMENT_POWER * p;
                // This is the acquisition posterior, not a decorative floor:
                // absent delivery evidence, all capture paths still imply a
                // conservative amount of master grain throughout the range.
                m_restore_p[b] = prior_missing_fraction(b) * p;
                eff_sum += m_char_p[b] + restore_gain * restore_gain
                         * m_restore_p[b];
            }
            m_eff_render = sqrt(eff_sum / float(MP_TONE_BINS)) / MP_FIELD_STD;
        }
    }
    barrier();

    if (lid == 0u && s_raster_changed != 0u) {
        m_geom_known = 0.0;
        m_geom_known_y = 0.0;
        m_geom_streak = 0.0;
        m_geom_streak_y = 0.0;
        m_geom_blackout = 0.0;
        m_geom_blackout_y = 0.0;
    }
    barrier();

    // mpv may redraw a paused frame without advancing source PTS. The shader
    // cannot infer that distinction from pixels, so shampv supplies this
    // machine-owned uniform. A newly loaded/inserted shader observes its held
    // source frame once after reset, so it can initialize without waiting for
    // unpause; established paused frames cannot train any posterior, advance
    // cadence, or alter their arrangement. Baked
    // look edits are the sole exception: rebuild the standing vocabulary once
    // under the same seed so pause-and-tune remains useful without animating.
    // The PARAM-conditioned return is dispatch-uniform and precedes all later
    // barriers, which is legal on FXC/D3D11.
    if (grain_pause > 0.5 && state_ok) {
        if (lid == 0u) {
            bool baked_params_changed =
                   abs(m_baked_grain_size - grain_size) > 1.0e-6
                || abs(m_baked_grain_contrast - grain_contrast) > 1.0e-6
                || abs(m_baked_value_warp - value_warp) > 1.0e-6
                || abs(m_baked_grain_base_sat - grain_base_sat) > 1.0e-6
                || abs(m_baked_grain_sharpness - grain_sharpness) > 1.0e-6
                || abs(m_baked_match_grain - match_grain) > 1.0e-6;
            m_regen = 0.0;
            if (baked_params_changed) {
                m_baked_grain_size = grain_size;
                m_baked_grain_contrast = grain_contrast;
                m_baked_value_warp = value_warp;
                m_baked_grain_base_sat = grain_base_sat;
                m_baked_grain_sharpness = grain_sharpness;
                m_baked_match_grain = match_grain;
                m_regen_pending = 1.0;
            }
            // A baked edit may have arrived while the generator was disabled.
            // Consume that sticky request as soon as gain/debug makes PASS 2
            // active, even if the baked cache already matches by then.
            bool gen_active = (grain_gain > 0.0 && match_grain > 0.0)
                           || debug_match > 0.5;
            if (gen_active && m_regen_pending > 0.5) {
                m_regen = 1.0;
                m_regen_pending = 0.0;
            }
            imageStore(out_image, ivec2(0), vec4(0.0));
        }
        return;
    }

    // A fixed full-raster probe discovers centred baked mattes and keeps cut
    // history independent of the active-picture mapping used by the grain
    // observer. Limited-range black arrives here near 16/255, not zero.
    if (lid == 0u) {
        s_picture_signal = 0u;
        s_raster_signal = 0u;
        s_candidate_inset_x = m_active_inset_x;
        s_candidate_inset_y = m_active_inset_y;
        s_candidate_valid_x = 0u;
        s_candidate_valid_y = 0u;
        s_candidate_immediate_x = 0u;
        s_candidate_immediate_y = 0u;
    }
    barrier();
    for (uint k = lid; k < uint(MP_GRID_N); k += nthreads) {
        uint gx = k % uint(MP_GRID_W);
        uint gy = k / uint(MP_GRID_W);
        vec2 probe_uv = (vec2(float(gx), float(gy)) + 0.5)
                      / vec2(float(MP_GRID_W), float(MP_GRID_H));
        float c = measure_luma(probe_uv);
        s_probe[k] = c;
        if (c > MP_BLACKOUT_CODE_MAX)
            atomicAdd(s_raster_signal, 1u);
        if (gx >= 16u && gx < 48u && gy >= 16u && gy < 48u
            && c > 0.10)
            atomicAdd(s_picture_signal, 1u);
    }
    barrier();

    if (lid == 0u) {
        s_scan_inset_x = clamp(m_active_inset_x, 0.0, MP_ACTIVE_INSET_MAX);
        s_scan_inset_y = clamp(m_active_inset_y, 0.0, MP_ACTIVE_INSET_MAX);
        s_scan_x0 = uint(floor(s_scan_inset_x * float(MP_GRID_W) + 0.5));
        s_scan_y0 = uint(floor(s_scan_inset_y * float(MP_GRID_H) + 0.5));
        s_scan_x1 = uint(MP_GRID_W) - s_scan_x0;
        s_scan_y1 = uint(MP_GRID_H) - s_scan_y0;
    }
    barrier();

    if (lid < uint(MP_GRID_H)) {
        uint dark = 0u;
        float lo = 1.0;
        float hi = 0.0;
        float sum = 0.0;
        for (uint x = s_scan_x0; x < s_scan_x1; x++) {
            float c = s_probe[int(lid) * MP_GRID_W + int(x)];
            if (c < MP_BAR_DARK_MAX) {
                dark++;
                sum += c;
                lo = min(lo, c);
                hi = max(hi, c);
            }
        }
        s_row_dark[lid] = dark;
        s_row_range[lid] = max(hi - lo, 0.0);
        s_row_level[lid] = sum / max(float(dark), 1.0);
    }
    if (lid < uint(MP_GRID_W)) {
        uint dark = 0u;
        float lo = 1.0;
        float hi = 0.0;
        float sum = 0.0;
        for (uint y = s_scan_y0; y < s_scan_y1; y++) {
            float c = s_probe[int(y) * MP_GRID_W + int(lid)];
            if (c < MP_BAR_DARK_MAX) {
                dark++;
                sum += c;
                lo = min(lo, c);
                hi = max(hi, c);
            }
        }
        s_col_dark[lid] = dark;
        s_col_range[lid] = max(hi - lo, 0.0);
        s_col_level[lid] = sum / max(float(dark), 1.0);
    }
    barrier();

    if (lid == 0u) {
        uint row_span = max(s_scan_x1 - s_scan_x0, 1u);
        uint col_span = max(s_scan_y1 - s_scan_y0, 1u);
        uint row_matte_min = (row_span * MP_BAR_DARK_SAMPLES + 63u) / 64u;
        uint col_matte_min = (col_span * MP_BAR_DARK_SAMPLES + 63u) / 64u;
        uint row_picture_dark = (row_span * MP_BAR_EDGE_PICTURE + 63u) / 64u;
        uint col_picture_dark = (col_span * MP_BAR_EDGE_PICTURE + 63u) / 64u;
        int top = 0;
        int bottom = 0;
        int left = 0;
        int right = 0;
        for (int y = 0; y < MP_GRID_H / 2; y++) {
            bool matte = s_row_dark[y] >= row_matte_min
                      && s_row_range[y] <= MP_BAR_RANGE_MAX
                      && abs(s_row_level[y] - s_row_level[0])
                         <= MP_BAR_LEVEL_MAX;
            if (!matte) break;
            top++;
        }
        for (int y = MP_GRID_H - 1; y >= MP_GRID_H / 2; y--) {
            bool matte = s_row_dark[y] >= row_matte_min
                      && s_row_range[y] <= MP_BAR_RANGE_MAX
                      && abs(s_row_level[y] - s_row_level[MP_GRID_H - 1])
                         <= MP_BAR_LEVEL_MAX;
            if (!matte) break;
            bottom++;
        }
        for (int x = 0; x < MP_GRID_W / 2; x++) {
            bool matte = s_col_dark[x] >= col_matte_min
                      && s_col_range[x] <= MP_BAR_RANGE_MAX
                      && abs(s_col_level[x] - s_col_level[0])
                         <= MP_BAR_LEVEL_MAX;
            if (!matte) break;
            left++;
        }
        for (int x = MP_GRID_W - 1; x >= MP_GRID_W / 2; x--) {
            bool matte = s_col_dark[x] >= col_matte_min
                      && s_col_range[x] <= MP_BAR_RANGE_MAX
                      && abs(s_col_level[x] - s_col_level[MP_GRID_W - 1])
                         <= MP_BAR_LEVEL_MAX;
            if (!matte) break;
            right++;
        }

        int top_support = 0, bottom_support = 0;
        int left_support = 0, right_support = 0;
        int full_top_support = 0, full_bottom_support = 0;
        int full_left_support = 0, full_right_support = 0;
        for (int d = 0; d < 3; d++) {
            int yt = min(top + d, MP_GRID_H - 1);
            int yb = max(MP_GRID_H - 1 - bottom - d, 0);
            int xl = min(left + d, MP_GRID_W - 1);
            int xr = max(MP_GRID_W - 1 - right - d, 0);
            if (s_row_dark[yt] < row_picture_dark) top_support++;
            if (s_row_dark[yb] < row_picture_dark) bottom_support++;
            if (s_col_dark[xl] < col_picture_dark) left_support++;
            if (s_col_dark[xr] < col_picture_dark) right_support++;
            if (s_row_dark[d] < row_picture_dark) full_top_support++;
            if (s_row_dark[MP_GRID_H - 1 - d] < row_picture_dark)
                full_bottom_support++;
            if (s_col_dark[d] < col_picture_dark) full_left_support++;
            if (s_col_dark[MP_GRID_W - 1 - d] < col_picture_dark)
                full_right_support++;
        }

        float picture_lo = 1.0;
        float picture_hi = 0.0;
        for (int y = 16; y < 48; y++) {
            for (int x = 16; x < 48; x++) {
                float c = s_probe[y * MP_GRID_W + x];
                picture_lo = min(picture_lo, c);
                picture_hi = max(picture_hi, c);
            }
        }
        bool signal_ok = s_picture_signal >= MP_BAR_SIGNAL_MIN
                      && picture_hi - picture_lo >= MP_BAR_PICTURE_RANGE;
        float coarse_y = float(min(top, bottom)) / float(MP_GRID_H);
        float coarse_x = float(min(left, right)) / float(MP_GRID_W);
        bool bars_y = signal_ok && top >= MP_BAR_MIN_CELLS
                   && bottom >= MP_BAR_MIN_CELLS
                   && abs(top - bottom) <= 8
                   && coarse_y <= MP_ACTIVE_INSET_MAX
                   && (top_support >= 2 || bottom_support >= 2);
        bool bars_x = signal_ok && left >= MP_BAR_MIN_CELLS
                   && right >= MP_BAR_MIN_CELLS
                   && abs(left - right) <= 8
                   && coarse_x <= MP_ACTIVE_INSET_MAX
                   && (left_support >= 2 || right_support >= 2);
        bool full_y = signal_ok && !bars_y
                   && full_top_support >= 2 && full_bottom_support >= 2;
        bool full_x = signal_ok && !bars_x
                   && full_left_support >= 2 && full_right_support >= 2;

        if (bars_x) {
            s_candidate_inset_x = coarse_x;
            s_candidate_valid_x = 1u;
            s_candidate_immediate_x = (min(left, right) >= 4
                                    && abs(left - right) <= 1) ? 1u : 0u;
        } else if (full_x) {
            s_candidate_inset_x = 0.0;
            s_candidate_valid_x = 1u;
            s_candidate_immediate_x = 1u;
        }
        if (bars_y) {
            s_candidate_inset_y = coarse_y;
            s_candidate_valid_y = 1u;
            s_candidate_immediate_y = (min(top, bottom) >= 4
                                    && abs(top - bottom) <= 1) ? 1u : 0u;
        } else if (full_y) {
            s_candidate_inset_y = 0.0;
            s_candidate_valid_y = 1u;
            s_candidate_immediate_y = 1u;
        }
    }
    barrier();

    // Refine a coarse 1/64 edge inside its transition cell. Sixteen samples
    // across each of eight sub-rows/sub-columns retain subtitle tolerance while
    // reducing active-height error to about one source pixel at 1080p.
    if (lid < 512u) {
        uint side = lid / 128u;
        uint q = lid % 128u;
        uint step = q / 16u;
        uint across = q % 16u;
        float across_uv = (float(across) + 0.5) / 16.0;
        vec2 uv = vec2(across_uv);
        float offset = ((float(step) + 0.5) / 8.0 - 0.5)
                     / float(MP_GRID_H);
        bool enabled = false;
        if (side == 0u && s_candidate_valid_y != 0u
            && s_candidate_inset_y > 0.0) {
            uv.x = s_scan_inset_x
                 + across_uv * (1.0 - 2.0 * s_scan_inset_x);
            uv.y = s_candidate_inset_y + offset;
            enabled = true;
        } else if (side == 1u && s_candidate_valid_y != 0u
                   && s_candidate_inset_y > 0.0) {
            uv.x = s_scan_inset_x
                 + across_uv * (1.0 - 2.0 * s_scan_inset_x);
            uv.y = 1.0 - (s_candidate_inset_y + offset);
            enabled = true;
        } else if (side == 2u && s_candidate_valid_x != 0u
            && s_candidate_inset_x > 0.0) {
            uv.x = s_candidate_inset_x + offset;
            uv.y = s_scan_inset_y
                 + across_uv * (1.0 - 2.0 * s_scan_inset_y);
            enabled = true;
        } else if (side == 3u && s_candidate_valid_x != 0u
            && s_candidate_inset_x > 0.0) {
            uv.x = 1.0 - (s_candidate_inset_x + offset);
            uv.y = s_scan_inset_y
                 + across_uv * (1.0 - 2.0 * s_scan_inset_y);
            enabled = true;
        }
        s_refine_probe[lid] = enabled ? measure_luma(uv) : 1.0;
    }
    barrier();

    if (lid == 0u) {
        if (s_candidate_valid_y != 0u && s_candidate_inset_y > 0.0) {
            int top_steps = 0;
            int bottom_steps = 0;
            for (int step = 0; step < 8; step++) {
                uint dark_t = 0u;
                uint dark_b = 0u;
                float lo_t = 1.0, hi_t = 0.0;
                float lo_b = 1.0, hi_b = 0.0;
                float sum_t = 0.0, sum_b = 0.0;
                for (int x = 0; x < 16; x++) {
                    float ct = s_refine_probe[step * 16 + x];
                    float cb = s_refine_probe[128 + step * 16 + x];
                    if (ct < MP_BAR_DARK_MAX) {
                        dark_t++;
                        sum_t += ct;
                        lo_t = min(lo_t, ct); hi_t = max(hi_t, ct);
                    }
                    if (cb < MP_BAR_DARK_MAX) {
                        dark_b++;
                        sum_b += cb;
                        lo_b = min(lo_b, cb); hi_b = max(hi_b, cb);
                    }
                }
                bool mt = dark_t >= 10u
                       && max(hi_t - lo_t, 0.0) <= MP_BAR_RANGE_MAX
                       && abs(sum_t / max(float(dark_t), 1.0)
                            - s_row_level[0]) <= MP_BAR_LEVEL_MAX;
                bool mb = dark_b >= 10u
                       && max(hi_b - lo_b, 0.0) <= MP_BAR_RANGE_MAX
                       && abs(sum_b / max(float(dark_b), 1.0)
                            - s_row_level[MP_GRID_H - 1]) <= MP_BAR_LEVEL_MAX;
                if (mt && top_steps == step) top_steps++;
                if (mb && bottom_steps == step) bottom_steps++;
            }
            float refine_sum = 0.0;
            float refine_count = 0.0;
            if (top_steps > 0 && top_steps < 8) {
                refine_sum += (float(top_steps) - 0.5) / 8.0 - 0.5;
                refine_count += 1.0;
            }
            if (bottom_steps > 0 && bottom_steps < 8) {
                refine_sum += (float(bottom_steps) - 0.5) / 8.0 - 0.5;
                refine_count += 1.0;
            }
            if (refine_count > 0.0) {
                s_candidate_inset_y += refine_sum / refine_count
                                     / float(MP_GRID_H);
            }
        }
        if (s_candidate_valid_x != 0u && s_candidate_inset_x > 0.0) {
            int left_steps = 0;
            int right_steps = 0;
            for (int step = 0; step < 8; step++) {
                uint dark_l = 0u;
                uint dark_r = 0u;
                float lo_l = 1.0, hi_l = 0.0;
                float lo_r = 1.0, hi_r = 0.0;
                float sum_l = 0.0, sum_r = 0.0;
                for (int y = 0; y < 16; y++) {
                    float cl = s_refine_probe[256 + step * 16 + y];
                    float cr = s_refine_probe[384 + step * 16 + y];
                    if (cl < MP_BAR_DARK_MAX) {
                        dark_l++;
                        sum_l += cl;
                        lo_l = min(lo_l, cl); hi_l = max(hi_l, cl);
                    }
                    if (cr < MP_BAR_DARK_MAX) {
                        dark_r++;
                        sum_r += cr;
                        lo_r = min(lo_r, cr); hi_r = max(hi_r, cr);
                    }
                }
                bool ml = dark_l >= 10u
                       && max(hi_l - lo_l, 0.0) <= MP_BAR_RANGE_MAX
                       && abs(sum_l / max(float(dark_l), 1.0)
                            - s_col_level[0]) <= MP_BAR_LEVEL_MAX;
                bool mr = dark_r >= 10u
                       && max(hi_r - lo_r, 0.0) <= MP_BAR_RANGE_MAX
                       && abs(sum_r / max(float(dark_r), 1.0)
                            - s_col_level[MP_GRID_W - 1]) <= MP_BAR_LEVEL_MAX;
                if (ml && left_steps == step) left_steps++;
                if (mr && right_steps == step) right_steps++;
            }
            float refine_sum = 0.0;
            float refine_count = 0.0;
            if (left_steps > 0 && left_steps < 8) {
                refine_sum += (float(left_steps) - 0.5) / 8.0 - 0.5;
                refine_count += 1.0;
            }
            if (right_steps > 0 && right_steps < 8) {
                refine_sum += (float(right_steps) - 0.5) / 8.0 - 0.5;
                refine_count += 1.0;
            }
            if (refine_count > 0.0) {
                s_candidate_inset_x += refine_sum / refine_count
                                     / float(MP_GRID_W);
            }
        }
        s_candidate_inset_x = clamp(s_candidate_inset_x, 0.0,
                                    MP_ACTIVE_INSET_MAX);
        s_candidate_inset_y = clamp(s_candidate_inset_y, 0.0,
                                    MP_ACTIVE_INSET_MAX);
        // A one-frame subtitle/logo can obscure an otherwise stable pending
        // edge exactly on a cut. Preserve that rectangle for cut normalization
        // while keeping candidate validity false for commit authority.
        if (s_candidate_valid_x == 0u
            && m_geom_streak >= MP_GEOM_X_BOOTSTRAP)
            s_candidate_inset_x = m_pending_inset_x;
        if (s_candidate_valid_y == 0u
            && m_geom_streak_y >= MP_GEOM_Y_BOOTSTRAP)
            s_candidate_inset_y = m_pending_inset_y;
    }
    barrier();

    // Fixed-coordinate cut evidence is normalized by the selected picture
    // rectangle, so static baked bars cannot cap the changed fraction.
    if (lid == 0u) {
        s_probe_count = 0u;
        s_probe_changed = 0u;
    }
    for (uint k = lid; k < uint(MP_TONE_BINS); k += nthreads) {
        s_luma_now[k] = 0u;
        s_luma_prev[k] = 0u;
    }
    barrier();
    for (uint k = lid; k < uint(MP_GRID_N); k += nthreads) {
        uint gx = k % uint(MP_GRID_W);
        uint gy = k / uint(MP_GRID_W);
        float cut_ix = s_candidate_inset_x;
        float cut_iy = s_candidate_inset_y;
        int x0 = int(floor(cut_ix * float(MP_GRID_W) + 0.5));
        int y0 = int(floor(cut_iy * float(MP_GRID_H) + 0.5));
        bool inside = int(gx) >= x0 && int(gx) < MP_GRID_W - x0
                   && int(gy) >= y0 && int(gy) < MP_GRID_H - y0;
        float c = s_probe[k];
        float p = prev_ready ? prev_probe[k] : c;
        if (inside) {
            atomicAdd(s_probe_count, 1u);
            atomicAdd(s_luma_now[tone_bin(c)], 1u);
            atomicAdd(s_luma_prev[tone_bin(p)], 1u);
            if (prev_ready && abs(c - p) > 0.018)
                atomicAdd(s_probe_changed, 1u);
        }
        prev_probe[k] = c;
    }
    barrier();

    if (lid == 0u) {
        float probe_count = max(float(s_probe_count), 1.0);
        float probe_changed = prev_ready
                            ? float(s_probe_changed) / probe_count : 0.0;
        float probe_hist = 0.0;
        if (prev_ready) {
            for (int b = 0; b < MP_TONE_BINS; b++)
                probe_hist += abs(float(s_luma_now[b])
                                - float(s_luma_prev[b]));
            probe_hist /= probe_count;
        }
        bool hard_cut = prev_ready && probe_changed > 0.75
                     && probe_hist > 0.25;

        float cell_x = 1.0 / float(MP_GRID_W);
        float cell_y = 1.0 / float(MP_GRID_H);
        bool pending_ready_x = m_geom_streak >= MP_GEOM_X_BOOTSTRAP;
        bool pending_ready_y = m_geom_streak_y >= MP_GEOM_Y_BOOTSTRAP;
        bool pending_match_y = pending_ready_y
                            && s_candidate_valid_y != 0u
                            && abs(s_candidate_inset_y - m_pending_inset_y)
                               <= 0.5 * cell_y;
        bool blackout_frame = s_raster_signal <= MP_BLACKOUT_SIGNAL_MAX;
        bool blackout_armed_x = m_geom_blackout >= 3.0;
        bool blackout_armed_y = m_geom_blackout_y >= 3.0;
        if (blackout_frame) {
            m_geom_blackout = min(m_geom_blackout + 1.0,
                                   MP_BLACKOUT_LATCH_MAX);
            m_geom_blackout_y = min(m_geom_blackout_y + 1.0,
                                     MP_BLACKOUT_LATCH_MAX);
        }

        bool valid_x = s_candidate_valid_x != 0u;
        bool valid_y = s_candidate_valid_y != 0u;
        bool diff_x = valid_x
                   && abs(s_candidate_inset_x - m_active_inset_x)
                      > 0.5 * cell_x;
        bool diff_y = valid_y
                   && abs(s_candidate_inset_y - m_active_inset_y)
                      > 0.5 * cell_y;
        if (!blackout_frame && valid_x)
            m_geom_blackout = diff_x
                            ? max(m_geom_blackout - 1.0, 0.0) : 0.0;
        if (!blackout_frame && valid_y)
            m_geom_blackout_y = diff_y
                              ? max(m_geom_blackout_y - 1.0, 0.0) : 0.0;

        if (diff_x) {
            bool same = abs(s_candidate_inset_x - m_pending_inset_x)
                      <= 0.5 * cell_x;
            if (same)
                m_geom_streak = min(m_geom_streak + 1.0, 65535.0);
            else {
                m_pending_inset_x = s_candidate_inset_x;
                m_geom_streak = 1.0;
            }
        } else if (valid_x) {
            m_pending_inset_x = m_active_inset_x;
            m_geom_streak = 0.0;
            m_geom_known = 1.0;
        } else {
            m_geom_streak = max(m_geom_streak - 1.0, 0.0);
        }

        if (diff_y) {
            bool same = abs(s_candidate_inset_y - m_pending_inset_y)
                      <= 0.5 * cell_y;
            if (same)
                m_geom_streak_y = min(m_geom_streak_y + 1.0, 65535.0);
            else {
                m_pending_inset_y = s_candidate_inset_y;
                m_geom_streak_y = 1.0;
            }
        } else if (valid_y) {
            m_pending_inset_y = m_active_inset_y;
            m_geom_streak_y = 0.0;
            m_geom_known_y = 1.0;
        } else {
            m_geom_streak_y = max(m_geom_streak_y - 1.0, 0.0);
        }

        bool commit_x = false;
        bool commit_y = false;
        float commit_inset_x = s_candidate_inset_x;
        float commit_inset_y = s_candidate_inset_y;
        if (diff_x) {
            commit_x = ((!prev_ready && s_candidate_immediate_x != 0u)
                     || ((hard_cut || blackout_armed_x)
                         && s_candidate_immediate_x != 0u)
                     || (m_geom_known < 0.5
                         && m_geom_streak >= MP_GEOM_X_BOOTSTRAP));
        } else if (hard_cut && !valid_x && pending_ready_x
                   && abs(m_pending_inset_x - m_active_inset_x)
                      > 0.5 * cell_x) {
            commit_x = true;
            commit_inset_x = m_pending_inset_x;
        }
        if (diff_y) {
            commit_y = ((!prev_ready && s_candidate_immediate_y != 0u)
                     || (hard_cut
                         && (s_candidate_immediate_y != 0u
                             || pending_match_y))
                     || blackout_armed_y
                     || (m_geom_known_y < 0.5
                         && m_geom_streak_y >= MP_GEOM_Y_BOOTSTRAP));
        } else if (hard_cut && !valid_y && pending_ready_y
                   && abs(m_pending_inset_y - m_active_inset_y)
                      > 0.5 * cell_y) {
            commit_y = true;
            commit_inset_y = m_pending_inset_y;
        }

        bool commit = commit_x || commit_y;
        if (commit_x) {
            m_active_inset_x = clamp(commit_inset_x, 0.0,
                                     MP_ACTIVE_INSET_MAX);
            m_pending_inset_x = m_active_inset_x;
            m_geom_streak = 0.0;
            m_geom_known = 1.0;
            m_geom_blackout = 0.0;
        }
        if (commit_y) {
            m_active_inset_y = clamp(commit_inset_y, 0.0,
                                     MP_ACTIVE_INSET_MAX);
            m_pending_inset_y = m_active_inset_y;
            m_geom_streak_y = 0.0;
            m_geom_known_y = 1.0;
            m_geom_blackout_y = 0.0;
        }
        s_active_inset_x = m_active_inset_x;
        s_active_inset_y = m_active_inset_y;
        s_probe_hist_l1 = probe_hist;
        s_probe_hard_cut = hard_cut ? 1u : 0u;
        s_history_ready = (prev_ready && !commit) ? 1u : 0u;
        m_geom_changed = commit ? 1.0 : 0.0;
    }
    barrier();

    if (lid == 0u) {
        s_valid_count = 0u;
        s_changed_count = 0u;
        s_fine_energy = 0u;
        s_mid_energy = 0u;
        s_coarse_energy = 0u;
        s_structure_fine = 0u;
        s_structure_broad = 0u;
        s_lk_gxx = 0u;
        s_lk_gyy = 0u;
        s_lk_gxy_p = 0u;
        s_lk_gxy_n = 0u;
        s_lk_bx_p = 0u;
        s_lk_bx_n = 0u;
        s_lk_by_p = 0u;
        s_lk_by_n = 0u;
    }
    for (uint k = lid;
         k < uint(MP_TONE_BINS * MP_BANDS * AMP_BINS); k += nthreads)
        s_hist[k] = 0u;
    for (uint k = lid; k < uint(AMP_BINS); k += nthreads)
        s_content_hist[k] = 0u;
    barrier();

    // Offsets are normalized picture-height coordinates expressed on the
    // finite MP_PICTURE_DENSITY analysis lattice. Derive the
    // horizontal UV pitch from the actual LUMA raster aspect so the crosses
    // remain isotropic on 4:3, scope and portrait sources instead of silently
    // source-raster mapping from the active picture rather than assuming an
    // output resolution. Three nested crosses provide a compact spectral body.
    vec2 active_inset = vec2(s_active_inset_x, s_active_inset_y);
    vec2 active_extent = vec2(1.0) - 2.0 * active_inset;
    float uvx_per_vtex = active_extent.y * HOOKED_size.y
                       / max(HOOKED_size.x * MP_PICTURE_DENSITY, 1.0);
    float uvy_per_vtex = active_extent.y / MP_PICTURE_DENSITY;
    vec2 dx1 = vec2(2.0 * uvx_per_vtex, 0.0);
    vec2 dy1 = vec2(0.0, 2.0 * uvy_per_vtex);
    vec2 dx3 = vec2(6.0 * uvx_per_vtex, 0.0);
    vec2 dy3 = vec2(0.0, 6.0 * uvy_per_vtex);
    vec2 dx6 = vec2(12.0 * uvx_per_vtex, 0.0);
    vec2 dy6 = vec2(0.0, 12.0 * uvy_per_vtex);

    for (uint k = lid; k < uint(MP_GRID_N); k += nthreads) {
        vec2 grid_uv = (vec2(float(k % uint(MP_GRID_W)),
                             float(k / uint(MP_GRID_W))) + 0.5)
                     / vec2(float(MP_GRID_W), float(MP_GRID_H));
        vec2 uv = active_inset + grid_uv * active_extent;
        float c = measure_luma(uv);
        float xm1 = measure_luma(uv - dx1);
        float xp1 = measure_luma(uv + dx1);
        float ym1 = measure_luma(uv - dy1);
        float yp1 = measure_luma(uv + dy1);
        float xm3 = measure_luma(uv - dx3);
        float xp3 = measure_luma(uv + dx3);
        float ym3 = measure_luma(uv - dy3);
        float yp3 = measure_luma(uv + dy3);
        float xm6 = measure_luma(uv - dx6);
        float xp6 = measure_luma(uv + dx6);
        float ym6 = measure_luma(uv - dy6);
        float yp6 = measure_luma(uv + dy6);

        float lp1 = (4.0 * c + xm1 + xp1 + ym1 + yp1) * 0.125;
        float lp3 = (4.0 * c + xm3 + xp3 + ym3 + yp3) * 0.125;
        float lp6 = (4.0 * c + xm6 + xp6 + ym6 + yp6) * 0.125;
        float band0 = c - lp1;
        float band1 = lp1 - lp3;
        float band2 = lp3 - lp6;
        float edge = 0.25 * (abs(xp6 - xm6) + abs(yp6 - ym6));
        bool flat_ok = c > 0.002 && c < 0.985 && edge < 0.026;
        bool prev_flat = s_history_ready != 0u && prev_grid[k] > 0.0;
        float prev_c = (s_history_ready != 0u)
                     ? max(abs(prev_grid[k]) - 1.0, 0.0) : c;

        int now_lb = tone_bin(c);
        if (s_history_ready != 0u && abs(c - prev_c) > 0.018)
            atomicAdd(s_changed_count, 1u);

        // Grid global-translation (single-step Lucas-Kanade) accumulation --
        // see the S4 veto note at the PAN defines. Gradients are taken at
        // +/-1 grid cell: at that spacing grain and fine texture alias away
        // and decorrelate, so their g*d products cancel across the lattice
        // while a coherent camera translation correlates. Runs before the
        // flat gate (the pan signal lives in coarse structure, edges
        // included). The border ring is skipped so no tap crosses into
        // mattes/bars and dilutes the fit. Signed sums split into +/- uint
        // pairs; the per-point clamp keeps 4096 x MP_LK_CLAMP inside uint32.
        uint lk_kx = uint(k) % uint(MP_GRID_W);
        uint lk_ky = uint(k) / uint(MP_GRID_W);
        if (s_history_ready != 0u && c > 0.002 && c < 0.985
            && lk_kx > 0u && lk_kx < uint(MP_GRID_W - 1)
            && lk_ky > 0u && lk_ky < uint(MP_GRID_H - 1)) {
            vec2 lk_step = active_extent
                         / vec2(float(MP_GRID_W), float(MP_GRID_H));
            float lk_gx = 0.5 * (measure_luma(uv + vec2(lk_step.x, 0.0))
                               - measure_luma(uv - vec2(lk_step.x, 0.0)));
            float lk_gy = 0.5 * (measure_luma(uv + vec2(0.0, lk_step.y))
                               - measure_luma(uv - vec2(0.0, lk_step.y)));
            float lk_d = c - prev_c;
            float lk_gxy = lk_gx * lk_gy;
            float lk_bx = -lk_gx * lk_d;
            float lk_by = -lk_gy * lk_d;
            atomicAdd(s_lk_gxx,
                      uint(min(lk_gx * lk_gx * MP_LK_SCALE, MP_LK_CLAMP)));
            atomicAdd(s_lk_gyy,
                      uint(min(lk_gy * lk_gy * MP_LK_SCALE, MP_LK_CLAMP)));
            if (lk_gxy >= 0.0)
                atomicAdd(s_lk_gxy_p,
                          uint(min(lk_gxy * MP_LK_SCALE, MP_LK_CLAMP)));
            else
                atomicAdd(s_lk_gxy_n,
                          uint(min(-lk_gxy * MP_LK_SCALE, MP_LK_CLAMP)));
            if (lk_bx >= 0.0)
                atomicAdd(s_lk_bx_p,
                          uint(min(lk_bx * MP_LK_SCALE, MP_LK_CLAMP)));
            else
                atomicAdd(s_lk_bx_n,
                          uint(min(-lk_bx * MP_LK_SCALE, MP_LK_CLAMP)));
            if (lk_by >= 0.0)
                atomicAdd(s_lk_by_p,
                          uint(min(lk_by * MP_LK_SCALE, MP_LK_CLAMP)));
            else
                atomicAdd(s_lk_by_n,
                          uint(min(-lk_by * MP_LK_SCALE, MP_LK_CLAMP)));
        }

        // One continuous delivery-rolloff observation over all content. This is
        // deliberately not a classifier: it merely records how much of the local
        // structure survives in the fine band relative to the six-texel band.
        float structure_fine = band0;
        float structure_broad = c - lp6;
        atomicAdd(s_structure_fine,
                  uint(min(structure_fine * structure_fine * 1.0e8, 1000000.0)));
        atomicAdd(s_structure_broad,
                  uint(min(structure_broad * structure_broad * 1.0e8, 1000000.0)));

        if (flat_ok) {
            float bands[MP_BANDS];
            bands[0] = band0; bands[1] = band1; bands[2] = band2;
            for (int j = 0; j < MP_BANDS; j++) {
                int ab = clamp(int(abs(bands[j]) / MP_AMP_MAX
                                   * float(AMP_BINS)), 0, AMP_BINS - 1);
                int hi = (now_lb * MP_BANDS + j) * AMP_BINS + ab;
                atomicAdd(s_hist[hi], 1u);
            }
            atomicAdd(s_valid_count, 1u);
            atomicAdd(s_fine_energy,
                      uint(min(band0 * band0 * 1.0e9, 1000000.0)));
            atomicAdd(s_mid_energy,
                      uint(min(band1 * band1 * 1.0e9, 1000000.0)));
            atomicAdd(s_coarse_energy,
                      uint(min(band2 * band2 * 1.0e9, 1000000.0)));

            if (prev_flat) {
                float dhp = abs(band0 - prev_grid_off[k]);
                int tb = clamp(int(dhp / MP_TEMP_MAX * float(AMP_BINS)),
                               0, AMP_BINS - 1);
                atomicAdd(s_content_hist[tb], 1u);
            }
        }

        // Sign stores the flat bit; abs(value)-1 stores every previous luma so
        // the next frame can reconstruct a cut histogram without another SSBO.
        prev_grid[k] = flat_ok ? 1.0 + c : -(1.0 + c);
        prev_grid_off[k] = band0;
    }
    barrier();

    if (lid == 0u) {
        float sigma_band[MP_TONE_BINS * MP_BANDS];
        uint tone_count[MP_TONE_BINS];
        float sigma_sum = 0.0;
        float sigma_weight = 0.0;
        for (int b = 0; b < MP_TONE_BINS; b++) {
            uint count0 = 0u;
            for (int a = 0; a < AMP_BINS; a++)
                count0 += s_hist[(b * MP_BANDS) * AMP_BINS + a];
            tone_count[b] = count0;
            for (int j = 0; j < MP_BANDS; j++) {
                uint count = 0u;
                int base = (b * MP_BANDS + j) * AMP_BINS;
                for (int a = 0; a < AMP_BINS; a++) count += s_hist[base + a];
                uint target = (count + 1u) / 2u;
                uint acc = 0u;
                int med_bin = 0;
                for (int a = 0; a < AMP_BINS; a++) {
                    acc += s_hist[base + a];
                    if (acc >= target) { med_bin = a; break; }
                }
                // Exact-zero bins remain exact zero; dequantizing their centre
                // would manufacture evidence on mathematically flat input.
                float med_abs = (med_bin == 0) ? 0.0
                              : (float(med_bin) + 0.5) * MP_AMP_MAX
                              / float(AMP_BINS);
                float sigma = med_abs * MP_MEDABS_TO_STD;
                if (j == 0) sigma *= MP_HP_TO_SOURCE;
                sigma_band[b * MP_BANDS + j] =
                    (count >= MP_MIN_BIN_SAMPLES) ? sigma : 0.0;
            }
            float sigma0 = sigma_band[b * MP_BANDS];
            if (count0 >= MP_MIN_BIN_SAMPLES) {
                sigma_sum += sigma0 * float(count0);
                sigma_weight += float(count0);
            }
        }
        float observed = (sigma_weight > 0.0) ? sigma_sum / sigma_weight : 0.0;

        uint tcount = 0u;
        for (int a = 0; a < AMP_BINS; a++) tcount += s_content_hist[a];
        uint ttarget = (tcount + 1u) / 2u;
        uint tacc = 0u;
        int tmed_bin = 0;
        for (int a = 0; a < AMP_BINS; a++) {
            tacc += s_content_hist[a];
            if (tacc >= ttarget) { tmed_bin = a; break; }
        }
        float temporal = (tcount == 0u || tmed_bin == 0) ? 0.0
                       : (float(tmed_bin) + 0.5) * MP_TEMP_MAX
                       / float(AMP_BINS) * MP_MEDABS_TO_STD
                       * MP_HP_TO_SOURCE * 0.70710678;
        float temporal_ratio = temporal / max(observed, 1.0e-6);
        float fine_e = float(s_fine_energy);
        float mid_e = float(s_mid_energy);
        float coarse_e = float(s_coarse_energy);
        float spectral_sum = max(fine_e + mid_e + coarse_e, 1.0);
        float frame_size = (0.95 * fine_e + 0.55 * mid_e
                          + 0.20 * coarse_e) / spectral_sum;
        float frame_hardness = fine_e / max(fine_e + mid_e, 1.0);
        float structure_ratio = float(s_structure_fine)
                              / max(float(s_structure_broad), 1.0);
        float coverage = float(s_valid_count) / float(MP_GRID_N);
        float changed = (s_history_ready != 0u)
                      ? float(s_changed_count) / float(MP_GRID_N) : 0.0;
        float hist_l1 = s_probe_hist_l1;
        bool hard_cut = s_probe_hard_cut != 0u;
        bool shot_boundary = !prev_ready || hard_cut;
        bool geometry_only = m_geom_changed > 0.5 && !shot_boundary;

        // Global-translation magnitude from the LK normal equations (2x2
        // solve). The shift is in grid-cell units; convert per axis to
        // picture-lattice samples so the measure is resolution-independent.
        // Magnitude
        // (not the vector) is EMA'd: alternating shake must hold the level.
        // Update skips boundary frames -- a same-geometry hard cut leaves
        // s_history_ready set while prev_grid holds pre-cut luma, so keying
        // the skip on shot_boundary keeps that garbage frame out of the EMA.
        float lk_Gxx = float(s_lk_gxx) / MP_LK_SCALE;
        float lk_Gyy = float(s_lk_gyy) / MP_LK_SCALE;
        float lk_Gxy = (float(s_lk_gxy_p) - float(s_lk_gxy_n)) / MP_LK_SCALE;
        float lk_BX = (float(s_lk_bx_p) - float(s_lk_bx_n)) / MP_LK_SCALE;
        float lk_BY = (float(s_lk_by_p) - float(s_lk_by_n)) / MP_LK_SCALE;
        float lk_det = lk_Gxx * lk_Gyy - lk_Gxy * lk_Gxy;
        float pan_px_inst = 0.0;
        if (lk_det > 1.0e-7) {
            float lk_sx = (lk_Gyy * lk_BX - lk_Gxy * lk_BY) / lk_det;
            float lk_sy = (lk_Gxx * lk_BY - lk_Gxy * lk_BX) / lk_det;
            float lk_cellx = (active_extent.x / float(MP_GRID_W))
                           / max(uvx_per_vtex, 1.0e-9);
            float lk_celly = MP_PICTURE_DENSITY / float(MP_GRID_H);
            pan_px_inst = sqrt(lk_sx * lk_sx * lk_cellx * lk_cellx
                             + lk_sy * lk_sy * lk_celly * lk_celly);
        }
        if (!shot_boundary && !geometry_only && s_history_ready != 0u)
            m_pan_px = (m_measured < 1.5) ? pan_px_inst
                     : mix(m_pan_px, pan_px_inst, 0.30);

        // Temporal similarity supplies authority, not amount. It gates title
        // learning and shot refinement but never provides a master-power target.
        float q_amp = smoothstep(0.00018, 0.00080, observed);
        float q_cov = smoothstep(0.08, 0.28, coverage);
        float q_random = 1.0 - smoothstep(0.35, 0.90,
                                          abs(log2(max(temporal_ratio, 0.01))));
        float q_still = 1.0 - smoothstep(0.03, 0.18, changed);
        // S4 evidence-quality veto. Applied as a SEPARATE factor on the
        // rise-capable lanes only -- never folded into q_cov/q_random/
        // q_still themselves, because spatial_reliability (the reduce-only
        // survivor lane) consumes the raw q_cov and vetoing a reduce-only
        // lane would RAISE rendered power on moving textured content.
        float q_source = 1.0 - smoothstep(MP_PAN_LO, MP_PAN_HI, m_pan_px);
        float static_gate_raw = (s_history_ready != 0u && !hard_cut)
                              ? q_cov * q_random * q_still : 0.0;
        float static_gate = static_gate_raw * q_source;
        float evidence_raw = q_amp * static_gate_raw;
        float evidence = q_amp * static_gate;

        // Estimate shot sensitivity after dividing out the title's luma curve.
        // A picture-population mean of sigma would confuse composition/exposure
        // with noise sensitivity, especially across bright/dark cuts. The
        // divisor must be EXACTLY the unit-gain curve presentation renders
        // (title curve blended with the per-bin master by local evidence) —
        // dividing by the bare per-bin master lets title-borrowed presence
        // re-enter through shot gain and double-count into the committed
        // presentation.
        float prior_base_p = MP_PRIOR_SIGMA * MP_PRIOR_SIGMA;
        float gain_log_sum = 0.0;
        float gain_weight = 0.0;
        for (int b = 0; b < MP_TONE_BINS; b++) {
            float sigma0 = sigma_band[b * MP_BANDS];
            float count_r = smoothstep(24.0, 96.0, float(tone_count[b]));
            float amp_r = smoothstep(0.00012, 0.00070, sigma0);
            float w = count_r * amp_r;
            if (w > 0.0) {
                float shape = prior_tone_shape(b);
                float title_unit = m_title_power * shape * shape;
                float local_q = smoothstep(0.10, 0.35, m_master_w[b]);
                // Floor the divisor at a quarter of the acquisition prior:
                // a clean-lane-drained title must not read the first grainy
                // shot as gain-4 sensitivity and then suppress its own
                // master establishment through the gain-normalized update.
                float unit_master = max(mix(title_unit, m_master_p[b],
                                            local_q),
                                        0.25 * shape * shape * prior_base_p);
                float ratio = sigma0 * sigma0
                            / max(unit_master, 1.0e-10);
                gain_log_sum += w * log2(clamp(ratio, 0.25, 16.0));
                gain_weight += w;
            }
        }
        float frame_gain_est = (gain_weight > 0.0)
                             ? exp2(gain_log_sum / gain_weight) : 1.0;
        float gain_weight_support = smoothstep(0.50, 2.0, gain_weight);
        // q_random is a continuous learning weight. Shot presentation uses a
        // steeper confidence curve: once temporal behaviour is convincingly
        // stochastic, amplitude comes from measured power, not from how far
        // inside the authenticity band this frame happened to land.
        float shot_auth = gain_weight_support * q_cov * q_still * q_source
                        * smoothstep(0.02, 0.15, q_random);

        // Title character is a hidden posterior. It only learns from stable,
        // photographic evidence and never decays because a shot or region is
        // difficult to observe.
        if (evidence > 0.0) {
            float a_title = 0.0015 * evidence;
            m_title_size = mix(m_title_size, frame_size, a_title);
            m_title_hardness = mix(m_title_hardness, frame_hardness, a_title);
            m_title_conf += 0.005 * evidence * (1.0 - m_title_conf);
        } else if (evidence_raw <= 0.0) {
            // Staleness: title confidence is re-earned across scenes (tau
            // ~4.6 min at 24p), never session-permanent. Odyssey benchmark:
            // conf saturated at 0.975 and held full learned authority plus a
            // 4x-annealed drain for the rest of the session. Decay can only
            // REDUCE authority, so motion/quiet stretches stay charter-safe;
            // a grainy title re-earns 0->0.5 in ~6 s of good evidence.
            // Freeze-not-drain: when the S4 veto is the ONLY suppressor
            // (evidence exists but is motion-poisoned) confidence HOLDS --
            // a continuously handheld grainy title must not bleed authority
            // for the length of the film. Decay is reserved for genuinely
            // evidence-free stretches.
            m_title_conf *= (1.0 - 1.5e-4);
        }

        // Grain presence is a title property, not eight independent detection
        // gates. One temporally authenticated tone may establish the title-wide
        // base; unavailable tones then inherit the same film-shaped curve.
        // Sustained, broadly observable near-zero evidence pulls the base back
        // toward the level it actually measures — at ANY confidence, so a
        // transient misread can always be walked back — while implausibly hot
        // evidence never enters. Ambiguous, moving and boundary evidence leave
        // the base unchanged.
        float presence_rise = 0.0;
        int presence_bins = 0;
        float clean_peak = 0.0;
        float represented_peak = 0.0;
        int clean_bins = 0;
        int clean_first = MP_TONE_BINS;
        int clean_last = -1;
        for (int b = 0; b < MP_TONE_BINS; b++) {
            float sigma0 = sigma_band[b * MP_BANDS];
            float count_r = smoothstep(24.0, 96.0, float(tone_count[b]));
            float amp_r = smoothstep(0.00012, 0.00070, sigma0);
            float plaus = 1.0 - smoothstep(MP_SIGMA_PLAUS_LO,
                                           MP_SIGMA_PLAUS_HI, sigma0);
            // Presence is a rise-only lane: the S4 veto applies in full.
            float reliability = count_r * amp_r * plaus
                              * q_cov * q_random * q_still * q_source;
            float master_auth = smoothstep(0.10, 0.35, m_master_w[b])
                              * smoothstep(0.08, 0.20, m_title_conf);
            if (reliability > 0.0 && master_auth > 0.0) {
                float shape = prior_tone_shape(b);
                float candidate_gain = m_master_p[b]
                                     / max(shape * shape * prior_base_p,
                                           1.0e-12);
                // The candidate is the measured bin's FULL level (capped by
                // the plausibility roof): shrinking the target instead of the
                // rate would equilibrate erased tones permanently below the
                // measured ones and hold a step between them. Caution lives
                // in the approach rate below, annealed by how many tones
                // corroborate, so one outlier bin transfers slowly while
                // broad agreement transfers at full weight.
                float candidate = min(candidate_gain, 3.24) * prior_base_p;
                float candidate_auth = reliability * master_auth;
                float rise = candidate_auth
                           * max(candidate - m_title_power, 0.0);
                presence_rise = max(presence_rise, rise);
                // Breadth counts bins that CORROBORATE the rise, not bins
                // that merely qualified: a single outlier tone must anneal
                // at the slow rate even when other tones are measurable.
                if (rise > 0.0) presence_bins++;
            }

            if (tone_count[b] >= MP_MIN_BIN_SAMPLES)
                represented_peak = max(represented_peak, sigma0);

            if (tone_count[b] >= 96u) {
                clean_bins++;
                clean_first = min(clean_first, b);
                clean_last = max(clean_last, b);
                clean_peak = max(clean_peak, sigma0);
            }
        }
        bool broad_clean = clean_bins >= 3
                        && clean_last - clean_first >= 2
                        && coverage >= 0.35
                        && q_still >= 0.80
                        && clean_peak <= 0.00018
                        && represented_peak <= 0.00030
                        && temporal <= 0.00012
                        && observed <= 0.00030;
        float clean_q = broad_clean
                      ? smoothstep(0.35, 0.55, coverage)
                      * smoothstep(0.80, 1.0, q_still)
                      * (1.0 - smoothstep(0.00012, 0.00018, clean_peak))
                      * (1.0 - smoothstep(0.00008, 0.00012, temporal))
                      * (1.0 - smoothstep(0.00018, 0.00030, observed))
                      : 0.0;
        bool title_update_ok = s_history_ready != 0u
                            && !shot_boundary && !geometry_only
                            && m_shot_age >= 24.0;
        if (title_update_ok && presence_rise > 0.0) {
            float breadth = mix(0.35, 1.0,
                                smoothstep(1.0, 3.0, float(presence_bins)));
            m_title_power += 0.010 * breadth * presence_rise;
        } else if (title_update_ok && clean_q > 0.0) {
            // Downward target is the level the clean window actually measures
            // (floored), never a fixed drain: reversibility means converging
            // to the evidence, not to zero. The rate carries the censoring
            // model: clean evidence is collectable ONLY from flat, still
            // regions — exactly where delivery erasure is near-certain — so
            // quiet flats get ~the survival complement (~0.12) of face-value
            // weight. A genuinely clean title still converges over an
            // episode; held cels and dark still scenes inside a grainy title
            // can no longer hole the base between presence refills (presence
            // runs ~40x faster). Field bug 2026-07-17 night: the previous
            // 0.002 full-rate lane drained titles to the floor during any
            // long quiet scene and whole shots rendered no grain at all.
            float clean_target = max(represented_peak * represented_peak,
                                     0.25 * prior_base_p);
            float clean_rate = 0.00025 * clean_q
                             * mix(1.0, 0.25,
                                   smoothstep(0.05, 0.50, m_title_conf));
            m_title_power = mix(m_title_power, clean_target, clean_rate);
        }
        // The floor keeps a drained title at half the cold-start bed: a
        // measured-clean master renders a subtle acquisition layer, never
        // nothing (charter: a zero-grain master is never authorized).
        m_title_power = clamp(m_title_power, 0.25 * prior_base_p,
                              3.24 * prior_base_p);

        // A real cut commits only the authenticated title presentation. Spatial
        // evidence on the boundary cannot distinguish grain from picture texture;
        // the next three frames are the sole fast refinement window for proving
        // shot-local sensitivity and character temporally.
        if (shot_boundary) {
            m_shot_age = 0.0;
            m_shot_gain = 1.0;
            m_shot_conf = 0.0;
            m_shot_size = m_title_size;
            m_shot_hardness = m_title_hardness;
            for (int b = 0; b < MP_TONE_BINS; b++)
                m_shot_obs_w[b] = 0.0;
        } else if (m_shot_age < 4.0 && !geometry_only) {
            float frame_gain = clamp(frame_gain_est, 0.50, 4.0);
            float gain_target = mix(1.0, frame_gain, shot_auth);
            m_shot_gain = mix(m_shot_gain, gain_target, 0.35);
            m_shot_conf = mix(m_shot_conf, shot_auth, 0.25);
            if (evidence > 0.0) {
                float shape_rate = 0.18 * evidence;
                m_shot_size = mix(m_shot_size, frame_size, shape_rate);
                m_shot_hardness = mix(m_shot_hardness, frame_hardness,
                                      shape_rate);
            }
        } else if (evidence > 0.0) {
            float frame_gain = clamp(frame_gain_est, 0.50, 4.0);
            m_shot_gain = mix(m_shot_gain, frame_gain, 0.0015 * evidence);
            m_shot_conf += 0.003 * evidence * (1.0 - m_shot_conf);
            m_shot_size = mix(m_shot_size, frame_size, 0.0008 * evidence);
            m_shot_hardness = mix(m_shot_hardness, frame_hardness,
                                  0.0008 * evidence);
        }
        // Grain presence establishes amount, not proof of delivery loss. The
        // legacy boost lane remains neutral until a genuine loss estimator can
        // authorize it without counting the same grain evidence twice.
        m_shot_restore_boost = 1.0;

        float title_sum = 0.0;
        float weight_sum = 0.0;
        float char_sum = 0.0;
        float restore_sum = 0.0;
        float target_missing_sum = 0.0;
        for (int b = 0; b < MP_TONE_BINS; b++) {
            float sigma0 = sigma_band[b * MP_BANDS];
            float pobs = sigma0 * sigma0;
            float count_r = smoothstep(24.0, 96.0, float(tone_count[b]));
            float amp_r = smoothstep(0.00012, 0.00070, sigma0);
            float plaus = 1.0 - smoothstep(MP_SIGMA_PLAUS_LO,
                                           MP_SIGMA_PLAUS_HI, sigma0);
            float spatial_reliability = count_r * amp_r * plaus * q_cov;
            float reliability_raw = spatial_reliability * q_random * q_still;
            float reliability = reliability_raw * q_source;

            // Best-preserved evidence updates the absolute title master curve.
            // Downward evidence is deliberately much slower because delivery
            // erasure is more likely than a physically noiseless master. The
            // master is TITLE-referenced: divide the current shot sensitivity
            // out of the observation, or one long high-gain shot leaks its
            // sensitivity into the persistent curve and, via presence, the
            // title-wide base. The absolute ceiling is the film-plausible
            // roof for sustained band-edge evidence the soft reject passes.
            if (reliability_raw > 0.0) {
                // Sensitivity normalization deflates hot shots only. Dividing
                // by gain < 1 INFLATES the learned target, and an over-learned
                // master reads exactly as gain < 1 at the next cut -- the
                // runaway measured on the Odyssey benchmark (master bins up
                // 3x, title pinned at its cap in ~2 min). A genuinely quiet
                // shot now learns conservatively low at the slow rate instead
                // -- recoverable, unlike the ratchet.
                float pobs_title = pobs / clamp(m_shot_gain, 1.0, 4.0);
                float clipped = clamp(pobs_title, 0.25 * m_master_p[b],
                                      4.0 * m_master_p[b]);
                clipped = min(clipped, MP_MASTER_P_MAX);
                // 3:1, not 10:1: a blind rate asymmetry rectifies fluctuating
                // evidence (fire/texture sigma jitter) into a one-way climb.
                // The erasure prior still earns a downward discount, but down
                // reads here carry measurable grain (amp_r > 0), unlike the
                // clean lane's censored flats. Calibration point for the
                // ProRes ground-truth pass.
                // The S4 veto gates the RISE and the authority earn only:
                // translation decorrelation can only INFLATE sigma, so a
                // sub-master read under motion is a valid one-sided bound
                // and keeps walking the level down (reversibility).
                float master_rate = (clipped > m_master_p[b])
                                  ? 0.003 * q_source : 0.001;
                m_master_p[b] = mix(m_master_p[b], clipped,
                                    master_rate * reliability_raw);
                m_master_w[b] += 0.005 * reliability
                               * (1.0 - m_master_w[b]);

            } else {
                // Staleness: per-bin authority is re-earned, not permanent.
                // Evidence-free stretches relax local_q back toward the title
                // curve (tau ~4.6 min at 24p). The learned LEVEL stays --
                // absence is not evidence of a clean master. Keyed on RAW
                // evidence: motion-vetoed frames HOLD authority rather than
                // drain it (freeze-not-drain, S4).
                m_master_w[b] *= (1.0 - 1.5e-4);
            }

            // A spatial survivor read may only reduce restoration. It is
            // therefore safe to acquire at a boundary before temporal
            // authentication — a cut into grain-rich surviving content must
            // not commit the full deficit on top of that grain for even one
            // window; title/master learning above remains temporal.
            if (shot_boundary && spatial_reliability > 0.0) {
                m_shot_obs_p[b] = pobs;
                m_shot_obs_w[b] = 0.80 * spatial_reliability;
            } else if (shot_boundary) {
                m_shot_obs_p[b] = pobs;
                m_shot_obs_w[b] = 0.0;
            } else {
                bool acquire = m_shot_age < 4.0 && !geometry_only;
                // Survivor tracking is reduce-only, so it stays on RAW
                // reliability -- the S4 veto never withholds evidence that
                // can only reduce restoration.
                float obs_support = acquire ? spatial_reliability
                                            : reliability_raw;
                if (obs_support > 0.0) {
                    float obs_rate = acquire ? 0.35 * obs_support
                                             : 0.0020 * obs_support;
                    m_shot_obs_p[b] = mix(m_shot_obs_p[b], pobs, obs_rate);
                    m_shot_obs_w[b] += (acquire ? 0.35 : 0.004) * obs_support
                                       * (1.0 - m_shot_obs_w[b]);
                }
            }

        }

        // Title-wide survived fraction -> ONE bed level. The added layer
        // keeps the rendition curve's SHAPE: per-bin survivor differencing
        // inverted the film bell — zero added exactly in the measurable
        // mids, full deficit at the crushed/clipped extremes — an
        // anti-physical tonal profile (author spec, 2026-07-17 night).
        // Survivors therefore modulate the bed's LEVEL through one
        // fidelity-discounted aggregate; per-bin evidence keeps informing
        // the master curve, never the added field's tonal shape. With
        // fidelity 0.5 the bed level stays in [~0.5, 1] x missing fraction:
        // an even, film-shaped bed that lossy survivors can halve but never
        // extinguish — surviving delivered grain is damaged evidence
        // (clumping, blocking, compressed patches), and no lossy delivery
        // is a ProRes master.
        float surv_wsum = 0.5;   // prior mass: no survivor evidence
        float surv_dsum = 0.5;   // means the full missing fraction
        for (int b = 0; b < MP_TONE_BINS; b++) {
            float w = clamp(m_shot_obs_w[b], 0.0, 1.0);
            if (w <= 0.0) continue;
            float shape = prior_tone_shape(b);
            float title_curve = m_title_power * shape * shape * m_shot_gain;
            float learned_master = m_master_p[b] * m_shot_gain;
            float local_q = smoothstep(0.10, 0.35, m_master_w[b]);
            float master_b = mix(title_curve, learned_master, local_q);
            float healthy = MP_SURVIVOR_FIDELITY
                          * min(m_shot_obs_p[b] / max(master_b, 1.0e-12),
                                1.0);
            surv_wsum += w;
            surv_dsum += w * (1.0 - healthy);
        }
        float bed_deficit = surv_dsum / surv_wsum;

        for (int b = 0; b < MP_TONE_BINS; b++) {
            float sigma0 = sigma_band[b * MP_BANDS];
            float count_r = smoothstep(24.0, 96.0, float(tone_count[b]));
            float amp_r = smoothstep(0.00012, 0.00070, sigma0);
            float plaus = 1.0 - smoothstep(MP_SIGMA_PLAUS_LO,
                                           MP_SIGMA_PLAUS_HI, sigma0);
            float spatial_reliability = count_r * amp_r * plaus * q_cov;
            float reliability_raw = spatial_reliability * q_random * q_still;
            float reliability = reliability_raw * q_source;
            float shape = prior_tone_shape(b);
            float title_curve = m_title_power * shape * shape * m_shot_gain;
            float learned_master = m_master_p[b] * m_shot_gain;
            float local_q = smoothstep(0.10, 0.35, m_master_w[b]);
            // Local evidence wins in BOTH directions: a confidently-measured
            // quiet bin must be allowed to render below the title curve, or
            // downward per-bin learning is presentation-inert.
            float master = mix(title_curve, learned_master, local_q);
            float missing_fraction = prior_missing_fraction(b);
            float obs_weight = clamp(m_shot_obs_w[b], 0.0, 1.0);
            float char_target = MP_COMPLEMENT_POWER * master;
            float restore_target = bed_deficit * missing_fraction * master;

            float char_up_rate, char_down_rate;
            float restore_up_rate, restore_down_rate;
            if (geometry_only) {
                // Re-framing the observer changes neither the title nor the
                // shot presentation on the commit frame.
                char_up_rate = 0.0;
                char_down_rate = 0.0;
                restore_up_rate = 0.0;
                restore_down_rate = 0.0;
            } else if (shot_boundary) {
                // Commit the title-shaped shot character on the boundary rather
                // than letting an eye-visible adjustment trail into the shot.
                // The boundary uses the authenticated title baseline, so complete
                // both presentation terms atomically at that neutral shot gain.
                char_up_rate = 1.0;
                char_down_rate = 1.0;
                restore_up_rate = 1.0;
                restore_down_rate = 1.0;
            } else if (m_shot_age < 4.0) {
                // Upward post-boundary refinement requires temporal
                // authentication; the survivor read is downward-only, so its
                // shedding stays motion-ungated — disappearing proof must
                // revoke toward the title baseline even inside a moving cut.
                char_up_rate = 0.20 * evidence;
                char_down_rate = 0.50 * max(m_shot_obs_w[b],
                                            reliability_raw);
                restore_up_rate = 0.35 * reliability;
                restore_down_rate = 0.50 * max(m_shot_obs_w[b],
                                               spatial_reliability);
            } else {
                // Once the shot is established, presentation moves on a
                // roughly ten-minute horizon. The observer may keep learning
                // quickly internally without making that learning visible;
                // title-lane movement surfaces at the next boundary commit,
                // never mid-shot. The survivor-driven pullback is the one
                // faster lane: it can only shed restoration that visible
                // surviving grain contradicts.
                char_up_rate = 0.00004 * static_gate;
                char_down_rate = 0.00008;
                restore_up_rate = 0.00006 * static_gate;
                restore_down_rate = max(0.00008, 0.04 * obs_weight);
            }
            float ar = (char_target > m_char_p[b])
                     ? char_up_rate : char_down_rate;
            float rr = (restore_target > m_restore_p[b])
                     ? restore_up_rate : restore_down_rate;
            m_char_p[b] = mix(m_char_p[b], char_target, ar);
            m_restore_p[b] = mix(m_restore_p[b], restore_target, rr);

            title_sum += master;
            weight_sum += m_master_w[b];
            char_sum += m_char_p[b];
            restore_sum += m_restore_p[b];
            target_missing_sum += restore_target;
        }

        float avg_w = weight_sum / float(MP_TONE_BINS);
        // The retained boost lane is neutral in the current estimator; keep
        // this arithmetic synchronized with OUTPUT for state compatibility.
        float boost_q = clamp(m_shot_restore_boost - 1.0, 0.0, 1.0);
        float effective_restore = (restore_gain <= 2.0)
                                ? mix(restore_gain,
                                      min(2.0, 2.0 * restore_gain), boost_q)
                                : restore_gain;
        float p_total = char_sum / float(MP_TONE_BINS)
                      + effective_restore * effective_restore
                      * restore_sum / float(MP_TONE_BINS);
        float p_restore = effective_restore * effective_restore
                        * restore_sum / float(MP_TONE_BINS);
        m_loss_mix = (p_total > 1.0e-12) ? p_restore / p_total : 0.0;
        m_loss_conf = clamp(target_missing_sum
                          / max(title_sum, 1.0e-12), 0.0, 1.0);
        m_est_missing = sqrt(max(restore_sum / float(MP_TONE_BINS), 0.0));
        // Underlying renderable power, deliberately independent of live
        // gain/match knobs. OUTPUT gates and scales those controls directly;
        // caching them here made pause-time off->on toggles inherit a stale
        // zero until playback resumed.
        m_eff_render = sqrt(max(p_total, 0.0)) / MP_FIELD_STD;
        m_eff_render = clamp(m_eff_render, 0.0, 0.75);
        m_observed = observed;
        m_temporal_support = temporal;
        m_structure_ratio = structure_ratio;
        m_tone_conf = max(m_title_conf, avg_w);
        m_coverage = coverage;
        m_motion = changed;
        m_cut_score = hist_l1;
        m_shot_age = min(m_shot_age + 1.0, 65535.0);
        m_measured = min(m_measured + 1.0, 65535.0);
        m_prev_ready = 1.0;

        // Two clocks, two jobs. m_arr_seed is the visible arrangement and
        // follows grain_rate; m_field_seed is the standing template vocabulary
        // and follows grain_gen_rate. A skipped template tick therefore still
        // presents a fresh block layout/jitter whenever the visible clock ticks.
        float prev_gen_frame = m_gen_frame;
        // Float SSBO counters stop accepting +1 at 2^24. Wrap well before
        // that boundary; the seed hash intentionally tolerates this roughly
        // four-day cycle at 24p, while the cadence clock keeps advancing.
        m_gen_frame += 1.0;
        if (m_gen_frame >= 8388608.0) m_gen_frame = 0.0;
        float visible_seed = floor(m_gen_frame * grain_rate);
        float prev_visible_seed = floor(prev_gen_frame * grain_rate);
        bool visible_tick = visible_seed != prev_visible_seed;
        if (visible_tick)
            m_arr_seed = visible_seed;

        bool scheduled_regen = visible_tick
            && floor(visible_seed * grain_gen_rate)
             != floor(prev_visible_seed * grain_gen_rate);
        bool baked_params_changed =
               abs(m_baked_grain_size - grain_size) > 1.0e-6
            || abs(m_baked_grain_contrast - grain_contrast) > 1.0e-6
            || abs(m_baked_value_warp - value_warp) > 1.0e-6
            || abs(m_baked_grain_base_sat - grain_base_sat) > 1.0e-6
            || abs(m_baked_grain_sharpness - grain_sharpness) > 1.0e-6
            || abs(m_baked_match_grain - match_grain) > 1.0e-6;
        // Size/hardness are baked into GRAIN_FIELD. Keep their fast post-cut
        // acquisition responsive even at a reduced steady-state gen rate.
        bool fast_character = shot_boundary
            || (!geometry_only && m_shot_age <= 4.0 && evidence > 0.0);
        if (scheduled_regen || fast_character || baked_params_changed) {
            if (visible_tick)
                m_field_seed = m_arr_seed;
            m_regen_pending = 1.0;
        }
        if (baked_params_changed) {
            m_baked_grain_size = grain_size;
            m_baked_grain_contrast = grain_contrast;
            m_baked_value_warp = value_warp;
            m_baked_grain_base_sat = grain_base_sat;
            m_baked_grain_sharpness = grain_sharpness;
            m_baked_match_grain = match_grain;
        }

        // PASS 2 can be disabled by its PARAM-only WHEN. Snapshot a sticky
        // request only when that pass will execute; it must never clear this
        // value itself because its workgroups have no global ordering.
        bool gen_active = (grain_gain > 0.0 && match_grain > 0.0)
                       || debug_match > 0.5;
        m_regen = 0.0;
        if (gen_active && m_regen_pending > 0.5) {
            m_regen = 1.0;
            m_regen_pending = 0.0;
        }
        // OUTPUT is the full presentation canvas after placement. Preserve the
        // source raster aspect so its active destination rectangle can be
        // recovered independently of window/display aspect ratio.
        m_source_aspect = HOOKED_size.x / max(HOOKED_size.y, 1.0);
        imageStore(out_image, ivec2(0), vec4(0.0));
    }
    barrier();
}

void hook() {
    lean_observe();
}

//!HOOK LUMA
//!BIND HOOKED
//!BIND GRAIN_STATE
//!BIND GRAIN_FIELD
//!SAVE GRAIN_GEN_TRIGGER
//!WIDTH 960
//!HEIGHT 540
//!COMPUTE 32 32
//!WHEN grain_gain match_grain * debug_match +
//!DESC Film Grain Match: GRAIN gen (960x540 template, source-locked)

// LUMA hook = mpv's FRESH group: runs once per SOURCE frame regardless of
// video-sync / display refresh (measured 24.x/s under both display-resample
// @120Hz and audio sync, 2026-07-06). The OUTPUT composite (redraw group,
// per-present) just fetches GRAIN_FIELD, so re-presents cost ~nothing and
// grain cadence can't ride the display refresh. TPL ARCHITECTURE
// (2026-07-10): grain is a normalized active-picture field represented on the
// current 2160-sample implementation lattice,
// whose width follows the committed active-picture aspect, but this
// pass only generates the physical 960x540 TEMPLATE — the composite
// assembles the picture field from per-block randomized template windows
// (AV1-FGS style; see the block shuffle there). Template texels REPRESENT
// picture-lattice samples, so every sigma below is calibrated in normalized
// active-picture-height units and
// generation cost scales with TEMPLATE area (~0.2 ms vs 3.4 at full size;
// 960x540 = the measured quality/perf knee, dev grain-genrate README).
//
// STORAGE-TEXTURE RECOVERY (2026-07-06): the earlier split SAVE'd GRAIN_FIELD
// from this fresh pass and BIND'd it in the OUTPUT redraw pass — but a SAVE'd
// texture is a per-frame transient and did NOT survive the fresh->redraw group
// gap, so grain generated but never reached the presented frame. Fix: GRAIN_FIELD
// is now a persistent, shader-owned TEXTURE+STORAGE image (declared
// top-of-file, like the GRAIN_STATE SSBO) that we imageStore into here and
// imageLoad from in the composite. Persistent storage retains its contents across
// presents, so a redraw (no fresh dispatch) reads the last-written field. This
// fresh pass still needs a dispatch grid, so it SAVEs a throwaway trigger
// texture (GRAIN_GEN_TRIGGER, never bound) purely to size the 960x540 dispatch.
// GRAIN_FIELD is rgba16f: rgb = final signed grain (bandpass + warp),
// a is unused.

// 9-tap support (was 7): the blue channel's base sigma (1.20) reaches ~1.15 even at
// the crisp neutral and higher when coarse, where 7 taps (clean only to sigma ~1.0)
// truncate the Gaussian into a box and ripple the spectrum. 9 taps hold sigma up to
// ~1.5 cleanly, covering the full fine-digital -> 16mm render range. Arrays/loops
// are parametrized on MAX_TAPS so the support can never drift out of sync.
#define MAX_TAPS 4

// Neutral correlation-length calibration. The physical quantity is this sigma
// divided by PICTURE_DENSITY: a fraction of active picture height. K=0.75 was
// calibrated against real grain plates and retains broadband energy to the
// current lattice Nyquist. Source-matched size is therefore independent of
// master and output raster dimensions.
#define PICTURE_DENSITY 2160.0
// MUST equal MP_PICTURE_DENSITY / PICTURE_DENSITY_OUT in PASS 1 / PASS 3.
#define K_NEUTRAL_NORM (0.75 / 2160.0)
#define K_NEUTRAL (K_NEUTRAL_NORM * PICTURE_DENSITY)
// Physical TEMPLATE dims. The noise seed wraps on these so the template is
// TOROIDAL: the composite's per-block window fetches (offset + flip + overlap
// halo) wrap on the template extent and must not show a seam. Wrapping the
// seed coordinate makes noise(grid)==noise(0); the separable DoG halo then
// wraps too (each workgroup regenerates its halo from global coords), so the
// seam is continuous. In-range coords (the whole template interior) are
// unchanged. MUST equal this pass's WIDTH/HEIGHT directives above (960 x 540)
// Same translation unit, no compile guard. This is not a picture resolution:
// template is a reusable vocabulary on the current synthesis lattice, not a
// picture resolution. The composite derives its dims from imageSize(GRAIN_FIELD) so it
// can't desync; only these two same-file sites are hand-kept in lockstep.
// (Directive prefix omitted here on purpose: the parser splits sections on
// that marker even inside a comment.)
#define GEN_GRID ivec2(960, 540)
// Per-channel render-sigma cap. The 9-tap support holds a Gaussian cleanly to ~1.5;
// beyond that it would truncate. The coarse extreme and a higher-density future
// res-scaling can push a channel past it, so cap GRACEFULLY (slightly finer than
// ideal at that extreme) rather than ripple the spectrum. The actual film range
// (fine digital .. ~16mm) stays under it, so this never touches normal operation.
#define SIGMA_MAX    1.5
// CONTRAST / "sandpaper" axis (2026-06-05). The generator (white noise -> ONE Gaussian
// blur) is LOWPASS (DC-peaked = soft cloud); real film grain is BANDPASS (suppressed DC,
// mid-freq peak) per the offline profiler + ITU-T H.274's frequency-filtering grain model.
// Fix = difference-of-Gaussians: grain = blur(s1) - a*blur(s1*BP_RATIO), which suppresses
// DC -> bandpass. s1 = the per-channel render sigma (size lever); a = BP_ALPHA*grain_contrast
// (the hardness/sandpaper dial). grain_contrast=0 -> a=0 -> grain=blur(s1) = BIT-IDENTICAL
// to the old lowpass (A/B-safe). The 2nd (wider) blur reuses the 9-tap machinery by
// REGENERATING the (reproducible) noise; s2 is capped at SIGMA_MAX so 9 taps still hold it.
// An analytic RMS norm (from the blur weights) keeps grain STRENGTH constant as contrast
// rises. Offline-locked to CyberCity (tools/mgt_bandpass_design.py): BP_RATIO 1.8, A0 0.60.
#define BP_RATIO     1.8
#define BP_ALPHA     0.60
#define RED_VARIANCE_SCALE   1.0
#define GREEN_VARIANCE_SCALE 1.0
#define BLUE_VARIANCE_SCALE  1.0
#define RED_SATURATION       0.6
#define GREEN_SATURATION     0.5
#define BLUE_SATURATION      0.80

const uvec2 isize = uvec2(gl_WorkGroupSize) + uvec2(2 * MAX_TAPS);

shared float grain_r[isize.y][isize.x];
shared float grain_g[isize.y][isize.x];
shared float grain_b[isize.y][isize.x];
shared float dyn_wr[2 * MAX_TAPS + 1];
shared float dyn_wg[2 * MAX_TAPS + 1];
shared float dyn_wb[2 * MAX_TAPS + 1];
shared float dyn_wr2[2 * MAX_TAPS + 1];   // bandpass DoG outer-blur weights (s2 = BP_RATIO*s1)
shared float dyn_wg2[2 * MAX_TAPS + 1];
shared float dyn_wb2[2 * MAX_TAPS + 1];
shared float bp_norm[3];                   // per-channel analytic RMS norm (amplitude-stable)
shared float vsum_sigma[3];                // per-channel RMS of vsum (= GRAIN_STD*s1c), for value_warp
shared float warp_renorm;                  // 1/sqrt(E[tanh^2(value_warp*Z)]) -- amplitude-preserving
shared float field_norm[3];                // size-invariant absolute field RMS


uint pcg_hash(uint s) {
    uint state = s * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_triangular(inout uint state, float variance_scale) {
    uint a = pcg_hash(state); state = a;
    uint b = pcg_hash(state); state = b;
    float u = float(a) * (1.0 / 4294967296.0);
    float v = float(b) * (1.0 / 4294967296.0);
    return (u + v - 1.0) * 0.612 * variance_scale;
}

float gaussian_weight(float dx, float sigma) {
    return exp(-0.5 * dx * dx / max(sigma * sigma, 1e-6));
}

void hook() {
    uint lid = gl_LocalInvocationIndex;
    uint num_threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

    // The observer already produced the complete amount and character record.
    // Generation is source-locked; m_regen is this frame's immutable request.
    uint frame_seed = uint(max(0.0, m_field_seed));
    bool regenerate = m_regen > 0.5;
    if (gl_GlobalInvocationID.x == 0u && gl_GlobalInvocationID.y == 0u)
        imageStore(out_image, ivec2(0), vec4(0.0));

    // PARAM-uniform skips only. m_eff_render may NOT join this guard: it is an SSBO
    // read, and an early return conditioned on buffer data puts every barrier
    // below into varying control flow for FXC (D3D11 X3663 -- the same class
    // the unconditional outer blur already works around; params are cbuffer
    // values and provably uniform, buffer loads are not). The skip it bought
    // was structurally dead anyway: the restoration bed keeps m_eff_render > 0
    // on real content, and OUTPUT still gates on it safely (no barriers there).
    if ((grain_gain <= 0.0 || match_grain <= 0.0) && debug_match <= 0.5)
        return;

    // Grain is generated on the picture-space implementation lattice. Shot
    // character is committed from the persistent title posterior at cuts,
    // then moves only imperceptibly.
    float k_neutral = K_NEUTRAL;
    float observed_size = mix(1.35, 0.65,
                              smoothstep(0.20, 0.85, m_shot_size));
    float source_size = mix(k_neutral, observed_size,
                            clamp(match_grain * grain_sharpness, 0.0, 1.0));
    // Delivery loss changes missing power only. It cannot morph size/hardness
    // inside a shot, which previously made motion-correlated loss look synthetic.
    float k_size = source_size;
    // grain_size: live size multiplier (1.0 = calibrated). <1 finer, >1 coarser. Lets
    // you correct the per-source size by eye -- CyberCity currently renders too FINE
    // (fineR read its hard edges as fineness); raising size also lets the bandpass bite.
    k_size *= grain_size;
    const float soften_eff = 0.0;

    // CONTRAST/BANDPASS: build inner (s1) AND outer (s2 = BP_RATIO*s1) blur weights.
    // bp_alpha is UNIFORM across the workgroup (grain_contrast/match_grain/
    // m_shot_hardness are uniform reads), and the outer blur below runs
    // UNCONDITIONALLY (X3663 rework), so every barrier stays in uniform
    // control flow. bp_alpha=0 -> vsum2 multiplied out in the combine,
    // bp_norm=1, grain = blur(s1) = the old lowpass generator (A/B-safe).
    float render_hardness = smoothstep(0.25, 0.82, m_shot_hardness);
    float bp_alpha = BP_ALPHA * grain_contrast * match_grain
                   * render_hardness;
    // DoG positivity guard: when both blur sigmas clamp at SIGMA_MAX the
    // kernels coincide and the covariance zero-crossing sits at alpha 1.0
    // (>1 for any distinct pair, by Cauchy-Schwarz). Capping alpha below 1
    // guarantees the combined field stays positively correlated with the
    // inner blur on every channel at every grain_size — the coarse-shot
    // blue-channel inversion (audit bug #6) becomes unreachable. bp_norm
    // re-normalizes RMS, so the cap only limits how much DC the hardest
    // settings strip.
    bp_alpha = min(bp_alpha, 0.95);
    if (regenerate && lid < uint(2 * MAX_TAPS + 1)) {
        int idx = int(lid);
        float dx = float(idx - MAX_TAPS);
        float s1r = mix(0.78, 1.02, soften_eff) * k_size;
        float s1g = mix(1.00, 1.22, soften_eff) * k_size;
        float s1b = mix(1.20, 1.40, soften_eff) * k_size;
        dyn_wr[idx]  = gaussian_weight(dx, min(s1r, SIGMA_MAX));
        dyn_wg[idx]  = gaussian_weight(dx, min(s1g, SIGMA_MAX));
        dyn_wb[idx]  = gaussian_weight(dx, min(s1b, SIGMA_MAX));
        dyn_wr2[idx] = gaussian_weight(dx, min(s1r * BP_RATIO, SIGMA_MAX));
        dyn_wg2[idx] = gaussian_weight(dx, min(s1g * BP_RATIO, SIGMA_MAX));
        dyn_wb2[idx] = gaussian_weight(dx, min(s1b * BP_RATIO, SIGMA_MAX));
    }
    barrier();
    if (regenerate && lid == 0u) {
        float nr = 0.0, ng = 0.0, nb = 0.0, nr2 = 0.0, ng2 = 0.0, nb2 = 0.0;
        for (int i = 0; i < 2 * MAX_TAPS + 1; i++) {
            nr += dyn_wr[i]; ng += dyn_wg[i]; nb += dyn_wb[i];
            nr2 += dyn_wr2[i]; ng2 += dyn_wg2[i]; nb2 += dyn_wb2[i];
        }
        for (int i = 0; i < 2 * MAX_TAPS + 1; i++) {
            dyn_wr[i] /= nr; dyn_wg[i] /= ng; dyn_wb[i] /= nb;
            dyn_wr2[i] /= nr2; dyn_wg2[i] /= ng2; dyn_wb2[i] /= nb2;
        }
        // Analytic RMS norm: Var(blur1 - a*blur2)/Var(blur1) on white noise. For a
        // separable 2D kernel, sum-of-squares and the cross-sum get squared. Keeps
        // grain STRENGTH constant as contrast rises (clean A/B). a=0 -> vr=1 -> norm=1.
        float s1c[3]; float s2c[3]; float s12c[3];
        s1c[0] = 0.0; s1c[1] = 0.0; s1c[2] = 0.0;
        s2c[0] = 0.0; s2c[1] = 0.0; s2c[2] = 0.0;
        s12c[0] = 0.0; s12c[1] = 0.0; s12c[2] = 0.0;
        for (int i = 0; i < 2 * MAX_TAPS + 1; i++) {
            s1c[0] += dyn_wr[i]*dyn_wr[i];  s2c[0] += dyn_wr2[i]*dyn_wr2[i];  s12c[0] += dyn_wr[i]*dyn_wr2[i];
            s1c[1] += dyn_wg[i]*dyn_wg[i];  s2c[1] += dyn_wg2[i]*dyn_wg2[i];  s12c[1] += dyn_wg[i]*dyn_wg2[i];
            s1c[2] += dyn_wb[i]*dyn_wb[i];  s2c[2] += dyn_wb2[i]*dyn_wb2[i];  s12c[2] += dyn_wb[i]*dyn_wb2[i];
        }
        for (int c = 0; c < 3; c++) {
            float r1 = s1c[c] * s1c[c];
            float vr = 1.0 + bp_alpha * bp_alpha * (s2c[c]*s2c[c]) / r1
                           - 2.0 * bp_alpha * (s12c[c]*s12c[c]) / r1;
            bp_norm[c] = inversesqrt(max(vr, 1e-4));
        }
        // value_warp amplitude bookkeeping (uniform; computed once per frame). vsum_sigma =
        // the per-channel RMS of vsum (bp_norm makes Var(vsum)=Var(vsum1)=GRAIN_STD^2*s1c^2;
        // verified ratio 1.000). warp_renorm = 1/sqrt(E[tanh^2(value_warp*Z)]), Z~N(0,1),
        // via fixed Gaussian quadrature -> the tanh warp preserves grain strength (RMS
        // stable to ~1% offline). value_warp<=0.05 -> renorm 1 (the warp branch is skipped).
        const vec3 GRAIN_STD = vec3(0.16223, 0.1742, 0.15653);
        vsum_sigma[0] = GRAIN_STD.x * s1c[0];
        vsum_sigma[1] = GRAIN_STD.y * s1c[1];
        vsum_sigma[2] = GRAIN_STD.z * s1c[2];
        // Canonical absolute RMS at the neutral picture-space kernel. Without
        // this normalization coarse kernels render less power and fine kernels
        // more power even when the observer requested the same sigma_plus.
        field_norm[0] = 0.08318138 / max(vsum_sigma[0], 1.0e-6);
        field_norm[1] = 0.06602582 / max(vsum_sigma[1], 1.0e-6);
        field_norm[2] = 0.04909565 / max(vsum_sigma[2], 1.0e-6);

        // Analytic pre-warp RGB covariance of the field this dispatch builds.
        // H.274/AV1 model grain per colour component, and the generator's
        // channel-specific size/RMS is intentional character. Keep it; carry
        // its six covariance terms so OUTPUT can prevent multiplicative
        // RGB compositing from turning that character into extra LUMA energy on
        // saturated colours. For separable 2-D kernels, every cross inner
        // product is the square of its 1-D inner product. value_warp is a
        // monotone marginal transform; the private empirical sweep finds that
        // combined legal artistic-override residuals stay within ~5%, so no fitted warp
        // heuristic belongs in this physical covariance record.
        if (gl_WorkGroupID.x == 0u && gl_WorkGroupID.y == 0u) {
        vec3 d11 = vec3(0.0), d12 = vec3(0.0);
        vec3 d21 = vec3(0.0), d22 = vec3(0.0);
        for (int i = 0; i < 2 * MAX_TAPS + 1; i++) {
            d11 += vec3(dyn_wr[i]  * dyn_wg[i],
                        dyn_wr[i]  * dyn_wb[i],
                        dyn_wg[i]  * dyn_wb[i]);
            d12 += vec3(dyn_wr[i]  * dyn_wg2[i],
                        dyn_wr[i]  * dyn_wb2[i],
                        dyn_wg[i]  * dyn_wb2[i]);
            d21 += vec3(dyn_wr2[i] * dyn_wg[i],
                        dyn_wr2[i] * dyn_wb[i],
                        dyn_wg2[i] * dyn_wb[i]);
            d22 += vec3(dyn_wr2[i] * dyn_wg2[i],
                        dyn_wr2[i] * dyn_wb2[i],
                        dyn_wg2[i] * dyn_wb2[i]);
        }
        float a2 = bp_alpha * bp_alpha;
        vec3 cross_energy = d11 * d11
                          - bp_alpha * (d12 * d12 + d21 * d21)
                          + a2 * d22 * d22;
        vec3 self_energy = vec3(
            s1c[0]*s1c[0] - 2.0*bp_alpha*s12c[0]*s12c[0] + a2*s2c[0]*s2c[0],
            s1c[1]*s1c[1] - 2.0*bp_alpha*s12c[1]*s12c[1] + a2*s2c[1]*s2c[1],
            s1c[2]*s1c[2] - 2.0*bp_alpha*s12c[2]*s12c[2] + a2*s2c[2]*s2c[2]);
        vec3 filter_corr = cross_energy * inversesqrt(max(
            vec3(self_energy.x * self_energy.y,
                 self_energy.x * self_energy.z,
                 self_energy.y * self_energy.z), vec3(1.0e-12)));

        const vec3 base_luma = vec3(0.299, 0.587, 0.114);
        const vec3 noise_scale = vec3(RED_VARIANCE_SCALE,
                                      GREEN_VARIANCE_SCALE,
                                      BLUE_VARIANCE_SCALE);
        vec3 lum_mix = base_luma * noise_scale;
        vec3 sat = vec3(RED_SATURATION, GREEN_SATURATION,
                        BLUE_SATURATION) * grain_base_sat;
        vec3 mix_r = mix(lum_mix, vec3(noise_scale.r, 0.0, 0.0), sat.r);
        vec3 mix_g = mix(lum_mix, vec3(0.0, noise_scale.g, 0.0), sat.g);
        vec3 mix_b = mix(lum_mix, vec3(0.0, 0.0, noise_scale.b), sat.b);
        vec3 base_corr = vec3(dot(mix_r, mix_g),
                              dot(mix_r, mix_b),
                              dot(mix_g, mix_b)) * inversesqrt(max(vec3(
            dot(mix_r, mix_r) * dot(mix_g, mix_g),
            dot(mix_r, mix_r) * dot(mix_b, mix_b),
            dot(mix_g, mix_g) * dot(mix_b, mix_b)), vec3(1.0e-12)));
        // field_norm uses the historical calibrated GRAIN_STD constants. The
        // base mix's true marginal standard deviation drifts when base_sat is
        // changed (and blue is slightly below the calibration even at the
        // default), so derive cap-only live diagonals from the actual mix. Do
        // not feed these into the established log-normal mean correction.
        const float TRIANGULAR_STD = 0.612 / sqrt(6.0);
        vec3 mix_std = TRIANGULAR_STD * sqrt(vec3(
            dot(mix_r, mix_r), dot(mix_g, mix_g), dot(mix_b, mix_b)));
        vec3 cap_field_std = vec3(0.08318138, 0.06602582, 0.04909565)
                           * mix_std / GRAIN_STD;
        vec3 cap_field_var = cap_field_std * cap_field_std;
        // Schur product of two Gram matrices is PSD algebraically. Project the
        // third correlation into the valid interval implied by the first two;
        // this removes tiny float32 cancellation excursions at coarse/clamped
        // kernel corners without the non-PSD risk of three independent clamps.
        vec3 field_corr = base_corr * filter_corr;
        float corr_rg = clamp(field_corr.x, -1.0, 1.0);
        float corr_rb = clamp(field_corr.y, -1.0, 1.0);
        float corr_gb_mid = corr_rg * corr_rb;
        float corr_gb_span = sqrt(max((1.0 - corr_rg * corr_rg)
                                    * (1.0 - corr_rb * corr_rb), 0.0));
        float corr_gb = clamp(field_corr.z,
                              corr_gb_mid - corr_gb_span,
                              corr_gb_mid + corr_gb_span);
        m_field_cov_rg = corr_rg * cap_field_std.r * cap_field_std.g;
        m_field_cov_rb = corr_rb * cap_field_std.r * cap_field_std.b;
        m_field_cov_gb = corr_gb * cap_field_std.g * cap_field_std.b;
        m_field_var_r = cap_field_var.r;
        m_field_var_g = cap_field_var.g;
        m_field_var_b = cap_field_var.b;
        }
        if (value_warp > 0.05) {
            float num = 0.0, den = 0.0;
            for (int j = -16; j <= 16; j++) {
                float z = float(j) * 0.25;
                float wpdf = exp(-0.5 * z * z);
                float t = tanh(value_warp * z);
                num += wpdf * t * t; den += wpdf;
            }
            warp_renorm = inversesqrt(max(num / den, 1e-4));
        } else {
            warp_renorm = 1.0;
        }
    }
    barrier();

    // --- inner blur(s1): generate noise -> separable blur -> vsum1 ---
    if (regenerate) for (uint i = lid; i < isize.y * isize.x; i += num_threads) {
        uvec2 local_pos = uvec2(i % isize.x, i / isize.x);
        ivec2 global_coord_i = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy)
                             + ivec2(local_pos) - ivec2(MAX_TAPS);
        uvec2 global_pos = uvec2((global_coord_i % GEN_GRID + GEN_GRID) % GEN_GRID);
        uint seed_init = (global_pos.x * 1664525u) + (global_pos.y * 22695477u)
                       + (frame_seed * 314159265u);
        float g_r = rand_triangular(seed_init, RED_VARIANCE_SCALE);
        float g_g = rand_triangular(seed_init, GREEN_VARIANCE_SCALE);
        float g_b = rand_triangular(seed_init, BLUE_VARIANCE_SCALE);
        float grain_lum = dot(vec3(g_r, g_g, g_b), vec3(0.299, 0.587, 0.114));
        grain_r[local_pos.y][local_pos.x] = mix(grain_lum, g_r, RED_SATURATION * grain_base_sat);
        grain_g[local_pos.y][local_pos.x] = mix(grain_lum, g_g, GREEN_SATURATION * grain_base_sat);
        grain_b[local_pos.y][local_pos.x] = mix(grain_lum, g_b, BLUE_SATURATION * grain_base_sat);
    }
    barrier();
    if (regenerate) for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
        float hsum_r = 0.0, hsum_g = 0.0, hsum_b = 0.0;
        for (int x = 0; x < 2 * MAX_TAPS + 1; x++) {
            hsum_r += dyn_wr[x] * grain_r[y][gl_LocalInvocationID.x + x];
            hsum_g += dyn_wg[x] * grain_g[y][gl_LocalInvocationID.x + x];
            hsum_b += dyn_wb[x] * grain_b[y][gl_LocalInvocationID.x + x];
        }
        grain_r[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_r;
        grain_g[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_g;
        grain_b[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_b;
    }
    barrier();
    float vsum1_r = 0.0, vsum1_g = 0.0, vsum1_b = 0.0;
    if (regenerate) for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
        vsum1_r += dyn_wr[y] * grain_r[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
        vsum1_g += dyn_wg[y] * grain_g[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
        vsum1_b += dyn_wb[y] * grain_b[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
    }

    // --- outer blur(s2) for the DoG: regenerate the SAME noise, blur with dyn_*2 ---
    // UNCONDITIONAL (barriers must not sit in varying control flow -> D3D X3663). When
    // grain_contrast=0, bp_alpha=0 so vsum2 is multiplied out in the combine (still
    // A/B-safe); we just always pay the cheap 2nd blur instead of skipping it.
    float vsum2_r = 0.0, vsum2_g = 0.0, vsum2_b = 0.0;
    barrier();
    {
        if (regenerate) for (uint i = lid; i < isize.y * isize.x; i += num_threads) {
            uvec2 local_pos = uvec2(i % isize.x, i / isize.x);
            ivec2 global_coord_i = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy)
                                 + ivec2(local_pos) - ivec2(MAX_TAPS);
            uvec2 global_pos = uvec2((global_coord_i % GEN_GRID + GEN_GRID) % GEN_GRID);
            uint seed_init = (global_pos.x * 1664525u) + (global_pos.y * 22695477u)
                           + (frame_seed * 314159265u);
            float g_r = rand_triangular(seed_init, RED_VARIANCE_SCALE);
            float g_g = rand_triangular(seed_init, GREEN_VARIANCE_SCALE);
            float g_b = rand_triangular(seed_init, BLUE_VARIANCE_SCALE);
            float grain_lum = dot(vec3(g_r, g_g, g_b), vec3(0.299, 0.587, 0.114));
            grain_r[local_pos.y][local_pos.x] = mix(grain_lum, g_r, RED_SATURATION * grain_base_sat);
            grain_g[local_pos.y][local_pos.x] = mix(grain_lum, g_g, GREEN_SATURATION * grain_base_sat);
            grain_b[local_pos.y][local_pos.x] = mix(grain_lum, g_b, BLUE_SATURATION * grain_base_sat);
        }
        barrier();
        if (regenerate) for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
            float hsum_r = 0.0, hsum_g = 0.0, hsum_b = 0.0;
            for (int x = 0; x < 2 * MAX_TAPS + 1; x++) {
                hsum_r += dyn_wr2[x] * grain_r[y][gl_LocalInvocationID.x + x];
                hsum_g += dyn_wg2[x] * grain_g[y][gl_LocalInvocationID.x + x];
                hsum_b += dyn_wb2[x] * grain_b[y][gl_LocalInvocationID.x + x];
            }
            grain_r[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_r;
            grain_g[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_g;
            grain_b[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_b;
        }
        barrier();
        if (regenerate) for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
            vsum2_r += dyn_wr2[y] * grain_r[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
            vsum2_g += dyn_wg2[y] * grain_g[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
            vsum2_b += dyn_wb2[y] * grain_b[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
        }
    }

    if (regenerate) {
        // Bandpass combine (DoG = blur(s1) - a*blur(s2)), RMS-normalized so
        // strength holds as measured hardness changes.
        float vsum_r = bp_norm[0] * (vsum1_r - bp_alpha * vsum2_r);
        float vsum_g = bp_norm[1] * (vsum1_g - bp_alpha * vsum2_g);
        float vsum_b = bp_norm[2] * (vsum1_b - bp_alpha * vsum2_b);

        // VALUE-DOMAIN contrast (value_warp): tanh the grain toward a
        // bimodal/high-per-grain-contrast marginal while preserving RMS.
        if (value_warp > 0.05) {
            vsum_r = vsum_sigma[0] * warp_renorm * tanh(value_warp * vsum_r / max(vsum_sigma[0], 1e-6));
            vsum_g = vsum_sigma[1] * warp_renorm * tanh(value_warp * vsum_g / max(vsum_sigma[1], 1e-6));
            vsum_b = vsum_sigma[2] * warp_renorm * tanh(value_warp * vsum_b / max(vsum_sigma[2], 1e-6));
        }
        vsum_r *= field_norm[0];
        vsum_g *= field_norm[1];
        vsum_b *= field_norm[2];

        // Final field store. The trigger dispatch rounds 540 up to 544 rows,
        // so guard the four out-of-range rows explicitly.
        ivec2 gpos = ivec2(gl_GlobalInvocationID.xy);
        if (all(lessThan(gpos, imageSize(GRAIN_FIELD)))) {
            imageStore(GRAIN_FIELD, gpos, vec4(vsum_r, vsum_g, vsum_b, 0.0));
            if (all(equal(gpos, ivec2(0))))
                m_field_valid = 1.0;
        }
    }
}

//!HOOK OUTPUT
//!BIND HOOKED
//!BIND GRAIN_STATE
//!BIND GRAIN_FIELD
//!COMPUTE 32 32
//!WHEN grain_gain match_grain * debug_match +
//!DESC Film Grain Match: OUTPUT composite + debug

// The per-present half: this is the shader's only pass in mpv's REDRAW group
// (re-runs per present -- up to display Hz under display-resample), so it
// stays fetch + key + apply. All grain generation and every scalar that used
// to be derived here (eff_render/mid/steep, conf, shape_w) now come from the
// source-locked gen pass via GRAIN_STATE / GRAIN_FIELD.
#define DENSITY_GAIN 1.0
#define DENSITY_SHADOW_FLOOR 0.015
#define PICTURE_DENSITY_OUT 2160.0
// MUST equal MP_PICTURE_DENSITY / PICTURE_DENSITY in PASS 1 / PASS 2.
#define TPL_BLOCK_NORM (64.0 / 2160.0) // active-picture-height fraction
#define TPL_OV_NORM    (8.0 / 2160.0)  // active-picture-height fraction
#define TPL_BLOCK (TPL_BLOCK_NORM * PICTURE_DENSITY_OUT)
#define TPL_OV    (TPL_OV_NORM * PICTURE_DENSITY_OUT)
#define MP_TONE_BINS_OUT 8
#define MP_FIELD_STD_OUT 0.0185
// MUST equal PASS 1's MP_STATE_MAGIC (same translation-unit-sync rule as
// MP_TONE_BINS_OUT / MP_FIELD_STD_OUT — no compile guard exists).
#define MP_STATE_MAGIC_OUT 0.949451

const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
const vec3 field_var = vec3(0.08318138 * 0.08318138,
                            0.06602582 * 0.06602582,
                            0.04909565 * 0.04909565);

// --- HDR (PQ BT.2020 output) domain bridge, grain_hdr=1. The grain model is
// measured on gamma-encoded SDR source codes at LUMA, but whenever the player
// target is PQ (target-trc=pq, or an HDR-signalled display) the OUTPUT pixels
// this pass receives are true PQ BT.2020: libplacebo runs OUTPUT hooks AFTER
// the conversion to the target colorspace, so measure and apply sit in
// different domains unless we bridge. Bridge per pixel: PQ code -> linear
// BT.2020 nits -> linear BT.709 -> SDR-equivalent 2.4-gamma code vs
// grain_ref_white; key + apply the model there; convert back and re-encode.
// Grain lands exactly as the SDR path wherever the image sits at SDR levels
// and fades to zero shortly above reference white (the measured bell
// extrapolates; grain does not persist into expanded highlight cores).
// Standard ST 2084 constants, so the round trip shares a transfer with
// whatever performed the PQ encode upstream.
float pq_eotf_nits(float e) {
    float p = pow(e, 1.0 / 78.84375);
    return 10000.0 * pow(max(p - 0.8359375, 0.0)
                         / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}
float pq_oetf_code(float nits) {
    float y = pow(clamp(nits / 10000.0, 0.0, 1.0), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), 78.84375);
}

vec3 bt2020_to_bt709(vec3 rgb) {
    return vec3(
         1.6604903 * rgb.r - 0.5876391 * rgb.g - 0.0728516 * rgb.b,
        -0.1245500 * rgb.r + 1.1328999 * rgb.g - 0.0083480 * rgb.b,
        -0.0181511 * rgb.r - 0.1005787 * rgb.g + 1.1187299 * rgb.b
    );
}

vec3 bt709_to_bt2020(vec3 rgb) {
    return vec3(
        0.6274040 * rgb.r + 0.3292820 * rgb.g + 0.0433136 * rgb.b,
        0.0690970 * rgb.r + 0.9195400 * rgb.g + 0.0113612 * rgb.b,
        0.0163916 * rgb.r + 0.0880132 * rgb.g + 0.8955950 * rgb.b
    );
}

vec3 signed_pow(vec3 v, float p) {
    return sign(v) * pow(abs(v), vec3(p));
}

float matched_grain_scale(float lum, float hdr_mode) {
    // Exposure-weighted bins devote two anchors to the compression-vulnerable
    // near-black/shadow range. Curves store absolute independent power.
    float p = clamp(sqrt(clamp(lum, 0.0, 1.0))
                    * float(MP_TONE_BINS_OUT) - 0.5,
                    0.0, float(MP_TONE_BINS_OUT - 1));
    int i0 = int(floor(p));
    int i1 = min(i0 + 1, MP_TONE_BINS_OUT - 1);
    float char_p = mix(m_char_p[i0], m_char_p[i1], fract(p));
    float restore_p = mix(m_restore_p[i0], m_restore_p[i1], fract(p));
    // Keep this expression identical to PASS 1's p_total accounting.
    // The retained boost lane is neutral in the current estimator; keep this
    // arithmetic synchronized with PASS 1 for state compatibility.
    float boost_q = clamp(m_shot_restore_boost - 1.0, 0.0, 1.0);
    float effective_restore = (restore_gain <= 2.0)
                            ? mix(restore_gain,
                                  min(2.0, 2.0 * restore_gain), boost_q)
                            : restore_gain;
    float power = char_p + effective_restore * effective_restore * restore_p;
    float sigma = grain_gain * match_grain * sqrt(max(power, 0.0));
    float black_lo = mix(0.0010, 0.00025, hdr_mode);
    float black_hi = mix(0.0120, 0.00600, hdr_mode);
    // ONE fade-to-white envelope for every chain (author decree 2026-07-19,
    // resolving charter-audit P2): between the shadow toe and upper fade,
    // amount follows the rendition curve -- no aesthetic shoulder.
    // grain_fade sets the work-domain luma where grain reaches zero.
    // Clip-limited chains (plain
    // SDR, or SDR in a PQ container) cap the top at 0.95: waiting for
    // mathematical 1.0 leaves a tiny moving tail in perceptual whites --
    // temporal playback reveals it and channel clipping makes it one-sided
    // (the 2026-07-15 live-white finding). With headroom the top is the
    // user's above-ref-white reach; we cannot know the upstream expansion's
    // tuning. The 0.96/1.10 start/top ratio is the retained stock geometry.
    // Aggressive low fades need a matching shadow toe: the stock black gate
    // reaches full grain almost immediately, which makes a 0.2 fade read as a
    // hard band. Below 0.5, widen the rise and shorten its full-strength shelf;
    // at 0.5 and above every edge is exactly the stock expression.
    // OUTPUT's symmetric room clamp and flash-guard ceiling key on the same
    // top; the black gate stays container-keyed on hdr_mode.
    float white_hdr = hdr_mode * step(0.5, grain_headroom);
    // Floored at the declared PARAM minimum: a degenerate override would
    // collapse the smoothstep edges below (NaN through the whole tone scale).
    float fade_user = max(grain_fade, 0.2);
    float fade_top = mix(min(fade_user, 0.95), fade_user, white_hdr);
    float stock_start = fade_top * (0.96 / 1.10);
    float low_fade_q = 1.0 - smoothstep(0.30, 0.50, fade_top);
    float toe_hi = mix(black_hi, max(black_hi, 0.45 * fade_top), low_fade_q);
    float fade_start = mix(stock_start, 0.60 * fade_top, low_fade_q);
    float shadow_toe = smoothstep(black_lo, toe_hi, lum);
    float white_fade = 1.0 - smoothstep(fade_start, fade_top, lum);
    float protection = shadow_toe * white_fade;
    float scale = sigma / MP_FIELD_STD_OUT * protection;
    // Density multiplication contributes one factor of luma. Divide it back
    // out so the absolute master-power curve survives into shadows; the small
    // floor hands off continuously to the mean-neutral pedestal below.
    if (density_combine > 0.5)
        scale /= max(lum, 0.015);
    return min(scale, 8.0);
}

// Multiplicative RGB density is not automatically colour-energy neutral. The
// small-signal working-code luma perturbation is
//   dY = scale * (luma_coeff * carrier_rgb)^T * grain_rgb,
// so its variance is q^T C q. On a neutral at the same luma q is simply the
// neutral level times luma_coeff. Because this generator intentionally gives
// red the strongest/finer component, the old path made equal-luma saturated
// red ~1.24x neutral RMS at defaults (up to ~1.42x at grain_base_sat=1).
// Carry the generator's current covariance and reduce only chromaticities above
// the neutral reference. Never boost a quiet direction: H.274/AV1 permit real
// per-component grain character, and this correction removes compositing bias
// rather than flattening that character. Signed-gamma out-of-709 intermediates
// in the PQ bridge must stay signed: their cancellation is part of the actual
// BT.709-domain luma perturbation, while the downward-only cap cannot amplify it.
// The covariance describes the canonical 2160-sample field; OUTPUT footprint
// filtering can make lower-density presentations more conservative, so this is
// a no-excess cap rather than a claim of resolution-wide exact normalization.
float density_colour_energy_cap(vec3 carrier) {
    float neutral_level = max(dot(luma_coeff, carrier), 0.0);
    if (neutral_level <= 1.0e-8)
        return 1.0;
    vec3 q = luma_coeff * carrier;
    vec3 cap_field_var = vec3(m_field_var_r, m_field_var_g, m_field_var_b);
    float carrier_var = dot(q * q, cap_field_var)
        + 2.0 * (q.r * q.g * m_field_cov_rg
               + q.r * q.b * m_field_cov_rb
               + q.g * q.b * m_field_cov_gb);
    float neutral_var = neutral_level * neutral_level * (
          dot(luma_coeff * luma_coeff, cap_field_var)
        + 2.0 * (luma_coeff.r * luma_coeff.g * m_field_cov_rg
               + luma_coeff.r * luma_coeff.b * m_field_cov_rb
               + luma_coeff.g * luma_coeff.b * m_field_cov_gb));
    if (carrier_var <= neutral_var * 1.0001)
        return 1.0;
    return min(sqrt(max(neutral_var, 0.0)
                  / max(carrier_var, 1.0e-12)), 1.0);
}

// Wrapped point fetch of the toroidal grain template. GRAIN_FIELD is a
// persistent storage image, so every filtered read below is assembled from
// explicit imageLoads. Coordinates are implementation-lattice samples, with
// sample i centred at i+0.5. Exact-density samples therefore retain the
// historical one-load path, while arbitrary footprint taps use the correct
// half-texel cell boundaries.
vec3 grain_point(vec2 fpos, ivec2 g) {
    ivec2 i = ivec2(floor(fpos));
    ivec2 a = (i % g + g) % g;                  // component-wise wrap
    return imageLoad(GRAIN_FIELD, a).rgb;
}

// Same PCG as the measure/gen units — the block shuffle below needs a few
// decorrelated words per visible tick.
uint pcg_hash(uint s) {
    uint state = s * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// One block's grain sample for an arbitrary picture-lattice position pos. b is
// the block index doing the presenting (possibly a neighbour of pos's own
// block, when evaluating the overlap bands — intra then exceeds [0,BLOCK)
// and the flip can go one period negative, bounded by -TPL_OV). One template
// extent is added before the fetch so the coordinate reaching grain_point's
// integer wrap is ALWAYS non-negative: GLSL leaves % formally undefined on
// negative operands, and while every real backend folds it correctly through
// the double-mod, the D3D11/SPIRV-Cross path is unvalidated — one vec2 add
// buys spec-clean portability (audit 2026-07-10). Wrap-equivalent, so output
// is bit-identical. Each block shows a randomized (integer offset + H/V
// flip) window of the toroidal template, re-hashed per visible tick.
vec3 tpl_sample(vec2 b, vec2 pos, float vseed, ivec2 g) {
    vec2 intra = pos - b * TPL_BLOCK;
    uint h0 = pcg_hash(uint(int(b.x)) * 374761393u
                     + uint(int(b.y)) * 3266489917u
                     + uint(vseed) * 668265263u);
    uint h1 = pcg_hash(h0);
    uint h2 = pcg_hash(h1);
    if ((h2 & 1u) != 0u) intra.x = TPL_BLOCK - intra.x;
    if ((h2 & 2u) != 0u) intra.y = TPL_BLOCK - intra.y;
    return grain_point(intra + vec2(g)
                       + vec2(float(h0 % uint(g.x)), float(h1 % uint(g.y))), g);
}

// Evaluate one point of the assembled picture-space field. Filtering must call this
// complete evaluator for every tap: filtering only the physical template and
// reusing the centre pixel's block weights would crossfade the wrong shuffled
// windows at block boundaries.
vec3 picture_point(vec2 pos, float vseed, ivec2 g) {
    vec2 b = floor(pos / TPL_BLOCK);
    vec2 u = pos - b * TPL_BLOCK;
    float wxp = 0.0, wxc = 1.0, wyp = 0.0, wyc = 1.0;
    if (u.x < TPL_OV) {
        float t = u.x / TPL_OV * 1.5707963;
        wxp = cos(t); wxc = sin(t);
    }
    if (u.y < TPL_OV) {
        float t = u.y / TPL_OV * 1.5707963;
        wyp = cos(t); wyc = sin(t);
    }
    vec3 field = (wxc * wyc) * tpl_sample(b, pos, vseed, g);
    if (wxp > 0.0)
        field += (wxp * wyc)
               * tpl_sample(b - vec2(1.0, 0.0), pos, vseed, g);
    if (wyp > 0.0)
        field += (wyp * wxc)
               * tpl_sample(b - vec2(0.0, 1.0), pos, vseed, g);
    if (wxp > 0.0 && wyp > 0.0)
        field += (wxp * wyp)
               * tpl_sample(b - vec2(1.0, 1.0), pos, vseed, g);
    return field;
}

// Tent reconstruction of the canonical field for outputs denser than its
// picture-space lattice. This reveals the same finite-bandwidth field at more
// sample points instead of repeating lattice samples. At an exact sample centre
// fract() is zero and the result is exactly picture_point(base).
vec3 picture_linear(vec2 pos, float vseed, ivec2 g) {
    vec2 base = floor(pos - 0.5) + 0.5;
    vec2 f = pos - base;
    vec3 c00 = picture_point(base, vseed, g);
    vec3 c10 = picture_point(base + vec2(1.0, 0.0), vseed, g);
    vec3 c01 = picture_point(base + vec2(0.0, 1.0), vseed, g);
    vec3 c11 = picture_point(base + vec2(1.0, 1.0), vseed, g);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// Sample the continuous picture field through one output-pixel footprint.
// output_step is the output pixel width in implementation-lattice samples:
//   == 1: one point per lattice sample
//    < 1: tent reconstruction between canonical samples
//  1..1.25: the same tent is the deterministic narrow-footprint integral
//   > 1.25: stratified integration, growing the grid with footprint span
// Adjacent estimators crossfade over a quarter-texel transition so changing
// window height cannot produce a visible grain-power step.
// The footprint average deliberately loses only power the output grid cannot
// resolve; it is not RMS-renormalized. A source/output gain would make the
// grain a property of the user's scaler instead of the title.
vec3 picture_grid2(vec2 centre, float output_step, float vseed, ivec2 g) {
    vec2 d = vec2(0.25 * output_step);
    return 0.25 * (
          picture_point(centre + vec2(-d.x, -d.y), vseed, g)
        + picture_point(centre + vec2( d.x, -d.y), vseed, g)
        + picture_point(centre + vec2(-d.x,  d.y), vseed, g)
        + picture_point(centre + vec2( d.x,  d.y), vseed, g));
}

vec3 picture_grid3(vec2 centre, float output_step, float vseed, ivec2 g) {
    vec3 integrated = vec3(0.0);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            vec2 q = (vec2(float(x), float(y)) + 0.5) / 3.0 - 0.5;
            integrated += picture_point(centre + q * output_step, vseed, g);
        }
    }
    return integrated * (1.0 / 9.0);
}

vec3 picture_grid4(vec2 centre, float output_step, float vseed, ivec2 g) {
    vec3 integrated = vec3(0.0);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            vec2 q = (vec2(float(x), float(y)) + 0.5) / 4.0 - 0.5;
            integrated += picture_point(centre + q * output_step, vseed, g);
        }
    }
    return integrated * (1.0 / 16.0);
}

vec3 picture_pixel(vec2 centre, float output_step, float vseed, ivec2 g) {
    if (abs(output_step - 1.0) < 0.000001)
        return picture_point(centre, vseed, g);
    if (output_step < 1.25)
        return picture_linear(centre, vseed, g);

    // Keep the sample pitch near one implementation-lattice sample as output gets
    // smaller. The grid grows while output pixel count falls by the same
    // square factor, so 720p/540p gain accuracy without an exploding total
    // evaluator count. Four per axis is the deliberate portability ceiling;
    // very small outputs remain an approximation rather than compiling large
    // dynamic loops into every backend.
    if (output_step < 1.5) {
        float t = smoothstep(1.25, 1.5, output_step);
        return mix(picture_linear(centre, vseed, g),
                   picture_grid2(centre, output_step, vseed, g), t);
    }
    if (output_step < 2.25)
        return picture_grid2(centre, output_step, vseed, g);
    if (output_step < 2.5) {
        float t = smoothstep(2.25, 2.5, output_step);
        return mix(picture_grid2(centre, output_step, vseed, g),
                   picture_grid3(centre, output_step, vseed, g), t);
    }
    if (output_step < 3.25)
        return picture_grid3(centre, output_step, vseed, g);
    if (output_step < 3.5) {
        float t = smoothstep(3.25, 3.5, output_step);
        return mix(picture_grid3(centre, output_step, vseed, g),
                   picture_grid4(centre, output_step, vseed, g), t);
    }
    return picture_grid4(centre, output_step, vseed, g);
}

void hook() {
    vec4 color = HOOKED_tex(HOOKED_pos);

    // State-validity guard: on RGB / no-LUMA sources PASS 1 never runs, so
    // GRAIN_STATE holds whatever the API left there (zero-init is common but
    // not guaranteed; NaN m_eff_render would pass a <=0 gate as false). An
    // unvalidated state must passthrough unconditionally — including debug,
    // whose rows would render garbage.
    if (!(abs(m_state_magic - MP_STATE_MAGIC_OUT) < 0.0001
          && abs(m_state_epoch - state_epoch) < 0.5
          && m_field_valid > 0.5)) {
        imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
        return;
    }

    // Real zero path: avoid template reads and, in HDR mode, avoid a decode/encode
    // round trip. PASS 1 still observes, so later evidence opens without a toggle-
    // induced reset or stale EMA.
    if ((grain_gain <= 0.0 || match_grain <= 0.0 || m_eff_render <= 0.0)
        && debug_match <= 0.5) {
        imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
        return;
    }

    // Source-locked grain lives in canonical active-picture coordinates.
    // First recover the full source-raster placement (display-added padding),
    // then nest the committed baked-picture rectangle inside it. Neither kind
    // of matte may resize or rephase the field.
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gsize = imageSize(GRAIN_FIELD);
    vec2 canvas = HOOKED_size;
    float source_aspect = (m_source_aspect > 0.01)
                        ? m_source_aspect
                        : canvas.x / max(canvas.y, 1.0);
    float canvas_aspect = canvas.x / max(canvas.y, 1.0);
    vec2 raster_size = canvas;
    vec2 raster_origin = vec2(0.0);
    if (canvas_aspect > source_aspect + 0.0001) {
        raster_size.x = canvas.y * source_aspect;
        raster_origin.x = 0.5 * (canvas.x - raster_size.x);
    } else if (canvas_aspect < source_aspect - 0.0001) {
        raster_size.y = canvas.x / source_aspect;
        raster_origin.y = 0.5 * (canvas.y - raster_size.y);
    }
    vec2 baked_inset = clamp(vec2(m_active_inset_x, m_active_inset_y),
                             vec2(0.0), vec2(0.24));
    vec2 active_origin = raster_origin + baked_inset * raster_size;
    vec2 active_size = (vec2(1.0) - 2.0 * baked_inset) * raster_size;
    vec2 pixel_centre = vec2(gid) + 0.5;
    // The detector commits the last confirmed matte sample. For masking,
    // advance half a refinement substep and round inward to whole pixels so
    // lifted mattes and bar-resident subtitles cannot receive grain.
    vec2 mask_guard = vec2(baked_inset.x > 0.0 ? 1.0 / 1024.0 : 0.0,
                           baked_inset.y > 0.0 ? 1.0 / 1024.0 : 0.0);
    vec2 mask_lo = ceil(raster_origin
                      + (baked_inset + mask_guard) * raster_size);
    vec2 mask_hi = floor(raster_origin + raster_size
                       - (baked_inset + mask_guard) * raster_size);
    if (debug_match <= 0.5
        && (float(gid.x) < mask_lo.x || float(gid.y) < mask_lo.y
         || float(gid.x) >= mask_hi.x || float(gid.y) >= mask_hi.y)) {
        imageStore(out_image, gid, color);
        return;
    }
    // gscale maps normalized picture coordinates onto the finite synthesis
    // lattice, not the physical template extent;
    // gsize only wraps each shuffled template-window fetch.
    float gscale = PICTURE_DENSITY_OUT / max(active_size.y, 1.0);
    vec2 fpos = (pixel_centre - active_origin) * gscale;
    // KNOWN LIMIT (audit 2026-07-10): the overlap blend preserves variance
    // exactly (weights' squares sum to 1) but not DISTRIBUTION SHAPE — a
    // weighted sum of independent warped samples is more Gaussian than its
    // inputs. At value_warp >= ~2 (deliberately bimodal grain) the bands
    // carry measurably softer per-grain contrast (kurtosis 1.19 -> 1.65
    // edge / 2.01 corner at warp 3), on ~23% of pixels. Invisible in
    // playback (the jitter moves the bands every tick) and borderline even
    // in adversarial freeze-frames at gain 12; accepted freeze-frame-only
    // limitation rather than restructuring warp to post-blend.
    // TPL BLOCK SHUFFLE (small-template architecture, AV1-FGS style).
    // Each normalized TPL_BLOCK-sized tile of the picture field presents a per-tile
    // randomized (integer offset + H/V flip) window of the physical
    // template, re-hashed every visible tick (see tpl_sample above).
    // Per-grain statistics are the template's texels (same DoG/warp
    // pipeline at the implementation-lattice scale); only the long-range arrangement
    // reuses template windows. The per-tick rehash supersedes the
    // full-size build's whole-field recycle transform. Two seam defenses,
    // both empirically required (dev/grain-genrate/README):
    //  - per-tick LATTICE PHASE JITTER (whole-texel origin shift hashed
    //    from vseed) so the boundary discontinuity never sits on the same
    //    pixels twice — kills temporal accumulation of the 64px grid;
    //  - AV1-style OVERLAP BLEND: within TPL_OV texels past a boundary,
    //    crossfade from the neighbour block's window with cos/sin weights
    //    (sum of squares = 1 -> grain RMS exactly preserved; adjacent
    //    windows are independent template regions) — kills the
    //    freeze-frame lattice (in-block DoG correlation breaks at raw
    //    boundaries: measured 1.65x gradient energy without this).
    // Continuity: at u == 0 the sample IS the neighbour's (weight 1) and
    // ramps out by u == TPL_OV. Outside the bands it is a single exact
    // texel fetch at gscale == 1.0, as before.
    float vseed = m_arr_seed;
    uint j0 = pcg_hash(uint(vseed) * 2246822519u + 3u);
    vec2 fj = fpos + vec2(float(j0 % 64u), float((j0 >> 8) % 64u));
    vec3 vsum = picture_pixel(fj, gscale, vseed, gsize);

    // grain_hdr bridge: work in the measured SDR domain (see helpers above).
    // Clamp the PQ input codes — YUV->RGB overshoot above 1.0 explodes the
    // PQ EOTF. SDR-equivalent codes may exceed 1.0 wherever an upstream stage
    // expanded highlights above reference white; the tone bell extrapolates
    // toward zero grain shortly above 1.0.
    bool hdr_bridge = grain_hdr > 0.5;
    vec3 work_rgb = color.rgb;
    if (hdr_bridge) {
        vec3 nits = vec3(pq_eotf_nits(clamp(color.r, 0.0, 1.0)),
                         pq_eotf_nits(clamp(color.g, 0.0, 1.0)),
                         pq_eotf_nits(clamp(color.b, 0.0, 1.0)));
        vec3 linear_709 = bt2020_to_bt709(nits / grain_ref_white);
        // An upstream SDR→HDR expansion usually stays inside the source 709
        // gamut, but wide-gamut expansion can produce valid 2020 colors
        // outside it. A signed gamma
        // extension preserves those negative intermediate 709 components so
        // the inverse/forward matrix pair remains a round trip.
        work_rgb = signed_pow(linear_709, 1.0 / 2.4);
    }

    float color_luma = dot(work_rgb, luma_coeff);
    float hdr_mode = hdr_bridge ? 1.0 : 0.0;
    float tone_scale = matched_grain_scale(color_luma, hdr_mode);
    vec3 scale_vec = vec3(tone_scale);
    vec3 pre_grain = work_rgb;
    vec3 grain_delta;
    if (density_combine > 0.5) {
        // Density multiplication cannot express an absolute noise floor below
        // the 0.015 divisor. Extend only picture-bearing near-black values with
        // the same zero-mean RGB perturbation; literal black/mattes stay exact.
        float ped_lo = mix(0.0010, 0.00025, hdr_mode);
        float ped_hi = mix(0.0060, 0.00300, hdr_mode);
        float pedestal_gate = smoothstep(ped_lo, ped_hi,
                                         max(color_luma, 0.0))
                            * (1.0 - smoothstep(0.015, 0.030,
                                                max(color_luma, 0.0)));
        float pedestal_signal = max(DENSITY_SHADOW_FLOOR
                                  - max(color_luma, 0.0), 0.0)
                              * pedestal_gate;
        vec3 x = DENSITY_GAIN * vsum * scale_vec;
        // Canonical-field log-normal bias correction. Footprint filtering
        // changes covariance, so non-native-density presentations retain a very small
        // conservative bias rather than pretending sum(weights^2) is exact
        // for the correlated DoG field.
        vec3 density_delta = exp(x - 0.5 * DENSITY_GAIN * DENSITY_GAIN
                               * tone_scale * tone_scale * field_var) - 1.0;
        grain_delta = work_rgb * density_delta;
        grain_delta += vec3(pedestal_signal) * density_delta;
        // One post-density scalar preserves channel log-normal shapes, their
        // zero-mean correction, hue speckle, and spatial/RMS ratios exactly.
        // At a neutral carrier the cap is identically one.
        grain_delta *= density_colour_energy_cap(
            work_rgb + vec3(pedestal_signal));
    } else {
        grain_delta = vsum * scale_vec;
    }

    if (!hdr_bridge || grain_headroom < 0.5) {
        // Luma can remain below 1.0 when only one or two RGB channels are
        // clipped. (With grain_headroom = 0 the same source clipping survives
        // inside a PQ container at work-domain 1.0, where the encode-side
        // ref-white clip below reproduces the display's clamp — so the same
        // defense applies.) Letting the display clamp those channels after grain would
        // remove positive excursions while retaining dark pits — temporally
        // conspicuous in tinted whites even though literal RGB white already
        // has tone_scale == 0. Limit every channel symmetrically to its code
        // headroom. Neutral highlights progressively share the tightest upper
        // headroom, so a tinted white with one clipped channel cannot retain
        // chromatic flicker; saturated colors keep their unclipped channels.
        // Under the bridge (headroom = 0) a negative signed-gamma component
        // (out-of-709 color) gets zero headroom = zero grain on that channel;
        // intentional — such values are fp/gamut noise for ceiling-limited content.
        vec3 channel_headroom = max(min(pre_grain, vec3(1.0) - pre_grain),
                                    vec3(0.0));
        float rgb_peak = max(max(pre_grain.r, pre_grain.g), pre_grain.b);
        float rgb_floor = min(min(pre_grain.r, pre_grain.g), pre_grain.b);
        float neutral_highlight = smoothstep(0.80, 0.95, rgb_floor);
        float shared_upper = max(1.0 - rgb_peak, 0.0);
        vec3 code_headroom = mix(channel_headroom, vec3(shared_upper),
                                 neutral_highlight);
        grain_delta = clamp(grain_delta, -code_headroom, code_headroom);
    } else {
        // CelFlare/PQ path: the same defense one octave up. The flash-guard
        // ceiling below is one-sided -- inside the open fade band a positive
        // excursion used to clip at the pixel's own level while the fade
        // still passed amplitude, leaving rectified negative-only pits on
        // expanded highlights (charter-audit P3), per-channel on tinted ones.
        // Bound the delta symmetrically against the sanctioned grain domain
        // instead: where grain cannot go up it may not go down. Neutral
        // highlights share the tightest upper room exactly like the SDR
        // limiter, scaled to the clamp top. No floor arm here -- the PQ
        // pedestal owns near-black and work-domain black is not a clip.
        // The clamp top floors at ref white: overshoot physics only exists
        // at/above 1.0. A grain_fade below ref white is a purely cosmetic
        // target owned by the LUMA fade -- enforcing it per channel would
        // delete grain from bright saturated channels (R 0.95 at luma 0.43)
        // while their neighbours keep full grain: colored, directional
        // suppression on ordinary unclipped content (re-base audit P1).
        float clamp_top = max(grain_fade, 1.0);
        vec3 up_room = max(vec3(clamp_top) - pre_grain, vec3(0.0));
        float rgb_peak = max(max(pre_grain.r, pre_grain.g), pre_grain.b);
        float rgb_floor = min(min(pre_grain.r, pre_grain.g), pre_grain.b);
        float neutral_highlight = smoothstep(0.80 * clamp_top,
                                             0.95 * clamp_top, rgb_floor);
        float shared_upper = max(clamp_top - rgb_peak, 0.0);
        vec3 code_room = mix(up_room, vec3(shared_upper), neutral_highlight);
        grain_delta = clamp(grain_delta, -code_room, code_room);
    }
    work_rgb += grain_delta;

    if (hdr_bridge) {
        // Grain is texture, not signal: it may never push a pixel above
        // max(its own pre-grain level, the sanctioned grain ceiling) -- ref
        // white for clip-limited content, the user's fade top under an
        // upstream expansion. The symmetric room clamp above already bounds
        // the delta inside that domain, so this stays a flash guard against
        // large density excursions re-encoding toward display peak
        // (design-audit finding, 2026-07-04), never a tone shaper.
        float ceil_top = (grain_headroom < 0.5) ? 1.0 : max(grain_fade, 1.0);
        work_rgb = min(work_rgb, max(pre_grain, vec3(ceil_top)));
        vec3 linear_709 = signed_pow(work_rgb, 2.4);
        vec3 out_nits = grain_ref_white * max(bt709_to_bt2020(linear_709), vec3(0.0));
        color.rgb = vec3(pq_oetf_code(out_nits.r), pq_oetf_code(out_nits.g),
                         pq_oetf_code(out_nits.b));
    } else {
        color.rgb = work_rgb;
    }

    if (debug_match > 0.5) {
        const int X_OFF = 24, Y_OFF = 400, BW = 10, BH = 10;
        const int ANCHOR = 10, NBITS = 16, NROWS = 52;
        ivec2 gid = ivec2(gl_GlobalInvocationID.xy) - ivec2(X_OFF, Y_OFF);
        if (gid.x >= 0 && gid.y >= 0
            && gid.x < ANCHOR + NBITS * BW && gid.y < NROWS * BH) {
            int row = gid.y / BH;
            float v = 0.0;
            if      (row == 0)  v = m_observed * 2000000.0;
            else if (row == 1)  v = sqrt(max(m_title_power, 0.0)) * 2000000.0;
            else if (row == 2)  v = m_shot_gain * 20000.0;
            else if (row == 3)  v = m_temporal_support * 2000000.0;
            else if (row == 4)  v = m_est_missing * 2000000.0;
            else if (row == 5)  v = m_loss_conf * 65000.0;
            else if (row == 6)  v = m_title_conf * 65000.0;
            else if (row == 7)  v = m_shot_conf * 65000.0;
            else if (row == 8)  v = m_loss_mix * 65000.0;
            else if (row == 9)  v = m_tone_conf * 65000.0;
            else if (row == 10) v = m_eff_render * 30000.0;
            else if (row == 11) v = m_structure_ratio * 1000000.0;
            else if (row == 12) v = m_coverage * 65000.0;
            else if (row == 13) v = m_motion * 30000.0;
            else if (row == 14) v = m_shot_size * 60000.0;
            else if (row == 15) v = m_shot_hardness * 60000.0;
            else if (row == 16) v = m_cut_score * 30000.0;
            else if (row == 17) v = m_shot_age;
            else if (row == 18) v = m_measured;
            else if (row == 19) v = m_gen_frame;
            else if (row < 28)  v = sqrt(max(m_master_p[row - 20], 0.0))
                                      * 2000000.0;
            else if (row < 36)  v = sqrt(max(m_restore_p[row - 28], 0.0))
                                      * 2000000.0;
            else if (row < 44)  v = sqrt(max(m_shot_obs_p[row - 36]
                                      * m_shot_obs_w[row - 36], 0.0))
                                      * 2000000.0;
            else if (row == 44) v = m_active_inset_x * 200000.0;
            else if (row == 45) v = m_active_inset_y * 200000.0;
            else if (row == 46) v = m_pending_inset_x * 200000.0;
            else if (row == 47) v = m_pending_inset_y * 200000.0;
            else if (row == 48) v = max(m_geom_streak, m_geom_streak_y);
            else if (row == 49) v = min(m_geom_known, m_geom_known_y) * 65000.0;
            else if (row == 50) v = m_geom_changed * 65000.0;
            else                v = m_pan_px * 2000.0;
            uint val = uint(clamp(v, 0.0, 65535.0));
            if (gid.x < ANCHOR)
                color.rgb = vec3(1.0);
            else {
                int b = (gid.x - ANCHOR) / BW;
                uint bit = (val >> uint(NBITS - 1 - b)) & 1u;
                color.rgb = (bit == 1u) ? vec3(1.0) : vec3(0.0);
            }
        }
    }

    imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
}
