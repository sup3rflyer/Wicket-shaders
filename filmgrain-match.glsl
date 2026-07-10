// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE
//
// Film Grain — MATCH (SDR) — adaptive grain restoration.
// ============================================================================
// Measures the SOURCE's surviving grain and auto-tunes the smooth-grain model
// to restore it, instead of applying a fixed tier. The grain shaders exist to
// reintroduce camera-original grain that compression/intermediates smoothed
// away (restoration, not decoration), so the match is COMPENSATORY: we measure
// what survived, then extrapolate UPWARD past it (RESTORE_GAIN) toward what the
// original likely was, and never drop below a baseline FLOOR (= the Light tier).
//
// What we measure and what it drives — SPATIAL flat-patch grain floor:
//   local 5x5 grain-std-vs-luma  ->  MID (peak location) + STEEPNESS (curve width)
//   robust plausible-band std    ->  INTENSITY (maps to restored output grain std)
// Temporal frame differences provide two safeguards: held/cut/motion rejection
// for the spatial model, and a conservative supplemental grain estimate when
// source grain survives mostly as frame-to-frame shimmer rather than flat-patch
// texture.
//
// Architecture (LUMA measure, OUTPUT render, one persistent buffer):
//   PASS 1 - Compute 32x32 at LUMA, measurement only: saves to a 1x1 dummy
//            (WIDTH/HEIGHT 1 -> a single workgroup; the LUMA plane itself is
//            untouched -- the old full-plane passthrough copy was the shader's
//            entire measurable GPU cost at 4K). Samples HOOKED.r across the
//            full frame and writes only GRAIN_STATE.
//   PASS 2 - Compute 32x32 at OUTPUT: renders grain onto HOOKED using the
//            LUMA-measured GRAIN_STATE.
//
// Runtime params. Only match_grain / debug_match / value_warp are on keys (F3 / Ctrl+F3 /
// Alt+F3 via shader-toggle.lua); the rest are tuned by editing the DEFAULT value under each
// param block below and reloading (the lua no longer overrides them). The param blocks are
// ordered to match these groups. (Comments cannot sit between the param blocks -- the parser
// rejects it -- so every param is documented HERE.)
//
//  == CONTROL ===============================================================
//   match_grain      0 = pure Light floor (bit-identical to the tier), 1 = matched; mix()
//                    between, so the on/off A/B is non-destructive.            [F3]
//   debug_match      machine-readable state-readout overlay.                   [Ctrl+F3]
//   state_epoch      harness reset token; bump once per source file.
//
//  == GRAIN LOOK (the dials to tune by eye) =================================
//   grain_gain       overall grain AMOUNT/strength (1 = calibrated; up to 12).
//   grain_size       grain SIZE: <1 finer, >1 coarser (1 = calibrated).
//   grain_contrast   spectral hardness: 0 = soft/lowpass, 1 = sandpaper bandpass, up to 2 =
//                    more DC removed / peppery (difference-of-Gaussians).
//   value_warp       VALUE-domain contrast: 0 = Gaussian (bit-identical), ~2 hard, ~3 extreme
//                    = bimodal/high-per-grain-contrast (CyberCity "harsh"). Amplitude-
//                    preserving; the value-domain cousin of grain_contrast.    [Alt+F3]
//   grain_sharpness  GLOBAL crispness dial: 1 = crisp 4K-film-scan (matched to plates),
//                    0 = BIT-IDENTICAL to the old soft Light look.
//   chroma_amp       amplitude of the coarse independent chroma layer (pure hue
//                    fluctuation, luma-removed; 0 = base grain only).
//   grain_rate       temporal cadence: fraction of SOURCE frames that re-seed the
//                    grain (1 = fresh grain every source frame / "on ones", the
//                    default; 0.5 = every 2nd source frame / "on twos"). Seeded
//                    from m_gen_frame so it's display-refresh independent.
//   grain_gen_rate   field REGENERATION cadence as a fraction of grain_rate ticks.
//                    0.5 (default) / 0.25 = regenerate every 2nd / 4th tick and
//                    present a randomized toroidal shift+flip of the standing
//                    field on the others — statistically fresh grain at
//                    ~half/quarter generation cost; 1 = regenerate every tick
//                    (exact pre-feature behavior). 0.25 is the practical
//                    minimum: steady-state character steps stay masked by the
//                    reseed; below it the savings asymptote while slow
//                    grain-regime crossfades start stepping. Amplitude/tone
//                    keying stays per-frame at any rate, warming/fast-EMA
//                    frames always regenerate at full rate, and a confident
//                    hard cut regenerates in place, so the measured grain
//                    character never goes stale across a cut.
//   grain_base_sat   per-channel independence of the BASE grain (the baked-in hue
//                    speckle). 0.25 = calibrated look; 0 = true mono. NB the RMS
//                    bookkeeping (value_warp / chroma_amp scaling) assumes 0.25, so
//                    off-default drifts per-channel grain RMS by a few percent.
//
//  == RESTORATION (how much grain to rebuild on degraded sources) ===========
//   restore_gain     upward extrapolation past surviving grain (compensatory). 1 = match only
//                    what survived; higher = rebuild more of the grain compression ate.
//   restore_floor    how readily low-confidence / uncertain areas still get grained.
//   grain_restore_taper  1 = taper restore_gain->1 as measured amplitude rises (the Lain
//                    over-apply fix). 0 = off (default).
//
//  == PIPELINE / SIZING =====================================================
//   density_combine  0 = additive, 1 = multiplicative density (grain rides brightness; default).
//   grid_snap        1 = snap the SIZE-measurement grid to whole pixels so the gated fineR
//                    matches the clean-field fineR the size law is tuned on (removes the
//                    half-pixel prefilter) + use the autotuned law. 0 = old half-pixel grid +
//                    old hand-set law (the exact pre-snap look). Default 1.
//
//  == TEMPORAL DYNAMICS / FIRING (behaviour, not look) ======================
//   grain_attack     EMA attack rate (how fast the grain level rises to a new scene).
//   grain_decay      EMA decay rate (how fast it falls; faster than attack = cut-safe).
//   hardcut_frac     fraction-of-frame-changed that counts as a HARD cut (re-converge).
//   pan_freeze       1 = freeze the grain-amplitude RISE during a real global pan (kills the
//                    canvas-pan overfire; established grain HOLDS). 0 = prior. A/B-safe.
// ============================================================================

// shampv shader API (plain comments to libplacebo). All params are DYNAMIC:
// glsl-shader-opts changes apply next frame, no recompile; bump state_epoch
// to invalidate the persisted GRAIN_STATE live.
//@shampv input sdr
//@shampv ref-white-param grain_ref_white
//@shampv toggle grain_hdr debug_match density_combine grid_snap
//@shampv measures LUMA

//!PARAM match_grain
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM debug_match
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM state_epoch
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 65535.0
0.0

//!PARAM grain_gain
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 12.0
2.0

//!PARAM grain_size
//!TYPE DYNAMIC float
//!MINIMUM 0.3
//!MAXIMUM 2.5
1.0

//!PARAM grain_contrast
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM value_warp
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 4.0
0.0

//!PARAM chroma_amp
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM grain_sharpness
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM grain_rate
//!TYPE DYNAMIC float
//!MINIMUM 0.1
//!MAXIMUM 1.0
1.0

//!PARAM grain_gen_rate
//!DESC Field regeneration cadence as a fraction of grain_rate ticks: 1 = regenerate every tick (exact pre-feature behavior), 0.5 (default) / 0.25 = regenerate every 2nd / 4th tick and present a randomized toroidal shift+flip of the standing field on the others (statistically fresh, ~half/quarter generation cost). 0.25 is the practical minimum (character steps stay masked); a confident hard cut always regenerates in place.
//!TYPE DYNAMIC float
//!MINIMUM 0.1
//!MAXIMUM 1.0
0.5

//!PARAM grain_base_sat
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.25

//!PARAM restore_gain
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 4.0
4.0

//!PARAM restore_floor
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.40

//!PARAM grain_restore_taper
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM density_combine
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM grid_snap
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM grain_attack
//!TYPE DYNAMIC float
//!MINIMUM 0.002
//!MAXIMUM 0.30
0.02

//!PARAM grain_decay
//!TYPE DYNAMIC float
//!MINIMUM 0.002
//!MAXIMUM 0.30
0.04

//!PARAM hardcut_frac
//!TYPE DYNAMIC float
//!MINIMUM 0.40
//!MAXIMUM 1.0
0.85

//!PARAM pan_freeze
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM grain_hdr
//!DESC HDR chain mode — set 1 when the output is PQ BT.2020 (the CelFlare sdr-to-hdr chain). Keys and applies grain in the measured SDR domain via a per-pixel PQ bridge; grain fades to zero shortly above reference white. 0 = exact prior SDR behavior.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM grain_ref_white
//!DESC SDR reference white in nits for the HDR bridge — must match hdr-reference-white (and CelFlare's cf_ref_white).
//!TYPE DYNAMIC float
//!MINIMUM 80.0
//!MAXIMUM 480.0
100.0

//!BUFFER GRAIN_STATE
//!VAR float m_intensity_raw
//!VAR float m_mid
//!VAR float m_steepness
//!VAR float m_log_avg
//!VAR float m_measured
//!VAR float m_prev_ready
//!VAR float m_state_magic
//!VAR float m_state_epoch
//!VAR float m_debug_tick
//!VAR float m_sat_frac
//!VAR float m_held_frames
//!VAR float m_distinct_frames
//!VAR float m_hold_run
//!VAR float m_content_held
//!VAR float m_content_distinct
//!VAR float m_content_hold_run
//!VAR float m_content_log_avg
//!VAR float m_coherence
//!VAR float m_coh_frac
//!VAR float m_coh_samples
//!VAR float m_measure_swatch_r
//!VAR float m_measure_swatch_g
//!VAR float m_measure_swatch_b
//!VAR float m_measure_stage
//!VAR float hist_amp[16]
//!VAR float hist_temporal[16]
//!VAR float hist_spatial[16]
//!VAR float hist_held[16]
//!VAR float m_held_count
//!VAR float m_film_level
//!VAR float m_grain_fine
//!VAR float m_source_height
//!VAR float m_pan_px
//!VAR float m_gen_frame
//!VAR float m_eff_render
//!VAR float m_eff_mid
//!VAR float m_eff_steep
//!VAR float m_conf
//!VAR float m_shape_w
//!VAR float m_regen
//!VAR float m_field_seed
//!VAR float prev_grid[9216]
//!VAR float prev_grid_off[9216]
//!STORAGE

//!TEXTURE GRAIN_FIELD
//!SIZE 3840 2160
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

#define MID_FLOOR    0.15
#define STEEP_FLOOR  2.6





#define TUKEY_SCALE 0.459

#define NBINS     16
#define GRID_W    96
#define GRID_H    96
#define GRID_N    (GRID_W * GRID_H)
#define AMP_BINS  64
#define AMP_MAX   0.05

#define NOISE_STD_PER_INTENSITY 0.030
#define BLACK_FLOOR   0.02
#define WHITE_CEIL    0.985
#define MIN_BIN_CELLS 48.0
#define GRAIN_LO_BIN  2
#define GRAIN_HI_BIN  14
#define PLAUSIBLE_BRIGHT_LO 0.88
#define PLAUSIBLE_BRIGHT_HI 0.985
#define HELD_EPS      0.0001
#define CUT_SAT_FRAC  0.60
#define SAT_BIN       (AMP_BINS * 3 / 4)

#define P25_TO_GRAINSTD (1.0 / 0.3186 / 1.41421356)
#define P50_TO_GRAINSTD (1.0 / 0.67448975 / 1.41421356)
#define SPATIAL_EDGE_GATE 0.015
#define SPATIAL_VAR_MAX   0.02
#define SPATIAL_PCTL      0.25
#define TEMPORAL_EDGE_GATE 0.006
#define TEMPORAL_PCTL      0.50
#define TEMPORAL_BIN_FLOOR 0.003
#define TEMPORAL_MIN_BINS  5
#define TEMPORAL_GAIN      1.25
#define TEMPORAL_RUN_MIN   4
#define TEMPORAL_RUN_FULL  7
// Spatial-incoherence motion gate. Grain is spatially incoherent; coherent motion
// (smoke, pans) and global luma drift are not. We measure the temporal change of
// the local spatial micro-gradient between a pixel and one COH_OFFSET_PX away:
//   d_pair = [cur(p) - cur(p+d)] - [prev(p) - prev(p+d)]
// Global DC drift cancels (p and p+d shift equally); only independent grain
// survives. incoherence = sum(d_pair^2) / sum(d_center^2 + d_offset^2):
// grain ~1.0 (full decorrelation), motion/drift ~0 (micro-gradient preserved).
// COH_OFFSET_PX must exceed the grain correlation length (so even coarse/blocky
// grain reads incoherent) yet stay under the motion coherence length.
#define COH_OFFSET_PX 12.0
#define COH_CLAMP     0.07
#define COH_SQ_SCALE  1.0e7
#define COH_MIN_NORM  20000.0
#define COH_LO        0.45
#define COH_HI        0.75
#define COH_ALPHA     0.25
#define TEMPORAL_SPATIAL_CEIL 0.004
#define TEMPORAL_NEEDS_SPATIAL_LO 0.0018
#define TEMPORAL_NEEDS_SPATIAL_HI 0.0028
#define SPATIAL_STATIC_GATE 0.004
#define SPATIAL_STATIC_RATIO_LO 0.45
#define SPATIAL_STATIC_RATIO_HI 0.80
#define CONTENT_HELD_EPS   0.0025
// STEP 2 held-cel firing gate (texture-safe). held_run = contiguous luma bins of
// the held-cel temporal-grain curve above the dither floor. Overlay grain fires
// via held_run directly; film grain fires via spatial amplitude but only when
// held_run corroborates (so busy-paper spatial can't ride the bypass).
// FILM-grain firing path (the high-value target: genuine camera grain compression
// smoothed unevenly -- Goku Midnight Eye / Cyber City Oedo 808). Driven by the
// flat-temporal LEVEL (band-median of the p25 transition-temporal curve), which
// measures whether grain ANIMATES on flats rather than whether texture EXISTS, so
// it rides past the spatial film-vs-busy-texture overlap that blocked every prior
// attempt. The per-scene level overlaps (clean motion scenes -- water/foliage/
// dissolves -- spike to 8-10e-3, same as film), so it is NOT used instantaneously:
// m_film_level is a cross-timeline EMA with asymmetric-SAFE rates (slow attack,
// faster decay). The slow attack underweights the isolated clean spikes (they are
// the minority of scenes) while persistent film grain accumulates. Offline EMA
// equilibrium (mgt_gate_table.py, pooled taxonomy): clean/texture settle
// 1.9-2.8e-3 (floor, ZERO false positives), film settles 4.4e-3 (fires ~0.55).
// LO sits 1.0e-3 above the highest clean equilibrium -> the false-positive margin.
#define FILM_ATTACK   0.015
#define FILM_DECAY    0.05
#define FILM_LEVEL_CEIL 0.008

// PAN-FREEZE gate (canvas-pan overfire fix). A baked artistic texture (e.g. the
// distressed-paint wall in Days with My Stepsister) translating under a pan is a
// per-frame grain TWIN -- amplitude, incoherence, held-run, size and even motion-
// compensated residual all overlap real grain (offline-confirmed). Its only tell is
// the GLOBAL pan itself. A grid-level single-step Lucas-Kanade translation estimate
// is grain-INSENSITIVE (grain decorrelates across grid neighbours so its gx*d/gy*d
// products average to ~0; a coherent pan does not) -> reads ~2-4px on the canvas pan
// vs ~0 on held grain incl. NieR's heavy overlay (offline grid-LK: canvas 1.5-2.5px,
// grain <=0.43px). We do NOT veto firing (can't separate per-frame); instead, while a
// real pan is present, FREEZE the RISE of m_intensity_raw -> the translating texture
// can't BUILD up, while grain already established simply HOLDS through the pan (and
// clean sources are floored regardless). LK reuses the cur_smooth neighbour samples
// (no extra fetches). pan_freeze=0 -> exact prior behaviour; gs=0 stays bit-identical.
// Calibrated to in-shader m_pan_px (smoke): static grain accumulation reads <=0.05
// (Goku, the prize film target, 0.01-0.02), while the canvas pan reads 0.10-0.26.
// LO/HI sit in that gap so static grain BUILDS unfrozen and the pan freezes; grain
// sources that read high are themselves panning, where freeze only HOLDS the level.
// (In-shader magnitudes are ~6x below the offline area-grid LK -- aliased texture
// dilutes Gxx on the point grid -- but the static-vs-pan separation is what matters.)
#define PAN_FREEZE_LO 0.07
#define PAN_FREEZE_HI 0.16
#define LK_SCALE      2.0e5

#define ALPHA_SLOW    0.03
#define ALPHA_FAST    0.20
#define ALPHA_MISSING 0.35
#define CUT_REREADY   8.0
#define STATE_MAGIC   0.5577819

#define STEEP_MIN     0.6
#define STEEP_MAX     9.0

const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);

shared uint s_hist[NBINS][AMP_BINS];
shared uint s_content_hist[AMP_BINS];
shared uint s_update_prev_grid;
shared uint s_coh_incoh;
shared uint s_coh_norm;
shared uint s_size_fine;
shared uint s_size_coarse;
shared uint s_lk_gxx;
shared uint s_lk_gyy;
shared uint s_lk_gxy_p;
shared uint s_lk_gxy_n;
shared uint s_lk_bx_p;
shared uint s_lk_bx_n;
shared uint s_lk_by_p;
shared uint s_lk_by_n;


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

float grain_scale(float lum, float mid, float steepness) {
    float d2 = steepness * (lum - mid) * (lum - mid);
    float t = 1.0 - d2 * TUKEY_SCALE;
    float curve = t > 0.0 ? t * t : 0.0;
    float protection = smoothstep(0.0, 0.12, lum);
    return curve * protection;
}

float gaussian_weight(float dx, float sigma) {
    return exp(-0.5 * dx * dx / max(sigma * sigma, 1e-6));
}

float measure_luma(vec2 uv) {
    return HOOKED_tex(uv).r;
}

void hook() {
    uint lid = gl_LocalInvocationIndex;
    uint num_threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    // The 1x1 SAVE target means exactly one workgroup dispatches: this pass is
    // measurement-only and never rewrites the LUMA plane.
    const bool is_measure_wg = true;

    bool state_ok = abs(m_state_magic - STATE_MAGIC) < 0.0001
                 && abs(m_state_epoch - state_epoch) < 0.5;
    bool prev_ready = state_ok && m_prev_ready > 0.5;

    // =========================================================================
    // MEASUREMENT — the single workgroup samples HOOKED_tex across full frame
    // =========================================================================

    if (is_measure_wg) {
        if (lid == 0u) {
            s_update_prev_grid = 0u;
            s_coh_incoh = 0u;
            s_coh_norm = 0u;
            s_size_fine = 0u;
            s_size_coarse = 0u;
            s_lk_gxx = 0u; s_lk_gyy = 0u;
            s_lk_gxy_p = 0u; s_lk_gxy_n = 0u;
            s_lk_bx_p = 0u; s_lk_bx_n = 0u;
            s_lk_by_p = 0u; s_lk_by_n = 0u;
        }
        for (uint k = lid; k < uint(NBINS * AMP_BINS); k += num_threads)
            s_hist[k / uint(AMP_BINS)][k % uint(AMP_BINS)] = 0u;
        for (uint k = lid; k < uint(AMP_BINS); k += num_threads)
            s_content_hist[k] = 0u;
    }
    barrier();

    if (is_measure_wg) {
        for (uint k = lid; k < uint(GRID_N); k += num_threads) {
            vec2 uv = (vec2(float(k % uint(GRID_W)), float(k / uint(GRID_W))) + 0.5)
                     / vec2(float(GRID_W), float(GRID_H));
            float cur_luma = measure_luma(uv);
            float prev_luma = prev_grid[k];

            if (prev_ready && cur_luma > BLACK_FLOOR && cur_luma < WHITE_CEIL) {
                float xm = measure_luma(uv - vec2(HOOKED_pt.x, 0.0));
                float xp = measure_luma(uv + vec2(HOOKED_pt.x, 0.0));
                float ym = measure_luma(uv - vec2(0.0, HOOKED_pt.y));
                float yp = measure_luma(uv + vec2(0.0, HOOKED_pt.y));
                float tslope = 0.5 * (abs(xp - xm) + abs(yp - ym));

                // Grid global-pan (single-step Lucas-Kanade) accumulation over ALL
                // valid cells -- see the PAN_FREEZE note. The pan signal lives in the
                // coarse STRUCTURE (the scene geometry that genuinely translates), so
                // this runs BEFORE the flat gate and uses GRID-NEIGHBOUR gradients
                // (+/- one grid cell ~= 11-20px): at that spacing the fine baked texture
                // aliases away (1px gradients are swamped by it and it DECORRELATES
                // under the pan, so it can't be fit by a translation), leaving the
                // coarse structure. d = -s*grad with s in GRID-CELL units. Grain is
                // temporally INCOHERENT so its grad*d products average to ~0 over the
                // grid; a coherent pan yields a consistent correlation. The grid
                // neighbours are reused by cur_smooth below. Signed sums split +/-.
                vec2 lk_step = vec2(1.0 / float(GRID_W), 1.0 / float(GRID_H));
                float gxp = measure_luma(uv + vec2(lk_step.x, 0.0));
                float gxm = measure_luma(uv - vec2(lk_step.x, 0.0));
                float gyp = measure_luma(uv + vec2(0.0, lk_step.y));
                float gym = measure_luma(uv - vec2(0.0, lk_step.y));
                float lk_d  = cur_luma - prev_luma;
                float lk_gx = 0.5 * (gxp - gxm);
                float lk_gy = 0.5 * (gyp - gym);
                float lk_gxy = lk_gx * lk_gy;
                float lk_bx = -lk_gx * lk_d;
                float lk_by = -lk_gy * lk_d;
                atomicAdd(s_lk_gxx, uint(min(lk_gx * lk_gx * LK_SCALE, 4.0e8)));
                atomicAdd(s_lk_gyy, uint(min(lk_gy * lk_gy * LK_SCALE, 4.0e8)));
                if (lk_gxy >= 0.0) atomicAdd(s_lk_gxy_p, uint(min(lk_gxy * LK_SCALE, 4.0e8)));
                else               atomicAdd(s_lk_gxy_n, uint(min(-lk_gxy * LK_SCALE, 4.0e8)));
                if (lk_bx >= 0.0)  atomicAdd(s_lk_bx_p, uint(min(lk_bx * LK_SCALE, 4.0e8)));
                else               atomicAdd(s_lk_bx_n, uint(min(-lk_bx * LK_SCALE, 4.0e8)));
                if (lk_by >= 0.0)  atomicAdd(s_lk_by_p, uint(min(lk_by * LK_SCALE, 4.0e8)));
                else               atomicAdd(s_lk_by_n, uint(min(-lk_by * LK_SCALE, 4.0e8)));

                if (tslope >= TEMPORAL_EDGE_GATE)
                    continue;
                float d_signed = cur_luma - prev_luma;
                float d = abs(d_signed);
                int lb = clamp(int(cur_luma * float(NBINS)), 0, NBINS - 1);
                int ab = clamp(int(d / AMP_MAX * float(AMP_BINS)), 0, AMP_BINS - 1);
                atomicAdd(s_hist[lb][ab], 1u);

                // Grain SHARPNESS/SIZE (sandpaper<->soft): compare grain high-freq
                // energy at the 1px vs 3px scale via Laplacians (below). Fine/sharp
                // grain (Goku sandpaper) keeps energy at 1px; coarse/soft grain (Nier)
                // spreads it to 3px. Reuses the flat (non-edge) gate above; the 1e6
                // scale keeps the uint atomics in range (per-sample clamp guards
                // heavy grain). The fine/coarse ratio is formed in the reduction.
                // SIZE measurement on a PIXEL-SNAPPED grid (grid_snap). The 96-grid
                // centres land at sub-pixel positions (half-pixel in x @4K), so the
                // bilinear measure_luma() softens the 1px lap MORE than the 3px lap and
                // the gated fineR reads ~2x too coarse at the fine end -- mismatching
                // the clean-field fineR the size law is tuned on (offline replica:
                // log-affine slope 0.73 unsnapped -> 0.996 snapped, R2 0.9999). Snapping
                // uv to the texel CENTRE makes measure_luma return the exact texel (no
                // prefilter); the +/-1px,+/-3px taps inherit exact texels. ONLY the size
                // path is snapped -- the shared tslope gate, the histogram and all
                // amplitude/temporal sampling keep the original uv (bit-identical, so the
                // validated look is preserved; grid_snap=0 restores it exactly). The
                // measure runs on a SINGLE workgroup, so the extra fetches are ~free.
                vec2 suv = (grid_snap > 0.5)
                         ? (floor(uv * HOOKED_size) + 0.5) * HOOKED_pt : uv;
                float cur_s = measure_luma(suv);
                float xms = measure_luma(suv - vec2(HOOKED_pt.x, 0.0));
                float xps = measure_luma(suv + vec2(HOOKED_pt.x, 0.0));
                float yms = measure_luma(suv - vec2(0.0, HOOKED_pt.y));
                float yps = measure_luma(suv + vec2(0.0, HOOKED_pt.y));
                float x3m = measure_luma(suv - vec2(3.0 * HOOKED_pt.x, 0.0));
                float x3p = measure_luma(suv + vec2(3.0 * HOOKED_pt.x, 0.0));
                float y3m = measure_luma(suv - vec2(0.0, 3.0 * HOOKED_pt.y));
                float y3p = measure_luma(suv + vec2(0.0, 3.0 * HOOKED_pt.y));
                // Grain-energy gate: skip grain-poor flat samples (compression
                // smooths grain in flats), so the ratio reflects grain, not residual.
                float e1 = (cur_s - xps) * (cur_s - xps) + (cur_s - xms) * (cur_s - xms)
                         + (cur_s - yps) * (cur_s - yps) + (cur_s - yms) * (cur_s - yms);
                if (e1 > 1.6e-5) {
                    // LAPLACIAN (2nd difference) at 1px vs 3px. The 2nd difference
                    // cancels the smooth linear gradient that biased a raw diff
                    // ratio coarse, isolating grain. ratio lap1^2/lap3^2 (formed in
                    // the reduction): HIGH = fine/sharp (sandpaper), LOW = coarse/
                    // soft. Validated offline (lap_g, Spearman +0.90 vs 16-bit GT).
                    float lap1 = 4.0 * cur_s - (xms + xps + yms + yps);
                    float lap3 = 4.0 * cur_s - (x3m + x3p + y3m + y3p);
                    atomicAdd(s_size_fine,   uint(min(lap1 * lap1 * 1.0e6, 2.0e5)));
                    atomicAdd(s_size_coarse, uint(min(lap3 * lap3 * 1.0e6, 2.0e5)));
                }

                // Spatial-incoherence: temporal change of the local micro-gradient
                // (p vs p+d). Global DC drift cancels; independent grain survives.
                float cur_off = measure_luma(uv + vec2(COH_OFFSET_PX * HOOKED_pt.x, 0.0));
                float d_off_signed = cur_off - prev_grid_off[k];
                float dc = clamp(d_signed, -COH_CLAMP, COH_CLAMP);
                float doff = clamp(d_off_signed, -COH_CLAMP, COH_CLAMP);
                float d_pair = dc - doff;
                atomicAdd(s_coh_incoh, uint(d_pair * d_pair * COH_SQ_SCALE));
                atomicAdd(s_coh_norm, uint((dc * dc + doff * doff) * COH_SQ_SCALE));

                uint gx = k % uint(GRID_W);
                uint gy = k / uint(GRID_W);
                uint km = (gx > 0u) ? k - 1u : k;
                uint kp = (gx + 1u < uint(GRID_W)) ? k + 1u : k;
                uint ku = (gy > 0u) ? k - uint(GRID_W) : k;
                uint kd = (gy + 1u < uint(GRID_H)) ? k + uint(GRID_W) : k;
                // reuse the grid-neighbour samples fetched above for the LK estimate
                float cur_smooth = (cur_luma + gxm + gxp + gym + gyp) * 0.2;
                float prev_smooth = (prev_grid[k] + prev_grid[km] + prev_grid[kp] + prev_grid[ku] + prev_grid[kd]) * 0.2;
                int cab = clamp(int(abs(cur_smooth - prev_smooth) / AMP_MAX * float(AMP_BINS)), 0, AMP_BINS - 1);
                atomicAdd(s_content_hist[cab], 1u);
            }
        }
    }
    barrier();

    if (is_measure_wg && lid == 0u) {
        vec3 input_center_rgb = HOOKED_tex(vec2(0.5, 0.5)).rgb;
        m_measure_swatch_r = input_center_rgb.r;
        m_measure_swatch_g = input_center_rgb.g;
        m_measure_swatch_b = input_center_rgb.b;
        m_measure_stage = 0.25;
        // Source LUMA height — the resolution the grain is MEASURED at. The render
        // pass scales the per-source grain by output/source so its on-screen size
        // matches the (libplacebo-scaled) image. In the AnimeJaNai pipeline LUMA is
        // post-upscale (= OUTPUT), so this is a no-op there; it corrects the per-
        // source size only when libplacebo scales the source to a different display.
        m_source_height = HOOKED_size.y;
        if (!state_ok || !prev_ready || frame == 0u) {
            m_mid = MID_FLOOR; m_steepness = STEEP_FLOOR; m_intensity_raw = 0.0;
            m_log_avg = 0.0; m_measured = 0.0; m_debug_tick = 0.0;
            m_prev_ready = 1.0; m_state_magic = STATE_MAGIC; m_state_epoch = state_epoch; m_sat_frac = 0.0;
            m_held_frames = 0.0; m_distinct_frames = 0.0; m_hold_run = 0.0;
            m_content_held = 0.0; m_content_distinct = 0.0; m_content_hold_run = 0.0; m_content_log_avg = 0.0;
            m_coherence = 0.0; m_coh_frac = 0.0; m_coh_samples = 0.0;
            m_held_count = 0.0; m_film_level = 0.0; m_pan_px = 0.0;
            m_gen_frame = 0.0;
            m_eff_render = 0.0; m_eff_mid = MID_FLOOR; m_eff_steep = STEEP_FLOOR;
            m_conf = 0.0; m_shape_w = 0.0;
            m_regen = 1.0; m_field_seed = 0.0;
            for (int b = 0; b < NBINS; b++) {
                hist_amp[b] = 0.0; hist_temporal[b] = 0.0; hist_spatial[b] = 0.0;
                hist_held[b] = 0.0;
            }
            s_update_prev_grid = 1u;
        } else {
            m_debug_tick += 1.0;
            // SOURCE-frame counter — advances once per source frame (this measure
            // pass is a LUMA hook = fresh group, 24/s). The OUTPUT pass seeds grain
            // from THIS, not the per-present `frame` builtin, so grain re-seeds at
            // the source rate on any display (fixes the display-refresh-dependent
            // cadence: `frame` ticks per present, so grain_rate 0.5 was ~60 Hz on a
            // 120 Hz display, ~24 Hz on the 47.952 Hz PC — verified 2026-07-06).
            m_gen_frame += 1.0;

            float binw = AMP_MAX / float(AMP_BINS);
            float total = 0.0, sum_abs = 0.0, sat = 0.0;
            float content_total = 0.0, content_sum_abs = 0.0;
            for (int lb = 0; lb < NBINS; lb++)
                for (int ab = 0; ab < AMP_BINS; ab++) {
                    float c = float(s_hist[lb][ab]);
                    total += c;
                    sum_abs += c * (float(ab) + 0.5) * binw;
                    if (ab >= SAT_BIN) sat += c;
                }
            for (int ab = 0; ab < AMP_BINS; ab++) {
                float c = float(s_content_hist[ab]);
                content_total += c;
                content_sum_abs += c * (float(ab) + 0.5) * binw;
            }
            float mean_abs = total > 0.0 ? sum_abs / total : 0.0;
            float sat_frac = total > 0.0 ? sat / total : 0.0;
            float content_mean_abs = content_total > 0.0 ? content_sum_abs / content_total : 0.0;

            bool no_flat = total < 1.0;
            bool held = no_flat || mean_abs < HELD_EPS;
            bool cut = sat_frac > CUT_SAT_FRAC;
            // Confident HARD cut (flashback to a different grain regime): most of
            // the frame changed at once. This is the ONLY thing that re-enters fast
            // convergence (resets the warmup counter), so the otherwise-stable grain
            // level can jump to the new regime. Busy/chaotic action stays well under
            // hardcut_frac, so it can never pump grain via this path.
            bool hard_cut = sat_frac > hardcut_frac;
            if (hard_cut) m_measured = min(m_measured, CUT_REREADY);
            bool content_held = content_total > 0.0 && content_mean_abs < CONTENT_HELD_EPS;
            // FIELD REGEN decision (single thread here -> race-free; the gen pass
            // consumes m_regen next dispatch, the composite m_field_seed). A
            // visible tick regenerates only every 1/grain_gen_rate ticks; skipped
            // ticks are presented as a randomized toroidal shift+flip of the
            // standing field by the composite (see the recycle transform there).
            // m_field_seed = the visible seed the field was generated AT: the gen
            // pass seeds its noise from it (so a hard-cut regen refreshes the
            // measured character while HOLDING the standing pattern), and the
            // composite derives the recycle transform from vseed - m_field_seed
            // (0 = identity = bit-exact prior behavior). hard_cut (the confident
            // regime-change detector above) forces the in-place regen so the
            // baked grain character can never go stale across a cut. And any
            // frame that advances a FAST-phase EMA regenerates at full rate:
            // the field-baked character (size law, soften, bandpass fire) moves
            // ALPHA_FAST ~20% on such a frame, and a recycled multi-tick step
            // would land several of those moves in one visible hit. Two fast
            // windows exist, each guarded AT the gate that actually advances
            // its EMAs: here, the held-cel histogram (content_held && !cut
            // advances m_held_count and hist_held -> grain_presence -> the
            // baked gs_fire) while m_held_count < 8; the main shape/amplitude
            // warming test (m_measured < 32) lives in the second reducer block
            // below, inside its real update_ok && valid_bins gate — NOT here,
            // and NOT keyed on the held/cut locals: m_measured only advances
            // when the shape histogram has >= 2 valid bins, so a flat/degenerate
            // source keeps m_measured pinned forever while its EMAs are equally
            // pinned — recycling there is exact, and gating on a mirror of the
            // wrong condition held the door open forever (found empirically on
            // the flat-gray harness source). Frames that advance NO EMA cannot
            // move the baked character at all; settled ALPHA_SLOW drift steps
            // ~3% per skipped tick through a pow-law map, masked by the pattern
            // reseed it always coincides with.
            float vseed = floor(m_gen_frame * grain_rate);
            float vprev = floor(max(m_gen_frame - 1.0, 0.0) * grain_rate);
            bool vtick = vseed != vprev;
            bool ema_fast = content_held && !cut && m_held_count < 8.0;
            bool regen_tick = floor(vseed * grain_gen_rate) != floor(vprev * grain_gen_rate)
                              || ema_fast;
            m_regen = ((vtick && regen_tick) || hard_cut) ? 1.0 : 0.0;
            if (vtick && regen_tick) m_field_seed = vseed;
            s_update_prev_grid = (!held || no_flat || cut) ? 1u : 0u;
            if (held && !no_flat && !cut) {
                m_held_frames += 1.0;
                m_hold_run += 1.0;
            } else if (!cut) {
                m_distinct_frames += 1.0;
                m_hold_run = 0.0;
            } else {
                m_hold_run = 0.0;
            }
            if (content_held && !cut) {
                m_content_held += 1.0;
                m_content_hold_run += 1.0;
            } else if (!cut && content_total > 0.0) {
                m_content_distinct += 1.0;
                m_content_hold_run = 0.0;
            } else {
                m_content_hold_run = 0.0;
            }

            m_log_avg = mean_abs;
            m_content_log_avg = content_mean_abs;
            m_sat_frac = sat_frac;

            // DIAGNOSTIC: absolute RMS of the high-passed temporal diff (d_pair,
            // grain estimate with DC drift + smooth motion removed) vs the raw
            // per-pixel diff (content + grain). If d_pair_rms separates clean from
            // grainy sources, it is a clean grain measure; the raw rms is content-
            // dominated. total = flat-sample count from s_hist.
            float coh_norm = float(s_coh_norm);
            float n_coh = max(total, 1.0);
            float d_pair_rms = sqrt((float(s_coh_incoh) / COH_SQ_SCALE) / n_coh);
            float d_raw_rms  = sqrt((coh_norm / COH_SQ_SCALE) / (2.0 * n_coh));
            m_coh_frac = d_pair_rms;            // [0,1.008] readout, ~grain scale
            m_coh_samples = d_raw_rms * 1.0e6;  // content+grain rms x1e6
            if (coh_norm >= COH_MIN_NORM && !held && !cut) {
                float incoh = float(s_coh_incoh) / coh_norm;
                m_coherence = (m_measured < 0.5) ? incoh
                                                 : mix(m_coherence, incoh, COH_ALPHA);
            }

            float center_luma = measure_luma(vec2(0.5, 0.5));
            hist_spatial[14] = total;
            hist_spatial[15] = center_luma;

            // Grid global-pan magnitude (px/frame) from the LK normal equations
            // (2x2 solve). Signed sums recombined from the +/- accumulators. The
            // shift is in grid-cell units -> px via the grid cell size. Light EMA
            // (0.30) smooths single-frame noise but tracks a pan within a few frames;
            // skip held/cut frames (stale prev_grid would inflate it). This feeds the
            // intensity-rise freeze below -- it is a MOTION estimate, not a grain one,
            // so it may be read promptly (the cross-timeline caveat is for grain gates).
            float lk_Gxx = float(s_lk_gxx) / LK_SCALE;
            float lk_Gyy = float(s_lk_gyy) / LK_SCALE;
            float lk_Gxy = (float(s_lk_gxy_p) - float(s_lk_gxy_n)) / LK_SCALE;
            float lk_BX  = (float(s_lk_bx_p)  - float(s_lk_bx_n))  / LK_SCALE;
            float lk_BY  = (float(s_lk_by_p)  - float(s_lk_by_n))  / LK_SCALE;
            float lk_det = lk_Gxx * lk_Gyy - lk_Gxy * lk_Gxy;
            float pan_px_inst = 0.0;
            if (lk_det > 1e-7) {
                float sgx = (lk_Gyy * lk_BX - lk_Gxy * lk_BY) / lk_det;  // grid-cell units
                float sgy = (lk_Gxx * lk_BY - lk_Gxy * lk_BX) / lk_det;
                float cellx = HOOKED_size.x / float(GRID_W);
                float celly = HOOKED_size.y / float(GRID_H);
                pan_px_inst = sqrt(sgx * sgx * cellx * cellx + sgy * sgy * celly * celly);
            }
            if (!held && !cut)
                m_pan_px = (m_measured < 0.5) ? pan_px_inst : mix(m_pan_px, pan_px_inst, 0.30);

            // Grain SHARPNESS ratio (fine/coarse = 1px-vs-3px energy), measured at
            // LUMA = pre-upscale source sharpness. (OUTPUT would read the libplacebo
            // /AI-blurred grain and lose the sandpaper character, so this MUST stay
            // on the LUMA pass.) High -> fine/sharp (sandpaper, Goku); low -> coarse
            // /soft (Nier). Slow EMA over distinct frames; the render pass reads
            // m_grain_fine and scales grain sigma toward it, resolution-normalized.
            // Applied only when the amplitude path is graining, so its value on
            // clean sources (no grain -> reads smooth/low) is harmlessly unused.
            if (!held && !cut && s_size_coarse > 0u) {
                float size_ratio = float(s_size_fine) / float(s_size_coarse);
                m_grain_fine = (m_measured < 0.5) ? size_ratio
                                                  : mix(m_grain_fine, size_ratio, 0.05);
            }

            if (!held && !cut) {
                bool first = m_measured < 0.5;
                float alpha = (m_measured < 8.0) ? ALPHA_FAST : ALPHA_SLOW;

                for (int lb = 0; lb < NBINS; lb++) {
                    float Nb = 0.0;
                    for (int ab = 0; ab < AMP_BINS; ab++)
                        Nb += float(s_hist[lb][ab]);

                    // Transition-temporal grain: p25 of |delta| on flat pixels,
                    // matching the offline source-frame estimator
                    // (mgt_temporal_probe.py). The low quartile tracks the grain
                    // floor and is robust to the motion-inflated high tail, so
                    // panning/static texture (Frieren) stops reading as grain
                    // while real grain survives as a smooth contiguous curve.
                    // p25 and p50 yield the SAME grain sigma on pure grain (the
                    // _TO_GRAINSTD constants renormalize each quantile); they
                    // differ only under motion, where p50 is pulled up and p25
                    // is not. hist_held below stays on p50 (no motion there).
                    float t_g = -1.0;
                    if (Nb >= MIN_BIN_CELLS) {
                        float target = 0.25 * Nb, acc = 0.0, q = AMP_MAX;
                        for (int ab = 0; ab < AMP_BINS; ab++) {
                            float c = float(s_hist[lb][ab]);
                            if (acc + c >= target) {
                                q = (float(ab) + (target - acc) / max(c, 1.0)) * binw;
                                break;
                            }
                            acc += c;
                        }
                        t_g = q * P25_TO_GRAINSTD;
                    }

                    hist_temporal[lb] = (t_g >= 0.0)
                        ? (first ? t_g : mix(hist_temporal[lb], t_g, alpha))
                        : (first ? 0.0 : mix(hist_temporal[lb], 0.0, ALPHA_MISSING));
                }

                // Flat-temporal FILM-grain LEVEL: band-median (nonzero) of the p25
                // transition-temporal curve over the plausible band [2..13] (matches
                // the debug-emitted range + the offline equilibrium validation),
                // accumulated into m_film_level with asymmetric-SAFE EMA rates so an
                // occasional clean motion scene (water/foliage/dissolve) can't lift it
                // -- only persistent film grain does (see the FILM_* define comment).
                float fl_vals[NBINS];
                int fl_n = 0;
                for (int fb = GRAIN_LO_BIN; fb <= 13; fb++) {
                    if (hist_temporal[fb] > 0.0 && fl_n < NBINS) {
                        fl_vals[fl_n] = hist_temporal[fb];
                        fl_n++;
                    }
                }
                if (fl_n > 0) {
                    for (int i = 1; i < NBINS; i++) {
                        if (i >= fl_n) break;
                        float key = fl_vals[i];
                        int j = i - 1;
                        for (int g = 0; g < NBINS; g++) {
                            if (j < 0 || fl_vals[j] <= key) break;
                            fl_vals[j + 1] = fl_vals[j];
                            j--;
                        }
                        fl_vals[j + 1] = key;
                    }
                    float ftl_inst = fl_vals[fl_n / 2];
                    // Upper plausibility ceiling. Real film grain's flat-temporal
                    // level is bounded (~0.004-0.007); smoke / fog / dissolves / fast
                    // flat-region motion read FAR higher (0.009-0.030) and would pull
                    // the EMA up within a single SUSTAINED scene (the slow attack only
                    // suppresses BRIEF spikes, not a 10-30s effects shot). Reject those
                    // frames as motion, not grain -> m_film_level just holds. Diverse-
                    // timestamp smoke proved clean sources spike to 9-30 on effects
                    // scenes (Fate smoke 18.8, Ninja 29.2) while real grain stays <7.
                    if (ftl_inst <= FILM_LEVEL_CEIL) {
                        float fl_alpha = (ftl_inst > m_film_level) ? FILM_ATTACK : FILM_DECAY;
                        m_film_level = (m_film_level <= 0.0) ? ftl_inst
                                                            : mix(m_film_level, ftl_inst, fl_alpha);
                    }
                }
            }

            // HELD-CEL TEMPORAL GRAIN (#2 discriminator, measurement only).
            // On a content-held cel the smooth image is static, so the per-pixel
            // high-freq temporal delta (s_hist, built this frame) is PURE grain +
            // dither, with NO motion contamination — unlike hist_temporal above,
            // which measures across content transitions. Real grain re-randomizes
            // every frame -> a smooth nonzero curve across luma; painted/static
            // texture is bit-identical frame to frame -> cancels to the dither
            // floor. This is the offline 16-bit source-frame probe, in-shader.
            // NOT yet wired to render — validating separation first.
            if (content_held && !cut) {
                bool hfirst = m_held_count < 0.5;
                float halpha = (m_held_count < 8.0) ? ALPHA_FAST : ALPHA_SLOW;
                for (int lb = 0; lb < NBINS; lb++) {
                    float Nb = 0.0;
                    for (int ab = 0; ab < AMP_BINS; ab++)
                        Nb += float(s_hist[lb][ab]);

                    float h_g = -1.0;
                    if (Nb >= MIN_BIN_CELLS) {
                        float target = TEMPORAL_PCTL * Nb, acc = 0.0, q = AMP_MAX;
                        for (int ab = 0; ab < AMP_BINS; ab++) {
                            float c = float(s_hist[lb][ab]);
                            if (acc + c >= target) {
                                q = (float(ab) + (target - acc) / max(c, 1.0)) * binw;
                                break;
                            }
                            acc += c;
                        }
                        h_g = q * P50_TO_GRAINSTD;
                    }

                    hist_held[lb] = (h_g >= 0.0)
                        ? (hfirst ? h_g : mix(hist_held[lb], h_g, halpha))
                        : (hfirst ? 0.0 : mix(hist_held[lb], 0.0, ALPHA_MISSING));
                }
                m_held_count += 1.0;
            }
        }
    }
    barrier();

    if (is_measure_wg && s_update_prev_grid != 0u) {
        for (uint k = lid; k < uint(GRID_N); k += num_threads) {
            vec2 uv = (vec2(float(k % uint(GRID_W)), float(k / uint(GRID_W))) + 0.5)
                     / vec2(float(GRID_W), float(GRID_H));
            prev_grid[k] = measure_luma(uv);
            prev_grid_off[k] = measure_luma(uv + vec2(COH_OFFSET_PX * HOOKED_pt.x, 0.0));
        }
    }
    barrier();

    // Spatial estimator: 5x5 local flat-patch std by luma. This drives the
    // measured tone curve. Temporal flat-patch frame differences can provide a
    // conservative supplemental intensity floor when many luma bins show
    // stochastic frame-to-frame grain. When spatial residual is high but the
    // temporal signal does not corroborate it, treat the excess as painted or
    // static texture rather than recoverable master grain.
    if (is_measure_wg) {
        for (uint k = lid; k < uint(NBINS * AMP_BINS); k += num_threads)
            s_hist[k / uint(AMP_BINS)][k % uint(AMP_BINS)] = 0u;
    }
    barrier();

    if (is_measure_wg) {
        for (uint k = lid; k < uint(GRID_N); k += num_threads) {
            vec2 uv = (vec2(float(k % uint(GRID_W)), float(k / uint(GRID_W))) + 0.5)
                     / vec2(float(GRID_W), float(GRID_H));

            float sy = 0.0, syy = 0.0, gx = 0.0, gy = 0.0;
            for (int yy = -2; yy <= 2; yy++) {
                for (int xx = -2; xx <= 2; xx++) {
                    vec2 suv = uv + vec2(float(xx), float(yy)) * HOOKED_pt;
                    float y = measure_luma(suv);
                    sy += y;
                    syy += y * y;
                    gx += y * float(xx);
                    gy += y * float(yy);
                }
            }

            float mean_y = sy / 25.0;
            float var_y = max(syy / 25.0 - mean_y * mean_y, 0.0);
            float slope = length(vec2(gx, gy)) / 50.0;

            if (mean_y > BLACK_FLOOR && mean_y < WHITE_CEIL
                && slope < SPATIAL_EDGE_GATE && var_y < SPATIAL_VAR_MAX) {
                // Remove the best local plane before estimating grain. Raw 5x5
                // variance treats small painted ramps/road/wall texture as
                // grain; the residual keeps high-frequency stochastic energy.
                float residual_var = max(var_y - 2.0 * slope * slope, 0.0);
                float amp = sqrt(residual_var);

                int lb = clamp(int(mean_y * float(NBINS)), 0, NBINS - 1);
                int ab = clamp(int(amp / AMP_MAX * float(AMP_BINS)), 0, AMP_BINS - 1);
                atomicAdd(s_hist[lb][ab], 1u);
            }
        }
    }
    barrier();

    if (is_measure_wg && lid == 0u) {
        float binw = AMP_MAX / float(AMP_BINS);
        bool update_ok = prev_ready;
        float spatial_peak = 1e-6;
        float spatial_level = 1e-6;
        int spatial_pb = (GRAIN_LO_BIN + GRAIN_HI_BIN) / 2;
        int valid_bins = 0;
        float level_vals[NBINS];
        float temporal_vals[NBINS];
        int level_n = 0;
        int temporal_n = 0;

        for (int lb = 0; lb < NBINS; lb++) {
            float Nb = 0.0;
            for (int ab = 0; ab < AMP_BINS; ab++)
                Nb += float(s_hist[lb][ab]);

            float s_g = 0.0;
            if (Nb >= MIN_BIN_CELLS) {
                valid_bins++;
                float target = SPATIAL_PCTL * Nb, acc = 0.0;
                for (int ab = 0; ab < AMP_BINS; ab++) {
                    float c = float(s_hist[lb][ab]);
                    if (acc + c >= target) {
                        s_g = (float(ab) + (target - acc) / max(c, 1.0)) * binw;
                        break;
                    }
                    acc += c;
                }
            }
            if (lb < 14)
                hist_spatial[lb] = s_g;

            if (lb >= GRAIN_LO_BIN && lb <= GRAIN_HI_BIN && s_g > 0.0) {
                float lum_b = (float(lb) + 0.5) / float(NBINS);
                float dark_w = smoothstep(0.02, 0.12, lum_b);
                float bright_w = 1.0 - smoothstep(PLAUSIBLE_BRIGHT_LO, PLAUSIBLE_BRIGHT_HI, lum_b);
                float plausible = dark_w * bright_w;
                float a = s_g * plausible;
                if (a > 0.0 && level_n < NBINS) {
                    level_vals[level_n] = a;
                    level_n++;
                }
                float t = hist_temporal[lb] * plausible;
                if (t >= TEMPORAL_BIN_FLOOR && temporal_n < NBINS) {
                    temporal_vals[temporal_n] = t;
                    temporal_n++;
                }
                if (a > spatial_peak) { spatial_peak = a; spatial_pb = lb; }
            }

            if (update_ok) {
                bool first = m_measured < 8.0;
                float alpha = (m_measured < 32.0) ? ALPHA_FAST : ALPHA_SLOW;
                if (Nb >= MIN_BIN_CELLS)
                    hist_amp[lb] = first ? s_g : mix(hist_amp[lb], s_g, alpha);
                else
                    hist_amp[lb] = first ? 0.0 : mix(hist_amp[lb], 0.0, ALPHA_MISSING);
            }
        }

        if (update_ok && valid_bins >= 2) {
            for (int i = 1; i < NBINS; i++) {
                if (i >= level_n) break;
                float key = level_vals[i];
                int j = i - 1;
                for (int guard = 0; guard < NBINS; guard++) {
                    if (j < 0 || level_vals[j] <= key) break;
                    level_vals[j + 1] = level_vals[j];
                    j--;
                }
                level_vals[j + 1] = key;
            }
            if (level_n > 0) {
                float level_floor = max(0.001, 0.25 * spatial_peak);
                int keep_start = 0;
                for (int i = 0; i < NBINS; i++) {
                    if (i >= level_n || level_vals[i] >= level_floor) break;
                    keep_start++;
                }
                int keep_n = level_n - keep_start;
                int level_idx = (keep_n >= 3)
                    ? keep_start + (3 * (keep_n - 1)) / 4
                    : (3 * (level_n - 1)) / 4;
                spatial_level = level_vals[level_idx];
            }

            // Contiguity gate: real grain is signal-independent, so its temporal
            // energy forms a smooth, contiguous curve across all luma bins.
            // Motion over smooth gradients produces temporal energy concentrated
            // in the few luma bands where moving edges land, leaving gaps. The
            // longest contiguous run of above-floor plausible bins separates them
            // (grain ~all bins; motion a short spiky run).
            int temporal_run = 0, temporal_run_max = 0;
            for (int lb = GRAIN_LO_BIN; lb <= GRAIN_HI_BIN; lb++) {
                if (hist_temporal[lb] >= TEMPORAL_BIN_FLOOR) {
                    temporal_run++;
                    temporal_run_max = max(temporal_run_max, temporal_run);
                } else {
                    temporal_run = 0;
                }
            }
            float temporal_contig = smoothstep(float(TEMPORAL_RUN_MIN),
                                               float(TEMPORAL_RUN_FULL),
                                               float(temporal_run_max));

            // Incoherence gate: pass temporal grain only when its diffs are
            // spatially incoherent (m_coherence ~1.0). Coherent moving texture
            // (smoke, pans) and DC drift read ~0 and are suppressed.
            float temporal_coh = smoothstep(COH_LO, COH_HI, m_coherence);

            float temporal_level = 0.0;
            if (temporal_n >= TEMPORAL_MIN_BINS) {
                for (int i = 1; i < NBINS; i++) {
                    if (i >= temporal_n) break;
                    float key = temporal_vals[i];
                    int j = i - 1;
                    for (int guard = 0; guard < NBINS; guard++) {
                        if (j < 0 || temporal_vals[j] <= key) break;
                        temporal_vals[j + 1] = temporal_vals[j];
                        j--;
                    }
                    temporal_vals[j + 1] = key;
                }
                temporal_level = temporal_vals[temporal_n / 2] * TEMPORAL_GAIN
                               * temporal_contig * temporal_coh;
            }

            float half_peak = 0.5 * spatial_peak;
            int lbf = spatial_pb, rbf = spatial_pb;
            for (int i = 0; i < NBINS; i++) {
                if (lbf > GRAIN_LO_BIN && hist_amp[lbf - 1] >= half_peak) lbf--;
                else break;
            }
            for (int i = 0; i < NBINS; i++) {
                if (rbf < GRAIN_HI_BIN && hist_amp[rbf + 1] >= half_peak) rbf++;
                else break;
            }

            float fwhm = float(rbf - lbf + 1) / float(NBINS);
            float d_half = max(0.5 * fwhm, 0.5 / float(NBINS));
            float steep = clamp(0.638 / (d_half * d_half), STEEP_MIN, STEEP_MAX);
            float mid = (float(spatial_pb) + 0.5) / float(NBINS);
            float temporal_support = temporal_level;
            if (spatial_level > TEMPORAL_SPATIAL_CEIL)
                temporal_level = 0.0;
            // Temporal grain needs SPATIAL corroboration. Real grain (and overlay
            // grain) has spatial texture AND temporal animation; smoke / fog /
            // dissolves are temporally active but spatially SMOOTH (near-zero 5x5
            // variance). Without this, the temporal-only path fired moving smoke as
            // grain (Fate ~19:08: spatial_level ~0.0011, m_int 0.16, eff 0.188) while
            // genuine soft grain sits at spatial_level >= 0.003 (NieR). Diverse-
            // timestamp smoke surfaced it; the gap is ~0.0012 (smoke) vs ~0.0030
            // (NieR). Costs the spatially-weakest soft grain (Lain) -- accepted
            // under-grain. temporal_support stays raw (it gates the static clamp).
            temporal_level *= smoothstep(TEMPORAL_NEEDS_SPATIAL_LO,
                                         TEMPORAL_NEEDS_SPATIAL_HI, spatial_level);
            float spatial_for_intensity = spatial_level;
            if (spatial_level > SPATIAL_STATIC_GATE) {
                float temporal_ratio = temporal_support / max(spatial_level, 1e-6);
                float temporal_ok = smoothstep(SPATIAL_STATIC_RATIO_LO,
                                               SPATIAL_STATIC_RATIO_HI,
                                               temporal_ratio);
                spatial_for_intensity = mix(min(spatial_level, SPATIAL_STATIC_GATE),
                                            spatial_level,
                                            temporal_ok);
            }
            float irraw = max(spatial_for_intensity, temporal_level) / NOISE_STD_PER_INTENSITY;

            bool first = m_measured < 8.0;
            bool warming = m_measured < 32.0;
            float alpha = warming ? ALPHA_FAST : ALPHA_SLOW;
            // Stable, asymmetric grain LEVEL, biased SAFE on the way up. The
            // measured intensity always RISES with grain_attack (slow) and FALLS
            // with grain_decay -- even right after a hard cut. A scene cut can no
            // longer pop grain upward: a genuine grainy flashback still builds in
            // ~1-2s, but a clean-but-busy new scene (or stale cross-cut temporal
            // corroboration that briefly mis-passes the static gate) is given time
            // to settle back down before any grain becomes visible. During the
            // post-cut/warmup window the DOWN direction is allowed to be fast
            // (ALPHA_FAST) so a grainy->clean cut sheds grain in a few frames --
            // removing grain is never "overkill". Tone-shape params (mid/steep)
            // keep the symmetric warmup alpha; they only shape grain, never add it.
            // Freeze the RISE of the intensity EMA while a real global pan is present
            // (m_pan_px): a translating baked texture can no longer BUILD grain up over
            // the shot, while grain already established simply HOLDS through the pan
            // (the FALL path is untouched -- shedding grain is never overkill). Clean
            // sources are floored regardless. pan_freeze=0 -> exact prior behaviour.
            float motion_freeze = pan_freeze * smoothstep(PAN_FREEZE_LO, PAN_FREEZE_HI, m_pan_px);
            float rise_alpha = grain_attack * (1.0 - motion_freeze);
            float int_alpha = (irraw > m_intensity_raw)
                            ? rise_alpha
                            : (warming ? ALPHA_FAST : grain_decay);
            if (first) {
                m_mid = mid; m_steepness = steep; m_intensity_raw = irraw;
            } else {
                m_mid = mix(m_mid, mid, alpha);
                m_steepness = mix(m_steepness, steep, alpha);
                m_intensity_raw = mix(m_intensity_raw, irraw, int_alpha);
            }
            // FIELD REGEN warming override (see the decision in the first
            // reducer block above). THIS branch is what actually advances
            // m_measured and the shape/amplitude EMAs (update_ok &&
            // valid_bins >= 2), so the ALPHA_FAST warming test lives here:
            // while m_measured < 32 (covers the < 8 snap phase too), any
            // frame that moves those EMAs forces a full regen so the
            // field-baked character tracks per-frame. Same thread as the
            // decision above -> plain program-order override, consumed by
            // the gen pass after this dispatch completes. On a non-tick
            // frame (grain_rate < 1) m_field_seed deliberately stays put:
            // the regen re-bakes IN PLACE (same seed -> same pattern,
            // freshly measured character), the hard-cut semantics.
            if (m_measured < 32.0) {
                m_regen = 1.0;
                float ov_vseed = floor(m_gen_frame * grain_rate);
                if (ov_vseed != floor(max(m_gen_frame - 1.0, 0.0) * grain_rate))
                    m_field_seed = ov_vseed;
            }
            m_measured += 1.0;
        }
    }
    barrier();

    if (lid == 0u)
        imageStore(out_image, ivec2(0), vec4(0.0));
}

//!HOOK LUMA
//!BIND HOOKED
//!BIND GRAIN_STATE
//!BIND GRAIN_FIELD
//!SAVE GRAIN_GEN_TRIGGER
//!WIDTH 3840
//!HEIGHT 2160
//!COMPUTE 32 32
//!DESC Film Grain Match: GRAIN gen (fixed 4K, source-locked)

// LUMA hook = mpv's FRESH group: runs once per SOURCE frame regardless of
// video-sync / display refresh (measured 24.x/s under both display-resample
// @120Hz and audio sync, 2026-07-06). The OUTPUT composite (redraw group,
// per-present) just fetches GRAIN_FIELD, so re-presents cost ~nothing and
// grain cadence can't ride the display refresh. The 3840x2160 grid is the
// grain's own resolution ("a fixed 4K scan"): generation cost is constant,
// and GRAIN_RES_REF scaling below collapses to 1 (the grid IS the 4K ref).
//
// STORAGE-TEXTURE RECOVERY (2026-07-06): the earlier split SAVE'd GRAIN_FIELD
// from this fresh pass and BIND'd it in the OUTPUT redraw pass — but a SAVE'd
// texture is a per-frame transient and did NOT survive the fresh->redraw group
// gap, so grain generated but never reached the presented frame. Fix: GRAIN_FIELD
// is now a persistent, shader-owned TEXTURE+STORAGE image (declared
// top-of-file, like the GRAIN_STATE SSBO) that we imageStore into here and
// imageLoad from in the composite. Persistent storage retains its contents across
// presents, so a redraw (no fresh dispatch) reads the last-written field. This
// fresh pass still needs a dispatch grid, so it SAVEs a throwaway 4K trigger
// texture (GRAIN_GEN_TRIGGER, never bound) purely to size the 3840x2160 dispatch.
// GRAIN_FIELD is rgba16f: rgb = final signed grain (bandpass + warp + chroma),
// a = the pre-warp LOWPASS grain's luma (debug A/B strips).

// 9-tap support (was 7): the blue channel's base sigma (1.20) reaches ~1.15 even at
// the crisp neutral and higher when coarse, where 7 taps (clean only to sigma ~1.0)
// truncate the Gaussian into a box and ripple the spectrum. 9 taps hold sigma up to
// ~1.5 cleanly, covering the full fine-digital -> 16mm render range. Arrays/loops
// are parametrized on MAX_TAPS so the support can never drift out of sync.
#define MAX_TAPS 4

#define INT_FLOOR    0.10
#define INT_CEIL     0.85
#define MID_FLOOR    0.15
#define STEEP_FLOOR  2.6

// Crisp NEUTRAL render sigma scale (the no-fire 4K-film-scan default). 0.75 puts the
// green-channel sigma at ~0.75 px @4K = a 35mm scan (matched to the real grain
// plates; holds broadband energy to Nyquist instead of the old soft-Gaussian
// crater at sigma 1.0). The grain-stock probe found film spans ~0.75 (35mm) .. 1.5
// (16mm) @4K; the per-source map nudges around this neutral by measured gauge.
#define K_NEUTRAL    0.75
// Output res the crisp default is calibrated for (grain sigma scales by
// OUTPUT_height/REF so it reads as a 4K scan at constant visual angle on any display).
#define GRAIN_RES_REF 2160.0
// Fixed grain-grid dims. The noise seed wraps on these so the field is TOROIDAL:
// the composite fetches gid % grid, and a display taller/wider than the grid
// (e.g. the 3456x2234 XDR wraps vertically at 2160) must not show a seam. Wrapping
// the seed coordinate makes noise(grid)==noise(0); the separable DoG halo then
// wraps too (each workgroup regenerates its halo from global coords), so the seam
// is continuous. In-range coords (the whole grid interior) are unchanged.
// MUST equal this pass's WIDTH/HEIGHT directives above (3840 x 2160) — same
// translation unit, no compile guard. The composite derives its dims from
// imageSize(GRAIN_FIELD) so it can't desync; only these two same-file sites are
// hand-kept in lockstep. (Directive prefix omitted here on purpose: the parser
// splits sections on that marker even inside a comment.)
#define GEN_GRID ivec2(3840, 2160)
// Per-channel render-sigma cap. The 9-tap support holds a Gaussian cleanly to ~1.5;
// beyond that it would truncate. The coarse extreme (low-fineR firing) and >4K
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
// Lain restore-gain taper thresholds on m_intensity_raw: below LO = full compensatory
// restore_gain; above HI = ~x1 (grain already heavy/intact, don't over-restore).
// Only active when grain_restore_taper>0. Needs feel-test tuning.
#define RESTORE_TAPER_LO 0.15
#define RESTORE_TAPER_HI 0.30

#define CONF_LOW     0.10
#define CONF_HIGH    0.35


#define RED_VARIANCE_SCALE 1.0
#define GREEN_VARIANCE_SCALE 1.0
#define BLUE_VARIANCE_SCALE 1.0

#define RED_SATURATION 0.6
#define GREEN_SATURATION 0.5
#define BLUE_SATURATION 0.4

#define GRAIN_LO_BIN  2
#define GRAIN_HI_BIN  14

// Spatial-incoherence motion gate. Grain is spatially incoherent; coherent motion
// (smoke, pans) and global luma drift are not. We measure the temporal change of
// the local spatial micro-gradient between a pixel and one COH_OFFSET_PX away:
//   d_pair = [cur(p) - cur(p+d)] - [prev(p) - prev(p+d)]
// Global DC drift cancels (p and p+d shift equally); only independent grain
// survives. incoherence = sum(d_pair^2) / sum(d_center^2 + d_offset^2):
// grain ~1.0 (full decorrelation), motion/drift ~0 (micro-gradient preserved).
// COH_OFFSET_PX must exceed the grain correlation length (so even coarse/blocky
// grain reads incoherent) yet stay under the motion coherence length.
#define COH_OFFSET_PX 12.0
#define SOFTEN_TEMPORAL_RATIO_LO 1.20
#define SOFTEN_TEMPORAL_RATIO_HI 2.50
#define SOFTEN_INTENSITY_HI 0.24
#define SOFTEN_MAX_BLEND 0.35
#define TEMPORAL_TONE_STEEPEN 1.75
#define SHAPE_AMP_LOW     0.08
#define SHAPE_AMP_HIGH    0.25
#define SHAPE_MAX_BLEND   0.40
// STEP 2 held-cel firing gate (texture-safe). held_run = contiguous luma bins of
// the held-cel temporal-grain curve above the dither floor. Overlay grain fires
// via held_run directly; film grain fires via spatial amplitude but only when
// held_run corroborates (so busy-paper spatial can't ride the bypass).
#define HELD_RUN_BIN_FLOOR 0.0021
#define HELD_RUN_LO        2.0
#define HELD_RUN_HI        5.0
#define HELD_CORROB_LO     1.0
#define HELD_CORROB_HI     3.0
#define SPAT_PRESENCE_LO   0.0050
#define SPAT_PRESENCE_HI   0.0080
// FILM-grain firing path (the high-value target: genuine camera grain compression
// smoothed unevenly -- Goku Midnight Eye / Cyber City Oedo 808). Driven by the
// flat-temporal LEVEL (band-median of the p25 transition-temporal curve), which
// measures whether grain ANIMATES on flats rather than whether texture EXISTS, so
// it rides past the spatial film-vs-busy-texture overlap that blocked every prior
// attempt. The per-scene level overlaps (clean motion scenes -- water/foliage/
// dissolves -- spike to 8-10e-3, same as film), so it is NOT used instantaneously:
// m_film_level is a cross-timeline EMA with asymmetric-SAFE rates (slow attack,
// faster decay). The slow attack underweights the isolated clean spikes (they are
// the minority of scenes) while persistent film grain accumulates. Offline EMA
// equilibrium (mgt_gate_table.py, pooled taxonomy): clean/texture settle
// 1.9-2.8e-3 (floor, ZERO false positives), film settles 4.4e-3 (fires ~0.55).
// LO sits 1.0e-3 above the highest clean equilibrium -> the false-positive margin.
#define FILM_LEVEL_LO 0.0038
#define FILM_LEVEL_HI 0.0050

// PAN-FREEZE gate (canvas-pan overfire fix). A baked artistic texture (e.g. the
// distressed-paint wall in Days with My Stepsister) translating under a pan is a
// per-frame grain TWIN -- amplitude, incoherence, held-run, size and even motion-
// compensated residual all overlap real grain (offline-confirmed). Its only tell is
// the GLOBAL pan itself. A grid-level single-step Lucas-Kanade translation estimate
// is grain-INSENSITIVE (grain decorrelates across grid neighbours so its gx*d/gy*d
// products average to ~0; a coherent pan does not) -> reads ~2-4px on the canvas pan
// vs ~0 on held grain incl. NieR's heavy overlay (offline grid-LK: canvas 1.5-2.5px,
// grain <=0.43px). We do NOT veto firing (can't separate per-frame); instead, while a
// real pan is present, FREEZE the RISE of m_intensity_raw -> the translating texture
// can't BUILD up, while grain already established simply HOLDS through the pan (and
// clean sources are floored regardless). LK reuses the cur_smooth neighbour samples
// (no extra fetches). pan_freeze=0 -> exact prior behaviour; gs=0 stays bit-identical.
// Calibrated to in-shader m_pan_px (smoke): static grain accumulation reads <=0.05
// (Goku, the prize film target, 0.01-0.02), while the canvas pan reads 0.10-0.26.
// LO/HI sit in that gap so static grain BUILDS unfrozen and the pan freezes; grain
// sources that read high are themselves panning, where freeze only HOLDS the level.
// (In-shader magnitudes are ~6x below the offline area-grid LK -- aliased texture
// dilutes Gxx on the point grid -- but the static-vs-pan separation is what matters.)


#define STEEP_MAX     9.0

const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
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
shared float chroma_scale[3];              // per-channel chroma layer scale (CHROMA layer, uniform)


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

    // =========================================================================
    // GRAIN RENDERING — all workgroups
    // =========================================================================

    float conf = smoothstep(CONF_LOW, CONF_HIGH, m_intensity_raw);

    float shape_min = 1e10, shape_max = 0.0;
    float temporal_peak_render = 0.0;
    float spatial_peak_render = 0.0;
    int held_run = 0, held_run_max = 0;
    for (int b = GRAIN_LO_BIN; b <= GRAIN_HI_BIN; b++) {
        if (hist_amp[b] > 0.0) {
            shape_min = min(shape_min, hist_amp[b]);
            shape_max = max(shape_max, hist_amp[b]);
        }
        spatial_peak_render = max(spatial_peak_render, hist_amp[b]);
        temporal_peak_render = max(temporal_peak_render, hist_temporal[b]);
        // Held-cel contiguous run: the texture-safe grain-presence signal. Grain
        // re-randomizes on held cels -> smooth contiguous curve across luma;
        // painted/static texture is bit-identical on held cels and a pan is NOT a
        // held cel, so texture leaves no held signature (held_run ~0).
        if (hist_held[b] >= HELD_RUN_BIN_FLOOR) {
            held_run++;
            held_run_max = max(held_run_max, held_run);
        } else {
            held_run = 0;
        }
    }

    // STEP 2 firing gate. Held-cel run is the per-scene grain/texture discriminator
    // the transition-run can't be (a paper pan reads as grain on the transition
    // curve but leaves no held-cel signature). Two firing paths:
    //   overlay grain -> high held_run directly (NieR/Lain/Clevatess).
    //   film grain    -> spatial amplitude, but CORROBORATED by some held-cel grain
    //                    so busy-paper spatial (held_run ~0) cannot ride the bypass.
    // Frieren-busy has high spatial yet held_run ~0, so BOTH paths veto it.
    float held_presence = smoothstep(HELD_RUN_LO, HELD_RUN_HI, float(held_run_max));
    float held_corrob   = smoothstep(HELD_CORROB_LO, HELD_CORROB_HI, float(held_run_max));
    float spat_presence = smoothstep(SPAT_PRESENCE_LO, SPAT_PRESENCE_HI, spatial_peak_render);
    float grain_presence = max(held_presence, spat_presence * held_corrob);

    // FILM-grain path (the high-value target). The cross-scene EMA m_film_level
    // (measured in PASS 1) clears FILM_LEVEL_LO only for genuine, persistent film
    // grain; clean/texture settle below it (validated equilibrium). Folded in as a
    // BOOST like the others -> can only add firing, never veto what already fires.
    // Amplitude is still bounded by tgt_int (= m_intensity_raw*restore_gain), so a
    // clean source with low spatial amplitude stays near floor even if it ever fired.
    float film_presence = smoothstep(FILM_LEVEL_LO, FILM_LEVEL_HI, m_film_level);
    grain_presence = max(grain_presence, film_presence);

    // grain_presence BOOSTS firing for texture-safe grain (high held_run, or film
    // amplitude corroborated by held grain) WITHOUT vetoing what the amplitude path
    // already fires: conf is kept as the floor via max(), so film grain (Goku/
    // Gunbuster) firing on m_int is never regressed. Clean texture has conf ~0 AND
    // grain_presence ~0 -> floor; the held boost only adds on top. Amplitude
    // (tgt_int) sets HOW MUCH: stronger measured grain -> more. The stable asymmetric
    // m_intensity_raw drives tgt_int, so firing inherits the no-pump-up dynamics;
    // match_grain=0 -> w=0 -> exact Light floor (A/B safe).
    float w = match_grain * max(conf, grain_presence * restore_floor);
    // restore_gain is COMPENSATORY (extrapolate up the grain compression SMOOTHED).
    // On sources whose heavy grain SURVIVED intact (e.g. Lain's escalating mild->heavy
    // shots), x3 OVER-restores and the amplitude EMA builds up + over-applies. TAPER
    // the gain toward 1.0 as the measured amplitude rises, so faint/smoothed grain
    // gets full compensation and already-heavy grain gets ~none. OFF by default
    // (grain_restore_taper=0 -> exact current behavior, A/B safe); enable to feel-test
    // the Lain over-apply fix. Thresholds need feel-test tuning.
    float gain_taper = mix(restore_gain, 1.0,
                           smoothstep(RESTORE_TAPER_LO, RESTORE_TAPER_HI, m_intensity_raw));
    float eff_gain = mix(restore_gain, gain_taper, grain_restore_taper);
    float tgt_int = clamp(m_intensity_raw * eff_gain, INT_FLOOR, INT_CEIL);
    float eff_render = mix(INT_FLOOR, tgt_int, w);
    // grain_gain: unclamped post-clamp intensity dial (restore_gain saturates at
    // INT_CEIL, so it can't push past it; this can). 1.0 = calibrated; raise to make
    // grain strongly visible for evaluation, then dial back. Scales image + debug patches.
    eff_render *= grain_gain;
    float shape_ratio = (shape_min > 0.0) ? shape_max / shape_min : 1.0;
    float shape_conf = smoothstep(1.2, 2.0, shape_ratio);

    float softness_ratio = temporal_peak_render / max(spatial_peak_render, 1e-6);
    float softness_temporal = smoothstep(SOFTEN_TEMPORAL_RATIO_LO,
                                         SOFTEN_TEMPORAL_RATIO_HI,
                                         softness_ratio);
    float softness_conf = smoothstep(CONF_LOW, SOFTEN_INTENSITY_HI, m_intensity_raw);
    float source_soften = SOFTEN_MAX_BLEND * softness_temporal * softness_conf;

    // Adaptive floor tone. When amplitude confidence is low (eff_render pinned
    // near the Light floor), the floor grain's tone curve still follows the
    // measured shape so borderline/soft-grain sources stop collapsing to generic
    // Light. AMPLITUDE STAYS PINNED — only eff_mid/eff_steep move. Gated by
    // SPATIAL amplitude (shape_amp_gate) and curve structure (shape_conf): these
    // separate clean (m_int ~0.05) and static texture (~0.08) from real grain
    // (~0.16+). NB: temporal incoherence (m_coherence) is NOT usable as a gate —
    // when actually measured it reads ~1.0 for clean AND grainy alike, because
    // the sub-LSB temporal diff is dither-dominated and dither is itself
    // incoherent (corollary of the amplitude-quantization keystone).
    float shape_amp_gate = smoothstep(SHAPE_AMP_LOW, SHAPE_AMP_HIGH, m_intensity_raw);
    float shape_floor_blend = SHAPE_MAX_BLEND * shape_amp_gate * shape_conf;
    float shape_w = max(conf * shape_conf, shape_floor_blend) * match_grain;

    float eff_mid = mix(MID_FLOOR, m_mid, shape_w);
    float eff_steep = mix(STEEP_FLOOR, m_steepness, shape_w);
    float temporal_tone = softness_temporal * softness_conf;
    eff_steep = min(STEEP_MAX, eff_steep * mix(1.0, TEMPORAL_TONE_STEEPEN, temporal_tone));

    // Seed from m_gen_frame (the measure pass's SOURCE-frame counter), not
    // content cadence — a content-cadence seed froze grain on held cels of
    // clean sources (no grain + identical cel -> counter stalls -> static
    // "dirt" look) — and NOT the `frame` builtin: `frame` ticks per PRESENT
    // (proven 2026-07-06 with a paused-frame parity probe), so under
    // display-resample it made grain_rate display-refresh-dependent — the
    // "0.5 = half rate" tuning was really "one reseed per source frame" on a
    // 47.952 Hz (2x 23.976) display and an over-fast ~60 Hz crawl at 120 Hz.
    // With the source-locked counter, grain_rate 1.0 = every source frame
    // ("on ones", the tuned look); 0.5 = on twos. The measured distinct
    // counter is retained for held-grain-cadence detection (twos/threes),
    // deferred until source-grain cadence is measured directly.
    // Seed value = m_field_seed, the visible-tick seed the measure pass pinned
    // for this field (equals floor(m_gen_frame * grain_rate) on every regular
    // regen tick, so at grain_gen_rate 1.0 this is the exact prior seed
    // sequence). On a hard-cut forced regen it deliberately KEEPS the standing
    // tick's seed: same noise pattern, freshly measured character.
    uint frame_seed = uint(max(0.0, m_field_seed));

    // Composite-side scalars, computed ONCE here (uniform math over
    // GRAIN_STATE + params) and passed through the state buffer so the
    // two translation units can't drift on duplicated defines. Written
    // by a single thread; the composite dispatch reads them after this
    // pass completes. Written BEFORE the regen gate below so amplitude/
    // tone keying stays fresh on every source frame even while the field
    // itself is held (grain_gen_rate < 1).
    if (gl_GlobalInvocationID.x == 0u && gl_GlobalInvocationID.y == 0u) {
        m_eff_render = eff_render;
        m_eff_mid    = eff_mid;
        m_eff_steep  = eff_steep;
        m_conf       = conf;
        m_shape_w    = shape_w;
        // The SAVE'd trigger target only exists to size the dispatch and is
        // never bound; write it from ONE invocation so we don't pay a second
        // full-4K store.
        imageStore(out_image, ivec2(0), vec4(0.0));
    }

    // FIELD REGEN GATE (grain_gen_rate). m_regen is a single SSBO value decided
    // by the measure pass, identical for every invocation of this dispatch, so
    // this is UNIFORM control flow — returning before the barriered generation
    // below is legal (each workgroup's invocations all take the same branch).
    // On skip frames the persistent GRAIN_FIELD keeps its last-generated
    // content and the composite presents it through the per-tick recycle
    // transform; only the scalar block above needed to run.
    if (m_regen < 0.5)
        return;

    // GRAIN SIZE/SHARPNESS + OPERATING POINT (re-anchored 2026-06-03, grain-stock
    // calibration). The generator MODEL is fine: tested against real 4K film plates,
    // its own grain at sigma ~0.75 green matches a 35mm scan to the FFT-ideal,
    // holding broadband energy to Nyquist. It only looked "soft" because it OPERATED
    // at the cratering neutral sigma 1.0 (+ the old 0.85 strength + source_soften
    // WIDENING). Fixes:
    //  - neutral sigma -> K_NEUTRAL (crisp 4K-film-scan default = "crisper everywhere"),
    //  - source_soften no longer widens (it fought crispness),
    //  - sigma scales by OUTPUT_height/GRAIN_RES_REF so the 4K-scan character holds
    //    at constant visual angle on ANY display,
    //  - clamp opened to [0.45,1.60]: sharper toward the zero-blur limit, coarser to 16mm.
    // gs = grain_sharpness*match_grain is the GLOBAL CRISPNESS DIAL and the A/B guard:
    // gs=0 (match off, or the dial at 0) -> k_neutral=1.0, k_size=1.0, no soften
    // change, no res scale => BIT-IDENTICAL to the old soft Light floor. gs=1 (default)
    // -> full 4K-scan crisp. The per-source map (gs_k_tgt) nudges finer/coarser around
    // the neutral by MEASURED gauge (sharper grain -> smaller sigma -> the sandpaper
    // bite; coarser -> larger). m_grain_fine = lap1^2/lap3^2 at LUMA, HIGH=fine/sharp.
    // fineR ceiling raised to 1.0 (white-noise level) and k floor to 0.40 so the
    // sharp end reaches very FINE digital-anime grain (finer/sharper than any film
    // stock), not just 35mm. Zero-blur isn't reached in practice but the RANGE is
    // there (white-noise fineR~1.0 -> k~0.43 -> sigma ~0.43 green @4K).
    float gs_fineR = clamp(m_grain_fine, 0.03, 1.0);
    // Size law  sigma = (R0 / fineR)^EXP, clamped [KLO,KHI]. With grid_snap the gated
    // fineR matches the clean-field fineR the autotune tuned against, so use the TUNED
    // law (mgt_autotune: R0 .135 EXP .317 clamp .639,1.571; size error -66%, LOSO 7/8
    // sizes). grid_snap=0 -> old half-pixel fineR -> the OLD hand-set law = the exact
    // pre-snap validated look (A/B; the two move together so neither half is orphaned).
    float law_R0  = (grid_snap > 0.5) ? 0.135 : 0.099;
    float law_EXP = (grid_snap > 0.5) ? 0.317 : 0.364;
    float law_KLO = (grid_snap > 0.5) ? 0.639 : 0.40;
    float law_KHI = (grid_snap > 0.5) ? 1.571 : 1.60;
    float gs_k_tgt = clamp(pow(law_R0 / gs_fineR, law_EXP), law_KLO, law_KHI);
    float gs_fire = smoothstep(0.12, 0.30, w);
    float gs = grain_sharpness * match_grain;
    // RESOLUTION NORMALIZATION (two different scalings, folded into the k mix).
    // Grain is generated at the FIXED 3840x2160 grid = GRAIN_RES_REF (the "4K
    // scan"), then the composite SAMPLES it scaled by grid_h/output_h -> so the
    // sigmas here are all in 4K-REFERENCE texels, and the composite converts to
    // display size (constant visual angle):
    //  - NEUTRAL (the clean-content 4K-film-scan default) is a fixed 4K-reference
    //    sigma -> vis_scale is identically 1 (the grid IS the 4K ref); the
    //    composite scale then makes it constant visual angle on any display, so
    //    on a 2160-tall display it matches the old OUTPUT-res generation exactly.
    //  - PER-SOURCE (gs_k_tgt) is measured in SOURCE-LUMA pixels -> scale by
    //    GRID/SOURCE so the grain (at the 4K grid) matches the source's grain
    //    stretched to 4K; the composite scale then carries it to the display.
    float src_h = (m_source_height > 1.0) ? m_source_height : HOOKED_size.y;
    float vis_scale = 1.0;                                        // grid IS the 4K ref
    float src_scale = 2160.0 / src_h;                             // GRID / SOURCE (2160 MUST equal GEN_GRID.y / //HEIGHT / //SIZE)
    float k_neutral = mix(1.0, K_NEUTRAL * vis_scale, gs);
    float k_size = mix(k_neutral, gs_k_tgt * src_scale, gs * gs_fire);
    // grain_size: live size multiplier (1.0 = calibrated). <1 finer, >1 coarser. Lets
    // you correct the per-source size by eye -- CyberCity currently renders too FINE
    // (fineR read its hard edges as fineness); raising size also lets the bandpass bite.
    k_size *= grain_size;
    float soften_eff = source_soften * (1.0 - gs);

    // CONTRAST/BANDPASS: build inner (s1) AND outer (s2 = BP_RATIO*s1) blur weights.
    // bp_alpha is UNIFORM across the workgroup (grain_contrast/match_grain/gs_fire are
    // all uniform), so the bp_alpha>0 branch below is uniform control flow -> the
    // barriers inside it are legal. bp_alpha=0 -> outer blur skipped, bp_norm=1,
    // grain = blur(s1) = the old lowpass generator, bit-identical (A/B-safe).
    float bp_alpha = BP_ALPHA * grain_contrast * match_grain * gs_fire;
    if (lid < uint(2 * MAX_TAPS + 1)) {
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
    if (lid == 0u) {
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
        // CHROMA layer scale (uniform, once/frame). The chroma is a coarse, INDEPENDENT,
        // blue-biased, luma-removed per-channel field that REUSES the DoG outer weights
        // (dyn_*2, already per-channel blue-biased coarse) -> no extra weight math. Normalize
        // the outer-blur RMS (chroma noise RMS 0.2498 * s2c[c]) to unit, then scale to
        // chroma_amp x luma-grain-RMS, blue-biased by CHROMA_W. GATED by gs_fire*match_grain
        // (mono unless grain confidently fires). chroma_amp=0 -> 0 -> exact mono (A/B-safe).
        // Amplitude/blue-bias match the offline-validated render_grain_rgb (vs the t3 GT).
        const vec3 CHROMA_W = vec3(0.8, 0.9, 1.4);
        float luma_rms = dot(luma_coeff, vec3(vsum_sigma[0], vsum_sigma[1], vsum_sigma[2]));
        float chroma_eff = chroma_amp * match_grain * gs_fire;
        for (int cc = 0; cc < 3; cc++) {
            chroma_scale[cc] = (s2c[cc] > 1e-8)
                ? chroma_eff * luma_rms * CHROMA_W[cc] / (0.2498 * s2c[cc]) : 0.0;
        }
    }
    barrier();

    // --- inner blur(s1): generate noise -> separable blur -> vsum1 ---
    for (uint i = lid; i < isize.y * isize.x; i += num_threads) {
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
    for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
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
    for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
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
        for (uint i = lid; i < isize.y * isize.x; i += num_threads) {
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
        for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
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
        for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
            vsum2_r += dyn_wr2[y] * grain_r[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
            vsum2_g += dyn_wg2[y] * grain_g[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
            vsum2_b += dyn_wb2[y] * grain_b[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
        }
    }

    // --- CHROMA layer: a coarse, INDEPENDENT, blue-biased per-channel field. REUSES the DoG
    // outer weights (dyn_*2 -- already per-channel blue-biased coarse) so no extra weight math;
    // the only new cost is one noise-gen + h/v blur. Per-channel-INDEPENDENT noise (distinct
    // seeds, also independent of the luma grain above) -> the cross-channel decorrelation that
    // makes it chroma. Added LUMA-REMOVED after the warp so the luma grain is untouched.
    // Unconditional (barriers can't sit in varying flow); chroma_scale=0 (chroma_amp=0 or
    // not firing) -> zero contribution -> exact mono. ---
    float chroma_raw_r = 0.0, chroma_raw_g = 0.0, chroma_raw_b = 0.0;
    barrier();
    {
        for (uint i = lid; i < isize.y * isize.x; i += num_threads) {
            uvec2 local_pos = uvec2(i % isize.x, i / isize.x);
            ivec2 global_coord_i = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy)
                                 + ivec2(local_pos) - ivec2(MAX_TAPS);
            uvec2 global_pos = uvec2((global_coord_i % GEN_GRID + GEN_GRID) % GEN_GRID);
            uint cbase = (global_pos.x * 1664525u) + (global_pos.y * 22695477u)
                       + (frame_seed * 314159265u);
            uint cr_s = cbase + 0x68bc21ebu;     // 3 INDEPENDENT per-channel streams (distinct
            uint cg_s = cbase + 0x02e5be93u;     // seeds -> decorrelated chroma, also distinct
            uint cb_s = cbase + 0x967a889bu;     // from the luma grain -> pure chroma layer)
            grain_r[local_pos.y][local_pos.x] = rand_triangular(cr_s, 1.0);
            grain_g[local_pos.y][local_pos.x] = rand_triangular(cg_s, 1.0);
            grain_b[local_pos.y][local_pos.x] = rand_triangular(cb_s, 1.0);
        }
        barrier();
        for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
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
        for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
            chroma_raw_r += dyn_wr2[y] * grain_r[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
            chroma_raw_g += dyn_wg2[y] * grain_g[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
            chroma_raw_b += dyn_wb2[y] * grain_b[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
        }
    }

    // PLANNED — luma-keyed grain COARSENESS (per-pixel size variation, ~free):
    // vsum1 (sigma s1) and vsum2 (1.8*s1) are the SAME noise at two sizes, both
    // already computed. A luma-keyed per-pixel mix between them varies apparent
    // grain size across the tone range (real stock grain is not size-constant
    // in luma) at the cost of one mix() — no extra generation. Needs: analytic
    // RMS renorm as a function of the blend (Var(mix) from the s1c/s2c/s12c
    // sums already computed above, same algebra as bp_norm), an interaction
    // story with the DoG (blend the pair pre- or post-combine), and the
    // direction/curve taken from real plates (stocks differ) — plate-matching
    // work, so parked until reference material is at hand.
    // bandpass combine (DoG = blur(s1) - a*blur(s2)), RMS-normalized so strength holds.
    float vsum_r = bp_norm[0] * (vsum1_r - bp_alpha * vsum2_r);
    float vsum_g = bp_norm[1] * (vsum1_g - bp_alpha * vsum2_g);
    float vsum_b = bp_norm[2] * (vsum1_b - bp_alpha * vsum2_b);

    // VALUE-DOMAIN contrast (value_warp): tanh the grain toward a bimodal/high-per-grain-
    // contrast marginal -- the CyberCity "harsh" character (negative excess-kurtosis).
    // Per channel: warped = sigma*renorm*tanh(warp * vsum / sigma) -> reshapes the value
    // distribution while preserving RMS (sigma + renorm computed above). value_warp=0 is
    // skipped -> BIT-IDENTICAL (A/B-safe). The value-domain cousin of grain_contrast.
    if (value_warp > 0.05) {
        vsum_r = vsum_sigma[0] * warp_renorm * tanh(value_warp * vsum_r / max(vsum_sigma[0], 1e-6));
        vsum_g = vsum_sigma[1] * warp_renorm * tanh(value_warp * vsum_g / max(vsum_sigma[1], 1e-6));
        vsum_b = vsum_sigma[2] * warp_renorm * tanh(value_warp * vsum_b / max(vsum_sigma[2], 1e-6));
    }

    // Add the CHROMA layer to the per-channel grain, LUMA-REMOVED: subtract the chroma's luma
    // part cl so REC709(vsum) is UNCHANGED -> the validated luma grain is untouched (chroma is
    // pure hue fluctuation). chroma_scale=0 (chroma_amp=0 / not firing) -> exact no-op.
    {
        float cs_r = chroma_raw_r * chroma_scale[0];
        float cs_g = chroma_raw_g * chroma_scale[1];
        float cs_b = chroma_raw_b * chroma_scale[2];
        float cl = dot(luma_coeff, vec3(cs_r, cs_g, cs_b));
        vsum_r += cs_r - cl;
        vsum_g += cs_g - cl;
        vsum_b += cs_b - cl;
    }

    // Final field store. rgb = the finished signed grain (bandpass + warp +
    // chroma); a = the pre-warp LOWPASS grain's luma so the debug A/B strips
    // can still show lowpass-vs-bandpass without a second saved texture.
    // (The composite-side scalars + the dispatch-trigger write moved above the
    // regen gate — they must run on every source frame, this store must not.)
    float lowpass_luma = dot(vec3(vsum1_r, vsum1_g, vsum1_b), luma_coeff);
    // Write the finished field into the PERSISTENT storage texture (survives across
    // presents). The 4K trigger dispatch rounds 2160 up to 2176 rows (COMPUTE 32),
    // so guard against the 16 out-of-range rows: an OOB imageStore is a spec-legal
    // silent no-op on both Vulkan and D3D11, but gate it explicitly rather than lean
    // on backend drop behavior (no barrier follows, so the divergent branch is safe).
    ivec2 gpos = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(gpos, imageSize(GRAIN_FIELD))))
        imageStore(GRAIN_FIELD, gpos, vec4(vsum_r, vsum_g, vsum_b, lowpass_luma));
}

//!HOOK OUTPUT
//!BIND HOOKED
//!BIND GRAIN_STATE
//!BIND GRAIN_FIELD
//!COMPUTE 32 32
//!DESC Film Grain Match: OUTPUT composite + debug

// The per-present half: this is the shader's only pass in mpv's REDRAW group
// (re-runs per present -- up to display Hz under display-resample), so it
// stays fetch + key + apply. All grain generation and every scalar that used
// to be derived here (eff_render/mid/steep, conf, shape_w) now come from the
// source-locked gen pass via GRAIN_STATE / GRAIN_FIELD.
#define DENSITY_GAIN 1.0
#define TUKEY_SCALE  0.459

const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);

// --- HDR (PQ BT.2020 output) domain bridge, grain_hdr=1 — for the CelFlare
// chain. The grain model is measured on gamma-encoded SDR source codes, but
// under the sdr-to-hdr retag the OUTPUT pixels are true PQ (CelFlare decodes,
// expands, PQ-encodes). Bridge per pixel: PQ code -> nits -> SDR-equivalent
// 2.4-gamma code vs grain_ref_white; key + apply the model there; re-encode.
// Grain lands exactly as the SDR path wherever the image sits at SDR levels
// (CelFlare holds midtones) and fades to zero shortly above reference white
// (the measured bell extrapolates; grain does not persist into expanded
// highlight cores). ST 2084
// constants match CelFlare's own encode so the round-trip shares a transfer.
float pq_eotf_nits(float e) {
    float p = pow(e, 1.0 / 78.84375);
    return 10000.0 * pow(max(p - 0.8359375, 0.0)
                         / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}
float pq_oetf_code(float nits) {
    float y = pow(clamp(nits / 10000.0, 0.0, 1.0), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), 78.84375);
}

float grain_scale(float lum, float mid, float steepness) {
    float d2 = steepness * (lum - mid) * (lum - mid);
    float t = 1.0 - d2 * TUKEY_SCALE;
    float curve = t > 0.0 ? t * t : 0.0;
    float protection = smoothstep(0.0, 0.12, lum);
    return curve * protection;
}

// Wrapped POINT fetch of the toroidal grain field. GRAIN_FIELD is a persistent
// storage image, so it is read via imageLoad (point-only — imageLoad has no
// bilinear path). At integer fpos (any 2160-tall display -> gscale 1.0) this is
// the exact texel, bit-identical to the old fetch. Sub/over-4K displays (gscale
// != 1.0) get a point-sampled (nearest) scale instead of the split's bilinear —
// slightly aliased grain scaling on non-2160 displays only; the author's targets
// are 2160-tall so the on-screen result is unchanged. The wrap (%) tiles the
// toroidal field when the scaled extent exceeds the grid (ultrawide / >4K width).
vec4 grain_point(vec2 fpos, ivec2 g) {
    ivec2 i = ivec2(floor(fpos - 0.5));
    ivec2 a = (i % g + g) % g;                  // component-wise wrap
    return imageLoad(GRAIN_FIELD, a);
}

// Same PCG as the measure/gen units — the recycle transform below needs a few
// decorrelated words per visible tick.
uint pcg_hash(uint s) {
    uint state = s * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void hook() {
    vec4 color = HOOKED_tex(HOOKED_pos);

    // Source-locked grain, SCALED to the display so it tracks image scale like
    // a real 4K source (constant visual angle), not a fixed display-pixel size.
    // gscale = grid_height / output_height, applied UNIFORMLY to both axes so
    // grain stays square; horizontal over-extent tiles via the wrap. On any
    // 2160-tall display (both the author's) gscale == 1.0 -> integer fpos ->
    // exact texel, crisp, bit-identical to the old 1:1 fetch. Sub-4K (gscale>1)
    // point-downsamples -> finer grain that shrinks with the display; >4K
    // (gscale<1) point-upsamples. Grid dims come from imageSize(GRAIN_FIELD) (not
    // a literal) so the composite can't desync from the gen pass's WIDTH/HEIGHT
    // directives (the cross-TU "constant sync" footgun; audit 2026-07-06).
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 gsize = imageSize(GRAIN_FIELD);
    float gscale = float(gsize.y) / max(HOOKED_size.y, 1.0);
    vec2 fpos = (vec2(gid) + 0.5) * gscale;
    // FIELD RECYCLE TRANSFORM (grain_gen_rate < 1). On visible ticks the gen
    // pass skipped, present a per-tick randomized toroidal shift + flip of the
    // standing field: offsets are hashed independently per tick (no coherent
    // drift, so nothing can read as motion/swimming) and displace uniformly
    // over the whole grid — far beyond the grain correlation length — so the
    // per-pixel frame-to-frame correlation is ~0, i.e. statistically fresh
    // grain for the cost of a few ALU ops. Flips kill any large-scale texture
    // familiarity; 90-degree rotations are excluded (the grid is not square).
    // The float mod() keeps the flip input in [0,grid) so the wrap in
    // grain_point never sees a negative coordinate (int % is undefined on
    // negatives); flipping about the grid extent maps texel centres to texel
    // centres and offsets are whole texels, so the gscale==1.0 exact-texel
    // property survives the transform. sub == 0 — every tick at the default
    // grain_gen_rate 1.0, and every regen tick otherwise — is the identity:
    // bit-identical to prior behavior.
    float vseed = floor(m_gen_frame * grain_rate);
    float sub = vseed - m_field_seed;
    if (sub > 0.5) {
        uint h1 = pcg_hash(uint(vseed));
        uint h2 = pcg_hash(h1);
        uint h3 = pcg_hash(h2);
        vec2 gext = vec2(gsize);
        fpos = mod(fpos, gext);
        if ((h3 & 1u) != 0u) fpos.x = gext.x - fpos.x;
        if ((h3 & 2u) != 0u) fpos.y = gext.y - fpos.y;
        fpos += vec2(float(h1 % uint(gsize.x)), float(h2 % uint(gsize.y)));
    }
    vec4 gfield4 = grain_point(fpos, gsize);
    vec3 vsum = gfield4.rgb;

    // grain_hdr bridge: work in the measured SDR domain (see helpers above).
    // Clamp the PQ input codes — YUV->RGB overshoot above 1.0 explodes the
    // PQ EOTF. SDR-equivalent codes may exceed 1.0 where CelFlare expanded;
    // the tone bell extrapolates toward zero grain shortly above 1.0.
    bool hdr_bridge = grain_hdr > 0.5;
    vec3 work_rgb = color.rgb;
    if (hdr_bridge) {
        vec3 nits = vec3(pq_eotf_nits(clamp(color.r, 0.0, 1.0)),
                         pq_eotf_nits(clamp(color.g, 0.0, 1.0)),
                         pq_eotf_nits(clamp(color.b, 0.0, 1.0)));
        work_rgb = pow(max(nits / grain_ref_white, vec3(0.0)), vec3(1.0 / 2.4));
    }

    float color_luma = dot(work_rgb, luma_coeff);
    float tone_scale = grain_scale(color_luma, m_eff_mid, m_eff_steep);
    vec3 scale_vec = vec3(tone_scale);
    vec3 pre_grain = work_rgb;
    if (density_combine > 0.5)
        work_rgb *= exp(m_eff_render * DENSITY_GAIN * vsum * scale_vec);
    else
        work_rgb += m_eff_render * vsum * scale_vec;

    if (hdr_bridge) {
        // Grain is texture, not signal: it may never push a pixel above
        // max(its own pre-grain level, reference white). The SDR path gets
        // this clip for free downstream at code 1.0; without it a broad
        // bright-keyed bell + a large density excursion re-encodes into the
        // PQ range ABOVE ref white and a grain frame can flash toward
        // display peak (design-audit finding, 2026-07-04).
        work_rgb = min(work_rgb, max(pre_grain, vec3(1.0)));
        vec3 out_nits = grain_ref_white * pow(max(work_rgb, vec3(0.0)), vec3(2.4));
        color.rgb = vec3(pq_oetf_code(out_nits.r), pq_oetf_code(out_nits.g),
                         pq_oetf_code(out_nits.b));
    } else {
        color.rgb = work_rgb;
    }

    if (debug_match > 0.5) {
        const int X_OFF = 24, Y_OFF = 400, BW = 10, BH = 10;
        const int ANCHOR = 10, NBITS = 16, NROWS = 96;
        ivec2 gid = ivec2(gl_GlobalInvocationID.xy) - ivec2(X_OFF, Y_OFF);
        if (gid.x >= 0 && gid.y >= 0
            && gid.x < ANCHOR + NBITS * BW && gid.y < NROWS * BH) {
            int row = gid.y / BH;
            float v = 0.0;
            if      (row == 0) v = m_intensity_raw * 30000.0;
            else if (row == 1) v = m_mid * 65000.0;
            else if (row == 2) v = m_steepness * 7000.0;
            else if (row == 3) v = m_gen_frame;  // was retired m_sat; row layout stable
            else if (row == 4) v = m_log_avg * 1000000.0;
            else if (row == 5) v = m_measured;
            else if (row == 6) v = m_debug_tick;
            else if (row == 7) v = m_eff_render * 90000.0;
            else if (row == 8) v = m_conf * 65000.0;
            else if (row == 9) v = m_prev_ready * 60000.0;
            else if (row < 26) v = hist_amp[row - 10] * 2000000.0;
            else if (row == 40) v = hist_spatial[14];
            else if (row == 41) v = hist_spatial[15] * 65000.0;
            else if (row < 42) v = hist_spatial[row - 26] * 2000000.0;
            else if (row == 42) v = m_sat_frac * 65000.0;
            else if (row <= 46) v = 0.0;  // rows 43-46 retired (diag_sat_*) — layout kept stable
            else if (row < 61)  v = hist_temporal[row - 47] * 2000000.0;
            else if (row == 61) v = m_held_frames;
            else if (row == 62) v = m_distinct_frames;
            else if (row == 63) v = m_hold_run;
            else if (row == 64) v = m_content_held;
            else if (row == 65) v = m_content_distinct;
            else if (row == 66) v = m_content_hold_run;
            else if (row == 67) v = m_content_log_avg * 1000000.0;
            else if (row == 68) v = m_coherence * 65000.0;
            else if (row == 69) v = m_coh_frac * 65000.0;
            else if (row == 70) v = m_coh_samples;
            else if (row == 71) v = m_measure_swatch_r * 65000.0;
            else if (row == 72) v = m_measure_swatch_g * 65000.0;
            else if (row == 73) v = m_measure_swatch_b * 65000.0;
            else if (row == 74) v = m_measure_stage * 65000.0;
            else if (row == 75) v = m_eff_mid * 65000.0;
            else if (row == 76) v = m_eff_steep * 7000.0;
            else if (row == 77) v = m_shape_w * 65000.0;
            else if (row < 92)  v = hist_held[row - 78] * 2000000.0;
            else if (row == 92) v = m_held_count;
            else if (row == 93) v = m_film_level * 2000000.0;
            else if (row == 94) v = m_grain_fine * 60000.0;
            else                v = m_pan_px * 10000.0;
            uint val = uint(clamp(v, 0.0, 65535.0));
            if (gid.x < ANCHOR)
                color.rgb = vec3(1.0);
            else {
                int b = (gid.x - ANCHOR) / BW;
                uint bit = (val >> uint(NBITS - 1 - b)) & 1u;
                color.rgb = (bit == 1u) ? vec3(1.0) : vec3(0.0);
            }
        }
        ivec2 swatch_pos = ivec2(gl_GlobalInvocationID.xy) - ivec2(X_OFF, Y_OFF + NROWS * BH + 10);
        if (swatch_pos.x >= 0 && swatch_pos.x < 60 && swatch_pos.y >= 0 && swatch_pos.y < 60)
            color.rgb = vec3(m_measure_swatch_r, m_measure_swatch_g, m_measure_swatch_b);
        else if (swatch_pos.x >= 70 && swatch_pos.x < 130 && swatch_pos.y >= 0 && swatch_pos.y < 60)
            color = HOOKED_tex(vec2(0.5, 0.5));

        // GRAIN-ON-CLEAN patches (right of the readout). The real source is already
        // grainy, so A/B is hard; these show the rendered grain on a FLAT base. Each
        // 120px patch = three 40px strips: CLEAN | LOWPASS | BANDPASS, so the bandpass
        // character is visible directly vs clean and vs the old lowpass. Slots: source-
        // center color, then gray 0.20 / 0.45 / 0.70 (grain is midtone-peaked -> mid shows
        // most). Crank Ctrl+Shift+F3 (gain) to make it clearly visible.
        ivec2 pp = ivec2(gl_GlobalInvocationID.xy) - ivec2(X_OFF + 210, Y_OFF);
        if (pp.x >= 0 && pp.x < 120 && pp.y >= 0 && pp.y < 430) {
            int slot = pp.y / 110;
            if (pp.y - slot * 110 < 100) {
                vec3 base = (slot == 0) ? vec3(m_measure_swatch_r, m_measure_swatch_g, m_measure_swatch_b)
                          : (slot == 1) ? vec3(0.20)
                          : (slot == 2) ? vec3(0.45)
                          :               vec3(0.70);
                float ts = grain_scale(dot(base, luma_coeff), m_eff_mid, m_eff_steep);
                // thirds: CLEAN | LOWPASS (blur s1) | BANDPASS (current grain_contrast),
                // so the bandpass character shows directly vs clean AND vs the old lowpass
                // without toggling. Crank Ctrl+Shift+F3 (gain) to make it clearly visible.
                // Lowpass strip is MONO since the split (GRAIN_FIELD.a carries only the
                // lowpass LUMA); the bandpass strip stays full per-channel.
                vec3 gfield = (pp.x < 40) ? vec3(0.0)
                            : (pp.x < 80) ? vec3(gfield4.a)
                            :               vsum;
                vec3 outc = base;
                if (density_combine > 0.5)
                    outc *= exp(m_eff_render * DENSITY_GAIN * gfield * ts);
                else
                    outc += m_eff_render * gfield * ts;
                color.rgb = outc;
            }
        }
    }

    imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
}
