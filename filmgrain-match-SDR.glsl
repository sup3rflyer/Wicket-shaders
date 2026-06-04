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
//   PASS 1 - Compute 32x32 at LUMA: one fixed 16:9 workgroup samples HOOKED.r
//            directly, builds temporal/spatial grain state, and writes only
//            GRAIN_STATE. No textures are passed across stages.
//   PASS 2 - Fragment passthrough at OUTPUT: captures display-resolution frame
//            as GRAIN_SRC for rendering/debug only.
//   PASS 3 - Compute 32x32 at OUTPUT: renders grain from GRAIN_SRC using the
//            LUMA-measured GRAIN_STATE.
//
// Runtime params (toggled live via glsl-shader-opts; see shader-toggle.lua):
//   match_grain   F3       0 = pure Light floor (bit-identical to the tier),
//                          1 = matched. mix() between, so A/B is non-destructive.
//   debug_match   Ctrl+F3  machine-readable state readout.
//   grain_sharpness Alt+F3  per-source grain sharpness ("sandpaper"<->soft) from
//                          the measured source character; 0 = prior fixed size.
//   restore_gain           upward extrapolation past surviving grain (the
//                          compensatory knob). 1.0 = match only what survived.
//   pan_freeze             1 = freeze the grain-amplitude RISE during a real global
//                          pan (kills the canvas-pan overfire build-up; grain already
//                          established HOLDS). 0 = prior behaviour. A/B-safe.
//   grain_restore_taper    1 = taper restore_gain->1 as measured amplitude rises
//                          (Lain over-apply fix). 0 = off (default).
//   density_combine        0 = additive, 1 = multiplicative density (default).
//   state_epoch            harness reset token; bump once per source file.
// ============================================================================

//!PARAM match_grain
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM debug_match
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM restore_gain
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 4.0
3.3

//!PARAM density_combine
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM state_epoch
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 65535.0
0.0

//!PARAM grain_attack
//!TYPE float
//!MINIMUM 0.002
//!MAXIMUM 0.30
0.02

//!PARAM grain_decay
//!TYPE float
//!MINIMUM 0.002
//!MAXIMUM 0.30
0.04

//!PARAM hardcut_frac
//!TYPE float
//!MINIMUM 0.40
//!MAXIMUM 1.0
0.85

//!PARAM restore_floor
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.40

//!PARAM grain_sharpness
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM grain_restore_taper
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM pan_freeze
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!BUFFER GRAIN_STATE
//!VAR float m_intensity_raw
//!VAR float m_mid
//!VAR float m_steepness
//!VAR float m_sat
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
//!VAR float diag_sat_amp_lo
//!VAR float diag_sat_amp_hi
//!VAR float diag_sat_count_lo
//!VAR float diag_sat_count_hi
//!VAR float hist_held[16]
//!VAR float m_held_count
//!VAR float m_film_level
//!VAR float m_grain_fine
//!VAR float m_source_height
//!VAR float m_pan_px
//!VAR float prev_grid[9216]
//!VAR float prev_grid_off[9216]
//!STORAGE

//!HOOK LUMA
//!BIND HOOKED
//!BIND GRAIN_STATE
//!COMPUTE 32 32
//!DESC Film Grain Match: LUMA measure

#define GRAIN_RATE 0.5
#define MAX_TAPS 3

#define INT_FLOOR    0.10
#define INT_CEIL     0.65
#define MID_FLOOR    0.15
#define STEEP_FLOOR  2.6
#define SATURATION   0.25

#define CONF_LOW     0.10
#define CONF_HIGH    0.35
#define DENSITY_GAIN 2.0


#define RED_VARIANCE_SCALE 1.0
#define GREEN_VARIANCE_SCALE 1.0
#define BLUE_VARIANCE_SCALE 1.0

#define RED_SATURATION 0.6
#define GREEN_SATURATION 0.5
#define BLUE_SATURATION 0.4

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
#define SOFTEN_TEMPORAL_RATIO_LO 1.20
#define SOFTEN_TEMPORAL_RATIO_HI 2.50
#define SOFTEN_INTENSITY_HI 0.24
#define SOFTEN_MAX_BLEND 0.35
#define TEMPORAL_TONE_STEEPEN 1.75
#define SHAPE_AMP_LOW     0.08
#define SHAPE_AMP_HIGH    0.25
#define SHAPE_MAX_BLEND   0.40
#define SAT_LO_CEIL       0.18
#define SAT_HI_FLOOR      0.45
#define SAT_MIN_CELLS     96.0
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
#define STATE_MAGIC   0.5567819

#define STEEP_MIN     0.6
#define STEEP_MAX     9.0

const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
const uvec2 isize = uvec2(gl_WorkGroupSize) + uvec2(2 * MAX_TAPS);

shared float grain_r[isize.y][isize.x];
shared float grain_g[isize.y][isize.x];
shared float grain_b[isize.y][isize.x];
shared float dyn_wr[2 * MAX_TAPS + 1];
shared float dyn_wg[2 * MAX_TAPS + 1];
shared float dyn_wb[2 * MAX_TAPS + 1];
shared uint s_hist[NBINS][AMP_BINS];
shared uint s_sat_hist[2][AMP_BINS];
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

float rgb_saturation(vec3 rgb) {
    float mx = max(rgb.r, max(rgb.g, rgb.b));
    float mn = min(rgb.r, min(rgb.g, rgb.b));
    return mx > 1e-5 ? (mx - mn) / mx : 0.0;
}

float gaussian_weight(float dx, float sigma) {
    return exp(-0.5 * dx * dx / max(sigma * sigma, 1e-6));
}

float measure_luma(vec2 uv) {
    return HOOKED_tex(uv).r;
}

float measure_saturation(vec2 uv) {
    return 0.0;
}

void hook() {
    uint lid = gl_LocalInvocationIndex;
    uint num_threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    uvec2 measure_wg = uvec2(min(30u, gl_NumWorkGroups.x - 1u), min(16u, gl_NumWorkGroups.y - 1u));
    bool is_measure_wg = (gl_WorkGroupID.xy == measure_wg);

    bool state_ok = abs(m_state_magic - STATE_MAGIC) < 0.0001
                 && abs(m_state_epoch - state_epoch) < 0.5;
    bool prev_ready = state_ok && m_prev_ready > 0.5;

    // =========================================================================
    // MEASUREMENT — center workgroup samples HOOKED_tex across full frame
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
        for (uint k = lid; k < uint(2 * AMP_BINS); k += num_threads)
            s_sat_hist[k / uint(AMP_BINS)][k % uint(AMP_BINS)] = 0u;
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
                float x3m = measure_luma(uv - vec2(3.0 * HOOKED_pt.x, 0.0));
                float x3p = measure_luma(uv + vec2(3.0 * HOOKED_pt.x, 0.0));
                float y3m = measure_luma(uv - vec2(0.0, 3.0 * HOOKED_pt.y));
                float y3p = measure_luma(uv + vec2(0.0, 3.0 * HOOKED_pt.y));
                // Grain-energy gate: skip grain-poor flat samples (compression
                // smooths grain in flats), so the ratio reflects grain, not residual.
                float e1 = (cur_luma - xp) * (cur_luma - xp) + (cur_luma - xm) * (cur_luma - xm)
                         + (cur_luma - yp) * (cur_luma - yp) + (cur_luma - ym) * (cur_luma - ym);
                if (e1 > 1.6e-5) {
                    // LAPLACIAN (2nd difference) at 1px vs 3px. The 2nd difference
                    // cancels the smooth linear gradient that biased a raw diff
                    // ratio coarse, isolating grain. ratio lap1^2/lap3^2 (formed in
                    // the reduction): HIGH = fine/sharp (sandpaper), LOW = coarse/
                    // soft. Validated offline (lap_g, Spearman +0.90 vs 16-bit GT).
                    float lap1 = 4.0 * cur_luma - (xm + xp + ym + yp);
                    float lap3 = 4.0 * cur_luma - (x3m + x3p + y3m + y3p);
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
                atomicAdd(s_sat_hist[0][cab], 1u);
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
            m_sat = 1.0; m_log_avg = 0.0; m_measured = 0.0; m_debug_tick = 0.0;
            m_prev_ready = 1.0; m_state_magic = STATE_MAGIC; m_state_epoch = state_epoch; m_sat_frac = 0.0;
            m_held_frames = 0.0; m_distinct_frames = 0.0; m_hold_run = 0.0;
            m_content_held = 0.0; m_content_distinct = 0.0; m_content_hold_run = 0.0; m_content_log_avg = 0.0;
            m_coherence = 0.0; m_coh_frac = 0.0; m_coh_samples = 0.0;
            diag_sat_amp_lo = 0.0; diag_sat_amp_hi = 0.0;
            diag_sat_count_lo = 0.0; diag_sat_count_hi = 0.0;
            m_held_count = 0.0; m_film_level = 0.0; m_pan_px = 0.0;
            for (int b = 0; b < NBINS; b++) {
                hist_amp[b] = 0.0; hist_temporal[b] = 0.0; hist_spatial[b] = 0.0;
                hist_held[b] = 0.0;
            }
            s_update_prev_grid = 1u;
        } else {
            m_debug_tick += 1.0;

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
                float c = float(s_sat_hist[0][ab]);
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
        for (uint k = lid; k < uint(2 * AMP_BINS); k += num_threads)
            s_sat_hist[k / uint(AMP_BINS)][k % uint(AMP_BINS)] = 0u;
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
            float sat = measure_saturation(uv);

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
                if (sat <= SAT_LO_CEIL)
                    atomicAdd(s_sat_hist[0][ab], 1u);
                else if (sat >= SAT_HI_FLOOR)
                    atomicAdd(s_sat_hist[1][ab], 1u);
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

            float sat_count_lo = 0.0, sat_count_hi = 0.0;
            for (int ab = 0; ab < AMP_BINS; ab++) {
                sat_count_lo += float(s_sat_hist[0][ab]);
                sat_count_hi += float(s_sat_hist[1][ab]);
            }

            float sat_amp_lo = 0.0, sat_amp_hi = 0.0;
            if (sat_count_lo >= SAT_MIN_CELLS) {
                float target = SPATIAL_PCTL * sat_count_lo, acc = 0.0;
                for (int ab = 0; ab < AMP_BINS; ab++) {
                    float c = float(s_sat_hist[0][ab]);
                    if (acc + c >= target) {
                        sat_amp_lo = (float(ab) + (target - acc) / max(c, 1.0)) * binw;
                        break;
                    }
                    acc += c;
                }
            }
            if (sat_count_hi >= SAT_MIN_CELLS) {
                float target = SPATIAL_PCTL * sat_count_hi, acc = 0.0;
                for (int ab = 0; ab < AMP_BINS; ab++) {
                    float c = float(s_sat_hist[1][ab]);
                    if (acc + c >= target) {
                        sat_amp_hi = (float(ab) + (target - acc) / max(c, 1.0)) * binw;
                        break;
                    }
                    acc += c;
                }
            }

            diag_sat_amp_lo = sat_amp_lo;
            diag_sat_amp_hi = sat_amp_hi;
            diag_sat_count_lo = sat_count_lo;
            diag_sat_count_hi = sat_count_hi;

            float sat_response = 1.0;
            if (sat_amp_lo > 0.0 && sat_amp_hi > 0.0)
                sat_response = clamp(sat_amp_hi / sat_amp_lo, 0.35, 1.25);

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
            // removing grain is never "overkill". Tone-shape params (mid/steep/sat)
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
                m_mid = mid; m_steepness = steep; m_intensity_raw = irraw; m_sat = sat_response;
            } else {
                m_mid = mix(m_mid, mid, alpha);
                m_steepness = mix(m_steepness, steep, alpha);
                m_intensity_raw = mix(m_intensity_raw, irraw, int_alpha);
                m_sat = mix(m_sat, sat_response, alpha);
            }
            m_measured += 1.0;
        }
    }
    barrier();

    vec4 pass_color = HOOKED_tex(HOOKED_pos);
    imageStore(out_image, ivec2(gl_GlobalInvocationID), pass_color);
}

//!HOOK OUTPUT
//!BIND HOOKED
//!SAVE GRAIN_SRC
//!DESC Film Grain Match: source capture
vec4 hook() { return HOOKED_tex(HOOKED_pos); }

//!HOOK OUTPUT
//!BIND GRAIN_SRC
//!BIND GRAIN_STATE
//!COMPUTE 32 32
//!DESC Film Grain Match: OUTPUT render + debug

#define GRAIN_RATE 1.0
// 9-tap support (was 7): the blue channel's base sigma (1.20) reaches ~1.15 even at
// the crisp neutral and higher when coarse, where 7 taps (clean only to sigma ~1.0)
// truncate the Gaussian into a box and ripple the spectrum. 9 taps hold sigma up to
// ~1.5 cleanly, covering the full fine-digital -> 16mm render range. Arrays/loops
// are parametrized on MAX_TAPS so the support can never drift out of sync.
#define MAX_TAPS 4

#define INT_FLOOR    0.10
#define INT_CEIL     0.65
#define MID_FLOOR    0.15
#define STEEP_FLOOR  2.6
#define SATURATION   0.25

// Crisp NEUTRAL render sigma scale (the no-fire 4K-film-scan default). 0.75 puts the
// green-channel sigma at ~0.75 px @4K = a 35mm scan (matched to the real grain
// plates; holds broadband energy to Nyquist instead of the old soft-Gaussian
// crater at sigma 1.0). The grain-stock probe found film spans ~0.75 (35mm) .. 1.5
// (16mm) @4K; the per-source map nudges around this neutral by measured gauge.
#define K_NEUTRAL    0.75
// Output res the crisp default is calibrated for (grain sigma scales by
// OUTPUT_height/REF so it reads as a 4K scan at constant visual angle on any display).
#define GRAIN_RES_REF 2160.0
// Per-channel render-sigma cap. The 9-tap support holds a Gaussian cleanly to ~1.5;
// beyond that it would truncate. The coarse extreme (low-fineR firing) and >4K
// res-scaling can push a channel past it, so cap GRACEFULLY (slightly finer than
// ideal at that extreme) rather than ripple the spectrum. The actual film range
// (fine digital .. ~16mm) stays under it, so this never touches normal operation.
#define SIGMA_MAX    1.5
// Lain restore-gain taper thresholds on m_intensity_raw: below LO = full compensatory
// restore_gain; above HI = ~x1 (grain already heavy/intact, don't over-restore).
// Only active when grain_restore_taper>0. Needs feel-test tuning.
#define RESTORE_TAPER_LO 0.15
#define RESTORE_TAPER_HI 0.30

#define CONF_LOW     0.10
#define CONF_HIGH    0.35
#define DENSITY_GAIN 2.0


#define RED_VARIANCE_SCALE 1.0
#define GREEN_VARIANCE_SCALE 1.0
#define BLUE_VARIANCE_SCALE 1.0

#define RED_SATURATION 0.6
#define GREEN_SATURATION 0.5
#define BLUE_SATURATION 0.4

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
#define SOFTEN_TEMPORAL_RATIO_LO 1.20
#define SOFTEN_TEMPORAL_RATIO_HI 2.50
#define SOFTEN_INTENSITY_HI 0.24
#define SOFTEN_MAX_BLEND 0.35
#define TEMPORAL_TONE_STEEPEN 1.75
#define SHAPE_AMP_LOW     0.08
#define SHAPE_AMP_HIGH    0.25
#define SHAPE_MAX_BLEND   0.40
#define SAT_LO_CEIL       0.18
#define SAT_HI_FLOOR      0.45
#define SAT_MIN_CELLS     96.0
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
#define STATE_MAGIC   0.5567819

#define STEEP_MIN     0.6
#define STEEP_MAX     9.0

const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
const uvec2 isize = uvec2(gl_WorkGroupSize) + uvec2(2 * MAX_TAPS);

shared float grain_r[isize.y][isize.x];
shared float grain_g[isize.y][isize.x];
shared float grain_b[isize.y][isize.x];
shared float dyn_wr[2 * MAX_TAPS + 1];
shared float dyn_wg[2 * MAX_TAPS + 1];
shared float dyn_wb[2 * MAX_TAPS + 1];


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

float rgb_saturation(vec3 rgb) {
    float mx = max(rgb.r, max(rgb.g, rgb.b));
    float mn = min(rgb.r, min(rgb.g, rgb.b));
    return mx > 1e-5 ? (mx - mn) / mx : 0.0;
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

    // Grain must animate every frame like real film grain — a held cel still
    // gets fresh grain each frame, because the film was exposed per frame. The
    // previous content-cadence seed froze grain on held cels of clean sources
    // (no grain + identical cel -> counter stalls -> static "dirt" look). Seed
    // from the per-frame counter so grain always dances at frame cadence. The
    // measured distinct counter is retained for held-grain-cadence detection
    // (twos/threes), deferred until source-grain cadence is measured directly.
    uint frame_seed = uint(max(0.0, floor(float(frame) * GRAIN_RATE)));

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
    float gs_k_tgt = clamp(pow(0.099 / gs_fineR, 0.364), 0.40, 1.60);
    float gs_fire = smoothstep(0.12, 0.30, w);
    float gs = grain_sharpness * match_grain;
    // RESOLUTION NORMALIZATION (two different scalings, folded into the k mix):
    //  - NEUTRAL (the clean-content 4K-film-scan default) is a fixed 4K-reference
    //    sigma -> scale by OUTPUT/GRAIN_RES_REF for constant visual angle on any display.
    //  - PER-SOURCE (gs_k_tgt) is measured in SOURCE-LUMA pixels -> scale by
    //    OUTPUT/SOURCE so the rendered grain matches the on-screen (libplacebo-scaled)
    //    image grain (a linear geometric stretch). NOT a "native 4K gauge" remap:
    //    we match what's actually on screen, which IS just the scaled source.
    // In the AnimeJaNai pipeline LUMA(post-upscale)=OUTPUT so both ratios are 1.0
    // (no-op); they only diverge when libplacebo scales the source to the display.
    float src_h = (m_source_height > 1.0) ? m_source_height : GRAIN_SRC_size.y;
    float vis_scale = GRAIN_SRC_size.y * (1.0 / GRAIN_RES_REF);   // OUTPUT / 4K-ref
    float src_scale = GRAIN_SRC_size.y / src_h;                   // OUTPUT / SOURCE
    float k_neutral = mix(1.0, K_NEUTRAL * vis_scale, gs);
    float k_size = mix(k_neutral, gs_k_tgt * src_scale, gs * gs_fire);
    float soften_eff = source_soften * (1.0 - gs);

    if (lid < uint(2 * MAX_TAPS + 1)) {
        int idx = int(lid);
        float dx = float(idx - MAX_TAPS);
        dyn_wr[idx] = gaussian_weight(dx, min(mix(0.78, 1.02, soften_eff) * k_size, SIGMA_MAX));
        dyn_wg[idx] = gaussian_weight(dx, min(mix(1.00, 1.22, soften_eff) * k_size, SIGMA_MAX));
        dyn_wb[idx] = gaussian_weight(dx, min(mix(1.20, 1.40, soften_eff) * k_size, SIGMA_MAX));
    }
    barrier();
    if (lid == 0u) {
        float nr = 0.0, ng = 0.0, nb = 0.0;
        for (int i = 0; i < 2 * MAX_TAPS + 1; i++) {
            nr += dyn_wr[i];
            ng += dyn_wg[i];
            nb += dyn_wb[i];
        }
        for (int i = 0; i < 2 * MAX_TAPS + 1; i++) {
            dyn_wr[i] /= nr;
            dyn_wg[i] /= ng;
            dyn_wb[i] /= nb;
        }
    }
    barrier();

    for (uint i = lid; i < isize.y * isize.x; i += num_threads) {
        uvec2 local_pos = uvec2(i % isize.x, i / isize.x);
        ivec2 global_coord_i = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy)
                             + ivec2(local_pos) - ivec2(MAX_TAPS);
        uvec2 global_pos = uvec2(global_coord_i);
        uint seed_init = (global_pos.x * 1664525u) + (global_pos.y * 22695477u)
                       + (frame_seed * 314159265u);
        float g_r = rand_triangular(seed_init, RED_VARIANCE_SCALE);
        float g_g = rand_triangular(seed_init, GREEN_VARIANCE_SCALE);
        float g_b = rand_triangular(seed_init, BLUE_VARIANCE_SCALE);
        float grain_lum = dot(vec3(g_r, g_g, g_b), vec3(0.299, 0.587, 0.114));
        grain_r[local_pos.y][local_pos.x] = mix(grain_lum, g_r, RED_SATURATION * SATURATION);
        grain_g[local_pos.y][local_pos.x] = mix(grain_lum, g_g, GREEN_SATURATION * SATURATION);
        grain_b[local_pos.y][local_pos.x] = mix(grain_lum, g_b, BLUE_SATURATION * SATURATION);
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

    float vsum_r = 0.0, vsum_g = 0.0, vsum_b = 0.0;
    for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
        vsum_r += dyn_wr[y] * grain_r[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
        vsum_g += dyn_wg[y] * grain_g[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
        vsum_b += dyn_wb[y] * grain_b[gl_LocalInvocationID.y + y][gl_LocalInvocationID.x + MAX_TAPS];
    }

    vec4 color = GRAIN_SRC_tex(GRAIN_SRC_pos);

    float color_luma = dot(color.rgb, luma_coeff);
    float color_sat = rgb_saturation(color.rgb);
    float tone_scale = grain_scale(color_luma, eff_mid, eff_steep);
    float eff_sat_response = mix(1.0, m_sat, w);
    float sat_gate = mix(1.0, eff_sat_response, smoothstep(SAT_LO_CEIL, SAT_HI_FLOOR, color_sat));
    vec3 vsum = vec3(vsum_r, vsum_g, vsum_b);
    vec3 scale_vec = vec3(tone_scale * sat_gate);
    if (density_combine > 0.5)
        color.rgb *= exp(eff_render * DENSITY_GAIN * vsum * scale_vec);
    else
        color.rgb += eff_render * vsum * scale_vec;

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
            else if (row == 3) v = m_sat * 65000.0;
            else if (row == 4) v = m_log_avg * 1000000.0;
            else if (row == 5) v = m_measured;
            else if (row == 6) v = m_debug_tick;
            else if (row == 7) v = eff_render * 90000.0;
            else if (row == 8) v = conf * 65000.0;
            else if (row == 9) v = m_prev_ready * 60000.0;
            else if (row < 26) v = hist_amp[row - 10] * 2000000.0;
            else if (row == 40) v = hist_spatial[14];
            else if (row == 41) v = hist_spatial[15] * 65000.0;
            else if (row < 42) v = hist_spatial[row - 26] * 2000000.0;
            else if (row == 42) v = m_sat_frac * 65000.0;
            else if (row == 43) v = diag_sat_amp_lo * 2000000.0;
            else if (row == 44) v = diag_sat_amp_hi * 2000000.0;
            else if (row == 45) v = diag_sat_count_lo;
            else if (row == 46) v = diag_sat_count_hi;
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
            else if (row == 75) v = eff_mid * 65000.0;
            else if (row == 76) v = eff_steep * 7000.0;
            else if (row == 77) v = shape_w * 65000.0;
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
            color = GRAIN_SRC_tex(vec2(0.5, 0.5));
    }

    imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
}
