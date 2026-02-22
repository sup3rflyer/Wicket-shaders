// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE

//!BUFFER SCENE_STATE
//!VAR float smoothed_avg
//!VAR float smoothed_max
//!VAR float smoothed_min
//!VAR float smoothed_log_avg
//!VAR float smoothed_contrast
//!VAR float smoothed_highlight_pop
//!VAR float scene_cut_lockout
//!VAR float prev_luma[96]
//!VAR float prev_cb[96]
//!VAR float prev_cr[96]
//!VAR float scene_highlight_peak
//!VAR float scene_exp_steepness
//!VAR float scene_knee_end
//!VAR float scene_master_knee
//!VAR float scene_type_val
//!VAR float scene_contrast
//!VAR float smoothed_avg_chroma
//!VAR float scene_B_highlight_peak
//!VAR float scene_B_exp_steepness
//!VAR float scene_B_knee_end
//!VAR float scene_B_master_knee
//!VAR float scene_B_type_val
//!VAR float scene_B_contrast
//!VAR float scene_exp_denom
//!VAR float scene_B_exp_denom
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!BIND SCENE_STATE
//!DESC CelFlare v2.5 (Scene-Adaptive SDR→HDR)

// =============================================================================
// USER CONTROLS
// =============================================================================

// --- Core Expansion ---
#define INTENSITY 1.5              // 0.5 = subtle, 1.0 = normal, 1.5+ = aggressive
#define CURVE_STEEPNESS 0.4        // 0.5 = gentle (lifts mids), 1.0 = adaptive, 1.5+ = punchy highlights

// --- Dynamic Intensity ---
// Per-scene multiplier on INTENSITY based on contrast. Flat/pastel scenes get gentler
// expansion, dramatic high-contrast scenes get more pop. Range kept tight to avoid
// visible fluctuation between dialogue cuts.
#define ENABLE_DYNAMIC_INTENSITY 1 // 0 = off (fixed INTENSITY), 1 = contrast-adaptive
#define DYN_INTENSITY_LOW  0.75    // Multiplier for flat/low-contrast scenes
#define DYN_INTENSITY_HIGH 1.15    // Multiplier for high-contrast/dramatic scenes
#define DYN_CONTRAST_LOW   2.2     // Contrast floor (stops) — below this = DYN_INTENSITY_LOW
#define DYN_CONTRAST_HIGH  5.5     // Contrast ceiling (stops) — above this = DYN_INTENSITY_HIGH

// --- Saturation Boost (Oklab) ---
// Counters the "silvery" desaturated look from expanding luminance without chroma.
// Uses Oklab color space for perceptually uniform chroma scaling with perfect hue preservation.
// Stevens' power law: perceived colorfulness ~ chroma * L^0.5, so chroma must scale by
// expansion^0.5 to maintain constant perceived colorfulness after luminance expansion.
// Applied uniformly across all saturation levels — multiplicative scaling naturally gives
// the largest absolute boost to the most saturated colors (faithful to colorist intent).
#define ENABLE_SAT_BOOST 1
#define SAT_BOOST_EXPONENT 0.267    // Compensates remaining gap: RGB expansion already provides cbrt(k) chroma, Stevens needs sqrt(k), so boost = k^(1/2 - 1/3) = k^(1/6)
#define SAT_BOOST_MAX 1.25         // Safety cap for very high expansion values
#define SAT_KNEE_OFFSET 0.10      // Extend chroma boost below luminance knee in linear luma (0 = disabled).
                                  // mix(luma, rgb) naturally gives zero change on neutrals and tapers
                                  // proportionally with existing chroma — no extra low-sat guard needed.

// --- Hue Correction (Oklab) ---
// Corrects hue shifts from saturation boost gamut clipping (e.g. yellow→green on fires).
// Blends post-expansion hue toward original in Oklab, preserving expanded chroma magnitude.
// Technique from renodx and PumboAutoHDR.
#define ENABLE_HUE_CORRECTION 0
#define HUE_CORRECTION_STRENGTH 1.0  // 0.0 = none, 1.0 = full

// --- Bezold-Brücke Warmth Compensation ---
// At higher output luminances, warm yellows and oranges perceptually shift toward green
// (Bezold-Brücke effect — M-cone gain increases relative to L-cone). Pre-rotates hues
// clockwise in Oklab ab plane to pre-compensate. Warm mask (b/(|a|+chroma)) targets
// yellow-orange while excluding reds (which need opposite correction) and blues (excluded
// naturally by positive-b gate). Scales with expansion as proxy for output luminance.
// NOTE: For peaks above ~400 nits a full hue map with atan2 is needed — at high luminance
// blues shift toward violet and reds need counter-clockwise correction.
#define ENABLE_BB_WARMTH 1
#define BB_WARMTH 0.1             // Rotation in radians (~3°); try 0.03–0.10

// --- Grain Stabilization ---
// Stabilizes expansion decision for grainy content by averaging similar neighbors.
// Operates in gamma space for perceptual uniformity.
#define ENABLE_GRAIN_STABLE 1

// --- Dither ---
// Masks 8-bit quantization artifacts that get amplified by expansion.
// Applied in PQ space (perceptually uniform).
#define ENABLE_DITHER 1
#define DITHER_STRENGTH 0.7        // Reduced (correct expansion needs less dither)
#define DITHER_TEMPORAL 1          // Animate noise per frame (integrates temporally)

// --- EOTF ---
// Gamma used for linearization. BT.1886 = 2.4 (broadcast standard), sRGB ≈ 2.2.
// If skin/orange appears darker with the shader than without, try lowering toward 2.2.
#define EOTF_GAMMA 2.3

// --- PQ Output ---
// Shader encodes to PQ BT.2020 directly to bypass libplacebo's SDR peak clipping.
// Must match hdr-reference-white in mpv.conf and Windows SDR brightness.
#define REFERENCE_WHITE 116.0
#define PQ_FAST_APPROX 1       // 0 = exact ST.2084, 1 = degree-7 polynomial (~2.4x faster PQ encoding)

// =============================================================================
// ADVANCED TUNING
// =============================================================================
// These parameters are pre-tuned. Modify only if you understand their effects.

// --- Grain Stabilization ---
// Grain bilateral filter runs in CelFlare-blur.glsl (pre-pass).
// See that file for tuning parameters (threshold, radius, edge detection, range).
#define EARLY_EXIT_GAMMA 0.10      // Skip very dark pixels (gamma) - below any possible knee

// --- Expansion Knee ---
#define MASTER_KNEE_OFFSET 0.15    // Onset is this many linear units below knee_end (see scene classification)

// --- Saturation Rolloff (Oklab chroma-based, separate from sat boost) ---
// Reduces expansion on already-saturated colors to prevent clipping.
#define ENABLE_SAT_ROLLOFF 0
#define SAT_THRESHOLD 0.25         // Normalized Oklab chroma threshold to start rolloff
#define SAT_POWER 6.0              // Rolloff curve steepness
#define SAT_ROLLOFF 0.40           // Reduces expansion on saturated colors

// --- Chroma-Adaptive Expansion (Helmholtz-Kohlrausch compensation) ---
// In warm yellow-orange band: low saturation → slight expansion compression (pale skin);
// mid-high saturation → slight lift (healthy/warm skin). Compensates for the H-K effect
// where chromatic colors appear perceptually brighter at the same luminance as neutrals.
// Pivot is the crossover normalized chroma (~pale/warm skin boundary). Lift is gated to
// avoid halos at the expansion zone onset. Compression lets the early exit handle result.
// WARNING: Edge contouring risk at sharp anime saturation boundaries. Keep subtle (≤0.15).
#define ENABLE_CHROMA_EXPAND 0
#define CHROMA_EXPAND_STRENGTH 0.40   // Max modifier magnitude; try 0.08–0.15
#define CHROMA_EXPAND_PIVOT 0.20      // Normalized Oklab chroma crossover (~pale skin)
#define CHROMA_EXPAND_RED_EXTEND 0.50 // Extend warm mask toward red/pink (0.0=yellow-orange only, 1.0=full red)

// =============================================================================
// INTERNAL PARAMETERS
// =============================================================================
// Scene analysis and expansion curve parameters.

// --- Temporal Smoothing ---
#define TEMPORAL_ALPHA 0.03        // Normal smoothing (lower = slower adaptation)
#define TEMPORAL_ALPHA_FAST 0.6    // Fast adaptation after scene cuts

// --- Scene Cut Detection (Gamma-Space Perceptual) ---
// Block-based: requires majority of samples to change in BOTH luma AND chroma.
#define BLOCK_LUMA_THRESH 0.09
#define BLOCK_CHROMA_THRESH 0.04
#define BLOCK_CHROMA_THRESH_SQ (BLOCK_CHROMA_THRESH * BLOCK_CHROMA_THRESH)
#define BLOCK_CHANGE_PCT 0.55      // Percentage of samples that must change (stricter to reduce false positives from fast camera motion)
#define LOCKOUT_FRAMES 6.0         // Frames to wait before detecting another cut

// --- Scene Sampling ---
#define SAMPLE_COLS 12             // 12x8 = 96 samples (~33% fewer, imperceptible on uniform content)
#define SAMPLE_ROWS 8

// --- Brightness Classification (log-average / key) ---
#define KEY_DARK 0.022
#define KEY_BRIGHT 0.065
#define KEY_VERY_BRIGHT 0.32             // Passthrough on bright daytime anime without killing speculars

// --- Contrast Classification (stops, cascading based on brightness) ---
#define CONTRAST_LOW_DARK 2.0
#define CONTRAST_LOW_BRIGHT 2.0
#define CONTRAST_HIGH_DARK 3.25
#define CONTRAST_HIGH_BRIGHT 4.2

// --- Specular Detection (cascading based on brightness) ---
#define HIGHLIGHT_ZONE_DARK 0.22         // Cel-shaded speculars sit lower in SDR than live-action
#define HIGHLIGHT_ZONE_BRIGHT 0.40       // Was 0.5 - speculars start lower in bright scenes
#define SPECULAR_THRESH_DARK 0.065       // Anime highlight population typically 5-11%
#define SPECULAR_THRESH_BRIGHT 0.22      // Bright anime palettes have lower relative highlight counts
#define SPECULAR_RELATIVE_FACTOR 2.0     // Also require Nx scene mean for highlight candidacy

// --- Expansion Curves by Scene Type ---
// Target: ~195-290 nits peak (1.8-2.7x at 108 nit reference white, with INTENSITY 1.2)

// Dark + Specular: Brightest highlights (candles, stars)
#define PEAK_DARK_SPECULAR 2.3
#define STEEP_DARK_SPECULAR 4.0
#define KNEE_DARK_SPECULAR 0.35

// Dark + Moody: Preserve mood, subtle pop
#define PEAK_DARK_MOODY 1.9
#define STEEP_DARK_MOODY 2.8
#define KNEE_DARK_MOODY 0.30

// Bright + Specular: Modest pop on already-bright content
#define PEAK_BRIGHT_SPECULAR 2.1
#define STEEP_BRIGHT_SPECULAR 6.0
#define KNEE_BRIGHT_SPECULAR 0.45

// Bright + Flat: Moderate (uniformly bright scenes, bright anime palettes)
#define PEAK_BRIGHT_FLAT 2.0
#define STEEP_BRIGHT_FLAT 7.0
#define KNEE_BRIGHT_FLAT 0.45

// High Contrast: Punchy but controlled
#define PEAK_HIGH_CONTRAST 2.2
#define STEEP_HIGH_CONTRAST 3.8
#define KNEE_HIGH_CONTRAST 0.35

// Default: Balanced middle ground
#define PEAK_DEFAULT 2.0
#define STEEP_DEFAULT 3.2
#define KNEE_DEFAULT 0.45

// Very Bright: Near passthrough
#define PEAK_VERY_BRIGHT 1.9
#define STEEP_VERY_BRIGHT 10.0
#define KNEE_VERY_BRIGHT 0.50


// =============================================================================
// DEBUG
// =============================================================================
// Enable one at a time to visualize different stages of the pipeline.

#define DEBUG_BYPASS 0             // Skip all processing, output original (gamma)
#define DEBUG_SHOW_STATS 0         // Show scene analysis bar graphs (top-left)
#define DEBUG_SHOW_SCENE_TYPE 0    // Show scene type color (top-right corner)
#define DEBUG_SHOW_EXPANSION 0     // Show expansion parameters (bottom-right)
#define DEBUG_SHOW_MASK 0          // Show expansion amount as grayscale
#define DEBUG_SHOW_SAT 0           // Show saturation rolloff factor
#define DEBUG_SHOW_GRAIN 0         // Show grain stabilization effect
#define DEBUG_SHOW_SAT_BOOST 0     // Show saturation boost amount
#define DEBUG_SHOW_DITHER 0        // Show dither magnitude and pattern
#define DEBUG_SHOW_HUE_SHIFT 0     // Show hue angle difference pre-correction

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// BT.709 luma coefficients (standard for SDR content)
float get_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// EOTF (gamma decode): gamma -> linear
vec3 eotf_gamma(vec3 v) {
    return pow(max(v, 0.0), vec3(EOTF_GAMMA));
}

float eotf_gamma(float v) {
    return pow(max(v, 0.0), EOTF_GAMMA);
}

// BT.709 to BT.2020 color space conversion (linear domain)
vec3 bt709_to_bt2020(vec3 rgb) {
    return vec3(
        0.6274040 * rgb.r + 0.3292820 * rgb.g + 0.0433136 * rgb.b,
        0.0690970 * rgb.r + 0.9195400 * rgb.g + 0.0113612 * rgb.b,
        0.0163916 * rgb.r + 0.0880132 * rgb.g + 0.8955950 * rgb.b
    );
}

// PQ (SMPTE ST 2084) OETF: linear light → perceptual code value
// Input: linear RGB normalized to 10000 nits (1.0 = 10000 nits)
vec3 pq_oetf(vec3 L) {
    const float m1 = 0.1593017578125;  // 2610/16384
    const float m2 = 78.84375;          // 2523/4096 * 128
    const float c1 = 0.8359375;         // 3424/4096
    const float c2 = 18.8515625;        // 2413/128
    const float c3 = 18.6875;           // 2392/128
    vec3 Lm1 = pow(max(L, 0.0), vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

// PQ OETF polynomial approximation: degree-7 minimax fit on sqrt(L) domain
// Valid for L in [0, 0.20] (0-2000 nits). Sub-1.2 ten-bit step accuracy for L > 0.005.
// Near-zero (< 5 nits) has larger error but those channels are not used in this shader.
// Cost: 1 sqrt + 7 FMA per channel vs 2 pow + 1 div per channel for exact PQ.
#if PQ_FAST_APPROX
vec3 pq_oetf_fast(vec3 L) {
    vec3 t = sqrt(max(L, 0.0));
    vec3 r = vec3(4830.3861664760);
    r = r * t - 8935.5954297213;
    r = r * t + 6836.4130114354;
    r = r * t - 2804.9691846594;
    r = r * t + 672.6577715456;
    r = r * t - 98.1828798096;
    r = r * t + 9.7074413362;
    r = r * t + 0.0677928739;
    return clamp(r, 0.0, 1.0);
}
#endif

// Gamma BT.709 → PQ BT.2020 (for passthrough/early-exit pixels)
vec3 gamma709_to_pq2020(vec3 rgb_gamma) {
    vec3 linear = eotf_gamma(rgb_gamma);
    vec3 bt2020 = bt709_to_bt2020(linear);
    return pq_oetf(bt2020 * (REFERENCE_WHITE / 10000.0));
}

// Linear BT.709 → PQ BT.2020 (for expanded pixels, values can exceed 1.0)
// Clamp in BT.2020 space (not BT.709) so out-of-709 colors that fit in the
// wider BT.2020 gamut are preserved — eliminates blue-channel clipping that
// shifts warm hues (yellow/orange/red) toward green/gold.
vec3 linear709_to_pq2020(vec3 rgb_linear) {
    vec3 bt2020 = max(bt709_to_bt2020(rgb_linear), 0.0);
    #if PQ_FAST_APPROX
        return pq_oetf_fast(bt2020 * (REFERENCE_WHITE / 10000.0));
    #else
        return pq_oetf(bt2020 * (REFERENCE_WHITE / 10000.0));
    #endif
}

// Hash without Sine (Dave Hoskins - https://www.shadertoy.com/view/4djSRW)
// Portable across GPUs, no visible pattern
float hashNoise(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Triangular PDF noise for dithering
// Sum of two uniform [0,1] distributions = triangular on [0,2], shifted to [-1,1]
float triangularNoise(vec2 p) {
    float r1 = hashNoise(p);
    float r2 = hashNoise(p + vec2(1.7, 3.1));
    return r1 + r2 - 1.0;
}

// =============================================================================
// OKLAB COLOR SPACE (Björn Ottosson, 2020)
// =============================================================================
// Perceptually uniform color space with excellent hue linearity.
// Saturation adjustments in Oklab preserve hue by design.

// Fast cube root via bit manipulation + Newton-Raphson (~4× faster than pow)
// Error: ~0.1% after one iteration - imperceptible for color conversion
float fast_cbrt(float x) {
    if (x <= 0.0) return 0.0;
    uint i = floatBitsToUint(x);
    i = i / 3u + 0x2a514067u;  // Initial approximation via exponent division
    float y = uintBitsToFloat(i);
    y = y * 0.666666667 + x / (3.0 * y * y);  // One Newton-Raphson iteration
    return y;
}

// Linear sRGB to Oklab
vec3 rgb_to_oklab(vec3 rgb) {
    // RGB to LMS (using sRGB primaries)
    float l = 0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b;
    float m = 0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b;
    float s = 0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b;

    // Cube root (perceptual nonlinearity) - using fast approximation
    float l_ = fast_cbrt(l);
    float m_ = fast_cbrt(m);
    float s_ = fast_cbrt(s);

    // LMS to Oklab
    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,  // L
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,  // a
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_   // b
    );
}

// Oklab to linear sRGB
vec3 oklab_to_rgb(vec3 lab) {
    // Oklab to LMS (cube root space)
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

    // Cube to undo perceptual nonlinearity
    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    // LMS to RGB
    return vec3(
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}

// Scene type constants (for debug display only)
#define SCENE_DARK_SPECULAR 0
#define SCENE_DARK_MOODY 1
#define SCENE_BRIGHT_SPECULAR 2
#define SCENE_BRIGHT_FLAT 3
#define SCENE_HIGH_CONTRAST 4
#define SCENE_DEFAULT 5
#define SCENE_VERY_BRIGHT 6

// =============================================================================
// MAIN PROCESSING
// =============================================================================

vec4 hook() {
    vec4 color = HOOKED_texOff(0);
    vec3 rgb_gamma = color.rgb;

    // -------------------------------------------------------------------------
    // SCENE STATISTICS (first pixel only for performance)
    // -------------------------------------------------------------------------
    vec2 pixel_pos = HOOKED_pos * HOOKED_size;
    bool is_first_pixel = (pixel_pos.x < 1.0 && pixel_pos.y < 1.0);

    if (is_first_pixel) {
        float Y_sum = 0.0;
        float Y_log_sum = 0.0;
        float Y_min = 1.0;
        float Y_max = 0.0;
        float highlight_count = 0.0;
        float valid_samples = 0.0;
        float chroma_sum = 0.0;
        float luma_changed_count = 0.0;
        float chroma_changed_count = 0.0;

        // Use smoothed log_avg for adaptive thresholds (linearized for consistency)
        float key_factor = smoothstep(KEY_DARK, KEY_BRIGHT, smoothed_log_avg);
        float adaptive_highlight_zone = mix(HIGHLIGHT_ZONE_DARK, HIGHLIGHT_ZONE_BRIGHT, key_factor);
        // Relative threshold: pixel must also exceed Nx scene mean to count as highlight.
        // Prevents inflated specular counts in uniformly bright scenes (e.g. bright anime palettes).
        // Uses arithmetic mean (more sensitive to bright outliers than geometric mean).
        float relative_highlight = smoothed_avg * SPECULAR_RELATIVE_FACTOR;

        float total_samples = float(SAMPLE_COLS * SAMPLE_ROWS);

        // Sample and compare to previous frame
        for (int y = 0; y < SAMPLE_ROWS; y++) {
            for (int x = 0; x < SAMPLE_COLS; x++) {
                int idx = y * SAMPLE_COLS + x;
                vec2 spos = vec2((float(x) + 0.5) / float(SAMPLE_COLS),
                                 (float(y) + 0.5) / float(SAMPLE_ROWS));

                // Sample in gamma space
                vec3 rgb_sample_gamma = HOOKED_tex(spos).rgb;
                float Y_gamma = get_luma(rgb_sample_gamma);

                // Compute chroma BEFORE black bar skip (needed for scene cut on all samples)
                float cb_gamma = rgb_sample_gamma.b - Y_gamma;
                float cr_gamma = rgb_sample_gamma.r - Y_gamma;

                // Scene cut detection runs on ALL 96 samples (including black bars)
                if (frame > 0) {
                    float luma_diff = abs(Y_gamma - prev_luma[idx]);
                    float cb_diff = cb_gamma - prev_cb[idx];
                    float cr_diff = cr_gamma - prev_cr[idx];
                    float chroma_diff_sq = cb_diff * cb_diff + cr_diff * cr_diff;

                    if (luma_diff > BLOCK_LUMA_THRESH) {
                        luma_changed_count += 1.0;
                    }
                    if (chroma_diff_sq > BLOCK_CHROMA_THRESH_SQ) {
                        chroma_changed_count += 1.0;
                    }
                }

                // Store gamma values for next frame (perceptual comparison)
                prev_luma[idx] = Y_gamma;
                prev_cb[idx] = cb_gamma;
                prev_cr[idx] = cr_gamma;

                // Linearize for statistics
                vec3 rgb_sample_linear = eotf_gamma(rgb_sample_gamma);
                float Y_linear = get_luma(rgb_sample_linear);

                // Accumulate chroma for B&W detection (all samples)
                chroma_sum += sqrt(cb_gamma * cb_gamma + cr_gamma * cr_gamma);

                // Black bar exclusion: skip near-black samples for scene statistics
                if (Y_linear < 0.001) continue;

                // Statistics in linear space (valid samples only)
                valid_samples += 1.0;
                Y_sum += Y_linear;
                Y_log_sum += log(max(Y_linear, 0.0001));
                Y_min = min(Y_min, Y_linear);
                Y_max = max(Y_max, Y_linear);

                if (Y_linear >= adaptive_highlight_zone && Y_linear > relative_highlight) {
                    highlight_count += 1.0;
                }
            }
        }

        // If too few valid samples (>75% black bars), keep previous frame's smoothed stats
        bool stats_valid = valid_samples >= 24.0;

        float current_avg = stats_valid ? Y_sum / valid_samples : smoothed_avg;
        float current_log_avg = stats_valid ? exp(Y_log_sum / valid_samples) : smoothed_log_avg;
        float current_highlight_pop = stats_valid ? highlight_count / valid_samples : smoothed_highlight_pop;
        float current_contrast = stats_valid ? log2(max(Y_max, 0.001) / max(Y_min, 0.001)) : smoothed_contrast;
        float current_Y_max = stats_valid ? Y_max : smoothed_max;
        float current_Y_min = stats_valid ? Y_min : smoothed_min;
        float current_avg_chroma = chroma_sum / total_samples;

        // Block-based scene cut detection (uses all 96 samples, not just valid)
        float luma_change_pct = luma_changed_count / total_samples;
        float chroma_change_pct = chroma_changed_count / total_samples;

        scene_cut_lockout = max(scene_cut_lockout - 1.0, 0.0);

        // B&W scene cut fallback: when current frame's avg chroma is very low, drop
        // the chroma requirement. Uses current frame (not smoothed) so color-to-B&W
        // hard cuts are detected immediately instead of waiting for smoothing to converge.
        bool is_bw = current_avg_chroma < 0.02;
        bool scene_cut = is_bw
            ? (luma_change_pct > BLOCK_CHANGE_PCT) && (scene_cut_lockout <= 0.0)
            : (luma_change_pct > BLOCK_CHANGE_PCT) &&
              (chroma_change_pct > BLOCK_CHANGE_PCT) &&
              (scene_cut_lockout <= 0.0);

        if (scene_cut) {
            scene_cut_lockout = LOCKOUT_FRAMES;
        }

        // Fast adaptation during scene cut AND lockout period (6 frames = 99.6% converged)
        float alpha = (scene_cut || scene_cut_lockout > 0.0) ? TEMPORAL_ALPHA_FAST : TEMPORAL_ALPHA;

        if (frame == 0) {
            smoothed_avg = current_avg;
            smoothed_max = current_Y_max;
            smoothed_min = current_Y_min;
            smoothed_log_avg = current_log_avg;
            smoothed_contrast = current_contrast;
            smoothed_highlight_pop = current_highlight_pop;
            smoothed_avg_chroma = current_avg_chroma;
            scene_cut_lockout = 0.0;
        } else {
            smoothed_avg = mix(smoothed_avg, current_avg, alpha);
            smoothed_max = mix(smoothed_max, current_Y_max, alpha);
            smoothed_min = mix(smoothed_min, current_Y_min, alpha);
            smoothed_log_avg = mix(smoothed_log_avg, current_log_avg, alpha);
            smoothed_contrast = mix(smoothed_contrast, current_contrast, alpha);
            smoothed_highlight_pop = mix(smoothed_highlight_pop, current_highlight_pop, alpha);
            smoothed_avg_chroma = mix(smoothed_avg_chroma, current_avg_chroma, alpha);
        }

        // ---------------------------------------------------------------------
        // SCENE CLASSIFICATION (computed once per frame, stored in buffer)
        // ---------------------------------------------------------------------
        float key = smoothed_log_avg;
        float contrast = smoothed_contrast;

        // Cascading specular threshold
        float key_factor_spec = smoothstep(KEY_DARK, KEY_BRIGHT, key);
        float adaptive_spec_threshold = mix(SPECULAR_THRESH_DARK, SPECULAR_THRESH_BRIGHT, key_factor_spec);

        // Soft specular factor
        float spec_blend_width = adaptive_spec_threshold * 0.5;
        float specular_factor = smoothstep(
            adaptive_spec_threshold - spec_blend_width,
            adaptive_spec_threshold + spec_blend_width,
            smoothed_highlight_pop
        );

        // Blend factors
        float brightness_factor = smoothstep(KEY_DARK, KEY_BRIGHT, key);
        float very_bright_factor = smoothstep(KEY_BRIGHT, KEY_VERY_BRIGHT, key);

        float adaptive_contrast_low = mix(CONTRAST_LOW_DARK, CONTRAST_LOW_BRIGHT, brightness_factor);
        float adaptive_contrast_high = mix(CONTRAST_HIGH_DARK, CONTRAST_HIGH_BRIGHT, brightness_factor);

        float contrast_factor = smoothstep(adaptive_contrast_low - 1.0, adaptive_contrast_low + 1.0, contrast);
        float high_contrast_factor = smoothstep(adaptive_contrast_high - 1.0, adaptive_contrast_high + 1.0, contrast);

        // Dark treatment
        float dark_peak = mix(PEAK_DARK_MOODY, PEAK_DARK_SPECULAR, specular_factor);
        float dark_steep = mix(STEEP_DARK_MOODY, STEEP_DARK_SPECULAR, specular_factor);
        float dark_knee = mix(KNEE_DARK_MOODY, KNEE_DARK_SPECULAR, specular_factor);

        // Bright treatment
        float bright_base_peak = mix(PEAK_BRIGHT_FLAT, PEAK_DEFAULT, contrast_factor);
        float bright_base_steep = mix(STEEP_BRIGHT_FLAT, STEEP_DEFAULT, contrast_factor);
        float bright_base_knee = mix(KNEE_BRIGHT_FLAT, KNEE_DEFAULT, contrast_factor);

        float bright_peak = mix(bright_base_peak, PEAK_BRIGHT_SPECULAR, specular_factor);
        float bright_steep = mix(bright_base_steep, STEEP_BRIGHT_SPECULAR, specular_factor);
        float bright_knee = mix(bright_base_knee, KNEE_BRIGHT_SPECULAR, specular_factor);

        // Dark <-> Bright blend
        float base_peak = mix(dark_peak, bright_peak, brightness_factor);
        float base_steep = mix(dark_steep, bright_steep, brightness_factor);
        float base_knee = mix(dark_knee, bright_knee, brightness_factor);

        // High contrast blend
        float pre_vb_peak = mix(base_peak, PEAK_HIGH_CONTRAST, high_contrast_factor);
        float pre_vb_steep = mix(base_steep, STEEP_HIGH_CONTRAST, high_contrast_factor);
        float pre_vb_knee = mix(base_knee, KNEE_HIGH_CONTRAST, high_contrast_factor);

        // Very bright blend
        float computed_peak = mix(pre_vb_peak, PEAK_VERY_BRIGHT, very_bright_factor);
        float computed_steep = mix(pre_vb_steep, STEEP_VERY_BRIGHT, very_bright_factor);
        float computed_knee = mix(pre_vb_knee, KNEE_VERY_BRIGHT, very_bright_factor);
        float computed_master = max(computed_knee - MASTER_KNEE_OFFSET, 0.05);
        float computed_exp_denom = exp(min(computed_steep * CURVE_STEEPNESS, 20.0)) - 1.0;

        // Compute scene type for debug visualization
        float st = float(SCENE_DEFAULT);
        if (very_bright_factor > 0.5) {
            st = float(SCENE_VERY_BRIGHT);
        } else if (brightness_factor < 0.3) {
            st = (specular_factor > 0.5) ? float(SCENE_DARK_SPECULAR) : float(SCENE_DARK_MOODY);
        } else if (brightness_factor > 0.7) {
            if (specular_factor > 0.5) st = float(SCENE_BRIGHT_SPECULAR);
            else if (contrast_factor < 0.5) st = float(SCENE_BRIGHT_FLAT);
        } else if (high_contrast_factor > 0.5) {
            st = float(SCENE_HIGH_CONTRAST);
        }

        // Double-buffer scene parameters: write to current frame's slot.
        // Other pixels read the OPPOSITE slot (previous frame's values),
        // eliminating the GPU race condition that causes grid artifacts on scene cuts.
        if (frame % 2 == 0) {
            scene_highlight_peak = computed_peak;
            scene_exp_steepness = computed_steep;
            scene_knee_end = computed_knee;
            scene_master_knee = computed_master;
            scene_type_val = st;
            scene_contrast = smoothed_contrast;
            scene_exp_denom = computed_exp_denom;
        } else {
            scene_B_highlight_peak = computed_peak;
            scene_B_exp_steepness = computed_steep;
            scene_B_knee_end = computed_knee;
            scene_B_master_knee = computed_master;
            scene_B_type_val = st;
            scene_B_contrast = smoothed_contrast;
            scene_B_exp_denom = computed_exp_denom;
        }

        // First pixel outputs original color after computing stats
        // This prevents visual artifacts from heavy computation on this pixel
        return vec4(gamma709_to_pq2020(color.rgb), color.a);
    }

    // Double-buffer: read from PREVIOUS frame's slot (opposite parity).
    // Eliminates GPU race where some warps see first-pixel's buffer write
    // and others don't, causing visible grid pattern on scene cuts.
    float highlight_peak, exp_steepness, knee_end, master_knee, db_contrast, exp_denom;
    int scene_type;
    if (frame % 2 == 0) {
        // Even frames: first pixel writes A, read from B (previous odd frame)
        highlight_peak = scene_B_highlight_peak;
        exp_steepness = scene_B_exp_steepness;
        knee_end = scene_B_knee_end;
        master_knee = scene_B_master_knee;
        scene_type = int(scene_B_type_val);
        db_contrast = scene_B_contrast;
        exp_denom = scene_B_exp_denom;
    } else {
        // Odd frames: first pixel writes B, read from A (previous even frame)
        highlight_peak = scene_highlight_peak;
        exp_steepness = scene_exp_steepness;
        knee_end = scene_knee_end;
        master_knee = scene_master_knee;
        scene_type = int(scene_type_val);
        db_contrast = scene_contrast;
        exp_denom = scene_exp_denom;
    }

    // Fallback defaults for frame 0 (opposite slot uninitialized)
    if (frame == 0) {
        highlight_peak = max(highlight_peak, 1.8);
        exp_steepness = max(exp_steepness, 3.0);
        knee_end = (knee_end < 0.1) ? 0.40 : knee_end;
        master_knee = max(master_knee, 0.25);
        scene_type = SCENE_DEFAULT;
        db_contrast = (db_contrast < 0.5) ? 3.0 : db_contrast;
        // exp_denom buffer slot is uninitialized (zero) on frame 0 — recompute from fallback steepness
        float fs_fallback = min(exp_steepness * CURVE_STEEPNESS, 20.0);
        exp_denom = (fs_fallback > 0.001) ? exp(fs_fallback) - 1.0 : 1.0;
    }

    // -------------------------------------------------------------------------
    // DEBUG OVERLAYS (Scene Analysis)
    // -------------------------------------------------------------------------
    #if DEBUG_SHOW_STATS
        vec2 pos = HOOKED_pos;
        float bar_width = 0.25;
        float bar_height = 0.015;
        float bar_gap = 0.005;
        float start_y = 0.0;

        // Recompute debug-only variables (only when debug enabled)
        float dbg_key = smoothed_log_avg;
        float dbg_brightness_factor = smoothstep(KEY_DARK, KEY_BRIGHT, dbg_key);
        float dbg_adaptive_contrast_low = mix(CONTRAST_LOW_DARK, CONTRAST_LOW_BRIGHT, dbg_brightness_factor);
        float dbg_adaptive_contrast_high = mix(CONTRAST_HIGH_DARK, CONTRAST_HIGH_BRIGHT, dbg_brightness_factor);
        float dbg_key_factor_spec = smoothstep(KEY_DARK, KEY_BRIGHT, dbg_key);
        float dbg_adaptive_spec_threshold = mix(SPECULAR_THRESH_DARK, SPECULAR_THRESH_BRIGHT, dbg_key_factor_spec);

        // Bar 1: Log-average (Key) - Green
        if (pos.x < bar_width && pos.y >= start_y && pos.y < start_y + bar_height) {
            float norm_x = pos.x / bar_width;
            float display_max = 0.40;  // Accommodate KEY_VERY_BRIGHT=0.32
            float val = clamp(smoothed_log_avg / display_max, 0.0, 1.0);
            vec3 bg = (norm_x < val) ? vec3(0.0, 0.5, 0.0) : vec3(0.1);

            float dark_mark = KEY_DARK / display_max;
            float bright_mark = KEY_BRIGHT / display_max;
            float very_bright_mark = KEY_VERY_BRIGHT / display_max;
            if (abs(norm_x - dark_mark) < 0.006) return vec4(gamma709_to_pq2020(vec3(0.0, 0.3, 0.6)), 1.0);
            if (abs(norm_x - bright_mark) < 0.006) return vec4(gamma709_to_pq2020(vec3(0.6, 0.4, 0.0)), 1.0);
            if (abs(norm_x - very_bright_mark) < 0.006) return vec4(gamma709_to_pq2020(vec3(0.6, 0.0, 0.0)), 1.0);
            if (abs(norm_x - val) < 0.008) return vec4(gamma709_to_pq2020(vec3(0.6, 0.6, 0.6)), 1.0);
            return vec4(gamma709_to_pq2020(bg), 1.0);
        }
        start_y += bar_height + bar_gap;

        // Bar 2: Contrast - Yellow
        if (pos.x < bar_width && pos.y >= start_y && pos.y < start_y + bar_height) {
            float norm_x = pos.x / bar_width;
            float contrast_display_max = 8.0;  // Was 12.0 - adjusted for new thresholds
            float val = clamp(smoothed_contrast / contrast_display_max, 0.0, 1.0);
            vec3 bg = (norm_x < val) ? vec3(0.5, 0.5, 0.0) : vec3(0.1);

            float low_mark = dbg_adaptive_contrast_low / contrast_display_max;
            float high_mark = dbg_adaptive_contrast_high / contrast_display_max;
            if (abs(norm_x - low_mark) < 0.006) return vec4(gamma709_to_pq2020(vec3(0.3, 0.3, 0.6)), 1.0);
            if (abs(norm_x - high_mark) < 0.006) return vec4(gamma709_to_pq2020(vec3(0.6, 0.3, 0.3)), 1.0);
            if (abs(norm_x - val) < 0.008) return vec4(gamma709_to_pq2020(vec3(0.6, 0.6, 0.6)), 1.0);
            return vec4(gamma709_to_pq2020(bg), 1.0);
        }
        start_y += bar_height + bar_gap;

        // Bar 3: Max luminance - Red
        if (pos.x < bar_width && pos.y >= start_y && pos.y < start_y + bar_height) {
            float norm_x = pos.x / bar_width;
            float val = clamp(smoothed_max, 0.0, 1.0);
            vec3 bg = (norm_x < val) ? vec3(0.5, 0.0, 0.0) : vec3(0.1);

            float zone_mark = mix(HIGHLIGHT_ZONE_DARK, HIGHLIGHT_ZONE_BRIGHT, dbg_brightness_factor);
            if (abs(norm_x - zone_mark) < 0.006) return vec4(gamma709_to_pq2020(vec3(0.6, 0.6, 0.0)), 1.0);
            if (abs(norm_x - val) < 0.008) return vec4(gamma709_to_pq2020(vec3(0.6, 0.6, 0.6)), 1.0);
            return vec4(gamma709_to_pq2020(bg), 1.0);
        }
        start_y += bar_height + bar_gap;

        // Bar 4: Highlight population - Magenta
        if (pos.x < bar_width && pos.y >= start_y && pos.y < start_y + bar_height) {
            float norm_x = pos.x / bar_width;
            float highlight_display_max = 0.30;  // Was 0.5 - adjusted for new thresholds (8-15%)
            float val = clamp(smoothed_highlight_pop / highlight_display_max, 0.0, 1.0);
            vec3 bg = (norm_x < val) ? vec3(0.5, 0.0, 0.5) : vec3(0.1);

            float spec_mark = dbg_adaptive_spec_threshold / highlight_display_max;
            if (abs(norm_x - spec_mark) < 0.006) return vec4(gamma709_to_pq2020(vec3(0.0, 0.6, 0.0)), 1.0);
            if (abs(norm_x - val) < 0.008) return vec4(gamma709_to_pq2020(vec3(0.6, 0.6, 0.6)), 1.0);
            return vec4(gamma709_to_pq2020(bg), 1.0);
        }
        start_y += bar_height + bar_gap;

        // Bar 5: Scene type indicator
        if (pos.x < bar_width && pos.y >= start_y && pos.y < start_y + bar_height) {
            float norm_x = pos.x / bar_width;
            vec3 type_color;
            if (scene_type == SCENE_DARK_SPECULAR) type_color = vec3(0.6, 0.4, 0.0);       // Orange
            else if (scene_type == SCENE_DARK_MOODY) type_color = vec3(0.3, 0.0, 0.5);    // Purple
            else if (scene_type == SCENE_BRIGHT_SPECULAR) type_color = vec3(0.6, 0.6, 0.0); // Yellow
            else if (scene_type == SCENE_BRIGHT_FLAT) type_color = vec3(0.4, 0.4, 0.4);   // Gray
            else if (scene_type == SCENE_HIGH_CONTRAST) type_color = vec3(0.0, 0.5, 0.5); // Cyan
            else if (scene_type == SCENE_VERY_BRIGHT) type_color = vec3(0.9, 0.9, 0.9);   // White (passthrough)
            else type_color = vec3(0.0, 0.5, 0.0);                                         // Green (default)

            float peak_norm = (highlight_peak - 1.5) / 2.0;  // Adjusted for new peak range
            if (norm_x < peak_norm) return vec4(gamma709_to_pq2020(type_color), 1.0);
            return vec4(gamma709_to_pq2020(vec3(0.1)), 1.0);
        }
    #endif

    #if DEBUG_SHOW_SCENE_TYPE
        if (HOOKED_pos.x > 0.9 && HOOKED_pos.y < 0.1) {
            vec3 type_color;
            if (scene_type == SCENE_DARK_SPECULAR) type_color = vec3(0.6, 0.4, 0.0);       // Orange
            else if (scene_type == SCENE_DARK_MOODY) type_color = vec3(0.3, 0.0, 0.5);    // Purple
            else if (scene_type == SCENE_BRIGHT_SPECULAR) type_color = vec3(0.6, 0.6, 0.0); // Yellow
            else if (scene_type == SCENE_BRIGHT_FLAT) type_color = vec3(0.4, 0.4, 0.4);   // Gray
            else if (scene_type == SCENE_HIGH_CONTRAST) type_color = vec3(0.0, 0.5, 0.5); // Cyan
            else if (scene_type == SCENE_VERY_BRIGHT) type_color = vec3(0.9, 0.9, 0.9);   // White (passthrough)
            else type_color = vec3(0.0, 0.5, 0.0);                                         // Green (default)
            return vec4(gamma709_to_pq2020(type_color), 1.0);
        }
    #endif

    #if DEBUG_BYPASS
        return vec4(gamma709_to_pq2020(color.rgb), color.a);
    #endif

    // -------------------------------------------------------------------------
    // PIXEL PROCESSING (Gamma Space)
    // -------------------------------------------------------------------------
    float Y_gamma = get_luma(rgb_gamma);

    // Early exit for very dark pixels (below any possible expansion knee)
    // 0.10 gamma ≈ 0.006 linear, well below any knee
    // Saves all grain sampling, linearization, and expansion math.
    #if !DEBUG_SHOW_MASK && !DEBUG_SHOW_SAT && !DEBUG_SHOW_GRAIN
        if (Y_gamma < EARLY_EXIT_GAMMA) return vec4(gamma709_to_pq2020(color.rgb), color.a);
    #else
        if (Y_gamma < 0.0001) return vec4(gamma709_to_pq2020(color.rgb), color.a);
    #endif

    float Y_decision_gamma = Y_gamma;

    // -------------------------------------------------------------------------
    // GRAIN STABILIZATION (Pre-filter Alpha Read)
    // -------------------------------------------------------------------------
    // CelFlare-blur.glsl pre-computes stabilized gamma luma and packs it into
    // MAIN's alpha channel. We just read it here — zero extra texture fetches.
    // If blur shader is not loaded, alpha = 1.0 (opaque video) — fall back to raw luma.
    #if ENABLE_GRAIN_STABLE
        Y_decision_gamma = (color.a > 0.99) ? Y_gamma : color.a;

        #if DEBUG_SHOW_GRAIN
            float grain_diff = abs(Y_gamma - Y_decision_gamma) * 20.0;
            return vec4(gamma709_to_pq2020(vec3(grain_diff, 0.0, 0.0)), 1.0);
        #endif
    #endif

    // -------------------------------------------------------------------------
    // LINEARIZE FOR EXPANSION
    // -------------------------------------------------------------------------
    // Convert from gamma to linear for expansion math
    float Y_decision = eotf_gamma(Y_decision_gamma);
    vec3 rgb_linear = eotf_gamma(rgb_gamma);

    // -------------------------------------------------------------------------
    // SATURATION (Oklab chroma - perceptually uniform)
    // -------------------------------------------------------------------------
    // Oklab chroma = sqrt(a² + b²), typically 0-0.35 for sRGB content
    vec3 oklab_orig = rgb_to_oklab(rgb_linear);
    float chroma_orig = sqrt(oklab_orig.y * oklab_orig.y + oklab_orig.z * oklab_orig.z);
    // Normalize to 0-1 range (0.35 ≈ max sRGB chroma) for threshold compatibility
    float sat_raw = chroma_orig / 0.35;
    float sat = pow(smoothstep(SAT_THRESHOLD, 1.0, sat_raw), SAT_POWER);

    #if DEBUG_SHOW_SAT
        return vec4(gamma709_to_pq2020(vec3(sat, chroma_orig * 3.0, 0.0)), 1.0);
    #endif

    // -------------------------------------------------------------------------
    // EXPANSION CURVE (Linear Space)
    // -------------------------------------------------------------------------
    // master_knee is pre-computed in first-pixel block and read from buffer

    float final_steepness = min(exp_steepness * CURVE_STEEPNESS, 20.0);  // Cap prevents exp() overflow

    float master_knee_t = step(master_knee, Y_decision);
    float master_expand_t = smoothstep(master_knee, 1.0, Y_decision);

    // Normalized exponential with L'Hôpital guard
    float master_curve = (final_steepness > 0.001)
        ? (exp(final_steepness * master_expand_t) - 1.0) / exp_denom
        : master_expand_t;

    float expansion = mix(1.0, mix(1.0, highlight_peak, master_curve), master_knee_t);

    // Saturation rolloff
    #if ENABLE_SAT_ROLLOFF
        expansion = mix(expansion, 1.0, sat * SAT_ROLLOFF);
    #endif

    // Intensity multiplier
    #if ENABLE_DYNAMIC_INTENSITY
        float dyn_factor = smoothstep(DYN_CONTRAST_LOW, DYN_CONTRAST_HIGH, db_contrast);
        float dyn_intensity = mix(DYN_INTENSITY_LOW, DYN_INTENSITY_HIGH, dyn_factor);
        expansion = 1.0 + (expansion - 1.0) * dyn_intensity * INTENSITY;
    #else
        expansion = 1.0 + (expansion - 1.0) * INTENSITY;
    #endif

    // Chroma-adaptive expansion: warm low-sat (pale skin) → compress;
    // warm mid-sat (healthy skin) → lift. Lift gated on expansion zone
    // to prevent halos at onset. Compression: early exit handles result.
    #if ENABLE_CHROMA_EXPAND
    {
        float ce_chroma_n = clamp(chroma_orig / 0.35, 0.0, 1.0);
        // Blend red axis (a+) into warm signal so pink/blush tones are included.
        // RED_EXTEND=0 → original yellow-only; 0.4 → red gets warm_ratio≈0.2 (enters smoothstep).
        float warm_signal = oklab_orig.z + CHROMA_EXPAND_RED_EXTEND * max(oklab_orig.y, 0.0);
        float warm_ratio  = warm_signal / (abs(oklab_orig.y) + chroma_orig + 0.001);
        // Gate on BOTH hue angle AND chroma magnitude. warm_ratio alone is a pure hue
        // direction measure — a near-neutral pixel with a tiny positive b still gets
        // warm_ratio≈1, but has no meaningful warm chroma to compensate for. H-K effect
        // is proportional to saturation, so correction should fade to zero at neutral.
        float warm_t = smoothstep(0.15, 0.65, warm_ratio) * smoothstep(0.05, 0.20, ce_chroma_n);
        float delta = CHROMA_EXPAND_STRENGTH * (ce_chroma_n - CHROMA_EXPAND_PIVOT) * warm_t;
        if (delta > 0.0) {
            expansion *= 1.0 + delta * smoothstep(1.001, 1.10, expansion);
        } else {
            expansion *= 1.0 + delta;
        }
    }
    #endif

    // -------------------------------------------------------------------------
    // EARLY EXIT: Skip remaining processing for non-expanded pixels
    // -------------------------------------------------------------------------
    // Saves ALU on dark/mid-tone pixels that don't receive expansion.
    // Threshold slightly above 1.0 to catch edge cases from sat rolloff.
    if (expansion < 1.001) {
        #if ENABLE_SAT_BOOST
        {
            float sat_knee = max(master_knee - SAT_KNEE_OFFSET, 0.0);
            float sat_reach = smoothstep(sat_knee, master_knee, Y_decision);
            if (SAT_KNEE_OFFSET > 0.0 && sat_reach > 0.001) {
                float Y_lin = get_luma(rgb_linear);
                vec3 rgb_boosted = mix(vec3(Y_lin), rgb_linear, 1.0 + sat_reach * 0.04);
                return vec4(linear709_to_pq2020(rgb_boosted), color.a);
            }
        }
        #endif
        return vec4(gamma709_to_pq2020(color.rgb), color.a);
    }

    #if DEBUG_SHOW_MASK
        float exp_amount = (expansion - 1.0) / 2.5;  // Adjusted for new range
        return vec4(gamma709_to_pq2020(vec3(exp_amount, exp_amount * 0.3, 0.0)), 1.0);
    #endif

    #if DEBUG_SHOW_EXPANSION
        if (HOOKED_pos.x > 0.75 && HOOKED_pos.y > 0.9) {
            float peak_norm = (highlight_peak - 1.5) / 2.0;
            float steep_norm = (exp_steepness - 1.5) / 2.5;
            return vec4(gamma709_to_pq2020(vec3(peak_norm, steep_norm, knee_end)), 1.0);
        }
    #endif

    // -------------------------------------------------------------------------
    // APPLY EXPANSION (Linear Space)
    // -------------------------------------------------------------------------
    vec3 rgb_expanded = rgb_linear * expansion;

    // -------------------------------------------------------------------------
    // SATURATION BOOST + HUE CORRECTION (Merged Single Oklab Pass)
    // -------------------------------------------------------------------------
    // Merged into one Oklab round-trip to save ~45 ALU and eliminate the
    // intermediate gamut clip that was distorting hue before correction could fix it.
    #if ENABLE_SAT_BOOST || ENABLE_HUE_CORRECTION || ENABLE_BB_WARMTH
    {
        vec3 oklab_exp = rgb_to_oklab(rgb_expanded);

        #if ENABLE_SAT_BOOST
            // Stevens' power law: uniform chroma scaling by expansion^0.5
            float sat_boost = min(pow(max(expansion, 0.0), SAT_BOOST_EXPONENT), SAT_BOOST_MAX);

            oklab_exp.y *= sat_boost;
            oklab_exp.z *= sat_boost;

            #if DEBUG_SHOW_SAT_BOOST
                float boost_viz = (sat_boost - 1.0) / (SAT_BOOST_MAX - 1.0);
                float chroma_out = sqrt(oklab_exp.y * oklab_exp.y + oklab_exp.z * oklab_exp.z);
                return vec4(gamma709_to_pq2020(vec3(boost_viz, chroma_out * 3.0, 0.0)), 1.0);
            #endif
        #endif

        #if ENABLE_HUE_CORRECTION
        {
            float chroma_post = length(oklab_exp.yz);
            float chroma_ref = length(oklab_orig.yz);

            if (chroma_post > 0.001 && chroma_ref > 0.001) {
                #if DEBUG_SHOW_HUE_SHIFT
                    vec2 dir_pre = oklab_exp.yz / chroma_post;
                    vec2 dir_ref = oklab_orig.yz / chroma_ref;
                    float angle_diff = acos(clamp(dot(dir_pre, dir_ref), -1.0, 1.0));
                    return vec4(gamma709_to_pq2020(vec3(angle_diff * 5.0, 0.0, 0.0)), 1.0);
                #endif

                // Blend hue direction toward original (pre-expansion) hue
                oklab_exp.yz = mix(oklab_exp.yz, oklab_orig.yz, HUE_CORRECTION_STRENGTH);

                // Restore expanded chroma magnitude (only hue angle changed)
                float chroma_corrected = length(oklab_exp.yz);
                if (chroma_corrected > 0.001) {
                    oklab_exp.yz *= chroma_post / chroma_corrected;
                }
            }
        }
        #endif

        // Bezold-Brücke pre-compensation: clockwise rotation in Oklab ab plane for
        // yellow-orange hues, proportional to expansion. Warm mask b/(|a|+chroma)
        // gives high weight to yellow, moderate to orange, near-zero to red.
        #if ENABLE_BB_WARMTH
        {
            float bb_chroma = length(oklab_exp.yz);
            if (bb_chroma > 0.001) {
                float warm_ratio = oklab_exp.z / (abs(oklab_exp.y) + bb_chroma);
                float warm_t = smoothstep(0.15, 0.65, warm_ratio);
                float exp_t  = smoothstep(0.05, 0.80, expansion - 1.0);
                // Clockwise rotation: Δa = +b·θ, Δb = −a·θ
                oklab_exp.yz += (BB_WARMTH * warm_t * exp_t) * vec2(oklab_exp.z, -oklab_exp.y);
                float new_chroma = length(oklab_exp.yz);
                if (new_chroma > 0.001) oklab_exp.yz *= bb_chroma / new_chroma;
            }
        }
        #endif

        // Convert back to linear RGB — allow out-of-709 negatives to pass through
        // to BT.2020 conversion where the wider gamut absorbs them
        rgb_expanded = oklab_to_rgb(oklab_exp);
    }
    #endif

    // -------------------------------------------------------------------------
    // ENCODE PQ BT.2020 OUTPUT
    // -------------------------------------------------------------------------
    vec3 rgb_pq = linear709_to_pq2020(rgb_expanded);

    // -------------------------------------------------------------------------
    // DITHER (PQ Space)
    // -------------------------------------------------------------------------
    #if ENABLE_DITHER
        float expansion_amount = expansion - 1.0;

        if (expansion_amount > 0.05) {
            float dither_magnitude = (1.0 / 1023.0) * sqrt(expansion_amount) * DITHER_STRENGTH;

            // PQ-level-aware dither: scale magnitude by cubic approximation of PQ derivative.
            // More dither at brighter PQ levels where banding from expanded 8-bit source is
            // more visible. Baseline unchanged at PQ ~0.5. Cost: 3 multiplies.
            float pq_scale = max(rgb_pq.g, 0.45) / 0.5;
            float pq_correction = pq_scale * pq_scale * pq_scale;
            dither_magnitude *= pq_correction;

            vec2 noise_coord = gl_FragCoord.xy;

            #if DITHER_TEMPORAL
                // Different irrationals for x and y break diagonal correlation
                noise_coord += vec2(float(frame) * 0.7548776662,
                                    float(frame) * 0.5698402917) * 100.0;
            #endif

            float noise = triangularNoise(noise_coord) * 0.5;
            rgb_pq += dither_magnitude * noise;

            #if DEBUG_SHOW_DITHER
                float mag_viz = dither_magnitude * 100.0;
                return vec4(gamma709_to_pq2020(vec3(mag_viz, mag_viz * 0.5 + noise * 0.5 + 0.25, 0.0)), 1.0);
            #endif
        }
    #endif

    return vec4(rgb_pq, color.a);
}

// =============================================================================
// PQ BT.2020 OUTPUT REFERENCE
// =============================================================================
// Input at MAIN: gamma-encoded BT.709 (BT.1886, gamma ~2.4)
// Output: PQ-encoded BT.2020 (libplacebo applies PQ EOTF to recover linear)
//
// | SDR Gamma | Linear | Nits (@100) | PQ Output |
// |-----------|--------|-------------|-----------|
// | 1.00      | 1.00   | 100         | ~0.501    |
// | 0.78      | 0.57   | 57          | ~0.433    |
// | 0.50      | 0.19   | 19          | ~0.328    |
// | 0.25      | 0.04   | 4.0         | ~0.191    |
// | Expanded (linear BT.709 > 1.0):              |
// | -         | 2.00   | 200         | ~0.558    |
// | -         | 2.70   | 270         | ~0.590    |
// =============================================================================
