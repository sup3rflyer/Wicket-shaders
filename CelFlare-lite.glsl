// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE

//!HOOK MAIN
//!BIND HOOKED
//!DESC CelFlare Lite v2.3 (Static SDR→HDR)

// =============================================================================
// CELFLARE LITE v2.3 - Static SDR→HDR Highlight Expansion (PQ BT.2020 Output)
// =============================================================================
//
// Lightweight variant of CelFlare v2.3. Uses the same processing pipeline
// (grain stabilization, Oklab sat boost + hue correction, PQ BT.2020 output)
// but with static expansion parameters instead of scene-adaptive analysis.
// No persistent buffers, no temporal state, no scene detection.
//
// WORKFLOW:
//   1. Read pre-filtered grain luma from alpha (CelFlare-blur.glsl pre-pass)
//   2. Linearize with BT.1886 EOTF for expansion math
//   3. Apply static expansion curve in linear space
//   4. Merged Oklab sat boost + hue correction
//   5. Convert BT.709 → BT.2020 gamut
//   6. Encode to PQ (ST.2084) for HDR output
//   7. PQ-aware dither
//
// REQUIREMENTS:
//   - mpv v0.41.0+ (for --hdr-reference-white support)
//   - vo=gpu-next with libplacebo
//
// CRITICAL SETTINGS:
//   Set hdr-reference-white to match your Windows SDR brightness (nits).
//   Set REFERENCE_WHITE below to match hdr-reference-white.
//   The vf-append line re-tags source metadata so libplacebo interprets
//   the shader's PQ BT.2020 output correctly (no pixel conversion).
//
//   Example mpv.conf profile:
//     [sdr-to-hdr]
//     profile-restore=copy
//     target-trc=pq
//     target-prim=bt.2020
//     target-peak=10000
//     hdr-reference-white=116    # Match your Windows SDR brightness
//     tone-mapping=clip
//     gamut-mapping-mode=clip
//     vf-append=format:gamma=pq:primaries=bt.2020
//     glsl-shaders-append=~~/shaders/CelFlare-blur.glsl
//     glsl-shaders-append=~~/shaders/CelFlare-lite.glsl
//
// =============================================================================

// =============================================================================
// USER CONTROLS
// =============================================================================

// --- Core Expansion (Static) ---
#define INTENSITY 1.4              // 0.5 = subtle, 1.0 = normal, 1.5+ = aggressive
#define HIGHLIGHT_PEAK 2.0         // Max expansion multiplier (default scene peak)
#define CURVE_STEEPNESS 3.5        // Exponential steepness (higher = more highlight-focused)
#define KNEE_END 0.40              // Where expansion begins (linear luminance)
#define MASTER_KNEE_OFFSET 0.15    // master_knee = KNEE_END - MASTER_KNEE_OFFSET

// --- Saturation Boost (Oklab) ---
// Counters the "silvery" desaturated look from expanding luminance without chroma.
// Uses Oklab color space for perceptually uniform chroma scaling with perfect hue preservation.
#define ENABLE_SAT_BOOST 1
#define SAT_BOOST_STRENGTH 1.15    // Boost for low-sat colors
#define SAT_BOOST_MAX 1.2          // Saturated enough to counter silvery expansion
#define SAT_HIGHLIGHT_ROLLOFF 0.8  // Normalized threshold for highlight desaturation
#define SAT_HIGHLIGHT_DESAT 0.8    // Target sat multiplier at peak (only for low-sat sources)

// --- Hue Correction (Oklab) ---
// Corrects hue shifts from saturation boost gamut clipping (e.g. yellow→green on fires).
#define ENABLE_HUE_CORRECTION 1
#define HUE_CORRECTION_STRENGTH 0.90  // 0.0 = none, 1.0 = full (0.75 recommended)

// --- Grain Stabilization ---
// Stabilizes expansion decision for grainy content by averaging similar neighbors.
// Reads pre-filtered luma from CelFlare-blur.glsl (alpha channel).
#define ENABLE_GRAIN_STABLE 1

// --- Dither ---
// Masks 8-bit quantization artifacts that get amplified by expansion.
// Applied in PQ space (perceptually uniform).
#define ENABLE_DITHER 1
#define DITHER_STRENGTH 0.7
#define DITHER_TEMPORAL 1          // Animate noise per frame (integrates temporally)

// --- EOTF ---
// Gamma used for linearization. BT.1886 = 2.4 (broadcast standard), sRGB ≈ 2.2.
#define EOTF_GAMMA 2.3

// --- PQ Output ---
// Shader encodes to PQ BT.2020 directly to bypass libplacebo's SDR peak clipping.
// Must match hdr-reference-white in mpv.conf and Windows SDR brightness.
#define REFERENCE_WHITE 116.0
#define PQ_FAST_APPROX 1       // 0 = exact ST.2084, 1 = degree-7 polynomial (~2.4x faster PQ encoding)

// =============================================================================
// ADVANCED TUNING
// =============================================================================

// --- Grain Stabilization Tuning (Gamma-Space Thresholds) ---
#define GRAIN_THRESHOLD 0.12       // Bilateral similarity threshold (tighter = edge-preserving)
#define GRAIN_BLUR_RADIUS 7.0      // Sample distance in blur pass (half-res pixels ≈ 14px original)
#define GRAIN_EDGE_LOW 0.06        // Edge gradient below this: fully blurred (grain area)
#define GRAIN_EDGE_HIGH 0.15       // Edge gradient above this: fully original (edge area)
#define GRAIN_RANGE_MIN 0.25       // Fixed lower limit (gamma space)
#define GRAIN_RANGE_MAX 0.95       // Upper limit (gamma space)
#define EARLY_EXIT_GAMMA 0.10      // Skip very dark pixels (gamma) - below any possible knee

// --- Saturation Rolloff (Oklab chroma-based, separate from sat boost) ---
// Reduces expansion on already-saturated colors to prevent clipping.
#define ENABLE_SAT_ROLLOFF 1
#define SAT_THRESHOLD 0.4          // HSV saturation threshold to start rolloff
#define SAT_POWER 10.0             // Rolloff curve steepness
#define SAT_ROLLOFF 0.80           // Reduces expansion on saturated colors

// =============================================================================
// DEBUG
// =============================================================================
// Enable one at a time to visualize different stages of the pipeline.

#define DEBUG_BYPASS 0             // Skip all processing, output PQ passthrough
#define DEBUG_SHOW_MASK 0          // Show expansion amount as grayscale
#define DEBUG_SHOW_SAT 0           // Show saturation rolloff factor
#define DEBUG_SHOW_GRAIN 0         // Show grain stabilization effect
#define DEBUG_SHOW_DITHER 0        // Show dither magnitude and pattern

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

// Inverse EOTF (gamma encode): linear -> gamma
vec3 oetf_gamma(vec3 v) {
    return pow(max(v, 0.0), vec3(1.0 / EOTF_GAMMA));
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
// Near-zero (< 5 nits) has larger error but those channels are invisible in highlights.
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
vec3 linear709_to_pq2020(vec3 rgb_linear) {
    vec3 bt2020 = bt709_to_bt2020(rgb_linear);
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

// Fast cube root via bit manipulation + Newton-Raphson (~4× faster than pow)
float fast_cbrt(float x) {
    if (x <= 0.0) return 0.0;
    uint i = floatBitsToUint(x);
    i = i / 3u + 0x2a514067u;
    float y = uintBitsToFloat(i);
    y = y * 0.666666667 + x / (3.0 * y * y);
    return y;
}

// Linear sRGB to Oklab
vec3 rgb_to_oklab(vec3 rgb) {
    float l = 0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b;
    float m = 0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b;
    float s = 0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b;

    float l_ = fast_cbrt(l);
    float m_ = fast_cbrt(m);
    float s_ = fast_cbrt(s);

    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

// Oklab to linear sRGB
vec3 oklab_to_rgb(vec3 lab) {
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    return vec3(
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    );
}

// =============================================================================
// MAIN PROCESSING
// =============================================================================

vec4 hook() {
    vec4 color = HOOKED_texOff(0);
    vec3 rgb_gamma = color.rgb;

    #if DEBUG_BYPASS
        return vec4(gamma709_to_pq2020(color.rgb), color.a);
    #endif

    float Y_gamma = get_luma(rgb_gamma);

    // Early exit for very dark pixels (below any possible expansion knee)
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
    #if ENABLE_GRAIN_STABLE
        Y_decision_gamma = color.a;

        #if DEBUG_SHOW_GRAIN
            float grain_diff = abs(Y_gamma - Y_decision_gamma) * 20.0;
            return vec4(gamma709_to_pq2020(vec3(grain_diff, 0.0, 0.0)), 1.0);
        #endif
    #endif

    // -------------------------------------------------------------------------
    // LINEARIZE FOR EXPANSION
    // -------------------------------------------------------------------------
    float Y_decision = eotf_gamma(Y_decision_gamma);
    vec3 rgb_linear = eotf_gamma(rgb_gamma);

    // -------------------------------------------------------------------------
    // SATURATION (Oklab chroma - perceptually uniform)
    // -------------------------------------------------------------------------
    vec3 oklab_orig = rgb_to_oklab(rgb_linear);
    float chroma_orig = sqrt(oklab_orig.y * oklab_orig.y + oklab_orig.z * oklab_orig.z);
    // Normalize to 0-1 range (0.35 ≈ max sRGB chroma) for threshold compatibility
    float sat_raw = chroma_orig / 0.35;
    float sat = pow(smoothstep(SAT_THRESHOLD, 1.0, sat_raw), SAT_POWER);

    #if DEBUG_SHOW_SAT
        return vec4(gamma709_to_pq2020(vec3(sat, chroma_orig * 3.0, 0.0)), 1.0);
    #endif

    // -------------------------------------------------------------------------
    // EXPANSION CURVE (Linear Space, Static Parameters)
    // -------------------------------------------------------------------------
    float master_knee = max(KNEE_END - MASTER_KNEE_OFFSET, 0.05);
    float final_steepness = min(CURVE_STEEPNESS, 20.0);  // Cap prevents exp() overflow

    float master_knee_t = step(master_knee, Y_decision);
    float master_expand_t = smoothstep(master_knee, 1.0, Y_decision);

    // Normalized exponential with L'Hôpital guard
    float master_curve = (final_steepness > 0.001)
        ? (exp(final_steepness * master_expand_t) - 1.0) / (exp(final_steepness) - 1.0)
        : master_expand_t;

    float expansion = mix(1.0, mix(1.0, HIGHLIGHT_PEAK, master_curve), master_knee_t);

    // Saturation rolloff
    #if ENABLE_SAT_ROLLOFF
        expansion = mix(expansion, 1.0, sat * SAT_ROLLOFF);
    #endif

    // Intensity multiplier
    expansion = 1.0 + (expansion - 1.0) * INTENSITY;

    // Early exit for non-expanded pixels
    if (expansion < 1.001) {
        return vec4(gamma709_to_pq2020(color.rgb), color.a);
    }

    #if DEBUG_SHOW_MASK
        float exp_amount = (expansion - 1.0) / 2.5;
        return vec4(gamma709_to_pq2020(vec3(exp_amount, exp_amount * 0.3, 0.0)), 1.0);
    #endif

    // -------------------------------------------------------------------------
    // APPLY EXPANSION (Linear Space)
    // -------------------------------------------------------------------------
    vec3 rgb_expanded = rgb_linear * expansion;

    // -------------------------------------------------------------------------
    // SATURATION BOOST + HUE CORRECTION (Merged Single Oklab Pass)
    // -------------------------------------------------------------------------
    // Merged into one Oklab round-trip to save ALU and eliminate the
    // intermediate gamut clip that would distort hue before correction.
    #if ENABLE_SAT_BOOST || ENABLE_HUE_CORRECTION
    {
        vec3 oklab_exp = rgb_to_oklab(rgb_expanded);

        #if ENABLE_SAT_BOOST
            float base_boost = sqrt(max(expansion, 0.0));
            float sat_boost = mix(1.0, base_boost, SAT_BOOST_STRENGTH);
            sat_boost = min(sat_boost, SAT_BOOST_MAX);
            sat_boost = mix(sat_boost, 1.0, sat);  // Don't boost already-saturated colors

            // Highlight rolloff using linear luminance
            float Y_expanded = get_luma(rgb_expanded);
            float Y_original = Y_expanded / max(expansion, 1.0);
            float highlight_t = smoothstep(SAT_HIGHLIGHT_ROLLOFF, 1.0, Y_original);
            float rolloff_strength = 1.0 - sat;
            sat_boost = mix(sat_boost, SAT_HIGHLIGHT_DESAT, highlight_t * rolloff_strength);

            // Scale a/b (chroma) while preserving L (lightness) and hue angle
            oklab_exp.y *= sat_boost;
            oklab_exp.z *= sat_boost;
        #endif

        #if ENABLE_HUE_CORRECTION
        {
            float chroma_post = length(oklab_exp.yz);
            float chroma_ref = length(oklab_orig.yz);

            if (chroma_post > 0.001 && chroma_ref > 0.001) {
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

        // Single conversion back to linear RGB (one gamut clip instead of two)
        rgb_expanded = max(oklab_to_rgb(oklab_exp), 0.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // ENCODE PQ BT.2020 OUTPUT
    // -------------------------------------------------------------------------
    vec3 rgb_pq = linear709_to_pq2020(max(rgb_expanded, 0.0));

    // -------------------------------------------------------------------------
    // DITHER (PQ Space)
    // -------------------------------------------------------------------------
    #if ENABLE_DITHER
        float expansion_amount = expansion - 1.0;

        if (expansion_amount > 0.05) {
            float dither_magnitude = (1.0 / 1023.0) * sqrt(expansion_amount) * DITHER_STRENGTH;

            // PQ-level-aware dither: scale magnitude by cubic approximation of PQ derivative.
            // More dither at brighter PQ levels where banding from expanded 8-bit source is
            // more visible. Cost: 3 multiplies.
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
