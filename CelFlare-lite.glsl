// Copyright (C) 2026 Agust Ari
// Licensed under GPL-3.0 — see LICENSE

//!HOOK MAIN
//!BIND HOOKED
//!DESC CelFlare Lite Grain Pre-filter

#define STABILIZE_OPACITY   0.85
#define GRAIN_THRESHOLD     0.28
#define GRAIN_BLUR_RADIUS   14.0

#define GRAIN_RANGE_MIN     0.35
#define GRAIN_RANGE_MAX     0.95
#define GRAIN_EDGE_LOW      0.20
#define GRAIN_EDGE_HIGH     0.45
#define GRAIN_EARLY_EXIT    0.25

#define BILATERAL_SHARPNESS 4.8

#define INNER_RING_BOOST    2.0

vec4 hook() {
    vec4 original = HOOKED_tex(HOOKED_pos);
    const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
    float Y_gamma = dot(original.rgb, luma_coeff);

    if (Y_gamma < GRAIN_EARLY_EXIT) {
        return vec4(original.rgb, Y_gamma);
    }

    float range_mask = smoothstep(GRAIN_RANGE_MIN - 0.05, GRAIN_RANGE_MIN, Y_gamma)
                     * (1.0 - smoothstep(GRAIN_RANGE_MAX, GRAIN_RANGE_MAX + 0.05, Y_gamma));

    if (range_mask < 0.01) {
        return vec4(original.rgb, Y_gamma);
    }

    vec2 pixel = floor(HOOKED_pos * HOOKED_size);
    float angle = fract(sin(dot(pixel, vec2(12.9898, 78.233))) * 43758.5453) * 6.2832;
    float ca = cos(angle);
    float sa = sin(angle);

    const vec2 outer[6] = vec2[6](
        vec2( 1.000, 0.000), vec2( 0.500, 0.866), vec2(-0.500, 0.866),
        vec2(-1.000, 0.000), vec2(-0.500, -0.866), vec2( 0.500, -0.866)
    );

    const vec2 inner[6] = vec2[6](
        vec2( 0.866, 0.500), vec2( 0.000, 1.000), vec2(-0.866, 0.500),
        vec2(-0.866, -0.500), vec2( 0.000, -1.000), vec2( 0.866, -0.500)
    );

    float asym_scale = mix(1.0, 7.0, smoothstep(0.55, 0.98, Y_gamma));
    float effective_sharpness = BILATERAL_SHARPNESS * mix(1.0, 0.82, smoothstep(0.60, 1.0, Y_gamma));

    float total_w = 1.0;
    float blurred = Y_gamma;
    float gx = 0.0, gy = 0.0, grad_w = 0.0;

    float raw_diff, asym, diff, w, s;
    vec2 rotated, offset;

    for (int i = 0; i < 6; i++) {
        vec2 h = outer[i];
        rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        offset = HOOKED_pt * GRAIN_BLUR_RADIUS * rotated;
        s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        raw_diff = s - Y_gamma;
        asym = raw_diff < 0.0 ? asym_scale : 1.0;
        diff = (raw_diff * asym) / GRAIN_THRESHOLD;
        w = exp(-effective_sharpness * diff * diff);
        blurred += s * w;
        total_w += w;

        gx += s * w * rotated.x;
        gy += s * w * rotated.y;
        grad_w += w;
    }

    float inv_grad_w = grad_w > 0.0 ? 1.0 / grad_w : 0.0;
    gx *= inv_grad_w;
    gy *= inv_grad_w;

    for (int i = 0; i < 6; i++) {
        vec2 h = inner[i];
        rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        offset = HOOKED_pt * GRAIN_BLUR_RADIUS * 0.5 * rotated;
        s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        raw_diff = s - Y_gamma;
        asym = raw_diff < 0.0 ? asym_scale : 1.0;
        diff = (raw_diff * asym) / GRAIN_THRESHOLD;
        w = exp(-effective_sharpness * diff * diff) * INNER_RING_BOOST;
        blurred += s * w;
        total_w += w;
    }

    blurred /= total_w;

    float edge = sqrt(gx * gx + gy * gy) / 2.7;
    float edge_mask = smoothstep(GRAIN_EDGE_LOW, GRAIN_EDGE_HIGH, edge);

    float Y_stabilized = mix(blurred, Y_gamma, edge_mask * 0.97);
    float Y_decision = mix(Y_gamma, Y_stabilized, range_mask * STABILIZE_OPACITY);

    return vec4(original.rgb, Y_decision);
}

//!HOOK MAIN
//!BIND HOOKED
//!DESC CelFlare Lite v3.0 (Static SDR->HDR)

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
// Perceptual chroma compensation via Stevens' power law in Oklab space.
// RGB expansion already scales chroma by cbrt(k); this adds the remaining k^(1/6) gap.
#define ENABLE_SAT_BOOST 0
#define SAT_BOOST_EXPONENT 0.267   // k^(1/6) base + empirical offset for PQ compression / Hunt effect
#define SAT_BOOST_MAX 1.25         // Safety cap
#define SAT_KNEE_OFFSET 0.10       // Extend chroma boost below luminance knee (linear luma, 0 = disabled)
#define SAT_KNEE_PEAK 0.04         // Max chroma boost at knee boundary (0.04 = 4%)

// --- Bezold-Brücke Warmth Compensation ---
// Pre-compensates for perceptual yellow→green hue shift at higher output luminances.
// Clockwise rotation in Oklab ab plane; warm mask targets yellow-orange, excludes red/blue.
#define ENABLE_BB_WARMTH 1
#define BB_WARMTH 0.03             // Rotation in radians; try 0.03–0.10

// --- Grain Stabilization ---
// Reads pre-filtered luma from CelFlare-blur.glsl (alpha channel).
#define ENABLE_GRAIN_STABLE 1

// --- Dither ---
// PQ-space dither to mask 8-bit banding amplified by expansion.
#define ENABLE_DITHER 1
#define DITHER_STRENGTH 0.7
#define DITHER_TEMPORAL 1          // Animate noise per frame

// --- EOTF ---
// BT.1886 = 2.4 (Rec.709 standard). sRGB ≈ 2.2.
#define EOTF_GAMMA 2.4

// --- PQ Output ---
// Direct PQ BT.2020 encoding. Must match hdr-reference-white in mpv.conf.
#define REFERENCE_WHITE 116.0
#define PQ_FAST_APPROX 1           // Degree-7 polynomial (~2.4x faster than exact ST.2084)

// =============================================================================
// ADVANCED TUNING
// =============================================================================

// --- Grain Stabilization ---
// Reads pre-filtered luma from CelFlare-blur.glsl (alpha channel).
#define EARLY_EXIT_GAMMA 0.25      // Skip dark pixels where chroma processing is imperceptible (~3 nits)

// --- Saturation Rolloff ---
// Reduces expansion on already-saturated colors to prevent gamut clipping.
#define ENABLE_SAT_ROLLOFF 1
#define SAT_THRESHOLD 0.4          // Normalized Oklab chroma threshold
#define SAT_POWER 10.0             // Rolloff curve steepness
#define SAT_ROLLOFF 0.80           // Max expansion reduction

// --- Chroma-Adaptive Luminance (H-K compensation) ---
// Lifts warm saturated skin and compresses pale skin to counter perceived luminance
// separation after expansion. Keep strength subtle.
#define ENABLE_CHROMA_EXPAND 1
#define CHROMA_EXPAND_STRENGTH 0.12   // Try 0.08–0.15 for anime
#define CHROMA_EXPAND_PIVOT 0.20      // Normalized Oklab chroma crossover (~pale/warm skin)
#define CE_CHROMA_CEIL 0.40           // Normalized chroma above which effect fades (0.40 = chroma 0.14)

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

    // Early exit for dark pixels where chroma processing is imperceptible
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
    float Y_decision = eotf_gamma(Y_decision_gamma);
    vec3 rgb_linear = eotf_gamma(rgb_gamma);

    // -------------------------------------------------------------------------
    // SATURATION (Oklab chroma, used by sat rolloff + chroma expand)
    // -------------------------------------------------------------------------
    vec3 oklab_orig = rgb_to_oklab(rgb_linear);
    float chroma_orig = sqrt(oklab_orig.y * oklab_orig.y + oklab_orig.z * oklab_orig.z);
    float sat_raw = chroma_orig / 0.35;  // Normalized to 0-1 (0.35 ≈ max sRGB chroma)
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

    // Below-knee chroma ramp + H-K compensation (before early exit)
    #if ENABLE_SAT_BOOST
        float sat_knee = max(master_knee - SAT_KNEE_OFFSET, 0.0);
        float knee_chroma = (SAT_KNEE_OFFSET > 0.0)
            ? smoothstep(sat_knee, master_knee, Y_decision) * SAT_KNEE_PEAK
            : 0.0;
    #endif

    #if ENABLE_CHROMA_EXPAND
        float ce_chroma_n = clamp(chroma_orig / 0.35, 0.0, 1.0);
        float skin_hue = smoothstep(-0.01, 0.03, oklab_orig.y)
                       * smoothstep(-0.01, 0.03, oklab_orig.z);
        float chroma_gate = smoothstep(0.05, 0.15, ce_chroma_n)
                          * (1.0 - smoothstep(CE_CHROMA_CEIL, CE_CHROMA_CEIL + 0.20, ce_chroma_n));
        float warm_t = skin_hue * chroma_gate;
        float ce_delta = CHROMA_EXPAND_STRENGTH * (ce_chroma_n - CHROMA_EXPAND_PIVOT) * warm_t;
    #else
        float ce_delta = 0.0;
    #endif

    // Early exit for non-expanded pixels
    if (expansion < 1.001) {
        #if ENABLE_SAT_BOOST || ENABLE_CHROMA_EXPAND
        {
            float early_sat = 0.0;
            #if ENABLE_SAT_BOOST
                early_sat = knee_chroma;
            #endif
            if (early_sat > 0.001 || abs(ce_delta) > 0.001) {
                vec3 rgb_adj = rgb_linear * (1.0 + ce_delta);
                float Y_adj = get_luma(rgb_adj);
                vec3 rgb_out = mix(vec3(Y_adj), rgb_adj, 1.0 + early_sat);
                return vec4(linear709_to_pq2020(rgb_out), color.a);
            }
        }
        #endif
        return vec4(gamma709_to_pq2020(color.rgb), color.a);
    }

    #if DEBUG_SHOW_MASK
        float exp_amount = (expansion - 1.0) / 2.5;
        return vec4(gamma709_to_pq2020(vec3(exp_amount, exp_amount * 0.3, 0.0)), 1.0);
    #endif

    // -------------------------------------------------------------------------
    // APPLY EXPANSION (Linear Space)
    // -------------------------------------------------------------------------
    #if ENABLE_CHROMA_EXPAND
        expansion *= (1.0 + ce_delta);
    #endif
    vec3 rgb_expanded = rgb_linear * expansion;

    // -------------------------------------------------------------------------
    // CHROMA PROCESSING (Single Oklab Round-Trip)
    // -------------------------------------------------------------------------
    #if ENABLE_SAT_BOOST || ENABLE_BB_WARMTH
    {
        vec3 oklab_exp = rgb_to_oklab(rgb_expanded);

        #if ENABLE_SAT_BOOST
            float sat_boost = min(pow(max(expansion, 0.0), SAT_BOOST_EXPONENT), SAT_BOOST_MAX);
            sat_boost = max(sat_boost, 1.0 + knee_chroma);

            oklab_exp.y *= sat_boost;
            oklab_exp.z *= sat_boost;
        #endif

        #if ENABLE_BB_WARMTH
        {
            float bb_chroma = length(oklab_exp.yz);
            if (bb_chroma > 0.001) {
                float warm_ratio = oklab_exp.z / (abs(oklab_exp.y) + bb_chroma);
                float warm_t = smoothstep(0.15, 0.65, warm_ratio);
                float exp_t  = smoothstep(0.05, 0.80, expansion - 1.0);
                oklab_exp.yz += (BB_WARMTH * warm_t * exp_t) * vec2(oklab_exp.z, -oklab_exp.y);
                float new_chroma = length(oklab_exp.yz);
                if (new_chroma > 0.001) oklab_exp.yz *= bb_chroma / new_chroma;
            }
        }
        #endif

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
