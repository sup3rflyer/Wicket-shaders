// CelFlare v4.0 — Illumination-Decomposition SDR→HDR Expansion
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// Expansion driven by regional illumination context (bright-biased blur of
// the luminance field) rather than per-pixel luminance. All pixels in a
// region get the same expansion multiplier — local contrast preserved by
// construction via multiplicative application.
//
// You MUST set REFERENCE_WHITE to match your SDR white and hdr-reference-white
// in mpv.conf.
//
// Search for "MAIN TUNING" in the expansion pass for user-facing controls.

//!BUFFER SCENE_STATE
//!VAR float smoothed_bright_frac
//!VAR float smoothed_scorch_frac
//!VAR float smoothed_contrast
//!VAR float smoothed_log_avg
//!VAR float scene_cut_lockout
//!VAR float prev_illum[16]
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!DESC CelFlare: Grain Pre-filter

#define STABILIZE_OPACITY   0.85
#define GRAIN_THRESHOLD     0.28
#define GRAIN_BLUR_RADIUS   24.0
#define GRAIN_RANGE_MIN     0.35
#define GRAIN_RANGE_MAX     0.95
#define GRAIN_EDGE_LOW      0.20
#define GRAIN_EDGE_HIGH     0.45
#define GRAIN_EARLY_EXIT    0.25
#define BILATERAL_SHARPNESS 6.0
#define INNER_RING_BOOST    3.0

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

    const vec2 inner[3] = vec2[3](
        vec2( 0.866, 0.500), vec2(-0.866, 0.500), vec2( 0.000, -1.000)
    );

    float asym_scale = mix(1.0, 7.0, smoothstep(0.55, 0.98, Y_gamma));
    float effective_sharpness = BILATERAL_SHARPNESS * mix(1.0, 0.82, smoothstep(0.60, 1.0, Y_gamma));

    float total_w = 1.0;
    float blurred = Y_gamma;
    float gx = 0.0, gy = 0.0, grad_w = 0.0;

    float raw_diff, asym, diff, w, s, d2, t;
    vec2 rotated, offset;

    for (int i = 0; i < 6; i++) {
        vec2 h = outer[i];
        rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        offset = HOOKED_pt * GRAIN_BLUR_RADIUS * rotated;
        s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        raw_diff = s - Y_gamma;
        asym = raw_diff < 0.0 ? asym_scale : 1.0;
        diff = (raw_diff * asym) / GRAIN_THRESHOLD;
        d2 = effective_sharpness * diff * diff;
        t = 1.0 - d2 * 0.25;
        w = t > 0.0 ? t * t : 0.0;
        blurred += s * w;
        total_w += w;

        gx += s * w * rotated.x;
        gy += s * w * rotated.y;
        grad_w += w;
    }

    float inv_grad_w = grad_w > 0.0 ? 1.0 / grad_w : 0.0;
    gx *= inv_grad_w;
    gy *= inv_grad_w;

    for (int i = 0; i < 3; i++) {
        vec2 h = inner[i];
        rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        offset = HOOKED_pt * GRAIN_BLUR_RADIUS * 0.5 * rotated;
        s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        raw_diff = s - Y_gamma;
        asym = raw_diff < 0.0 ? asym_scale : 1.0;
        diff = (raw_diff * asym) / GRAIN_THRESHOLD;
        d2 = effective_sharpness * diff * diff;
        t = 1.0 - d2 * 0.25;
        w = (t > 0.0 ? t * t : 0.0) * INNER_RING_BOOST;
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

// =============================================================================
// PASS 2: COLOR DOWNSAMPLE (1/4 resolution)
// =============================================================================
// 4-tap bilinear from RAW pixel RGB (gamma-space), not grain-stabilized alpha.
// The illumination field is blurred at sigma=25 on 1/4 res — grain is
// destroyed by the Gaussian. RGB stored for clip diffusion; Y extracted
// downstream via dot product.

//!HOOK MAIN
//!BIND HOOKED
//!SAVE CELFLARE_DS
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC CelFlare: Downsample 1/4

vec4 hook() {
    vec2 pt = HOOKED_pt;
    vec3 rgb  = HOOKED_tex(HOOKED_pos + vec2(-pt.x, -pt.y)).rgb;
    rgb      += HOOKED_tex(HOOKED_pos + vec2( pt.x, -pt.y)).rgb;
    rgb      += HOOKED_tex(HOOKED_pos + vec2(-pt.x,  pt.y)).rgb;
    rgb      += HOOKED_tex(HOOKED_pos + vec2( pt.x,  pt.y)).rgb;
    rgb *= 0.25;
    return vec4(rgb, 1.0);
}

// =============================================================================
// PASS 3: ILLUMINATION BLUR H (1/4 resolution)
// =============================================================================
// Separable Gaussian. sigma=20 at 1/4 res = effective ~80px at 1080p full
// res. BRIGHT_BIAS must be 0 for correct separability — any nonzero value
// makes weights data-dependent, creating visible outline artifacts at every
// bright/dark boundary. The halo guard in the expansion pass handles
// boundary protection instead.

//!HOOK MAIN
//!BIND CELFLARE_DS
//!SAVE CELFLARE_BLUR_H
//!WIDTH CELFLARE_DS.w
//!HEIGHT CELFLARE_DS.h
//!DESC CelFlare: Illumination Blur H

// sigma=20, radius=40, stride=2 bilinear tap merging (20 pairs, 41 fetches)

vec4 hook() {
    const float go[20] = {
        1.4990625011, 3.4978125140, 5.4965625542, 7.4953126373,
        9.4940627791, 11.4928129950, 13.4915633008, 15.4903137120,
        17.4890642443, 19.4878149131, 21.4865657342, 23.4853167231,
        25.4840678954, 27.4828192666, 29.4815708524, 31.4803226681,
        33.4790747295, 35.4778270520, 37.4765796511, 39.4753325423
    };
    const float gw[20] = {
        1.9937632601, 1.9690117179, 1.9252307163, 1.8637044098,
        1.7862039805, 1.6949029750, 1.5922761869, 1.4809886391,
        1.3637815864, 1.2433622741, 1.1223035003, 1.0029579299,
        0.8873907200, 0.7773324819, 0.6741530676, 0.5788552548,
        0.4920862280, 0.4141638659, 0.3451143082, 0.2847170585
    };

    // Precomputed: 1.0 + 2.0 * sum(gw[]) = 47.98460032
    #define BLUR_INV_WSUM 0.02084002

    vec2 pt = CELFLARE_DS_pt;
    vec2 pos = CELFLARE_DS_pos;

    vec3 s0 = CELFLARE_DS_tex(pos).rgb;
    vec3 sum = s0;

    for (int i = 0; i < 20; i++) {
        vec3 sp = CELFLARE_DS_tex(pos + vec2(go[i] * pt.x, 0.0)).rgb;
        vec3 sn = CELFLARE_DS_tex(pos - vec2(go[i] * pt.x, 0.0)).rgb;
        sum += (sp + sn) * gw[i];
    }

    return vec4(sum * BLUR_INV_WSUM, 1.0);
}

// =============================================================================
// PASS 4: ILLUMINATION BLUR V (1/4 resolution)
// =============================================================================

//!HOOK MAIN
//!BIND CELFLARE_BLUR_H
//!SAVE CELFLARE_ILLUM
//!WIDTH CELFLARE_BLUR_H.w
//!HEIGHT CELFLARE_BLUR_H.h
//!DESC CelFlare: Illumination Blur V

vec4 hook() {
    const float go[20] = {
        1.4990625011, 3.4978125140, 5.4965625542, 7.4953126373,
        9.4940627791, 11.4928129950, 13.4915633008, 15.4903137120,
        17.4890642443, 19.4878149131, 21.4865657342, 23.4853167231,
        25.4840678954, 27.4828192666, 29.4815708524, 31.4803226681,
        33.4790747295, 35.4778270520, 37.4765796511, 39.4753325423
    };
    const float gw[20] = {
        1.9937632601, 1.9690117179, 1.9252307163, 1.8637044098,
        1.7862039805, 1.6949029750, 1.5922761869, 1.4809886391,
        1.3637815864, 1.2433622741, 1.1223035003, 1.0029579299,
        0.8873907200, 0.7773324819, 0.6741530676, 0.5788552548,
        0.4920862280, 0.4141638659, 0.3451143082, 0.2847170585
    };

    // Precomputed: 1.0 + 2.0 * sum(gw[]) = 47.98460032
    #define BLUR_INV_WSUM 0.02084002

    vec2 pt = CELFLARE_BLUR_H_pt;
    vec2 pos = CELFLARE_BLUR_H_pos;

    vec3 s0 = CELFLARE_BLUR_H_tex(pos).rgb;
    vec3 sum = s0;

    for (int i = 0; i < 20; i++) {
        vec3 sp = CELFLARE_BLUR_H_tex(pos + vec2(0.0, go[i] * pt.y)).rgb;
        vec3 sn = CELFLARE_BLUR_H_tex(pos - vec2(0.0, go[i] * pt.y)).rgb;
        sum += (sp + sn) * gw[i];
    }

    return vec4(sum * BLUR_INV_WSUM, 1.0);
}

// =============================================================================
// PASS 5: FRAME STATS (1x1 render pass)
// =============================================================================
// Samples illumination field at 4x4 grid. Computes frame-level metrics that
// modulate the expansion curve: average illumination, bright fraction (replaces
// the entire 7-type scene classifier from v3.2), and scene cut detection.

//!HOOK MAIN
//!BIND HOOKED
//!BIND SCENE_STATE
//!BIND CELFLARE_ILLUM
//!SAVE CELFLARE_STATS
//!WIDTH 1
//!HEIGHT 1
//!DESC CelFlare: Frame Stats

#define KNEE            0.40
#define ENABLE_SCORCH   0
#define SCORCH_THRESH   0.80

#define TEMPORAL_ALPHA      0.03
#define TEMPORAL_ALPHA_FAST 0.9
#define LOCKOUT_FRAMES      6.0
#define ILLUM_CHANGE_THRESH 0.06
#define SCENE_CUT_PCT       0.50

float get_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec4 hook() {
    float illum_sum = 0.0;
    float bright_count = 0.0;
    float change_count = 0.0;
    float illum_min = 1.0;
    float illum_max = 0.0;
    float log_luma_sum = 0.0;
    int valid_luma = 0;
    #if ENABLE_SCORCH
    float scorch_count = 0.0;
    #endif

    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int idx = y * 4 + x;
            vec2 spos = vec2((float(x) + 0.5) / 4.0,
                             (float(y) + 0.5) / 4.0);

            float illum = dot(CELFLARE_ILLUM_tex(spos).rgb, vec3(0.2126, 0.7152, 0.0722));
            illum_sum += illum;
            illum_min = min(illum_min, illum);
            illum_max = max(illum_max, illum);

            if (illum > KNEE) bright_count += 1.0;
            #if ENABLE_SCORCH
            if (illum > SCORCH_THRESH) scorch_count += 1.0;
            #endif

            if (frame > 0) {
                if (abs(illum - prev_illum[idx]) > ILLUM_CHANGE_THRESH)
                    change_count += 1.0;
            }
            prev_illum[idx] = illum;

            // Log-average from source pixels (skip black bars)
            vec3 rgb = HOOKED_tex(spos).rgb;
            float Y = get_luma(rgb);
            if (Y > 0.001) {
                log_luma_sum += log(max(Y, 1e-6));
                valid_luma++;
            }
        }
    }

    float avg_illum = illum_sum / 16.0;
    float bright_frac = bright_count / 16.0;

    // Contrast: dynamic range from illumination field (stable, noise-free)
    float contrast = (illum_min > 0.001)
        ? log2(max(illum_max / illum_min, 1.0))
        : 0.0;

    // Log-average: perceptual brightness key from source pixels
    float log_avg = (valid_luma > 4)
        ? exp(log_luma_sum / float(valid_luma))
        : avg_illum;

    // Scene cut detection
    scene_cut_lockout = max(scene_cut_lockout - 1.0, 0.0);
    float change_pct = change_count / 16.0;
    bool scene_cut = (change_pct > SCENE_CUT_PCT) && (scene_cut_lockout <= 0.0);

    if (scene_cut) scene_cut_lockout = LOCKOUT_FRAMES;

    float alpha = (scene_cut || scene_cut_lockout > 0.0)
                  ? TEMPORAL_ALPHA_FAST : TEMPORAL_ALPHA;

    if (frame == 0) {
        smoothed_bright_frac = 0.0;
        smoothed_contrast = contrast;
        smoothed_log_avg = log_avg;
        scene_cut_lockout = 0.0;
        #if ENABLE_SCORCH
        smoothed_scorch_frac = 0.0;
        #endif
    } else {
        smoothed_bright_frac = mix(smoothed_bright_frac, bright_frac, alpha);
        smoothed_contrast = mix(smoothed_contrast, contrast, alpha);
        smoothed_log_avg = mix(smoothed_log_avg, log_avg, alpha);
        #if ENABLE_SCORCH
        float scorch_frac = scorch_count / 16.0;
        smoothed_scorch_frac = mix(smoothed_scorch_frac, scorch_frac, alpha);
        #endif
    }

    return vec4(0);
}

// =============================================================================
// PASS 6: ILLUMINATION EXPANSION (full resolution)
// =============================================================================
// Core change from v3.2: expansion curve evaluated at Y_illum (bright-biased
// blur of regional luminance at ~100px effective scale) instead of per-pixel Y.
// All pixels in a bright region get the bright region's expansion — local
// contrast preserved by construction through multiplicative application.
//
// Scene adaptation via bright_frac (fraction of illumination above knee)
// replaces the 7-type scene classifier. Continuous, no arbitrary boundaries.

//!HOOK MAIN
//!BIND HOOKED
//!BIND SCENE_STATE
//!BIND CELFLARE_STATS
//!BIND CELFLARE_ILLUM
//!DESC CelFlare v4.0

// =============================================
//  MAIN TUNING — start here
// =============================================
#define INTENSITY       1.3     // Global scaling (0.5 subtle · 1.0 normal · 1.5 aggressive)
#define KNEE            0.40    // Expansion onset — midtones below this stay near SDR
#define MAX_EXPANSION   3.5     // Hard ceiling (3.5 = ~406 nits at 116 ref white)
#define SCORCH_BOOST    0.00    // Extra lift for near-white highlights (0 = off)

// =============================================
//  SPATIALLY-MODULATED CURVE — regional adaptation
// =============================================
// Expansion is always f(Y_pixel) — monotonic remapping, no 8-bit banding.
// Y_illum modulates the curve SHAPE: bright regions get gentle/broad curves
// (preserving face gradients), dark regions get steep/concentrated curves
// (highlight pop). Uses linear ramp + pow(t, gamma) — no smoothstep
// inflection point in the face brightness range.
//
// Nit targets at REFERENCE_WHITE=116:
//   Reference white (Y≈0.85): 150–200 nits
//   Highlights (Y≈0.90–0.95): 230–290 nits
//   Specular (Y≈0.95–1.00):   300–350 nits
//   Midtones (Y<0.50):         near SDR (~no expansion)
#define PEAK_BRIGHT     3.0     // Expansion peak for bright regions
#define PEAK_DARK       3.5     // Expansion peak for dark regions (highlight headroom)
#define GAMMA_BRIGHT    1.5     // Curve shape: moderate concentration
#define GAMMA_DARK      3.0     // Curve shape: strong highlight concentration
#define PEAK_ATTEN      0.35    // How much bright_frac further reduces peak (0.0–0.6)
#define BRIGHT_FRAC_REF 0.30    // Bright fraction where scene adaptation plateaus

// =============================================
//  DYNAMIC INTENSITY — contrast-driven expansion scaling
// =============================================
// Flat/pastel scenes (low contrast) get softer expansion.
// Dramatic/high-contrast scenes (deep shadows + bright highlights) get punchier.
// Driven by smoothed_contrast (log2 dynamic range in stops).
#define ENABLE_DYNAMIC_INTENSITY 1
#define DYN_CONTRAST_LOW    2.5     // Below this: flat scene, minimum intensity
#define DYN_CONTRAST_HIGH   5.5     // Above this: dramatic scene, maximum intensity
#define DYN_INTENSITY_LOW   0.75    // Multiplier for flat scenes
#define DYN_INTENSITY_HIGH  1.10    // Multiplier for dramatic scenes

// =============================================
//  APL MODULATION — brightness-driven expansion scaling
// =============================================
// Dark scenes: neutral (gamma handles midtone suppression).
// Bright scenes: dampened to prevent washing out.
// Driven by smoothed_log_avg (perceptual brightness key).
#define ENABLE_APL_MOD      1
#define APL_KEY_DARK        0.03    // Below this: dark scene multiplier
#define APL_KEY_BRIGHT      0.30    // Above this: bright scene multiplier
#define APL_BOOST_DARK      1.00    // Neutral for dark scenes (gamma_dark suppresses midtones)
#define APL_DAMPEN_BRIGHT   0.70    // Dampen bright scenes

// =============================================
//  SPECULAR BONUS — near-white highlight pop
// =============================================
// Extra expansion for near-white, low-saturation pixels (specular
// reflections, bright sky, white objects). Purely per-pixel f(Y) — no
// illumination field involvement (which caused spatial shadow contours).
//
// Wide smoothstep ramp (SPEC_Y_LOW to 1.0) + squaring ensures the bonus
// onset is vanishingly gentle: derivative is zero at SPEC_Y_LOW, and the
// per-8-bit-step expansion change stays sub-visible across the ramp.
// The bonus concentrates naturally toward Y=1.0.
#define ENABLE_SPECULAR_BONUS 0
#define SPEC_Y_LOW          0.40    // Ramp onset (wide = gentle 8-bit steps)
#define SPEC_CHROMA_FLOOR   0.15    // Below this: full bonus (white/near-neutral)
#define SPEC_CHROMA_CEIL    0.50    // Above this: no bonus (highly saturated only)
#define SPEC_BOOST          0.90    // Peak expansion bonus at Y=1.0

// Clip diffusion: blend near-white toward illum field to soften SDR clip edges
#define CLIP_DIFFUSION       0.30    // Max blend at Y=1.0 (0.0=off, 0.5=strong)
#define CLIP_DIFFUSION_FLOOR 0.85    // Y_gamma below: no softening

// =============================================
//  CHROMA — expansion color behavior
// =============================================
// Chroma amplification attenuation for saturated pixels. Full cbrt(expansion)
// on chroma causes saturated colors to appear perceptually brighter than
// desaturated highlights at the same luminance expansion (Helmholtz-Kohlrausch).
// CHROMA_SCALE reduces this: 1.0 = full cbrt, 0.5 = half, 0.0 = chroma frozen.
// Only affects already-saturated pixels — near-neutrals always get full cbrt.
#define CHROMA_SCALE        1.00

// --- Warm Tone Compensation ---
#define ENABLE_WARM_PROTECT 0
#define WP_LUM_BOOST        0.20
#define WP_HUE_COS          0.7317  // cos(0.75) — precomputed unit vector for hue center
#define WP_HUE_SIN          0.6816  // sin(0.75)
#define WP_HUE_POWER        2.0     // Sharpness of hue window (higher = narrower)
#define WP_CHROMA_MIN       0.08
#define WP_CHROMA_MAX       0.14
#define WP_LUM_FLOOR        0.40
#define WP_BRIGHT_FRAC_LOW  0.05
#define WP_BRIGHT_FRAC_HIGH 0.30

#define ENABLE_PALE_SKIN 1
#define ENABLE_PS_COMPRESS  0       // Pale skin expansion compression (0 = off)
#define PS_COMPRESS         0.00
#define PS_SAT_BOOST        0.10
#define PS_BRIGHT_FLOOR     0.50
#define PS_CHROMA_CEIL      0.03

// =============================================
//  OUTPUT — encoding
// =============================================
#define REFERENCE_WHITE 116.0
#define PQ_FAST_APPROX  1
#define EOTF_GAMMA      2.4
#define ENABLE_GRAIN_STABLE 1
#define EARLY_EXIT_GAMMA    0.25

// =============================================
//  DEBUG
// =============================================
#define DEBUG_BYPASS        0
#define DEBUG_SHOW_ILLUM    0   // Illumination field as grayscale
#define DEBUG_SHOW_EXPANSION 0  // Expansion amount as heat map
#define DEBUG_SHOW_SCORCH   0   // Scorch contribution
#define DEBUG_SHOW_DETAIL   0   // Spatial vs per-pixel: green=spatial, red=per-pixel fallback
#define DEBUG_SHOW_SPECULAR 0   // Specular bonus: cyan = spec strength
#define DEBUG_SHOW_WP       0   // Warm tone protection
#define DEBUG_SHOW_STATS    0   // avg_illum + bright_frac + contrast + log_avg bars

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

float get_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 eotf_gamma(vec3 v) {
    return pow(max(v, 0.0), vec3(EOTF_GAMMA));
}

float eotf_gamma(float v) {
    return pow(max(v, 0.0), EOTF_GAMMA);
}

vec3 bt709_to_bt2020(vec3 rgb) {
    return vec3(
        0.6274040 * rgb.r + 0.3292820 * rgb.g + 0.0433136 * rgb.b,
        0.0690970 * rgb.r + 0.9195400 * rgb.g + 0.0113612 * rgb.b,
        0.0163916 * rgb.r + 0.0880132 * rgb.g + 0.8955950 * rgb.b
    );
}

vec3 pq_oetf(vec3 L) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 Lm1 = pow(max(L, 0.0), vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

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

vec3 gamma709_to_pq2020(vec3 rgb_gamma) {
    vec3 linear = eotf_gamma(rgb_gamma);
    vec3 bt2020 = bt709_to_bt2020(linear);
    return pq_oetf(bt2020 * (REFERENCE_WHITE / 10000.0));
}

vec3 linear709_to_pq2020(vec3 rgb_linear) {
    vec3 bt2020 = max(bt709_to_bt2020(rgb_linear), 0.0);
    #if PQ_FAST_APPROX
        return pq_oetf_fast(bt2020 * (REFERENCE_WHITE / 10000.0));
    #else
        return pq_oetf(bt2020 * (REFERENCE_WHITE / 10000.0));
    #endif
}

// =============================================================================
// OKLAB COLOR SPACE (Bjorn Ottosson, 2020)
// =============================================================================

float fast_cbrt(float x) {
    if (x <= 0.0) return 0.0;
    uint i = floatBitsToUint(x);
    i = i / 3u + 0x2a514067u;
    float y = uintBitsToFloat(i);
    y = y * 0.666666667 + x / (3.0 * y * y);
    return y;
}

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
// ILLUMINATION FIELD UPSAMPLING (from 1/4 res)
// =============================================================================
// The illumination field is a sigma~100px Gaussian — extremely smooth.
// BSPLINE_UPSAMPLE 1: C2 cubic B-spline (4 fetches, smooth gradients)
// BSPLINE_UPSAMPLE 0: Hardware bilinear (1 fetch, sufficient for smooth fields)
#define BSPLINE_UPSAMPLE 0

vec3 upsample_illum_rgb() {
    #if BSPLINE_UPSAMPLE
    vec2 tc = CELFLARE_ILLUM_pos / CELFLARE_ILLUM_pt - 0.5;
    vec2 f  = fract(tc);
    tc = (floor(tc) + 0.5) * CELFLARE_ILLUM_pt;

    vec2 f2  = f * f, f3 = f2 * f;
    vec2 w0  = (1.0 - f) * (1.0 - f) * (1.0 - f) * (1.0 / 6.0);
    vec2 w1  = (4.0 - 6.0 * f2 + 3.0 * f3) * (1.0 / 6.0);
    vec2 w2  = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) * (1.0 / 6.0);
    vec2 w3  = f3 * (1.0 / 6.0);

    vec2 s01 = w0 + w1, s23 = w2 + w3;
    vec2 p01 = tc + (-1.0 + w1 / s01) * CELFLARE_ILLUM_pt;
    vec2 p23 = tc + ( 1.0 + w3 / s23) * CELFLARE_ILLUM_pt;

    return
        CELFLARE_ILLUM_tex(vec2(p01.x, p01.y)).rgb * s01.x * s01.y +
        CELFLARE_ILLUM_tex(vec2(p23.x, p01.y)).rgb * s23.x * s01.y +
        CELFLARE_ILLUM_tex(vec2(p01.x, p23.y)).rgb * s01.x * s23.y +
        CELFLARE_ILLUM_tex(vec2(p23.x, p23.y)).rgb * s23.x * s23.y;
    #else
    return CELFLARE_ILLUM_tex(CELFLARE_ILLUM_pos).rgb;
    #endif
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

    // -------------------------------------------------------------------------
    // DEBUG: Stats overlay (top-left bars)
    // -------------------------------------------------------------------------
    #if DEBUG_SHOW_STATS
    {
        vec2 pos = HOOKED_pos;
        if (pos.x < 0.15 && pos.y < 0.10) {
            float bar_x = pos.x / 0.15;
            vec3 dbg = vec3(0.05);
            float row = pos.y / 0.10;
            if (row < 0.333) {
                // Row 1: bright_frac (yellow)
                if (bar_x < smoothed_bright_frac) dbg = vec3(0.6, 0.6, 0.0);
            } else if (row < 0.666) {
                // Row 2: contrast / 8 stops (orange)
                if (bar_x < smoothed_contrast / 8.0) dbg = vec3(0.7, 0.4, 0.0);
            } else {
                // Row 3: log_avg (green)
                if (bar_x < smoothed_log_avg) dbg = vec3(0.0, 0.6, 0.0);
            }
            return vec4(gamma709_to_pq2020(dbg), 1.0);
        }
    }
    #endif

    // -------------------------------------------------------------------------
    // PIXEL LUMA
    // -------------------------------------------------------------------------
    float Y_gamma = get_luma(rgb_gamma);

    // -------------------------------------------------------------------------
    // ILLUMINATION FIELD (before early exit — clip diffusion needs this)
    // -------------------------------------------------------------------------
    vec3 illum_rgb = upsample_illum_rgb();
    float Y_illum = get_luma(illum_rgb);

    #if DEBUG_SHOW_ILLUM
        return vec4(gamma709_to_pq2020(illum_rgb), 1.0);
    #endif

    if (Y_gamma < EARLY_EXIT_GAMMA) return vec4(gamma709_to_pq2020(color.rgb), color.a);

    // -------------------------------------------------------------------------
    // GRAIN STABILIZATION
    // -------------------------------------------------------------------------
    float Y_decision_gamma = Y_gamma;
    #if ENABLE_GRAIN_STABLE
        Y_decision_gamma = (color.a > 0.99) ? Y_gamma : color.a;
    #endif

    // -------------------------------------------------------------------------
    // CLIP DIFFUSION — soften SDR clip edges via illumination field
    // -------------------------------------------------------------------------
    // Only modifies rgb_gamma — Y_decision_gamma stays grain-stabilized.
    {
        float soften = smoothstep(CLIP_DIFFUSION_FLOOR, 1.0, Y_gamma) * CLIP_DIFFUSION;
        rgb_gamma = mix(rgb_gamma, illum_rgb, soften);
        Y_gamma = get_luma(rgb_gamma);
    }

    // Cheap gamma-space chroma for specular detection (max-min channel spread)
    #if ENABLE_SPECULAR_BONUS
        float chroma_orig_gamma = max(rgb_gamma.r, max(rgb_gamma.g, rgb_gamma.b))
                                - min(rgb_gamma.r, min(rgb_gamma.g, rgb_gamma.b));
    #endif

    // -------------------------------------------------------------------------
    // SPATIALLY-MODULATED PER-PIXEL EXPANSION
    // -------------------------------------------------------------------------
    // Expansion is f(Y_pixel) — monotonic remapping, no 8-bit banding.
    // Y_illum modulates the curve parameters: bright regions get gentle/broad
    // curves (preserving face gradients), dark regions get steep/concentrated
    // curves (highlight pop in dark scenes).
    //
    // Linear ramp + pow(t, gamma): no smoothstep inflection. For gamma >= 1,
    // the derivative is monotonically increasing — no local maximum in the
    // face brightness range that would create visible contours.

    // Scene-level adaptation (from SSBO — uniform per frame)
    float bf = smoothstep(0.0, BRIGHT_FRAC_REF, smoothed_bright_frac);

    // Regional adaptation (from illumination field — varies per pixel)
    float spatial_t = Y_illum;
    float local_peak = mix(PEAK_DARK, PEAK_BRIGHT, spatial_t);
    local_peak *= (1.0 - PEAK_ATTEN * bf);  // scene-level dampening on top
    float local_gamma = mix(GAMMA_DARK, GAMMA_BRIGHT, spatial_t);

    // Per-pixel expansion curve: linear ramp from KNEE, shaped by pow(gamma)
    float t = max(Y_decision_gamma - KNEE, 0.0) / (1.0 - KNEE);
    t = pow(max(t, 0.0), local_gamma);
    float expansion = 1.0 + (local_peak - 1.0) * t * INTENSITY;

    #if DEBUG_SHOW_DETAIL
    {
        float exp_contrib = max(expansion - 1.0, 0.0) * 2.0;
        return vec4(gamma709_to_pq2020(vec3(0.0, exp_contrib, 0.0)), 1.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // STEP 3: DYNAMIC INTENSITY (contrast-driven scaling)
    // -------------------------------------------------------------------------
    // Flat scenes get softer expansion, dramatic scenes get punchier.
    #if ENABLE_DYNAMIC_INTENSITY
    {
        float dyn_factor = smoothstep(DYN_CONTRAST_LOW, DYN_CONTRAST_HIGH, smoothed_contrast);
        float dyn_intensity = mix(DYN_INTENSITY_LOW, DYN_INTENSITY_HIGH, dyn_factor);
        expansion = 1.0 + (expansion - 1.0) * dyn_intensity;
    }
    #endif

    // -------------------------------------------------------------------------
    // STEP 4: APL MODULATION (brightness-driven scaling)
    // -------------------------------------------------------------------------
    // Dark scenes get more headroom, bright scenes get dampened.
    #if ENABLE_APL_MOD
    {
        float apl_t = smoothstep(APL_KEY_DARK, APL_KEY_BRIGHT, smoothed_log_avg);
        float apl_factor = mix(APL_BOOST_DARK, APL_DAMPEN_BRIGHT, apl_t);
        expansion = 1.0 + (expansion - 1.0) * apl_factor;
    }
    #endif

    // -------------------------------------------------------------------------
    // SCORCH — near-white highlight boost
    // -------------------------------------------------------------------------
    #if ENABLE_SCORCH
    {
        float scorch_drive = min(Y_decision_gamma, Y_illum);
        float scorch_region = smoothstep(0.70, 0.92, scorch_drive);
        float scorch_pixel = smoothstep(SCORCH_PIXEL_FLOOR, SCORCH_PIXEL_FLOOR + 0.10, Y_decision_gamma);
        float scorch_t = scorch_region * scorch_pixel;
        scorch_t *= scorch_t;
        float scorch_shutoff = 1.0 - smoothstep(SCORCH_SHUTOFF_BEGIN, SCORCH_SHUTOFF_END, smoothed_scorch_frac);
        expansion += SCORCH_BOOST * scorch_t * scorch_shutoff;

        #if DEBUG_SHOW_SCORCH
        {
            float sc = scorch_t * scorch_shutoff;
            return vec4(gamma709_to_pq2020(vec3(sc, sc * 0.6, 0.0)), 1.0);
        }
        #endif
    }
    #endif

    // -------------------------------------------------------------------------
    // SPECULAR BONUS — near-white highlight pop
    // -------------------------------------------------------------------------
    // Purely f(Y_pixel): wide smoothstep ramp squared so derivative is zero
    // at onset — no 8-bit step contours. No Y_illum, no context ratio.
    #if ENABLE_SPECULAR_BONUS
    float spec_strength;
    {
        float spec_ramp = smoothstep(SPEC_Y_LOW, 1.0, Y_decision_gamma);
        float spec_chroma_w = 1.0 - smoothstep(SPEC_CHROMA_FLOOR, SPEC_CHROMA_CEIL, chroma_orig_gamma);
        spec_strength = SPEC_BOOST * spec_ramp * spec_ramp * spec_chroma_w;
        expansion += spec_strength;
    }
    #endif

    #if DEBUG_SHOW_SPECULAR && ENABLE_SPECULAR_BONUS
    {
        return vec4(gamma709_to_pq2020(vec3(0.0, spec_strength, spec_strength)), 1.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // EXPANSION CAP
    // -------------------------------------------------------------------------
    expansion = min(expansion, MAX_EXPANSION);

    // -------------------------------------------------------------------------
    // LINEARIZE
    // -------------------------------------------------------------------------
    vec3 rgb_linear = eotf_gamma(rgb_gamma);
    #if ENABLE_GRAIN_STABLE
    float Y_decision = (color.a > 0.99) ? get_luma(rgb_linear) : eotf_gamma(color.a);
    #else
    float Y_decision = get_luma(rgb_linear);
    #endif

    // -------------------------------------------------------------------------
    // OKLAB (for chroma detection)
    // -------------------------------------------------------------------------
    vec3 oklab_orig = rgb_to_oklab(rgb_linear);
    float chroma_orig = sqrt(oklab_orig.y * oklab_orig.y + oklab_orig.z * oklab_orig.z);

    // -------------------------------------------------------------------------
    // WARM TONE COMPENSATION + PALE SKIN PROTECTION
    // -------------------------------------------------------------------------
    #if ENABLE_WARM_PROTECT || ENABLE_PALE_SKIN
        // Dot-product hue detector: cos(angle) between pixel's ab vector and
        // target hue direction. Replaces atan()+exp() Gaussian (~25 fewer SFU
        // cycles). pow() controls window width — higher = narrower acceptance.
        float inv_chroma = (chroma_orig > 1e-6) ? (1.0 / chroma_orig) : 0.0;
        float cos_dh = (oklab_orig.y * WP_HUE_COS + oklab_orig.z * WP_HUE_SIN) * inv_chroma;
        float wp_hue_w = pow(max(cos_dh, 0.0), WP_HUE_POWER);
        float wp_gate = smoothstep(WP_BRIGHT_FRAC_LOW, WP_BRIGHT_FRAC_HIGH, smoothed_bright_frac);
    #endif

    #if ENABLE_WARM_PROTECT
        float wp_chroma_w = smoothstep(WP_CHROMA_MIN - 0.03, WP_CHROMA_MIN + 0.03, chroma_orig)
                          * (1.0 - smoothstep(WP_CHROMA_MAX, WP_CHROMA_MAX + 0.08, chroma_orig));
        float wp_lum_w = smoothstep(0.0, WP_LUM_FLOOR, Y_decision);
        float wp_w = wp_hue_w * wp_chroma_w * wp_lum_w;
        expansion += WP_LUM_BOOST * wp_w * wp_gate;
    #endif

    #if ENABLE_PALE_SKIN
        float ps_chroma_w = smoothstep(0.015, 0.06, chroma_orig)
                          * (1.0 - smoothstep(0.04, PS_CHROMA_CEIL + 0.04, chroma_orig));
        float ps_bright_w = smoothstep(PS_BRIGHT_FLOOR, PS_BRIGHT_FLOOR + 0.15, Y_decision);
        float ps_w = wp_hue_w * ps_chroma_w * ps_bright_w * wp_gate;
        #if ENABLE_PS_COMPRESS
        expansion = mix(expansion, 1.0, PS_COMPRESS * ps_w);
        #endif
        float ps_sat = PS_SAT_BOOST * ps_w;
    #endif

    #if DEBUG_SHOW_WP && (ENABLE_WARM_PROTECT || ENABLE_PALE_SKIN)
    {
        float wp_mag = 0.0;
        float ps_mag = 0.0;
        #if ENABLE_WARM_PROTECT
            wp_mag = WP_LUM_BOOST * wp_w * wp_gate * 10.0;
        #endif
        #if ENABLE_PALE_SKIN
            ps_mag = PS_COMPRESS * ps_w * 10.0;
        #endif
        return vec4(gamma709_to_pq2020(vec3(ps_mag, wp_mag, 0.0)), color.a);
    }
    #endif

    // -------------------------------------------------------------------------
    // EARLY EXIT: non-expanded pixels
    // -------------------------------------------------------------------------
    if (expansion < 1.001) {
        return vec4(gamma709_to_pq2020(color.rgb), color.a);
    }

    #if DEBUG_SHOW_EXPANSION
    {
        float exp_amount = (expansion - 1.0) / 2.5;
        return vec4(gamma709_to_pq2020(vec3(exp_amount, exp_amount * 0.3, 0.0)), 1.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // APPLY EXPANSION (Oklab Space)
    // -------------------------------------------------------------------------
    // L always scales by cbrt(expansion) (equivalent to linear RGB multiply).
    // Chroma scales by a REDUCED amount for saturated pixels — full cbrt
    // causes saturated colors to gain perceptual brightness from chroma
    // amplification that desaturated highlights don't get, inverting contrast
    // (yellow balloon > white specular, blonde hair > white shine).
    //
    // CHROMA_SCALE controls the attenuation: 1.0 = full cbrt (v3.2 behavior),
    // 0.5 = half chroma amplification, 0.0 = no chroma change.
    // Attenuation weighted by existing saturation — near-neutrals get full
    // cbrt (they need chroma to stay balanced), saturated pixels get reduced.
    vec3 oklab_exp = oklab_orig;
    float cbrt_exp = fast_cbrt(expansion);
    oklab_exp.x *= cbrt_exp;
    float sat_norm = smoothstep(0.10, 0.25, chroma_orig);
    float chroma_factor = mix(cbrt_exp, mix(1.0, cbrt_exp, CHROMA_SCALE), sat_norm);
    oklab_exp.yz *= chroma_factor;

    #if ENABLE_PALE_SKIN
        oklab_exp.yz *= (1.0 + ps_sat);
    #endif

    vec3 rgb_expanded = oklab_to_rgb(oklab_exp);

    // -------------------------------------------------------------------------
    // ENCODE PQ BT.2020 OUTPUT
    // -------------------------------------------------------------------------
    vec3 rgb_pq = linear709_to_pq2020(rgb_expanded);

    #if PQ_FAST_APPROX
    {
        float onset_blend = smoothstep(1.001, 1.05, expansion);
        if (onset_blend < 1.0) {
            vec3 rgb_pq_pass = gamma709_to_pq2020(rgb_gamma);
            rgb_pq = mix(rgb_pq_pass, rgb_pq, onset_blend);
        }
    }
    #endif

    // -------------------------------------------------------------------------
    // PQ-AWARE DITHER — break 8-bit source banding after expansion
    // -------------------------------------------------------------------------
    // 8-bit source has 256 gamma steps. After 2-3× expansion, each step is
    // amplified and becomes visible in 10-bit PQ output. Triangular dither
    // (sum of two uniforms) randomizes sub-step placement.
    // Integer hash with frame-based temporal variation (no sin() SFU calls).
    {
        uvec2 pixel = uvec2(floor(HOOKED_pos * HOOKED_size));
        uint seed = pixel.x + pixel.y * uint(HOOKED_size.x) + uint(frame) * 747796405u;
        // PCG hash (two rounds for two independent uniforms)
        seed = seed * 747796405u + 2891336453u;
        uint h1 = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
        h1 = (h1 >> 22u) ^ h1;
        seed = seed * 747796405u + 2891336453u;
        uint h2 = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
        h2 = (h2 >> 22u) ^ h2;
        float n1 = float(h1) / 4294967295.0;
        float n2 = float(h2) / 4294967295.0;
        float tri_noise = n1 + n2 - 1.0;  // triangular [-1, 1]
        // Scale: 1 ten-bit PQ step
        rgb_pq += tri_noise * (1.0 / 1023.0);
    }

    return vec4(rgb_pq, color.a);
}
