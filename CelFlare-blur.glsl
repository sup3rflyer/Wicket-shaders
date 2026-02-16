// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE
//
// CelFlare Grain Pre-filter — Bilateral luma stabilization packed into alpha
// Load BEFORE CelFlare.glsl or CelFlare-lite.glsl in shader list.
// Writes stabilized gamma luma to alpha channel for CelFlare's expansion decision.
// Pixels outside grain range get original luma in alpha (no extra fetches).

//!HOOK MAIN
//!BIND HOOKED
//!DESC CelFlare Grain Pre-filter (bilateral luma → alpha)

// Grain pre-filter parameters (authoritative — CelFlare/Lite read the result only)
#define GRAIN_THRESHOLD 0.32
#define GRAIN_BLUR_RADIUS 20.0
#define GRAIN_RANGE_MIN 0.35
#define GRAIN_RANGE_MAX 0.95
#define GRAIN_EDGE_LOW 0.22
#define GRAIN_EDGE_HIGH 0.60
#define EARLY_EXIT_GAMMA 0.10

// Gaussian bilateral sharpness: higher = tighter acceptance, lower = wider.
// 2.0 ≈ similar effective width to the old linear 0.22 threshold.
// Try 1.5 for more averaging, 3.0 for tighter edge preservation.
#define BILATERAL_SHARPNESS 8.0

vec4 hook() {
    vec4 original = HOOKED_tex(HOOKED_pos);
    const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
    float Y_gamma = dot(original.rgb, luma_coeff);

    // Early exit: very dark pixels don't need stabilization
    if (Y_gamma < EARLY_EXIT_GAMMA) {
        return vec4(original.rgb, Y_gamma);
    }

    // Range check: only stabilize pixels in grain-sensitive range
    float range_mask = smoothstep(GRAIN_RANGE_MIN - 0.05, GRAIN_RANGE_MIN, Y_gamma)
                     * (1.0 - smoothstep(GRAIN_RANGE_MAX, GRAIN_RANGE_MAX + 0.05, Y_gamma));

    if (range_mask < 0.01) {
        return vec4(original.rgb, Y_gamma);
    }

    // Per-pixel rotation angle from spatial hash (breaks any visible sampling pattern)
    float angle = fract(sin(dot(floor(HOOKED_pos * HOOKED_size), vec2(12.9898, 78.233))) * 43758.5453) * 6.2832;
    float ca = cos(angle);
    float sa = sin(angle);

    // 12-tap dual-ring bilateral blur:
    //   Outer ring (6 samples, radius):       hex at 0°/60°/120°/180°/240°/300°
    //   Inner ring (6 samples, radius×0.5):   hex at 30°/90°/150°/210°/270°/330°
    // Two offset rings give good spatial coverage while preserving structure.
    float radius = GRAIN_BLUR_RADIUS;

    // Outer hex ring: 0°, 60°, 120°, 180°, 240°, 300°
    const vec2 outer[6] = vec2[6](
        vec2( 1.000,  0.000),
        vec2( 0.500,  0.866),
        vec2(-0.500,  0.866),
        vec2(-1.000,  0.000),
        vec2(-0.500, -0.866),
        vec2( 0.500, -0.866)
    );
    // Inner hex ring (30° offset): 30°, 90°, 150°, 210°, 270°, 330°
    const vec2 inner[6] = vec2[6](
        vec2( 0.866,  0.500),
        vec2( 0.000,  1.000),
        vec2(-0.866,  0.500),
        vec2(-0.866, -0.500),
        vec2( 0.000, -1.000),
        vec2( 0.866, -0.500)
    );

    // Asymmetric rejection scale — computed once, applied per-sample
    // In upper luma range, darker samples are penalized up to 2× to protect highlights
    float asym_scale = mix(1.0, 2.0, smoothstep(0.75, 0.95, Y_gamma));

    float total_w = 1.0;
    float blurred = Y_gamma;
    // Bilateral-weighted edge gradient from outer ring (longest baseline = best estimate)
    float gx = 0.0;
    float gy = 0.0;
    float grad_w = 0.0;

    float raw_diff, asym, diff, w, s;
    vec2 rotated, offset;

    // Outer ring (full radius, base spatial weight)
    for (int i = 0; i < 6; i++) {
        vec2 h = outer[i];
        rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        offset = HOOKED_pt * radius * rotated;
        s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        raw_diff = s - Y_gamma;
        asym = raw_diff < 0.0 ? asym_scale : 1.0;
        diff = (raw_diff * asym) / GRAIN_THRESHOLD;
        w = exp(-BILATERAL_SHARPNESS * diff * diff);
        blurred += s * w;
        total_w += w;
        gx += s * w * rotated.x;
        gy += s * w * rotated.y;
        grad_w += w;
    }

    // Normalize gradient by bilateral weight sum so grain outliers don't inflate it
    float inv_grad_w = grad_w > 0.0 ? 1.0 / grad_w : 0.0;
    gx *= inv_grad_w;
    gy *= inv_grad_w;

    // Inner ring (0.5× radius, stronger spatial boost — anchors the average)
    for (int i = 0; i < 6; i++) {
        vec2 h = inner[i];
        rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        offset = HOOKED_pt * radius * 0.5 * rotated;
        s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        raw_diff = s - Y_gamma;
        asym = raw_diff < 0.0 ? asym_scale : 1.0;
        diff = (raw_diff * asym) / GRAIN_THRESHOLD;
        w = exp(-BILATERAL_SHARPNESS * diff * diff) * 1.4;
        blurred += s * w;
        total_w += w;
    }

    blurred /= total_w;

    // Edge magnitude from bilateral-weighted outer hex gradient
    float edge = sqrt(gx * gx + gy * gy) / 3.0;

    // Edge-aware mixing: at edges, prefer original to avoid halos
    float edge_mask = smoothstep(GRAIN_EDGE_LOW, GRAIN_EDGE_HIGH, edge);
    float Y_stabilized = mix(blurred, Y_gamma, edge_mask);

    // Blend with range mask for smooth transition at boundaries
    float Y_decision = mix(Y_gamma, Y_stabilized, range_mask);

    return vec4(original.rgb, Y_decision);
}
