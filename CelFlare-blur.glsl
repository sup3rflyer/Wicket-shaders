// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE
//
// CelFlare Grain Pre-filter — Detail-First tuning (lower risk of smoothing real shading/texture)
// Load BEFORE CelFlare.glsl or CelFlare-lite.glsl
//!HOOK MAIN
//!BIND HOOKED
//!DESC CelFlare Grain Pre-filter (detail preservation focus)

#define GRAIN_THRESHOLD     0.28
#define GRAIN_BLUR_RADIUS   14.0          // smaller kernel = less chance to average real detail

#define GRAIN_RANGE_MIN     0.35
#define GRAIN_RANGE_MAX     0.95
#define GRAIN_EDGE_LOW      0.20
#define GRAIN_EDGE_HIGH     0.45
#define EARLY_EXIT_GAMMA    0.10

#define BILATERAL_SHARPNESS 4.8           

#define INNER_RING_BOOST    2.0
#define ROTATION_BLOCK_SIZE 8.0          

vec4 hook() {
    vec4 original = HOOKED_tex(HOOKED_pos);
    const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
    float Y_gamma = dot(original.rgb, luma_coeff);
    
    if (Y_gamma < EARLY_EXIT_GAMMA) {
        return vec4(original.rgb, Y_gamma);
    }

    float range_mask = smoothstep(GRAIN_RANGE_MIN - 0.05, GRAIN_RANGE_MIN, Y_gamma)
                     * (1.0 - smoothstep(GRAIN_RANGE_MAX, GRAIN_RANGE_MAX + 0.05, Y_gamma));
                     
    if (range_mask < 0.01) {
        return vec4(original.rgb, Y_gamma);
    }

    vec2 block = floor(HOOKED_pos * HOOKED_size / ROTATION_BLOCK_SIZE);
    float angle = fract(sin(dot(block, vec2(12.9898, 78.233))) * 43758.5453) * 6.2832;
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

    // Less extra smoothing in brights = more detail kept
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
    // even stronger edge fallback
    float Y_decision = mix(Y_gamma, Y_stabilized, range_mask);

    return vec4(original.rgb, Y_decision);
}