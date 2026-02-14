// CelFlare Grain Pre-filter — Bilateral luma stabilization packed into alpha
// Load BEFORE CelFlare.glsl in shader list.
// Writes stabilized gamma luma to alpha channel for CelFlare's expansion decision.
// Pixels outside grain range get original luma in alpha (no extra fetches).

//!HOOK MAIN
//!BIND HOOKED
//!DESC CelFlare Grain Pre-filter (bilateral luma → alpha)

// Must match CelFlare.glsl defines
#define GRAIN_THRESHOLD 0.22
#define GRAIN_RANGE_MIN 0.55
#define GRAIN_RANGE_MAX 0.95
#define GRAIN_EDGE_LOW 0.07
#define GRAIN_EDGE_HIGH 0.60
#define EARLY_EXIT_GAMMA 0.10

vec4 hook() {
    vec4 original = HOOKED_tex(HOOKED_pos);
    vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);
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
    //   Outer ring (6 samples, radius): hex at 0°/60°/120°/180°/240°/300°
    //   Inner ring (6 samples, radius×0.5): hex at 30°/90°/150°/210°/270°/330°
    // Two interlocking rings give finer spatial coverage, preserving grain texture.
    float radius = 7.0;

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

    float total_w = 1.0;
    float blurred = Y_gamma;
    // Edge gradient from outer ring only (longer baseline = better estimate)
    float gx = 0.0;
    float gy = 0.0;

    // Outer ring (full radius, weight 1.0)
    for (int i = 0; i < 6; i++) {
        vec2 h = outer[i];
        vec2 rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        vec2 offset = HOOKED_pt * radius * rotated;
        float s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        float w = max(1.0 - abs(s - Y_gamma) / GRAIN_THRESHOLD, 0.0);
        blurred += s * w;
        total_w += w;
        gx += s * rotated.x;
        gy += s * rotated.y;
    }

    // Inner ring (half radius, weight 0.7 — closer samples less influential)
    for (int i = 0; i < 6; i++) {
        vec2 h = inner[i];
        vec2 rotated = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        vec2 offset = HOOKED_pt * radius * 0.5 * rotated;
        float s = dot(HOOKED_tex(HOOKED_pos + offset).rgb, luma_coeff);
        float w = max(1.0 - abs(s - Y_gamma) / GRAIN_THRESHOLD, 0.0);
        blurred += s * w;
        total_w += w;
    }

    blurred /= total_w;

    // Edge magnitude from outer hex gradient (normalized)
    float edge = sqrt(gx * gx + gy * gy) / 3.0;

    // Edge-aware mixing: at edges, prefer original to avoid halos
    float edge_mask = smoothstep(GRAIN_EDGE_LOW, GRAIN_EDGE_HIGH, edge);
    float Y_stabilized = mix(blurred, Y_gamma, edge_mask);

    // Blend with range mask for smooth transition at boundaries
    float Y_decision = mix(Y_gamma, Y_stabilized, range_mask);

    return vec4(original.rgb, Y_decision);
}
