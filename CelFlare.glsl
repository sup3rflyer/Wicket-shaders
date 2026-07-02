// CelFlare v5.3 — Illumination-Decomposition SDR→HDR Expansion
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// Two-layer architecture:
//  1. Base curve: per-pixel monotonic f(Y) with shape (peak, gamma)
//     modulated by an illumination field (sigma ~80px Gaussian of regional
//     luminance). Gradient-preserving — local contrast survives by
//     construction via multiplicative expansion in Oklab.
//  2. Specular bonus: additive, scene-detected, per-pixel ramp on the peak
//     channel V = max(R,G,B). Restores HDR pop where SDR compressed most
//     (saturated primaries at signal ceiling, bright specular highlights).
//
// You MUST set REFERENCE_WHITE to match your SDR white and hdr-reference-white
// in mpv.conf.
//
// Search for "MAIN TUNING" in the expansion pass for user-facing controls.

//!BUFFER SCENE_STATE
//!VAR float smoothed_bright_frac
//!VAR float smoothed_spec_signal
//!VAR float smoothed_contrast
//!VAR float smoothed_log_avg
//!VAR float smoothed_growth_mode
//!VAR float scene_cut_lockout
//!VAR float smoothed_spec_natural
//!VAR float pump_fast
//!VAR float pump_slow
//!VAR float pump_env
//!VAR float prev_illum[144]
//!VAR float pump_fast_cell[144]
//!VAR float pump_slow_cell[144]
//!VAR float pump_env_cell[144]
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!COMPUTE 16 16
//!DESC CelFlare: Grain Pre-filter

// Role: stabilize EXPANSION DECISIONS across grainy pixels sharing a common
// underlying luma — NOT to clean the image. Goal is that SDR grain, after
// expansion, looks like SDR grain scaled by the regional expansion factor
// (same character, just brighter). Equal-luma neighbors must get equal
// expansion so grain passes through the multiplicative curve intact.
//
// Kept inside "one semantic patch" by a moderate radius + dense angular
// sampling (12 taps in a 36 px-diameter disc). Larger radii cross into
// adjacent features that happen to share luma and start *averaging* the
// expansion decision across unrelated regions, which visibly steps the
// expansion at feature boundaries.
//
// Compute layout: 16x16 workgroup, 256 threads. Each workgroup pre-loads
// a 54x54 luma tile into shared memory (cooperative, ~11 loads/thread),
// then each thread does 12 manual-bilinear shared reads instead of 12
// scattered texture fetches. Amortized fetch count is ~12.4/pixel (2916
// tile loads / 256 threads + 1 center RGB) — similar to the fragment
// path's ~13; the win is coalesced sequential loads + shared-memory
// bilinear, not raw fetch count. A workgroup VOTE skips the tile load
// entirely when all 256 pixels are outside the stabilization range
// (dark scenes, letterbox, blown-out whites) — the common case.
// Halo = radius+1 (the +1 is required for the bilinear floor(p)+1 lookup
// at workgroup edge — easy bug if you set HALO == GRAIN_BLUR_RADIUS).
// Math is bit-for-bit equivalent to the v4.7 fragment path: same sample
// positions, same bilinear weights (hardware tex did fract(uv*size) on
// the same coords).
//
// ALPHA PROTOCOL: this pass stores its decision luma in alpha as
// Y * 0.5 (stabilized Y_decision, or raw Y_gamma on the exit paths).
// The 0.5 encode keeps every legitimate value representable — the old
// scheme stored Y directly and PASS 6 used `a > 0.99` as an "unstabilized"
// sentinel, which collided with real stabilized lumas in (0.99, 1.0]
// (grain pits next to super-white can stabilize above 0.99) and silently
// fell back to raw Y there. PASS 6 decodes with a plain * 2.0. Assumes a
// float intermediate FBO for MAIN (true under gpu-next fp16); a unorm FBO
// would clamp at Y=2.0 instead of 1.0 — strictly more headroom than before.
#define STABILIZE_OPACITY   0.85
#define GRAIN_THRESHOLD     0.28
#define GRAIN_BLUR_RADIUS   18     // 9 px inner / 18 px outer — stays inside one patch
#define GRAIN_RANGE_MIN     0.35
#define GRAIN_RANGE_MAX     0.95
#define GRAIN_EDGE_LOW      0.20
#define GRAIN_EDGE_HIGH     0.45
// Matches the lower edge of range_mask's smoothstep; below this range_mask
// is identically 0 so stabilization would no-op anyway.
#define GRAIN_EARLY_EXIT    0.30
#define BILATERAL_SHARPNESS 6.0
#define INNER_RING_BOOST    2.0    // Inner carries 2:1 weight over outer (inner 12 : outer 6)
// Empirical edge normalization. Brings first-moment gradient magnitude
// (gx,gy) from ~0.4*luma_contrast into the GRAIN_EDGE_LOW..HIGH range.
// Lower value -> more pixels marked as edges (stabilization suppressed).
#define GRAIN_EDGE_NORM     2.7

#define BLOCK_W   16
#define BLOCK_H   16
#define HALO      (GRAIN_BLUR_RADIUS + 1)
#define TILE_W    (BLOCK_W + 2 * HALO)
#define TILE_H    (BLOCK_H + 2 * HALO)
#define TILE_SIZE (TILE_W * TILE_H)
#define THREADS   (BLOCK_W * BLOCK_H)

shared float tile_y[TILE_SIZE];
shared uint  wg_active;   // workgroup vote: any pixel in stabilization range?

float tile_bilinear(vec2 pos) {
    // pos is in tile-local pixel coords. Manual bilinear mirrors hardware
    // tex sampling at the same continuous coord — fract(pos) gives the
    // same sub-texel weights the GPU would compute internally.
    vec2 f = fract(pos);
    ivec2 i = ivec2(floor(pos));
    int b = i.x + i.y * TILE_W;
    float c00 = tile_y[b];
    float c10 = tile_y[b + 1];
    float c01 = tile_y[b + TILE_W];
    float c11 = tile_y[b + TILE_W + 1];
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void hook() {
    const vec3 luma_coeff = vec3(0.2126, 0.7152, 0.0722);

    ivec2 g_pixel = ivec2(gl_GlobalInvocationID.xy);
    uint  lid     = gl_LocalInvocationIndex;

    // This thread's own center pixel — RGB needed for passthrough output,
    // grabbed off the texture path (one fetch) rather than from the luma
    // tile (which only stores Y).
    vec2 g_uv = (vec2(g_pixel) + 0.5) * HOOKED_pt;
    vec4 original = HOOKED_tex(g_uv);
    float Y_gamma = dot(original.rgb, luma_coeff);

    // -------- workgroup vote: skip the tile when nothing is in range --------
    // The cooperative load (~2916 fetches + shared stores) is the pass's
    // dominant cost, and it ran unconditionally even when every pixel in the
    // block would early-exit — i.e. most of a dark frame, letterbox, blown
    // whites. Vote first on each thread's own (already-fetched) center luma.
    // All three barriers sit in workgroup-uniform control flow: the vote
    // result is uniform by construction, so the conditional return is
    // D3D11/SPIRV-Cross safe.
    if (lid == 0u) wg_active = 0u;
    barrier();
    if (Y_gamma >= GRAIN_EARLY_EXIT && Y_gamma < GRAIN_RANGE_MAX + 0.05)
        wg_active = 1u;   // concurrent same-value stores — benign by design
    barrier();
    if (wg_active == 0u) {
        imageStore(out_image, g_pixel, vec4(original.rgb, Y_gamma * 0.5));
        return;
    }

    // -------- cooperative luma tile load --------
    // tile_origin is the top-left corner of the tile in image pixel coords.
    // Edge workgroups will sample outside the image; HOOKED_tex's clamp
    // mode handles those without special casing (matches fragment behavior).
    ivec2 tile_origin = ivec2(gl_WorkGroupID.xy) * ivec2(BLOCK_W, BLOCK_H) - ivec2(HALO);
    for (uint i = lid; i < uint(TILE_SIZE); i += uint(THREADS)) {
        int lx = int(i) % TILE_W;
        int ly = int(i) / TILE_W;
        ivec2 src = tile_origin + ivec2(lx, ly);
        vec2 uv = (vec2(src) + 0.5) * HOOKED_pt;
        tile_y[i] = dot(HOOKED_tex(uv).rgb, luma_coeff);
    }

    barrier();

    // -------- early exits --------
    // Lower: dark pixels, range_mask = 0 below 0.30.
    // Upper: super-white (Y >= 1.00), range_mask = 0 above 0.95+0.05.
    //        Skips the smoothstep + bilateral entirely for upscaler overshoot.
    if (Y_gamma < GRAIN_EARLY_EXIT || Y_gamma >= GRAIN_RANGE_MAX + 0.05) {
        imageStore(out_image, g_pixel, vec4(original.rgb, Y_gamma * 0.5));
        return;
    }

    float range_mask = smoothstep(GRAIN_RANGE_MIN - 0.05, GRAIN_RANGE_MIN, Y_gamma)
                     * (1.0 - smoothstep(GRAIN_RANGE_MAX, GRAIN_RANGE_MAX + 0.05, Y_gamma));

    if (range_mask < 0.01) {
        imageStore(out_image, g_pixel, vec4(original.rgb, Y_gamma * 0.5));
        return;
    }

    // -------- per-pixel angle hash --------
    // Same hash the fragment version used; gl_GlobalInvocationID.xy is the
    // pixel coord directly (no floor(HOOKED_pos*HOOKED_size) needed).
    vec2 pixel_f = vec2(g_pixel);
    float angle = fract(sin(dot(pixel_f, vec2(12.9898, 78.233))) * 43758.5453) * 6.2832;
    float ca = cos(angle);
    float sa = sin(angle);

    // -------- bilateral parameters (hoisted, division-free) --------
    // Original: diff = rd*asym/TH; d2 = sharp*diff^2; t = 1 - d2*0.25
    //   -> t = 1 - 0.25*sharp*(rd*asym)^2/TH^2 = 1 - weight_k * rd^2 * asym^2
    // Algebraically identical, removes the per-sample divide and fuses
    // three multiplies into one constant.
    float asym_scale = mix(1.0, 7.0, smoothstep(0.55, 0.98, Y_gamma));
    float effective_sharpness = BILATERAL_SHARPNESS * mix(1.0, 0.82, smoothstep(0.60, 1.0, Y_gamma));
    float weight_k = 0.25 * effective_sharpness / (GRAIN_THRESHOLD * GRAIN_THRESHOLD);
    float asym_scale_sq = asym_scale * asym_scale;

    // -------- 3 antipodal pairs per ring --------
    // Original outer[6] and inner[6] are each 3 antipodal pairs at the
    // same basis directions, just signed. Rotate the basis once per pair,
    // sample +r and -r explicitly. 6 rotation matmuls saved per pixel.
    const vec2 outer_basis[3] = vec2[3](
        vec2( 1.000, 0.000),
        vec2( 0.500, 0.866),
        vec2(-0.500, 0.866)
    );
    const vec2 inner_basis[3] = vec2[3](
        vec2( 0.866, 0.500),
        vec2( 0.000, 1.000),
        vec2(-0.866, 0.500)
    );

    // Tile-local center of this thread's pixel.
    vec2 tile_center = vec2(gl_LocalInvocationID.xy) + vec2(HALO);

    float blurred = Y_gamma;
    float total_w = 1.0;
    float gx = 0.0, gy = 0.0, grad_w = 0.0;

    // Outer ring (radius R) — gradient projection uses rotated.xy directly
    // (the - sample contributes via -r.x, -r.y, hence the gx -=/gy -= form).
    for (int i = 0; i < 3; i++) {
        vec2 h = outer_basis[i];
        vec2 r = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        vec2 o = r * float(GRAIN_BLUR_RADIUS);

        // + sample
        float s_p  = tile_bilinear(tile_center + o);
        float rd_p = s_p - Y_gamma;
        float a2_p = rd_p < 0.0 ? asym_scale_sq : 1.0;
        float t_p  = max(0.0, 1.0 - weight_k * rd_p * rd_p * a2_p);
        float w_p  = t_p * t_p;
        blurred += s_p * w_p;
        total_w += w_p;
        gx += rd_p * w_p * r.x;
        gy += rd_p * w_p * r.y;
        grad_w += w_p;

        // - sample (antipodal), gradient direction is -r
        float s_m  = tile_bilinear(tile_center - o);
        float rd_m = s_m - Y_gamma;
        float a2_m = rd_m < 0.0 ? asym_scale_sq : 1.0;
        float t_m  = max(0.0, 1.0 - weight_k * rd_m * rd_m * a2_m);
        float w_m  = t_m * t_m;
        blurred += s_m * w_m;
        total_w += w_m;
        gx -= rd_m * w_m * r.x;
        gy -= rd_m * w_m * r.y;
        grad_w += w_m;
    }

    // Inner ring (radius R/2). Per original tuning:
    //  - blur weights take INNER_RING_BOOST (inner ring carries 2:1 weight)
    //  - gradient weights use the un-boosted w so the boost only amplifies
    //    blur trust, not gradient estimation
    //  - gradient direction scaled by 2.0 to match outer-ring units
    //    (raw_diff at half radius is half the linear-gradient response;
    //    multiply back to keep gx/gy comparable across rings).
    for (int i = 0; i < 3; i++) {
        vec2 h = inner_basis[i];
        vec2 r = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        vec2 o = r * (float(GRAIN_BLUR_RADIUS) * 0.5);

        // + sample
        float s_p  = tile_bilinear(tile_center + o);
        float rd_p = s_p - Y_gamma;
        float a2_p = rd_p < 0.0 ? asym_scale_sq : 1.0;
        float t_p  = max(0.0, 1.0 - weight_k * rd_p * rd_p * a2_p);
        float w_p  = t_p * t_p;
        blurred += s_p * w_p * INNER_RING_BOOST;
        total_w += w_p * INNER_RING_BOOST;
        gx += rd_p * w_p * r.x * 2.0;
        gy += rd_p * w_p * r.y * 2.0;
        grad_w += w_p;

        // - sample
        float s_m  = tile_bilinear(tile_center - o);
        float rd_m = s_m - Y_gamma;
        float a2_m = rd_m < 0.0 ? asym_scale_sq : 1.0;
        float t_m  = max(0.0, 1.0 - weight_k * rd_m * rd_m * a2_m);
        float w_m  = t_m * t_m;
        blurred += s_m * w_m * INNER_RING_BOOST;
        total_w += w_m * INNER_RING_BOOST;
        gx -= rd_m * w_m * r.x * 2.0;
        gy -= rd_m * w_m * r.y * 2.0;
        grad_w += w_m;
    }

    float inv_grad_w = grad_w > 0.0 ? 1.0 / grad_w : 0.0;
    gx *= inv_grad_w;
    gy *= inv_grad_w;

    blurred /= total_w;

    float edge = sqrt(gx * gx + gy * gy) * (1.0 / GRAIN_EDGE_NORM);
    float edge_mask = smoothstep(GRAIN_EDGE_LOW, GRAIN_EDGE_HIGH, edge);

    // Collapsed mix: was mix(Y_gamma, mix(blurred, Y_gamma, edge_mask),
    // range_mask * STABILIZE_OPACITY). Expands algebraically to
    //   Y_gamma + (1 - edge_mask) * range_mask * STAB * (blurred - Y_gamma)
    // Same result, one MAD instead of two.
    float stab = (1.0 - edge_mask) * range_mask * STABILIZE_OPACITY;
    float Y_decision = Y_gamma + (blurred - Y_gamma) * stab;

    // Alpha = Y * 0.5 encode (see header) — keeps stabilized values in
    // (0.99, 1.0+] representable for PASS 6 instead of colliding with the
    // old "unstabilized" sentinel.
    imageStore(out_image, g_pixel, vec4(original.rgb, Y_decision * 0.5));
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
// PASS 5: FRAME STATS (compute, 144-thread parallel reduction)
// =============================================================================
// Samples illumination field on a 16×9 grid. Computes frame-level metrics that
// modulate the expansion curve: average illumination, bright fraction (replaces
// the entire 7-type scene classifier from v3.2), and scene cut detection.
//
// Compute layout: COMPUTE 16 9 dispatches one workgroup of 144 threads
// against the 1×1 output. Each thread owns one grid cell — sampling, tier
// classification, and per-cell scene-cut delta all happen in parallel. Thread
// 0 performs the final 144-element reduction and writes SCENE_STATE.
//
// Scene cut: the previous prev_illum[16] separate 4×4 grid was
// redundant — the 16×9 stats grid already covers the frame at higher density.
// prev_illum was widened to [144] so each lane stores its own slot (no
// cross-lane access -> race-free), and change_pct is computed over all 144.
// SCENE_CUT_PCT semantics are preserved: fraction of cells whose illum moved
// by > ILLUM_CHANGE_THRESH. Spatial density is higher (1/16-w × 1/9-h vs
// 1/4 × 1/4), threshold tuning is unchanged.

//!HOOK MAIN
//!BIND HOOKED
//!BIND SCENE_STATE
//!BIND CELFLARE_ILLUM
//!SAVE CELFLARE_STATS
//!WIDTH 1
//!HEIGHT 1
//!COMPUTE 16 9
//!DESC CelFlare: Frame Stats

// Scene-brightness classifier threshold for bright_frac. Deliberately ABOVE
// PASS 6's expansion KNEE (0.30): v4.5 retuned the expansion onset down but
// kept the stats classifier (and everything tuned against bright_frac —
// PEAK_ATTEN, BRIGHT_FRAC_REF, growth frac_floor) calibrated at 0.40.
// Renamed from KNEE so a future retune of one can't silently miss the other.
#define BRIGHT_STAT_THRESH  0.40

// Specular detection (source brightness tiers, no illum field)
#define HIGHLIGHT_THRESH    0.75    // Source highlight tier
#define SPECULAR_THRESH     0.92    // Source specular tier
#define SPEC_FRAC_MIN       0.007   // ~1/144 noise floor
#define SPEC_FRAC_MAX       0.10    // Fade begins (sparkle clusters can reach 12-15%)
#define SPEC_FRAC_CEIL      0.16    // Full shutoff (large bright skies)
#define SPARSE_SPEC_CEIL    0.02    // Below this spec_frac, "sparse points against dark" fires alongside tier_gate — catches candles/LEDs/stars

// Bright-scene specular recovery. When most pixels exceed SPECULAR_THRESH the
// normal detection collapses (shutoff fires, tier_ratio kills). A stricter
// 0.97 threshold re-establishes tier separation: "above 0.97 in a bright
// scene IS specular" relative to the scene. Cases recovered: chrome at noon,
// sun glints on water, headlights against daylight — currently lost to
// whiteout. Driven by smoothed_log_avg (temporally damped, so the recovery
// transition is smooth and doesn't introduce a velocity step into spec_vel).
// Supplements only, never reduces — uses max(spec_raw, bs_raw).
#define BRIGHT_SPEC_THRESH    0.97  // Super-specular threshold for bright scenes
#define BRIGHT_SCENE_LOW      0.20  // smoothed_log_avg below: normal detection only
#define BRIGHT_SCENE_HIGH     0.35  // smoothed_log_avg above: bright fallback active
// Recovery's own scene-fraction shutoff. Tightened to discriminate "sparse
// specular against bright" (chrome at noon, headlights against daylight,
// sun glints on water — bs_frac ≤ ~0.10) from "broad white surface" (cel-art
// shirts/walls at Y=1.0, snow vistas — bs_frac ≥ ~0.20). The earlier wide
// window (0.50–0.85) treated cel-art whites as legitimate specular bodies
// and lifted them ~20–40 nits via SPEC_PEAK_BRIGHT, which read as "too hot"
// on anime even though the spatial curve alone was on target.
#define BRIGHT_SPEC_FRAC_MAX  0.15  // Recovery starts fading at 15% of cells > 0.97
#define BRIGHT_SPEC_FRAC_CEIL 0.40  // Recovery fully off at 40% of cells > 0.97
// Recovery counts NEAR-WHITE cells only. Every case in the recovery's spec
// (chrome, sun glints, headlights) is near-neutral by nature. With
// ENABLE_SATURATED_SPEC the counters run on V = max(R,G,B), so without a
// saturation fence a large bright SATURATED surface (pink carpet, a 1080p
// WEB test scene) supplies V>0.97 cells and impersonates "sparse chrome at noon"
// — keeping smoothed_spec_signal ~1 on a scene the NORMAL path correctly
// shut off (spec_frac > SPEC_FRAC_CEIL), which re-arms the per-pixel
// saturated-spec ramp on the very field that tripped the shutoff: 8-bit
// 4:2:0 Cr noise in V (unstabilizable by the luma-transplant V_stable)
// then flickers on the ramp = speckle. Saturated EMISSIVE scenes
// (LEDs/lasers/fire in the dark) are unaffected: they fire via the normal
// path's sparse_bonus/tier_gate, not the bright-scene recovery. A genuinely
// near-white V>0.97 cell always passes (sat 0.30 is far above specular
// whites; the carpet's qualifying cells measured sat 0.40–0.88).
#define BRIGHT_SPEC_SAT_MAX   0.30  // s_bright_spec counts only cells with sat below this

// Saturated-channel spec detection. Count pixels using V = max(R,G,B) rather
// than Y so pure saturated primaries (red LED Y=0.21 but V=1.0) qualify as
// specular tier. V >= Y always, so neutrals behave identically. Scene-level
// gating (tier_ratio, shutoff) still suppresses red-dominated scenes.
#define ENABLE_SATURATED_SPEC 1

// Velocity-adaptive temporal alpha. Stable scenes get heavy smoothing (SLOW),
// quick lighting changes get faster adaptation (MID), scene cuts get nearly
// instant lock-on (FAST). vel_mag = max(|Δbright_frac|, |Δlog_avg|) drives
// the SLOW→MID interpolation; cut detection overrides to FAST regardless.
//
// KNOWN LIMITATION (accepted 2026-06-10): all temporal constants are
// per-RENDERED-VIDEO-FRAME and tuned for ~24p content. Display refresh is
// irrelevant (mpv renders each video frame once regardless of vsync rate),
// but 60fps CONTENT runs every EMA ~2.5x faster and shrinks per-frame
// velocities ~2.5x (growth-mode under-fires). The failure direction is
// conservative — snappier adaptation, less pop, never artifacts. No time
// source exists in user shaders (probed: only `frame`/`random`; no PTS
// uniform in current mpv/libplacebo), and in-shader fps estimation is
// unsound for anime (held cels on twos/threes are pixel-identical to
// transport duplicates). If mpv ever exposes a PTS uniform: store prev_pts
// in the SSBO, dt = clamp(pts - prev_pts, 0.0, 0.5), alpha_eff =
// 1 - pow(1 - alpha24, dt*24), lockout in seconds, velocity gates scaled
// by 24*dt — thread-0-only, and EMAs become immune to redraw double-ticks,
// pause, and seeks for free.
#define TEMPORAL_ALPHA_SLOW 0.03    // Stable scenes (= prior TEMPORAL_ALPHA)
#define TEMPORAL_ALPHA_MID  0.12    // Quick lighting / brightness shifts
#define TEMPORAL_ALPHA_FAST 0.9     // Scene cut + lockout
#define ADAPT_DELTA_LOW     0.02    // Below: slow alpha
#define ADAPT_DELTA_HIGH    0.10    // Above: mid alpha
#define LOCKOUT_FRAMES      6.0
#define ILLUM_CHANGE_THRESH 0.06
#define SCENE_CUT_PCT       0.50

// Growth-mode discriminator. Detects expanding bright objects (fireball,
// crash-zoom on backlit window) so PASS 6 can bypass the bright-scene
// dampeners that otherwise suppress the most impressive moment of the event.
// Signature: spec_vel rising faster than bright_vel (hot core saturates
// first) AND contrast climbing (dynamic range grows with the object, vs
// a uniform fade-to-white where min catches max). frac_floor suppresses
// false-positives on sub-percent pixel fractions (fading title text).
#define GROWTH_SPEC_BIAS       0.4    // weight of bright_vel subtracted from spec_vel
#define GROWTH_SIG_LOW         0.015  // smoothstep onset on (spec_vel - bias*bright_vel)
#define GROWTH_SIG_HIGH        0.06
#define GROWTH_C_GATE_HIGH     0.25   // smoothstep saturation on contrast_vel
#define GROWTH_FRAC_FLOOR_LOW  0.04   // smoothed_bright_frac required to activate
#define GROWTH_FRAC_FLOOR_HIGH 0.10
#define GROWTH_SHUTOFF_LIFT    0.6    // 0 = no spec_shutoff bypass during growth, 1 = full lift

// Light-pump detector — augments sudden SUSTAINED brightening (explosion
// bloom, train exiting a tunnel, spell charge-up). Temporal band-pass on
// the frame's illum-V statistic: a moderate-alpha fast lane minus a slow
// baseline lane. The difference is positive only during a multi-frame RISE
// (so the pump tracks and augments the source's own attack), ≈0 at steady
// state, and self-releases once brightness plateaus (fast lane catches the
// slow one). The moderate fast alpha is what rejects 2-3 frame flashes
// (lightning, muzzle): they reverse before the fast lane builds, so drive
// stays under the onset.
// PUMP_ALPHA_FAST is the primary flash-vs-sustained dial: LOWER = more flash
// rejection but slower response to genuine attacks.
#define PUMP_ALPHA_FAST     0.12   // fast lane (~8 frame time constant). Sets ATTACK speed — lower = gentler ramp
#define PUMP_ALPHA_SLOW     0.04   // slow baseline lane (~25 frame time constant)
#define PUMP_DRIVE_LOW      0.03   // band-pass onset — below this, no pump
#define PUMP_DRIVE_HIGH     0.20   // band-pass saturation — full pump needs a steep rise (reserves full for violent events)
// The drive statistic is the p-NORM (power mean) of the 144 σ80 illum-V cell
// samples, not their arithmetic mean. V^p mass is dominated by highlight
// content, so a LOCALIZED bright event (two cells of fire flaring in a dark
// scene) swings the p-norm several times more than it moves the mean — the
// scalar now FIRES for regional events and the subtractive mask localizes
// the pump to the cells that are actually brightening (multi-fire scenes
// pump per-fire again). Occluder robustness IMPROVES at the same time: a
// dark object covering fraction f of bright content dips the p-norm by only
// 1-(1-f)^(1/p) (f=0.4 → ~12% at p=4, vs ~40%·V_bg for the mean), so a
// reveal recovery has LESS drive than before, not more. A frame-UNIFORM
// rise moves p-norm and mean identically, so DRIVE_LOW/HIGH keep their
// tuned meaning for global events.
// p=1 = frame mean (pre-v5.3 behavior); higher p → drive keyed ever harder
// to the brightest regions (p→∞ = brightest cell). Measured at the σ80
// scale the rise uses the FULL V range (region transitioning INTO
// brightness) — this is what a per-pixel specular gate can't do (no
// headroom above the clip point to measure velocity in).
// LOCALIZED drive (drive_loc) — field-tested addendum: the p-norm alone
// proved insufficient. It is "change of an aggregate," so its sensitivity
// to new bright mass is DILUTED by the standing bright mass already in
// frame (Δpnorm ≈ ΔM/(4·M^0.75)): a 2-cell fire igniting moves it ~0.14 in
// a black frame but ~0.005 in any scene that already holds a sky patch or
// window — under onset. Field report: masks opened everywhere, scalar
// never fired. So the scalar takes a SECOND onset source, an AGGREGATE OF
// per-cell changes (signed p-mean of the same fast−slow cell lanes the
// mask runs):
//   drive_loc = sign(S)·(|S|/144)^(1/p),  S = Σ sign(d_i)·|d_i|^p
// Static content has d≈0 and contributes NOTHING — standing bright mass is
// invisible, so a localized event registers near its own cell amplitude
// (2 cells at d=0.3 → drive_loc ≈ 0.10) on ANY baseline. The SIGN is the
// load-bearing safety: a moving occluder/pan pairs every wake-cell rise
// with a leading-edge fall → the signed sum CANCELS (the old min(rise,fall)
// trans-cov insight, embedded as arithmetic). A rectified (positive-only)
// aggregate would re-open the crossing-trail class — KEEP IT SIGNED.
// Fade-to-white: all cells rise coherently → drive_loc ≈ global drive →
// the contrast guard mutes it as before. Idle wobble: |0.005|^4 ≈ 6e-10,
// annihilated. pump_gate takes max(drive, drive_loc). RELEASE: a purely-
// local event never moves the global lanes, so without a local release its
// env would linger ~29s behind closed masks and any later mask opening
// (incl. an occluder wake) would inherit stale amplitude. When drive_loc<0
// (net local fall) the env releases at the FALLING CELLS' OWN velocity
// ratio (fast_cell/slow_cell, |d|^p-weighted) — the same fractional-drop
// semantics as the global rel and the mask's r, dimensionally on the LOCAL
// axis. (Do NOT divide drive_loc by the frame level instead: a cell-delta
// over an absolute frame level guillotines the env in one frame on dark
// frames — audited Rule-2 violation.) A held light has d≈0 → no release;
// a balanced crossing has drive_loc≈0 → no release (conservative hold).
// TRADE: sign cancellation fails on NET-ASYMMETRIC transitions — a large
// dark occluder EXITING frame (reveal with no paired cover), a bright
// object ENTERING (window on a pan). Those fire at a lower size threshold
// than before. Not a new artifact class (photometrically = tunnel exit, a
// designed target: the frame genuinely brightens and the pump eases in and
// out) — judge on real content. If it reads wrong, lower P toward 2
// (drive_loc needs bigger events); at p=1 both statistics collapse to the
// frame mean = v5.2 behavior. The mask still can't ADD under any p.
#define PUMP_DRIVE_P        4.0    // highlight weighting of BOTH drive statistics. A/B 2.0-6.0
// Spatial pump (SUBTRACTIVE): per-cell band-pass on the COARSE σ80 CELFLARE_ILLUM V
// (sampled at the cell center — the SAME field the scalar means over the frame) →
// a [0,1] "is this region brightening" mask. PASS 6 MULTIPLIES the global scalar
// pump by it, so spatial can only SUPPRESS the scalar where a region isn't
// brightening — never add pump. 0 = scalar-only (mask disabled).
// ⚠ FOOTGUN — this define is DUPLICATED in PASS 6 (line ~1149) and the two MUST
// match: no compile-time guard. A PASS5=0/PASS6=1 mismatch COMPILES but leaves
// pump_env_cell unwritten → PASS 6 reads a stale/garbage mask and the pump
// misbehaves. (Found flipped to 0 twice this session — keep BOTH at the same value.)
#define ENABLE_SPATIAL_PUMP 1
#define PUMP_CELL_DEADZONE  0.01   // per-cell drive dead-zone: idle-wobble cells read true zero (anti competing-highlight leak)
// Per-cell drive thresholds (independent of the scalar's PUMP_DRIVE_LOW/HIGH). The
// cell driver is the σ80 V AT the cell vs the scalar's frame p-norm, so a
// LOCALIZED event swings its own cell far more than it moves any frame stat → the
// per-cell drive for a real regional event is comparable-to-larger than the
// scalar's. LOW sits a touch above the scalar onset to reject σ80 neighbour-bleed
// (a rising event bleeding ~half-amplitude into an adjacent cell) and idle wobble;
// HIGH is where a genuine in-cell brightening saturates. Retuned for σ80 (the old
// 0.02/0.18 were for the deleted sharp MID driver). A/B LOW 0.03-0.05, HIGH 0.10-0.16.
#define PUMP_CELL_DRIVE_LOW  0.04   // per-cell band-pass onset (mask starts opening — region is brightening)
#define PUMP_CELL_DRIVE_HIGH 0.14   // per-cell saturation (mask fully open — region clearly brightening)
// SUBTRACTIVE model (2026-07-02): the per-cell env is a [0,1] SUPPRESSOR — PASS 6
// multiplies the global scalar pump by it, so spatial can only REMOVE pump from
// non-brightening regions, never ADD it. That structurally kills the whole reveal/
// ghost/occluder-trail artifact class (no global event → scalar ~0 → product ~0,
// regardless of any local rise), which is why the additive-mask machinery is gone:
// no novelty gate, no habitual-V memory, no established-exemption, no localized-
// confidence gate, no motion crossfade — all existed only to stop an additive mask
// from over-firing. Pump AMPLITUDE comes from the global scalar; since v5.3 its
// onset is max(p-norm band-pass, drive_loc — a signed p-mean aggregate of the
// per-cell drives; see PUMP_DRIVE_P block), so a localized event fires the scalar
// regardless of standing bright content — the mask then confines the pump to the
// brightening cells. Amplitude is still shared (one scalar), but per-region TIMING
// is each cell's own band-pass env, so multiple independent events (several fires)
// pump on their own rhythms.
// Release is VELOCITY-MATCHED, not clock-based: the primary release is sourced
// from the negative half of the band-pass (the source's own luminance fall —
// see the reducer), so the pump eases out in lockstep with the source and a
// HELD light does not release on a timer. This floor is the only clock left —
// an imperceptibly slow geometric relaxation so an indefinitely-held light
// settles like the eye adjusting, not like an animated dim. Keep it near 1.0.
// half-life = ln(0.5)/ln(F):  0.999 ≈ 29s @24p   0.9995 ≈ 58s   1.0 = pure hold
#define PUMP_ADAPT_FLOOR    0.999   // held-light relaxation; raise toward 1.0 = even slower/imperceptible
// Fade guard via CONTRAST RETENTION (not coverage). A genuine event keeps a
// hot core against dark surround/smoke → contrast stays high → pump allowed.
// A fade-to-white (or fade-to-any-uniform-colour) collapses contrast as the
// field goes uniform → pump muted. This distinguishes "explosion fills frame"
// (keep) from "frame whites out" (mute) — which coverage alone cannot — and
// because contrast collapses gradually during a fade, the mute eases in on its
// own (controlled release, no slew machinery needed). contrast = log2 dynamic
// range of the illumination field, in stops.
#define PUMP_CONTRAST_LOW   1.0    // below this (≈uniform): pump fully muted
#define PUMP_CONTRAST_HIGH  2.5    // above this (structured frame): full pump
// Optional coverage backstop for the degenerate uniform-but-high-contrast case
// (rare). Kept for A/B; not wired by default — the contrast gate supersedes it.
//#define PUMP_COVER_HIGH     0.62   // bright_frac above which the pump tapers
//#define PUMP_COVER_FULL     0.85   // bright_frac at/above which the pump is fully muted

float get_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Per-lane scratch for the cross-lane reduction. 8 arrays × 144 elements ×
// 4 bytes (mix of float and uint, all 32-bit) = 4608 bytes — well under any
// GPU's 16-32 KB shared-memory floor.
shared float s_illum[144];
shared float s_log_luma[144];
shared uint  s_valid[144];
shared uint  s_bright[144];
shared uint  s_spec[144];
shared uint  s_high[144];
shared uint  s_change[144];
shared uint  s_bright_spec[144];    // BRIGHT_SPEC_THRESH=0.97 count for bright-scene recovery
shared float s_illum_v[144];        // max(R,G,B) of the illum field — V-aware pump driver/guard
#if ENABLE_SPATIAL_PUMP
shared uint  s_pump_reset;          // thread-0 publishes per-cell pump reset (cut/lockout/frame0)
#endif

void hook() {
    uint lid = gl_LocalInvocationIndex;    // 0..143
    uint ix  = gl_LocalInvocationID.x;     // 0..15
    uint iy  = gl_LocalInvocationID.y;     // 0..8

    // Cell center in normalized texture coordinates — identical to the
    // (x+0.5)/16, (y+0.5)/9 positions the original double loop sampled.
    vec2 spos = vec2((float(ix) + 0.5) / 16.0,
                     (float(iy) + 0.5) / 9.0);

    // -------- per-cell sampling (parallel) --------
    vec3  illum_rgb = CELFLARE_ILLUM_tex(spos).rgb;
    float Y_ill   = get_luma(illum_rgb);
    float V_ill   = max(max(illum_rgb.r, illum_rgb.g), illum_rgb.b);  // V-aware pump driver/guard
    vec3  rgb_src = HOOKED_tex(spos).rgb;
    float Y_src   = get_luma(rgb_src);

    #if ENABLE_SATURATED_SPEC
    float intensity_src = max(max(rgb_src.r, rgb_src.g), rgb_src.b);
    #else
    float intensity_src = Y_src;
    #endif

    bool valid = Y_src > 0.001;
    // Near-white fence for the bright-scene recovery counter (see
    // BRIGHT_SPEC_SAT_MAX). Saturation is computed from V/min of rgb_src
    // regardless of ENABLE_SATURATED_SPEC (under !SAT_SPEC the gate is a
    // structural no-op anyway: Y_src > 0.97 forces near-neutral RGB).
    float v_src   = max(max(rgb_src.r, rgb_src.g), rgb_src.b);
    float sat_src = (v_src > 1e-6)
        ? (v_src - min(min(rgb_src.r, rgb_src.g), rgb_src.b)) / v_src
        : 0.0;
    s_illum[lid]    = Y_ill;
    s_illum_v[lid]  = V_ill;
    s_log_luma[lid] = valid ? log(max(Y_src, 1e-6)) : 0.0;
    s_valid[lid]    = valid ? 1u : 0u;
    s_bright[lid]   = (Y_ill > BRIGHT_STAT_THRESH)   ? 1u : 0u;
    s_spec[lid]        = (intensity_src > SPECULAR_THRESH) ? 1u : 0u;
    s_high[lid]        = (intensity_src > HIGHLIGHT_THRESH) ? 1u : 0u;
    s_bright_spec[lid] = (intensity_src > BRIGHT_SPEC_THRESH &&
                          sat_src < BRIGHT_SPEC_SAT_MAX) ? 1u : 0u;

    // Scene-cut delta. Each lane reads + writes its own prev_illum slot —
    // no cross-lane SSBO traffic, so race-free even though the buffer is
    // declared coherent. The barrier below still gates s_change visibility
    // for the reducer.
    s_change[lid]   = (frame > 0 &&
                      abs(Y_ill - prev_illum[lid]) > ILLUM_CHANGE_THRESH)
                      ? 1u : 0u;
    prev_illum[lid] = Y_ill;

    barrier();

    // -------- thread-0 reduction + SSBO update --------
    // 144 serial adds on ALU registers is trivially cheap compared to the
    // 144 texture fetches we just parallelized away. Could be replaced with
    // a subgroupAdd ladder for sub-microsecond savings if profiling demands.
    if (lid == 0u) {
        float illum_sum         = 0.0;
        float log_luma_sum      = 0.0;
        uint  valid_luma        = 0u;
        uint  bright_count      = 0u;
        uint  spec_count        = 0u;
        uint  high_count        = 0u;
        uint  change_count      = 0u;
        uint  bright_spec_count = 0u;
        float illum_min         = 1.0;
        float illum_max         = 0.0;
        float illum_v_psum      = 0.0;
        float illum_v_min       = 1.0;
        float illum_v_max       = 0.0;

        for (uint i = 0u; i < 144u; i++) {
            float yi           = s_illum[i];
            illum_sum         += yi;
            illum_min          = min(illum_min, yi);
            illum_max          = max(illum_max, yi);
            // Floor at 0: upstream ringing can undershoot slightly negative and
            // GLSL pow() is undefined for x<0 — a single NaN here would persist
            // in pump_fast/pump_slow (SSBO) until the next scene cut.
            float vi           = max(s_illum_v[i], 0.0);
            illum_v_psum      += pow(vi, PUMP_DRIVE_P);
            illum_v_min        = min(illum_v_min, vi);
            illum_v_max        = max(illum_v_max, vi);
            log_luma_sum      += s_log_luma[i];
            valid_luma        += s_valid[i];
            bright_count      += s_bright[i];
            spec_count        += s_spec[i];
            high_count        += s_high[i];
            change_count      += s_change[i];
            bright_spec_count += s_bright_spec[i];
        }

        const float N_SAMPLES = 144.0;
        float avg_illum   = illum_sum / N_SAMPLES;
        float bright_frac = float(bright_count) / N_SAMPLES;

        // Contrast: dynamic range from illumination field (stable, noise-free)
        float contrast = (illum_min > 0.001)
            ? log2(max(illum_max / illum_min, 1.0))
            : 0.0;

        // V-aware pump signals on max(R,G,B) of the illum field, so saturated
        // colored events (blue spell, red blast — low luma, high V) both DRIVE
        // the pump and SURVIVE its contrast guard. The drive statistic is the
        // p-NORM of the cell samples (see PUMP_DRIVE_P): highlight-weighted so
        // a localized bright event registers, occluder dips compressed. The
        // contrast guard keeps plain min/max dynamic range. Kept separate from
        // avg_illum/contrast, which growth-mode and APL still consume on the
        // luma axis.
        float pnorm_illum_v = pow(illum_v_psum / N_SAMPLES, 1.0 / PUMP_DRIVE_P);
        float contrast_v    = (illum_v_min > 0.001)
            ? log2(max(illum_v_max / illum_v_min, 1.0))
            : 0.0;

        // Log-average: perceptual brightness key from source pixels
        float log_avg = (valid_luma > 4u)
            ? exp(log_luma_sum / float(valid_luma))
            : avg_illum;

        // Specular signal: present when small fraction at specular tier
        float spec_frac          = float(spec_count) / N_SAMPLES;
        float highlight_frac_src = float(high_count) / N_SAMPLES;
        float spec_onset         = smoothstep(0.0, SPEC_FRAC_MIN, spec_frac);
        float spec_shutoff       = 1.0 - smoothstep(SPEC_FRAC_MAX, SPEC_FRAC_CEIL, spec_frac);
        // Tier separation: specular must be rarer than highlights.
        // Works for "bright point in bright surround" (candle with glow, chrome on
        // mid-bright surface) but fails when specular points have no sub-specular
        // halo (isolated candle flame in dark room, stars, distant LEDs).
        float tier_ratio   = 1.0 - spec_frac / max(highlight_frac_src, 0.001);
        float tier_gate    = smoothstep(0.3, 0.7, tier_ratio);
        // Sparse-points-against-dark fallback: fires when specular pixels are
        // very rare in the frame, regardless of tier ratio. Gated separately by
        // spec_onset so pure noise floor doesn't trigger.
        float sparse_bonus = 1.0 - smoothstep(0.0, SPARSE_SPEC_CEIL, spec_frac);
        float tier_mode    = max(tier_gate, sparse_bonus);

        // Bright-scene recovery: parallel detection at the stricter 0.97 tier
        // with its own (wider) shutoff and looser tier-ratio gate. In a sunny
        // daylight scene where normal spec_shutoff/tier_ratio collapse, this
        // restores separation against the brightest fraction. The
        // bright_scene envelope is driven by smoothed_log_avg (not the
        // instantaneous avg_illum) so transitions like fade-ups or sunrise
        // ramps don't introduce a step into spec_raw_natural that would
        // false-trigger the growth-mode discriminator via spec_vel.
        // Supplements only — uses max(spec_raw, bs_raw).
        float bs_frac     = float(bright_spec_count) / N_SAMPLES;
        float bs_onset    = smoothstep(0.0, SPEC_FRAC_MIN, bs_frac);
        float bs_shutoff  = 1.0 - smoothstep(BRIGHT_SPEC_FRAC_MAX,
                                             BRIGHT_SPEC_FRAC_CEIL, bs_frac);
        float bs_tier_r   = 1.0 - bs_frac / max(highlight_frac_src, 0.001);
        float bs_gate     = smoothstep(0.2, 0.5, bs_tier_r);
        float bs_raw      = bs_onset * bs_shutoff * bs_gate;
        float bright_scene = smoothstep(BRIGHT_SCENE_LOW, BRIGHT_SCENE_HIGH, smoothed_log_avg);

        // Natural (pre-bypass) form of spec_raw — fed to the velocity calc so
        // the growth-mode discriminator runs on unboosted signal (avoids a
        // positive feedback loop with the shutoff lift below). Recovery is
        // applied here too so velocity reflects the real scene-relative spec.
        float spec_raw_natural = spec_onset * spec_shutoff * tier_mode;
        spec_raw_natural = mix(spec_raw_natural,
                               max(spec_raw_natural, bs_raw), bright_scene);

        // Scene cut detection — 144-sample grid (reuses the stats grid).
        // SCENE_CUT_PCT (0.50) interpretation stays "majority of cells moved";
        // denser sampling makes detection more robust against localized motion
        // that the old 4×4 grid could fully miss between cell centers.
        float change_pct  = float(change_count) / N_SAMPLES;
        scene_cut_lockout = max(scene_cut_lockout - 1.0, 0.0);
        bool scene_cut    = (change_pct > SCENE_CUT_PCT) && (scene_cut_lockout <= 0.0);
        if (scene_cut) scene_cut_lockout = LOCKOUT_FRAMES;

        // ---- Velocity-driven adaptation ----
        // Signed velocities — instantaneous minus previous smoothed (the EMA
        // lag is itself a velocity proxy). Used twice: (a) magnitude drives
        // an adaptive base alpha (slow on still scenes, mid on quick changes),
        // (b) signed components feed the growth-mode discriminator.
        // spec_vel measures against smoothed_spec_natural (the un-lifted EMA),
        // NOT smoothed_spec_signal: the latter is updated with the growth-
        // LIFTED spec_raw, so using it as the baseline inflated it during
        // growth events, drove spec_vel artificially negative, and quenched
        // (or pumped) growth-mode before long events finished.
        float bright_vel   = bright_frac - smoothed_bright_frac;
        float spec_vel     = spec_raw_natural - smoothed_spec_natural;
        float contrast_vel = contrast - smoothed_contrast;
        float log_avg_vel  = log_avg - smoothed_log_avg;

        float vel_mag    = max(abs(bright_vel), abs(log_avg_vel));
        float base_alpha = mix(TEMPORAL_ALPHA_SLOW, TEMPORAL_ALPHA_MID,
                               smoothstep(ADAPT_DELTA_LOW, ADAPT_DELTA_HIGH, vel_mag));
        // Post-cut alpha decays FAST -> MID across the lockout window. Lock-on
        // converges in 1-2 frames at 0.9; holding 0.9 for all 6 frames let a
        // single-frame event inside the window (muzzle flash, strobe, white
        // impact frame) couple ~1:1 into every EMA as a visible pulse.
        float alpha;
        if (scene_cut) {
            alpha = TEMPORAL_ALPHA_FAST;
        } else if (scene_cut_lockout > 0.0) {
            alpha = mix(TEMPORAL_ALPHA_MID, TEMPORAL_ALPHA_FAST,
                        scene_cut_lockout / LOCKOUT_FRAMES);
        } else {
            alpha = base_alpha;
        }

        // Growth-mode discriminator. Fires when:
        //   - spec_vel rises faster than bright_vel (hot core saturates first
        //     — fireball, backlit window growing — vs uniform fade-to-white
        //     where the ratio at steady state is preserved)
        //   - contrast_vel positive (dynamic range expanding, not collapsing)
        //   - smoothed_bright_frac above a floor (avoids fading title-card
        //     text at sub-percent pixel fractions)
        float growth_sig   = spec_vel - GROWTH_SPEC_BIAS * bright_vel;
        float c_gate       = smoothstep(0.0, GROWTH_C_GATE_HIGH, contrast_vel);
        float frac_floor   = smoothstep(GROWTH_FRAC_FLOOR_LOW, GROWTH_FRAC_FLOOR_HIGH,
                                        smoothed_bright_frac);
        float growth_mode_instant = smoothstep(GROWTH_SIG_LOW, GROWTH_SIG_HIGH, growth_sig)
                                  * c_gate * frac_floor;

        // Update smoothed_growth_mode FIRST so shutoff_eff can read the
        // just-updated, temporally-smoothed value — same characteristic as
        // PASS 6's PEAK_ATTEN and APL bypasses. The earlier dual-track design
        // (instant for shutoff, smoothed for PASS 6) traded clarity for a
        // "leading edge" benefit that the downstream EMA on
        // smoothed_spec_signal mostly absorbed anyway, while exposing PASS 5
        // to single-frame spikes (camera flashes, muzzle flashes). Hard-reset
        // on cuts because bright_vel/spec_vel against a stale-scene baseline
        // would false-positive growth_mode for one frame; lockout already
        // provides fast adaptation via TEMPORAL_ALPHA_FAST.
        bool gm_reset = (frame == 0) || scene_cut || (scene_cut_lockout > 0.0);
        smoothed_growth_mode = gm_reset
            ? 0.0
            : mix(smoothed_growth_mode, growth_mode_instant, alpha);

        float shutoff_eff = mix(spec_shutoff, 1.0,
                                GROWTH_SHUTOFF_LIFT * smoothed_growth_mode);
        float spec_raw    = spec_onset * shutoff_eff * tier_mode;
        // Apply bright-scene recovery to the lifted form as well — same
        // bs_raw and bright_scene weighting.
        spec_raw = mix(spec_raw, max(spec_raw, bs_raw), bright_scene);

        // ---- Light-pump band-pass (sudden sustained brightening) ----
        // Two fixed-alpha EMAs of pnorm_illum_v (highlight-weighted p-norm of
        // illum V — see PUMP_DRIVE_P; V = max(R,G,B) so saturated colored
        // events register); their positive difference is the pump drive.
        // Reset both lanes to pnorm_illum_v on frame 0 and across cut/lockout
        // so a cut transient or stale baseline can't manufacture drive (a
        // hard cut to a brighter scene must NOT pump). Self-contained — does
        // not touch the smoothed_* init block.
        bool pump_reset = (frame == 0) || scene_cut || (scene_cut_lockout > 0.0);
        if (pump_reset) {
            pump_fast = pnorm_illum_v;
            pump_slow = pnorm_illum_v;
            pump_env  = 0.0;
        } else {
            pump_fast = mix(pump_fast, pnorm_illum_v, PUMP_ALPHA_FAST);
            pump_slow = mix(pump_slow, pnorm_illum_v, PUMP_ALPHA_SLOW);
            float drive      = pump_fast - pump_slow;                    // SIGNED velocity
            float drive_eff  = max(0.0, drive);
            #if ENABLE_SPATIAL_PUMP
            // Localized onset: signed p-mean AGGREGATE of the per-cell drives
            // (see PUMP_DRIVE_P block). Reads the cell lanes' PREVIOUS-frame
            // values (this reducer runs before the cell block updates them —
            // 1 frame of lag on 8/25-frame EMAs, immaterial). Deadzone shrinks
            // |d| symmetrically (both signs), so cancellation is preserved.
            // Sign is the reveal safety: rise+fall of a crossing occluder/pan
            // cancels; only net-asymmetric brightening survives. ONSET ONLY —
            // release stays on the global lanes + per-cell mask closure.
            float loc_sum = 0.0;
            float fall_w = 0.0, fall_ratio = 0.0;
            for (uint i = 0u; i < 144u; i++) {
                float cf = pump_fast_cell[i];
                float cs = pump_slow_cell[i];
                float di = cf - cs;
                float m  = max(abs(di) - PUMP_CELL_DEADZONE, 0.0);
                float w  = pow(m, PUMP_DRIVE_P);
                loc_sum += sign(di) * w;
                // Falling cells also publish their own velocity ratio (the same
                // fractional drop the mask's release uses) for the local release.
                if (di < 0.0 && cs > 1e-3) {
                    fall_w     += w;
                    fall_ratio += w * (cf / cs);
                }
            }
            float drive_loc = sign(loc_sum) * pow(abs(loc_sum) / N_SAMPLES, 1.0 / PUMP_DRIVE_P);
            drive_eff = max(drive_eff, drive_loc);
            #endif
            float pump_gate  = smoothstep(PUMP_DRIVE_LOW, PUMP_DRIVE_HIGH, drive_eff);
            // Contrast-retention fade guard on the SAME V axis as the driver
            // (else a colored event would be driven but muted): high while the
            // frame keeps a hot core vs dark surround (explosion, spell),
            // collapses toward 0 as the field goes uniform (fade-to-white or
            // fade-to-colour). Events are never muted; fades ease out.
            float cover_gate = smoothstep(PUMP_CONTRAST_LOW, PUMP_CONTRAST_HIGH, contrast_v);
            // Velocity-matched release. The NEGATIVE half of the band-pass is
            // the source itself falling: while drive<0, pump_fast/pump_slow is
            // <1 by exactly the source's recent fractional drop, so pump_env
            // eases out in lockstep with the source (fast on a cut, gentle on a
            // bloom-fade). A steady source gives drive≈0 → ratio 1 → NO release
            // (this is what made a held light's old timed release look like an
            // animated dim). PUMP_ADAPT_FLOOR is the only clock left: it relaxes
            // an indefinitely-held light imperceptibly slowly (eye-adapting).
            float rel = (drive < 0.0 && pump_slow > 1e-3) ? (pump_fast / pump_slow) : 1.0;
            #if ENABLE_SPATIAL_PUMP
            // Local release — velocity-matched on the LOCAL axis. When the net
            // local drive is a fall (a dying fire), release at the falling
            // cells' own fast/slow ratio, |d|^p-weighted — the identical
            // fractional-drop semantics as the global rel and the mask's r.
            // Closes the linger hole (a purely-local event's env otherwise
            // persisted ~29s behind closed masks; any mask opening in that
            // window — incl. a large-occluder wake — inherited stale
            // amplitude) WITHOUT the frame-level division that guillotined
            // the env on dark frames (see PUMP_DRIVE_P block). A held light
            // (d≈0) and a balanced crossing (net≈0) release nothing.
            // fall_w floor is normal-range (not 0.0): forecloses a mixed-FTZ
            // denormal 0/0 → NaN that would persist in pump_env's SSBO.
            if (drive_loc < 0.0 && fall_w > 1e-8)
                rel = min(rel, clamp(fall_ratio / fall_w, 0.0, 1.0));
            #endif
            pump_env = max(pump_env * rel * PUMP_ADAPT_FLOOR, pump_gate) * cover_gate;
        }

        if (frame == 0) {
            smoothed_bright_frac  = 0.0;
            smoothed_spec_signal  = 0.0;
            smoothed_spec_natural = 0.0;
            smoothed_contrast     = contrast;
            smoothed_log_avg      = log_avg;
            scene_cut_lockout     = 0.0;
        } else {
            smoothed_bright_frac  = mix(smoothed_bright_frac, bright_frac, alpha);
            smoothed_spec_signal  = mix(smoothed_spec_signal, spec_raw, alpha);
            smoothed_spec_natural = mix(smoothed_spec_natural, spec_raw_natural, alpha);
            smoothed_contrast     = mix(smoothed_contrast, contrast, alpha);
            smoothed_log_avg      = mix(smoothed_log_avg, log_avg, alpha);
        }

        // Publish the per-cell pump reset flag (cut / lockout / frame 0) so all
        // lanes reset their cell band-pass coherently in the spatial block below.
        #if ENABLE_SPATIAL_PUMP
        s_pump_reset = pump_reset ? 1u : 0u;
        #endif

        // Dummy write satisfies the 1×1 SAVE target; SSBO above is the
        // real product. Other lanes are out-of-bounds for the image and
        // would be no-ops, but the guard avoids 143 redundant store ops.
        imageStore(out_image, ivec2(0), vec4(0));
    }

    // -------- per-cell pump band-pass (SUBTRACTIVE spatial mask) --------
    // All 144 lanes; each owns one 16×9 grid cell and touches only its own SSBO
    // slot (race-free). The barrier gates s_pump_reset visibility (published by
    // thread-0). Emits pump_env_cell[144] = per-cell "is this region BRIGHTENING"
    // ∈[0,1]; PASS 6 bilinear-upsamples it and MULTIPLIES the global scalar pump by
    // it. Spatial is SUBTRACTIVE: it can only SUPPRESS the scalar in non-brightening
    // regions — it can never ADD pump. So a re-exposed background (a reveal), a pan,
    // an occluder wake, etc. can never MANUFACTURE pump: if the scene isn't globally
    // brightening the scalar is ~0 and the product is ~0, regardless of any local
    // rise. This is why the whole additive-mask machinery (novelty gate, habitual-V
    // memory, established-exemption, localized-confidence gate, motion crossfade) is
    // GONE — it only existed to stop an additive mask from over-firing on reveals.
    //
    // Driver = coarse σ80 CELFLARE_ILLUM V at the cell center (V_ill, sampled above)
    // — the SAME field the scalar means over the frame, read per-cell. A cell that is
    // rising builds env (mask→1, keeps the scalar); a static/darkening cell decays to
    // 0 (mask→0, suppresses the scalar). V=max channel → colored blooms register.
    #if ENABLE_SPATIAL_PUMP
    barrier();
    {
        float v = V_ill;
        if (s_pump_reset != 0u) {
            pump_fast_cell[lid] = v;
            pump_slow_cell[lid] = v;
            pump_env_cell[lid]  = 0.0;
        } else {
            pump_fast_cell[lid] = mix(pump_fast_cell[lid], v, PUMP_ALPHA_FAST);
            pump_slow_cell[lid] = mix(pump_slow_cell[lid], v, PUMP_ALPHA_SLOW);
            float d = pump_fast_cell[lid] - pump_slow_cell[lid];
            // Dead-zone: idle-wobble cells (drive under the zone) read true zero.
            float a = smoothstep(PUMP_CELL_DRIVE_LOW, PUMP_CELL_DRIVE_HIGH, max(0.0, d - PUMP_CELL_DEADZONE));
            // Velocity-matched release: the negative half of the band-pass is the
            // region's own fall, so the mask eases out in lockstep (a suppressor that
            // re-engages as a region stops brightening). Max-held + adapt floor.
            float r = (d < 0.0 && pump_slow_cell[lid] > 1e-3)
                    ? (pump_fast_cell[lid] / pump_slow_cell[lid]) : 1.0;
            pump_env_cell[lid] = max(pump_env_cell[lid] * r * PUMP_ADAPT_FLOOR, a);
        }
    }
    #endif
}

// =============================================================================
// PASS 6: ILLUMINATION EXPANSION (full resolution)
// =============================================================================
// Core change from v3.2: expansion curve evaluated at Y_illum (bright-biased
// blur of regional luminance at ~100px effective scale) instead of per-pixel Y.
// All pixels in a bright region get the bright region's expansion — local
// contrast preserved by construction through multiplicative application.
//
// Scene adaptation via bright_frac (fraction of illumination above PASS 5's
// BRIGHT_STAT_THRESH 0.40 — deliberately above the expansion KNEE 0.30)
// replaces the 7-type scene classifier. Continuous, no arbitrary boundaries.
//
// CELFLARE_STATS is bound only as the explicit data dependency on PASS 5 —
// the stats themselves arrive through the SCENE_STATE SSBO. Do not remove
// the bind without verifying pass ordering/visibility on every backend.

//!HOOK MAIN
//!BIND HOOKED
//!BIND SCENE_STATE
//!BIND CELFLARE_STATS
//!BIND CELFLARE_ILLUM
//!DESC CelFlare v5.3

// =============================================
//  MAIN TUNING — start here
// =============================================
#define INTENSITY       1.0     // Global scaling — PEAK defines expansion directly at 1.0
#define KNEE            0.30    // Expansion onset — midtones below this stay near SDR


// =============================================
//  SPATIALLY-MODULATED CURVE — regional adaptation
// =============================================
// Expansion is always f(Y_pixel) — monotonic remapping, no 8-bit banding.
// Y_illum modulates the curve SHAPE: bright regions get gentle/broad curves
// (preserving highlight gradients), dark regions get steep/concentrated curves
// (highlight pop). Linear ramp + pow(t, gamma) — derivative monotonically
// increasing for gamma >= 1, no inflection in the face brightness range.
//
// Gradient-preservation principle: the spatial curve is the primary authority
// on tonal relationships. APL and Dynamic are gentle scene adjustments (~10%),
// not aggressive dampeners. Specular bonus adds HDR pop on top.
//
// Nit targets at REFERENCE_WHITE=116 (spatial curve only, pre-APL/spec):
//   Peak (Y=1.00):             ~278–313 nits
//   Highlights (Y≈0.90–0.95):  180–250 nits
//   Reference white (Y≈0.85):  145–155 nits
//   Midtones (Y≤0.50):         near SDR (negligible lift)
#define PEAK_BRIGHT     2.4     // Expansion peak for bright regions (~278 nits pre-APL)
#define PEAK_DARK       2.7     // Expansion peak for dark regions (~313 nits pre-APL)
#define GAMMA_BRIGHT    2.1     // Gentler ramp through 0.85–0.95 — peak preserved at Y=1.0
#define GAMMA_DARK      2.3     // Matching gradualness in dark scenes

// Saturated-brightness credit on the base ramp. BT.709 luma under-credits
// the brightness of saturated R/B-dominant colors (G carries 0.7152 of the
// weight), so a bright saturated accent inside a bright field gets left
// behind by the convex ramp and reads as a dark stain after expansion.
// Measured (cheek blush on bright skin, 1080p WEB scene): blush V=0.952 vs
// skin V=0.994 — nearly equal peak channel — but Y 0.735 vs 0.942 put them
// at x1.23 vs x1.84 expansion, amplifying the artist's 1.67x local contrast
// to 2.51x on glass: pink darkened relative to its surround = purple bruise.
// Perception agrees with V, not Y (Helmholtz–Kohlrausch: saturated colors
// look brighter than their luminance), and the artist placed the accent at
// the top of its channel range — so saturated pixels get a BOUNDED credit
// from stabilized Y toward stabilized V on the ramp input. This is the base-
// curve sibling of the spec path's v_drive escape, at a fraction of the
// noise exposure: the base ramp's slope is ~10x gentler than the spec ramp
// and the credit halves the coupling, so WEB-grade 4:2:0 chroma noise in V
// works out to ~1-2 nits of wobble (vs the full-ramp speckle that killed
// raw-V spec drivers). Near-neutrals are bit-identical (sat gate = 0).
// The Y floor fade keeps the EARLY_EXIT_GAMMA boundary conservative-EXACT:
// at/below BASE_V_Y_LO the credit is 0, so any pixel the early exit skips
// would have computed expansion = 1.0 anyway — no contour at the boundary.
// Consequence: dim saturated emissives (red LED Y~0.21) get no BASE credit;
// that regime belongs to the spec escape, which already covers V >= 0.88.
#define ENABLE_BASE_V_CREDIT 1
#define BASE_V_CREDIT        0.75   // fraction of the Y->V gap credited at full gate
#define BASE_V_SAT_LO        0.10   // sat_gamma gate (same band as the spec v_drive)
#define BASE_V_SAT_HI        0.30
#define BASE_V_Y_LO          0.32   // luma floor fade-in: 0 at/below the early exit (KNEE)
#define BASE_V_Y_HI          0.48
#define PEAK_ATTEN      0.12    // Gentle bright_frac dampening (spatial curve adapts)
#define BRIGHT_FRAC_REF 0.40    // Bright fraction where scene adaptation plateaus

// =============================================
//  DYNAMIC INTENSITY — contrast-driven expansion scaling
// =============================================
// Narrow range — gentle scene adaptation that preserves gradient relationships.
// Flat/pastel scenes get slightly softer, dramatic scenes slightly punchier.
// Driven by smoothed_contrast (log2 dynamic range in stops).
#define ENABLE_DYNAMIC_INTENSITY 1
#define DYN_CONTRAST_LOW    2.5     // Below this: flat scene, minimum intensity
#define DYN_CONTRAST_HIGH   5.5     // Above this: dramatic scene, maximum intensity
#define DYN_INTENSITY_LOW   0.90    // Multiplier for flat scenes (gentle)
#define DYN_INTENSITY_HIGH  1.15    // Multiplier for dramatic scenes (gentle)

// =============================================
//  APL MODULATION — brightness-driven expansion scaling
// =============================================
// Dark scenes: neutral (gamma handles midtone suppression).
// Bright scenes: gently reduced to preserve gradient while avoiding washout.
// Driven by smoothed_log_avg (perceptual brightness key).
#define ENABLE_APL_MOD      1
#define APL_KEY_DARK        0.03    // Below this: dark scene multiplier
#define APL_KEY_BRIGHT      0.30    // Above this: bright scene multiplier
#define APL_BOOST_DARK      1.25    // Neutral for dark scenes (gamma_dark suppresses midtones)
#define APL_DAMPEN_BRIGHT   0.65    // Bright-scene reduction — full white ≈ 206 nits at REFERENCE_WHITE=116 (Y_illum≈0.7, bf=1)
// Mid-scene notch: parabolic dampener peaking at apl_t=0.5 (smoothed_log_avg
// ≈ 0.16 — normally-lit interiors, mid-key cinematic). Prevents pale skin /
// fabric / hair from looking "illuminated" in those scenes by trimming a few
// percent off the APL multiplier. Endpoints (dark and bright) unaffected.
// Applied BEFORE the growth-mode bypass so expanding-object events in a
// mid-key scene still get full pop.
#define MID_APL_DAMPEN      0.08    // Peak reduction at apl_t=0.5 (~8% off expansion)

// =============================================
//  VELOCITY-GATED DAMPENER BYPASS — expanding-object HDR pop
// =============================================
// PASS 5 sets smoothed_growth_mode in [0,1] when an expanding bright object
// (fireball, crash-zoom on backlit window) is detected — distinguished from
// fade-to-white by requiring rising contrast and a non-trivial pixel-fraction
// floor. When growth_mode is high, the bright-scene dampeners that would
// otherwise progressively suppress expansion through the event are pulled
// back toward neutral. Spec_shutoff lift is applied upstream in PASS 5.
//
// Bypass strengths name what fraction of each dampener is removed at full
// growth_mode (1.0 = dampener fully neutralised, 0.0 = no bypass).
#define ENABLE_GROWTH_BYPASS    1
#define GROWTH_PEAK_ATTEN_BYPASS 0.7   // PEAK_ATTEN scaling: (1 - this * growth_mode)
#define GROWTH_APL_BYPASS        0.8   // APL factor → mix toward 1.0 by (this * growth_mode)

// =============================================
//  LIGHT PUMP — augment sudden sustained brightening
// =============================================
// PASS 5 sets pump_env in [0,1] during a multi-frame brightness RISE
// (explosion bloom, train exiting a tunnel, spell charge-up) and ≈0 otherwise.
// Here it multiplies the fully-formed expansion (post base/APL/spec) by a
// brightness-weighted gain, giving the rising bright region an exposure-like
// punch on top of the normal curve. Self-releases when brightness plateaus
// (pump_env → 0). Distinct from the growth bypass: that REMOVES suppression on
// a sustained expanding object; this ADDS gain on the rising EDGE. A fireball
// can trigger both (pump on the bloom rise, growth-mode on the sustain) — if
// they stack too hot, gate PUMP_STRENGTH down by smoothed_growth_mode.
//
// EXAGGERATED defaults for first validation — the gain ceil dominates so the
// effect is unmissable. Drop PUMP_STRENGTH to ~0.3-0.6 for the subtle target.
#define ENABLE_LIGHT_PUMP   1
#define PUMP_STRENGTH       1      // gain per unit pump_env at full pixel weight. At = CEIL the response is
                                   // PROPORTIONAL (peak reserved for full-detection events, not roof-pinned);
                                   // > CEIL slams moderate events to the roof (aggressive); subtle ≈ 0.4
#define PUMP_Y_LOW          0.35   // per-pixel weight onset — low/broad so the whole bright region lifts (not a pinpoint)
#define PUMP_GAIN_CEIL      1.5    // hard cap on the pump multiplier (safety against runaway expansion)
#define PUMP_GROWTH_DAMP    0.6    // down-gate pump where growth-mode already lifts expansion (anti double-stack on fireballs)
// Spatial pump (SUBTRACTIVE). ⚠ MUST match the PASS 5 copy (line ~587) — no compile
// guard; a mismatch leaves pump_env_cell unwritten and PASS 6 reads a garbage mask.
// PASS 5 produces the per-cell brightening mask (pump_env_cell[144], 16×9); this pass
// bilinear-samples it and MULTIPLIES the global scalar pump by it (only suppresses,
// never adds). 0 = scalar-only.
#define ENABLE_SPATIAL_PUMP 1
// SUBTRACTIVE apply (see PASS 5): pump_local = pump_env × mask, where mask is the
// bilinear per-cell brightening env ∈[0,1]. The scalar (pump_env) supplies amplitude
// AND its own cover/velocity guards; the mask only SUPPRESSES it where a region isn't
// brightening. No crossfade / conf / cover knobs here — all lived in the additive
// model. Mask fractional coords are smoothstep-eased (C1) to kill bilinear seams.

// =============================================
//  SPECULAR BONUS — scene-detected, per-pixel bloom
// =============================================
// Stats pass samples 16×9 grid, counting source pixels in highlight
// (>0.75) and specular (>0.92) tiers. Specular signal fires when a
// small fraction qualifies AND specular is rarer than highlights
// (tier separation = real specular, not just a bright scene).
//
// Per-pixel ramp selects WHICH pixels (smoothstep on Y_pixel).
// Scene-level APL drives peak/gamma — dark scenes get more pop,
// bright scenes stay controlled. No per-pixel Y_illum modulation
// (caused edge halos where bright met dark instead of bloom-like
// center-out falloff). Bypasses APL/dynamic intensity dampening.
#define ENABLE_SPECULAR_BONUS 1
// Saturated-channel spec drive — mirrors PASS 5's tier counting on
// V = max(R,G,B), so saturated primaries (red LED Y=0.21, V=1.0) qualify
// for the ramp and escape the Y-based early exit.
// NOTE: each HOOK block (pass) is a SEPARATE compilation unit — this define
// must exist HERE. The PASS 5 copy is not visible to this pass, and an
// undefined identifier inside #if silently evaluates to 0: that is exactly
// how this path was compiled out of the expansion pass from v4.7 through
// v5.0 while the detection half (PASS 5) kept running. If you add an
// #if-gated feature spanning passes, duplicate the define in every pass that
// tests it. (Do not write the literal directive prefix in prose anywhere in
// this file — the libplacebo parser splits sections on it even mid-comment.)
#define ENABLE_SATURATED_SPEC 0
#define SPEC_Y_LOW          0.80    // Ramp onset — widened to soften "crunchy" clipped speculars.
                                    // Builds the spec as a gradient INTO the clipped core from
                                    // below; peak at Y=1.0 is unchanged, so no attenuation of the
                                    // source clip. Lower = wider/gentler phase-in (floor ~0.75
                                    // before spec starts catching bright-but-not-specular pixels).
#define SPEC_Y_LOW_MID_BUMP 0.05    // Parabolic bump pushes onset to 0.93 at spec_apl=0.5,
                                    // which corresponds to smoothed_log_avg ≈ 0.16 —
                                    // normally-lit interiors, dusk exteriors, mid-key
                                    // cinematic lighting. Adds selectivity in scenes
                                    // with abundant mid-bright surfaces (lampshades,
                                    // faces, fabrics, TV screens) by requiring V > 0.93
                                    // before spec fires. Endpoints (dark/bright APL)
                                    // are unaffected — bump is zero at spec_apl=0 and 1.
#define SPEC_PEAK_DARK      1.9     // Specular boost in dark scenes (highlight pop)
#define SPEC_PEAK_BRIGHT    0.7     // Specular boost in bright scenes (modest — eye whites
                                    // and hair highlights kept perceptually cool)
#define SPEC_GAMMA_DARK     1.3     // Gentler concentration in dark scenes — broadens the
                                    // ramp across the V range so the transition feels
                                    // less like a discrete edge. Mid-spec values (V≈0.93)
                                    // get ~+20% of their old strength; peak unchanged.
#define SPEC_GAMMA_BRIGHT   1.1     // Near-linear ramp in bright scenes — gradual phase-in.
                                    // Walks back part of the v5.0 hair-trim hardening
                                    // (was 1.25); paired with SPEC_PEAK_BRIGHT=0.7 the
                                    // total bright-scene spec stays modest.
#define SPEC_APL_LOW        0.03    // Scene APL below → dark-scene specular params
#define SPEC_APL_HIGH       0.30    // Scene APL above → bright-scene specular params
// Saturation gate. Genuine specular is near-white; bright clothing/hair/skin
// have chroma at high luminance. Gamma-space max−min saturation. Scene-aware:
// dark scenes preserve saturated emissives (red LEDs, blue lasers should still
// pop); bright scenes suppress aggressively (in daylight "bright + colored" is
// almost always a real surface, not a specular event).
#define SPEC_SAT_LOW          0.05  // Below: near-white → no attenuation
#define SPEC_SAT_HIGH         0.25  // Above: colored surface → full attenuation
#define SPEC_SAT_ATTEN_DARK   0.20  // Dark scenes: gentle (red LEDs lose only 20%)
#define SPEC_SAT_ATTEN_BRIGHT 0.80  // Bright scenes: strong (colored objects suppressed)
// Emissive carve-out: high-V pixels are almost always light sources (tungsten
// R=1.0,G=0.85,B=0.7 → sat=0.30; sodium street lamps; fire; candle cores).
// Those should pop even in bright scenes despite being chromatic — the carve
// disables the sat gate as V approaches 1.0.
#define SPEC_SAT_EMISSIVE_LOW  0.92  // V_gamma below: full sat gate applies
#define SPEC_SAT_EMISSIVE_HIGH 0.99  // V_gamma above: gate fully disabled
// Super-white bonus: upscaler Y_gamma>1.0 is direct signal evidence that the
// source was compressed — reward it proportionally. Gain is conservative;
// ceil caps runaway on extreme super-white. At default settings, peak
// nits (dark scene, full signal, Y=1.2) rises from ~534 to ~545.
#define SPEC_OVERSHOOT_GAIN 0.3
#define SPEC_RAMP_CEIL      1.10    // Hard cap on ramp; safety against extreme overshoot

// Clip diffusion: blend near-white toward illum field to soften SDR clip edges
#define ENABLE_CLIP_DIFFUSION 0      // GLSL #if needs integer — toggle this when tuning CLIP_DIFFUSION > 0
#define CLIP_DIFFUSION       0.00    // Max blend at Y=1.0 (0.0=off, 0.5=strong)
#define CLIP_DIFFUSION_FLOOR 0.95    // Y_gamma below: no softening

// =============================================
//  CHROMA — expansion color behavior
// =============================================
// Chroma amplification attenuation for saturated pixels. Full cbrt(expansion)
// on chroma causes saturated colors to appear perceptually brighter than
// desaturated highlights at the same luminance expansion (Helmholtz-Kohlrausch).
// CHROMA_SCALE reduces this: 1.0 = full cbrt, 0.5 = half, 0.0 = chroma frozen.
// Only affects already-saturated pixels — near-neutrals always get full cbrt.
// DISABLED at current defaults: ENABLE_CHROMA_ATTEN 0 == the exact behavior
// of CHROMA_SCALE 1.00 (chroma_factor degenerates to cbrt_exp for every
// pixel — verified algebraically), which is the long-standing validated
// look. The #if guard makes that explicit instead of leaving dead per-pixel
// math whose elimination depended on the compiler folding mix(x, x, t).
#define ENABLE_CHROMA_ATTEN 0
#define CHROMA_SCALE        1.00

// Near-neutral fast path. For desaturated pixels the Oklab roundtrip
// (rgb_to_oklab → manipulations → oklab_to_rgb) degenerates to a uniform
// scale on linear RGB because:
//   - chroma attenuation gate (sat_norm) returns 0 → chroma_factor = cbrt_exp
//   - warm shift gated by chroma > WS_CHROMA_FLOOR (=0.015)
//   - pale skin gated by chroma > 0.015 (smoothstep onset)
// Under those conditions, oklab_exp = oklab_orig × cbrt_exp uniformly, and
// oklab_to_rgb returns rgb_linear × expansion exactly. Bypass cost is one
// max-min subtract at pass entry. Bound (brute-forced over hue/level for
// pixels reachable past the early exit): max Oklab chroma at sat_gamma=0.04
// is 0.0237 (darkish desaturated magenta) — that CAN clear the 0.015 WS/PS
// gates, so the bypass is not exactly equivalent there. Worst-case seam:
// warm-shift displacement <= chroma*theta = 0.0237*0.06 = 0.0014 in (a,b),
// sub-JND and comparable to the fast_cbrt noise floor. Accepted.
#define ENABLE_OKLAB_BYPASS 1
#define SAT_BYPASS_THRESH   0.04

// --- Warm Shift: Bezold-Brücke Hue Compensation ---
// Rotates warm hues (yellow-green to near-red) toward red in Oklab to
// compensate for the psychovisual green shift at higher luminance.
// Driven by illumination field (regional, not per-pixel Y) — nearby dark
// pixels in bright warm regions get compensated because the B-B shift is
// a spatial perceptual effect. b_norm scaling (sine of hue from +a axis)
// prevents overshoot: near-red pixels barely rotate, yellows rotate fully.
#define ENABLE_WARM_SHIFT    1
#define WS_HUE_COS          0.3420  // cos(70°) — center of warm range in Oklab
#define WS_HUE_SIN          0.9397  // sin(70°)
#define WS_HUE_POWER        1.2     // Hue window width (lower = wider, ~50° each side)
#define WS_STRENGTH          0.06   // Max rotation in radians (~3.4° at full drive)
#define WS_ILLUM_LOW         0.35   // Y_illum below: no shift (dark region)
#define WS_ILLUM_HIGH        0.80   // Y_illum above: full shift
#define WS_CHROMA_FLOOR      0.015  // Skip near-neutrals (unstable hue)

#define ENABLE_PALE_SKIN    1
#define ENABLE_PS_COMPRESS  0       // GLSL #if needs integer — toggle this when tuning PS_COMPRESS > 0
#define PS_HUE_COS          0.7317  // cos(43°) — warm hue center for skin detection
#define PS_HUE_SIN          0.6816  // sin(43°)
#define PS_HUE_POWER        2.0     // Sharpness of hue window
#define PS_BRIGHT_FRAC_LOW  0.05
#define PS_BRIGHT_FRAC_HIGH 0.30
#define PS_COMPRESS         0.00    // Expansion compression strength (applied when ENABLE_PS_COMPRESS=1)
#define PS_SAT_BOOST        0.20
#define PS_BRIGHT_FLOOR     0.50
#define PS_CHROMA_CEIL      0.03

// =============================================
//  OUTPUT — encoding
// =============================================
#define REFERENCE_WHITE 116.0
#define PQ_FAST_APPROX  1
#define EOTF_GAMMA      2.4
#define ENABLE_GRAIN_STABLE 1
// Early-exit luma bound. == KNEE is exact for the base curve: PASS 1 writes
// RAW luma (alpha encode) below its GRAIN_EARLY_EXIT (0.30), so for
// Y_gamma < KNEE the decision luma equals Y_gamma, t = 0, and expansion is
// exactly 1.0 through dynamic/APL (both scale expansion-1). Must stay
// <= PASS 1's GRAIN_EARLY_EXIT or stabilized decisions could cross KNEE.
#define EARLY_EXIT_GAMMA    KNEE

// =============================================
//  DEBUG
// =============================================
#define DEBUG_BYPASS         0
#define DEBUG_SHOW_ILLUM     0   // Illumination field as grayscale
#define DEBUG_SHOW_EXPANSION 0  // Expansion amount as heat map
#define DEBUG_SHOW_DETAIL    0   // Spatial vs per-pixel: green=spatial, red=per-pixel fallback
#define DEBUG_SHOW_SPECULAR  0   // Specular bonus: cyan = spec strength
#define DEBUG_SHOW_PUMP      0   // Light pump: red = scene pump_env, green = per-pixel applied gain
#define DEBUG_SHOW_WP        0   // Warm shift + pale skin
#define DEBUG_SHOW_STATS     0   // avg_illum + bright_frac + contrast + log_avg bars

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
    // Passthrough + early-exit path — black floor critical. ALWAYS use
    // exact pq_oetf: the fast polynomial's constant term evaluates to
    // ~0.068 at L=0, which decodes to ~0.12 nits, lifting letterbox bars
    // and true blacks. See v3.0.1 changelog. The sub-1-LSB asymmetry this
    // creates with linear709_to_pq2020 in the onset_blend 1.001..1.05
    // region is imperceptible.
    vec3 linear = eotf_gamma(rgb_gamma);
    vec3 bt2020 = max(bt709_to_bt2020(linear), 0.0);
    return pq_oetf(bt2020 * (REFERENCE_WHITE / 10000.0));
}

// Fast-poly low-end repair. The polynomial's error explodes below ~30 nits:
// +25..42 ten-bit LSB under 0.5 nits (an effective ~0.2-0.35 nit per-channel
// black floor), ±5 LSB through 1-10 nits. The expansion-onset blend in the
// hook body only covers expansion < 1.05 — dark CHANNELS of strongly-expanded
// saturated pixels (e.g. the blue channel of a red emissive) went through the
// raw polynomial. Blend small channels to the exact OETF; above PQ_EXACT_HIGH
// the pure polynomial's |err| stays ≲ 1.8 LSB (its design accuracy). The
// branch is coherent (dark channels cluster spatially) and the exact path
// costs 2 pow per channel on that minority.
#define PQ_EXACT_LOW    0.0015   // L normalized (≈15 nits): fully exact below
#define PQ_EXACT_HIGH   0.0030   // L normalized (≈30 nits): fully fast above

vec3 linear709_to_pq2020(vec3 rgb_linear) {
    vec3 bt2020 = max(bt709_to_bt2020(rgb_linear), 0.0);
    vec3 L = bt2020 * (REFERENCE_WHITE / 10000.0);
    #if PQ_FAST_APPROX
        vec3 pq = pq_oetf_fast(L);
        if (min(min(L.r, L.g), L.b) < PQ_EXACT_HIGH) {
            vec3 w = smoothstep(PQ_EXACT_LOW, PQ_EXACT_HIGH, L);
            pq = mix(pq_oetf(L), pq, w);
        }
        return pq;
    #else
        return pq_oetf(L);
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
        if (pos.x < 0.15 && pos.y < 0.12) {
            float bar_x = pos.x / 0.15;
            vec3 dbg = vec3(0.05);
            float row = pos.y / 0.12;
            if (row < 0.25) {
                // Row 1: bright_frac (yellow)
                if (bar_x < smoothed_bright_frac) dbg = vec3(0.6, 0.6, 0.0);
            } else if (row < 0.50) {
                // Row 2: contrast / 8 stops (orange)
                if (bar_x < smoothed_contrast / 8.0) dbg = vec3(0.7, 0.4, 0.0);
            } else if (row < 0.75) {
                // Row 3: log_avg (green)
                if (bar_x < smoothed_log_avg) dbg = vec3(0.0, 0.6, 0.0);
            } else {
                // Row 4: spec_signal (cyan)
                if (bar_x < smoothed_spec_signal) dbg = vec3(0.0, 0.6, 0.6);
            }
            // White outline + 25/50/75 tick marks for visual reference.
            const float thick = 0.0015;
            bool on_edge = pos.x < thick || pos.x > 0.15 - thick
                        || pos.y < thick || pos.y > 0.12 - thick;
            bool on_tick = (abs(bar_x - 0.25) < 0.005)
                        || (abs(bar_x - 0.50) < 0.005)
                        || (abs(bar_x - 0.75) < 0.005);
            if (on_edge) dbg = vec3(1.0);
            else if (on_tick) dbg = mix(dbg, vec3(1.0), 0.4);
            return vec4(gamma709_to_pq2020(dbg), 1.0);
        }
    }
    #endif

    // -------------------------------------------------------------------------
    // PIXEL LUMA / PEAK CHANNEL
    // -------------------------------------------------------------------------
    float Y_gamma = get_luma(rgb_gamma);
    // V_gamma (peak channel): lets saturated primaries (red LED Y=0.21,
    // V=1.0) escape the Y-based early exit and qualify for the spec ramp.
    // sat_gamma (V − min): gamma-space saturation. Reused by the spec
    // sat gate and as the gate for the Oklab fast path on near-neutrals.
    float V_gamma   = max(max(rgb_gamma.r, rgb_gamma.g), rgb_gamma.b);
    float min_gamma = min(min(rgb_gamma.r, rgb_gamma.g), rgb_gamma.b);
    float sat_gamma = V_gamma - min_gamma;

    // -------------------------------------------------------------------------
    // ILLUMINATION FIELD
    // -------------------------------------------------------------------------
    // Fetched AFTER the early exit unless clip diffusion (or the illum debug
    // view) needs it for every pixel — with both disabled, dark pixels skip
    // the texture fetch entirely.
    #if ENABLE_CLIP_DIFFUSION || DEBUG_SHOW_ILLUM
    vec3 illum_rgb = upsample_illum_rgb();
    float Y_illum = get_luma(illum_rgb);

    #if DEBUG_SHOW_ILLUM
        return vec4(gamma709_to_pq2020(illum_rgb), 1.0);
    #endif
    #endif

    // Early exit: no base expansion (Y < KNEE) AND no saturated-channel
    // signal that could ever reach the spec ramp.
    #if ENABLE_SATURATED_SPEC
    // SPEC_Y_LOW is the LOWEST possible spec onset across scene-APL (the
    // mid-APL bump only raises it), so this guard never exits a pixel that
    // could fire spec in any scene — conservative-exact, unlike the earlier
    // V < onset+bump form, which sacrificed dim emissives with V in
    // [SPEC_Y_LOW, SPEC_Y_LOW + bump) at endpoint APLs (the documented v5.0
    // ~4% loss — that loss class no longer exists).
    if (Y_gamma < EARLY_EXIT_GAMMA && V_gamma < SPEC_Y_LOW) {
        return vec4(gamma709_to_pq2020(color.rgb), 1.0);
    }
    #else
    if (Y_gamma < EARLY_EXIT_GAMMA) return vec4(gamma709_to_pq2020(color.rgb), 1.0);
    #endif

    #if !(ENABLE_CLIP_DIFFUSION || DEBUG_SHOW_ILLUM)
    vec3 illum_rgb = upsample_illum_rgb();
    float Y_illum = get_luma(illum_rgb);
    #endif

    // -------------------------------------------------------------------------
    // GRAIN STABILIZATION
    // -------------------------------------------------------------------------
    float Y_decision_gamma = Y_gamma;
    #if ENABLE_GRAIN_STABLE
        // PASS 1 stores its decision luma (stabilized Y_decision, or raw
        // Y_gamma on its exit paths) as alpha * 0.5 — decode with * 2.0.
        // Replaces the old `a > 0.99 ? Y_gamma : a` sentinel, which collided
        // with legitimate stabilized lumas in (0.99, 1.0] and silently fell
        // back to raw Y there (frame-to-frame decision flicker exactly in
        // the spec-ramp band).
        Y_decision_gamma = color.a * 2.0;
    #endif

    // -------------------------------------------------------------------------
    // CLIP DIFFUSION — soften SDR clip edges via illumination field
    // -------------------------------------------------------------------------
    // Only modifies rgb_gamma — Y_decision_gamma stays grain-stabilized.
    #if ENABLE_CLIP_DIFFUSION
    {
        float soften = smoothstep(CLIP_DIFFUSION_FLOOR, 1.0, Y_gamma) * CLIP_DIFFUSION;
        rgb_gamma = mix(rgb_gamma, illum_rgb, soften);
        Y_gamma = get_luma(rgb_gamma);
    }
    #endif

    // Grain-stabilized peak channel: the luma stabilizer's own measured
    // correction transplanted onto V (cancels achromatic grain exactly;
    // residual chroma noise is unfixable here — PASS 1 has no chroma
    // decision). Computed once for both consumers: the base-ramp V credit
    // below and the saturated-spec driver/emissive carve in the spec block.
    // Placed AFTER clip diffusion so it matches the spec block's previous
    // local computation bit-exactly in every config (diffusion rewrites
    // Y_gamma; V_gamma stays pre-diffusion).
    float V_stable = V_gamma + (Y_decision_gamma - Y_gamma);

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
    // Growth-mode bypass: an expanding hot region should NOT lose peak as it
    // grows — that's the moment the user wants the impressive HDR pop. Pulls
    // PEAK_ATTEN back toward zero in proportion to smoothed_growth_mode.
    #if ENABLE_GROWTH_BYPASS
    float peak_atten_eff = PEAK_ATTEN * (1.0 - GROWTH_PEAK_ATTEN_BYPASS * smoothed_growth_mode);
    #else
    float peak_atten_eff = PEAK_ATTEN;
    #endif
    local_peak *= (1.0 - peak_atten_eff * bf);  // scene-level dampening on top
    float local_gamma = mix(GAMMA_DARK, GAMMA_BRIGHT, spatial_t);

    // Per-pixel expansion curve: linear ramp from KNEE, shaped by pow(gamma).
    // Ramp input = stabilized luma, plus the bounded saturated-brightness
    // credit toward stabilized V (see BASE_V_CREDIT block). max() makes the
    // credit strictly lift-only; all three gate terms are smoothsteps of
    // continuous per-pixel quantities, so the drive stays contour-free.
    #if ENABLE_BASE_V_CREDIT
    float vcredit_w = BASE_V_CREDIT
                    * smoothstep(BASE_V_SAT_LO, BASE_V_SAT_HI, sat_gamma)
                    * smoothstep(BASE_V_Y_LO, BASE_V_Y_HI, Y_decision_gamma);
    float base_drive = mix(Y_decision_gamma,
                           max(Y_decision_gamma, V_stable), vcredit_w);
    #else
    float base_drive = Y_decision_gamma;
    #endif
    float t = max(base_drive - KNEE, 0.0) / (1.0 - KNEE);
    t = pow(min(t, 1.0), local_gamma);  // clamp for upscaler super-whites
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
        // Mid-scene notch: parabolic dampener at apl_t=0.5. Trims the APL
        // multiplier on mid-key interiors / dusk exteriors specifically;
        // endpoints unaffected. Applied BEFORE growth bypass so a fireball
        // in a mid-key scene can still drive apl_factor back to 1.0.
        float mid_notch = MID_APL_DAMPEN * apl_t * (1.0 - apl_t) * 4.0;
        apl_factor *= 1.0 - mid_notch;
        // Growth-mode bypass: pull apl_factor back toward 1.0 (no APL
        // adjustment) during an expanding-object event. The bypass is
        // symmetric across DARK/BRIGHT — in dark scenes APL_BOOST=1.25
        // also gets pulled toward 1.0, so a growth event in a dark scene
        // loses a small amount of dark-bias boost, which is the right
        // tradeoff (the spatial curve already handles dark-scene boost).
        #if ENABLE_GROWTH_BYPASS
        apl_factor = mix(apl_factor, 1.0, GROWTH_APL_BYPASS * smoothed_growth_mode);
        #endif
        expansion = 1.0 + (expansion - 1.0) * apl_factor;
    }
    #endif

    // -------------------------------------------------------------------------
    // SPECULAR BONUS — scene-detected, spatially-modulated
    // -------------------------------------------------------------------------
    // Scene gate: smoothed_spec_signal from stats pass (16×9 tier detection).
    // Per-pixel ramp: smoothstep on Y selects which pixels get boost.
    // Y_illum modulates peak and gamma (same pattern as base curve) —
    // adds continuous spatial variation that breaks 8-bit quantization.
    // Added AFTER APL/dynamic intensity — not subject to those dampeners.
    #if ENABLE_SPECULAR_BONUS
    float spec_strength;
    {
        // Scene-level APL drives peak/gamma — dark *scenes* get more pop,
        // but no spatial edge bias within a single bright region.
        float spec_apl = smoothstep(SPEC_APL_LOW, SPEC_APL_HIGH, smoothed_log_avg);
        float spec_peak = mix(SPEC_PEAK_DARK, SPEC_PEAK_BRIGHT, spec_apl);
        float spec_gamma = mix(SPEC_GAMMA_DARK, SPEC_GAMMA_BRIGHT, spec_apl);
        // Drive signal: stabilized Y, with a saturation-gated peak-channel
        // escape. Saturated primaries at peak channel (V=1.0) qualify even
        // when Y is low, so red LEDs and blue lasers get emissive pop. The
        // v_drive gate keeps NEAR-NEUTRALS on the grain-stabilized luma
        // exactly: a bare max(Y_dec, V) would let raw V rectify every
        // positive grain excursion past the stabilizer (asymmetric sparkle
        // + DC lift) on bright neutrals, where V ≈ raw Y.
        //
        // V itself must ALSO be grain-stabilized: a bright saturated sky
        // (sat ~0.4-0.7, V ~0.95) drives v_drive to 1.0, putting raw V on
        // the steep part of the spec ramp — per-frame grain in the peak
        // channel became per-pixel spec flicker (speckling sky). Grain is
        // overwhelmingly achromatic, so the luma stabilizer's own measured
        // correction (Y_decision - Y_raw) transplants onto V directly.
        // Where PASS 1 didn't stabilize (Y < its 0.30 early exit — the red
        // LED / laser case — or super-white) the correction is exactly 0
        // and V passes through raw, scene-damped by smoothed_spec_signal
        // as before.
        // The escape is additionally FADED OUT by stabilized luma: it exists
        // for DIM saturated emissives (red LED Y=0.21, blue laser Y=0.07,
        // magenta neon Y=0.29) whose Y can never reach the ramp. A bright
        // saturated FIELD (blue/cyan sky: Y ~0.70, V ~0.95) is per-pixel
        // indistinguishable from an emissive, and its V carries Cb chroma
        // noise that NO luma-derived stabilization can remove (PASS 1 has no
        // chroma decision) — so above the fade band the driver returns to
        // the grain-stabilized luma exactly, which zeroes the ramp there.
        // High-Y saturated emissives (green laser 0.72, cyan 0.79) lose the
        // escape, same as every version before v5.1 — accepted.
        #define SPEC_V_ESCAPE_Y_LO 0.45
        #define SPEC_V_ESCAPE_Y_HI 0.60
        #if ENABLE_SATURATED_SPEC
        // V_stable hoisted to the base-curve entry (shared with the
        // BASE_V_CREDIT ramp input) — same value, computed once.
        float v_drive = smoothstep(0.10, 0.30, sat_gamma)
                      * (1.0 - smoothstep(SPEC_V_ESCAPE_Y_LO,
                                          SPEC_V_ESCAPE_Y_HI, Y_decision_gamma));
        float spec_driver = mix(Y_decision_gamma,
                                max(Y_decision_gamma, V_stable), v_drive);
        #else
        float spec_driver = Y_decision_gamma;
        #endif
        // APL-tiered onset: parabolic bump pushes the ramp threshold up in
        // mid-bright scenes where lots of pixels are 0.88–0.93 but aren't
        // genuine specular. Endpoints stay at SPEC_Y_LOW. Bump peaks at
        // spec_apl=0.5 with value SPEC_Y_LOW_MID_BUMP.
        float spec_y_low = SPEC_Y_LOW
                         + SPEC_Y_LOW_MID_BUMP * spec_apl * (1.0 - spec_apl) * 4.0;
        float spec_t = smoothstep(spec_y_low, 1.0, spec_driver);
        // Super-white overshoot: upscaler can produce signal > 1.0. Treat
        // it as evidence that the source was SDR-clipped — add linear bonus
        // on top of the saturated ramp, scene-signal-gated as everything
        // else. Hard-capped so extreme overshoot can't blow the nit budget.
        float overshoot = max(spec_driver - 1.0, 0.0);
        float spec_ramp = min(pow(spec_t, spec_gamma) + overshoot * SPEC_OVERSHOOT_GAIN,
                              SPEC_RAMP_CEIL);
        spec_strength = spec_peak * spec_ramp * smoothed_spec_signal;

        // Saturation gate. Genuine specular is near-white. Bright colored
        // objects (red shirts under sun, blonde hair, saturated sky) shouldn't
        // get a "specular pop". Pairs with bright-scene recovery upstream:
        // recovery turns spec back on in daylight, sat-gate keeps it from
        // firing on the red car under the sun. Emissive carve-out preserves
        // light sources at V ≥ 0.92 (tungsten/fire/sodium) from being gated.
        // V_gamma + sat_gamma already computed at PASS 6 entry (shared with
        // the Oklab fast-path bypass). The carve uses the grain-stabilized
        // V where available — its 0.92-0.99 band has a ~14x/unit slope, so
        // raw-V grain made the sat gate itself flicker on saturated skies.
        #if ENABLE_SATURATED_SPEC
        float emissive_carve = smoothstep(SPEC_SAT_EMISSIVE_LOW,
                                          SPEC_SAT_EMISSIVE_HIGH, V_stable);
        #else
        float emissive_carve = smoothstep(SPEC_SAT_EMISSIVE_LOW,
                                          SPEC_SAT_EMISSIVE_HIGH, V_gamma);
        #endif
        float sat_atten = mix(SPEC_SAT_ATTEN_DARK, SPEC_SAT_ATTEN_BRIGHT, spec_apl)
                        * (1.0 - emissive_carve);
        spec_strength *= 1.0 - smoothstep(SPEC_SAT_LOW, SPEC_SAT_HIGH, sat_gamma) * sat_atten;

        expansion += spec_strength;
    }
    #endif

    #if DEBUG_SHOW_SPECULAR && ENABLE_SPECULAR_BONUS
    {
        return vec4(gamma709_to_pq2020(vec3(0.0, spec_strength, spec_strength)), 1.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // LIGHT PUMP — augment sudden sustained brightening (post-spec)
    // -------------------------------------------------------------------------
    // Multiplicative exposure-like gain on the fully-formed expansion, weighted
    // by per-pixel brightness so the rising bright region lifts while shadows
    // hold (preserves contrast — the explosion doesn't wash the dark frame to
    // grey). pump_env is the PASS 5 band-pass: nonzero only during a multi-
    // frame brightness rise, self-releasing at plateau. Capped for safety.
    #if ENABLE_LIGHT_PUMP
    float pump_gain = 0.0;
    {
        float pump_w = smoothstep(PUMP_Y_LOW, 1.0, Y_decision_gamma);
        // Spatial vs scene-global drive. Spatial: bilinear-sample the per-cell
        // mask (16×9) at this pixel, gated by global event confidence (pump_env,
        // which already carries the fade/cover guard). Localizes the pump to the
        // brightening region and rejects across-pans. Legacy: the global scalar
        // lifts every bright pixel uniformly.
        #if ENABLE_SPATIAL_PUMP
        vec2  pg  = vec2(HOOKED_pos.x * 16.0 - 0.5, HOOKED_pos.y * 9.0 - 0.5);
        vec2  pgf = fract(pg);
        pgf = pgf * pgf * (3.0 - 2.0 * pgf);    // #5: smoothstep-ease → C1, kills bilinear kink seams
        ivec2 pi0 = clamp(ivec2(floor(pg)),     ivec2(0), ivec2(15, 8));
        ivec2 pi1 = clamp(ivec2(floor(pg)) + 1, ivec2(0), ivec2(15, 8));
        float m00 = pump_env_cell[pi0.y * 16 + pi0.x];
        float m10 = pump_env_cell[pi0.y * 16 + pi1.x];
        float m01 = pump_env_cell[pi1.y * 16 + pi0.x];
        float m11 = pump_env_cell[pi1.y * 16 + pi1.x];
        float mask = mix(mix(m00, m10, pgf.x), mix(m01, m11, pgf.x), pgf.y);
        // SUBTRACTIVE: the mask (per-cell "is this region brightening" ∈[0,1]) only
        // SUPPRESSES the global scalar pump — it can't add pump. pump_env already
        // carries the scalar's amplitude, contrast/cover guard, and velocity release.
        // A brightening region has mask→1 (keeps the scalar); a static/darkening one
        // decays to 0 (suppressed). A reveal/pan/occluder-wake can't manufacture pump:
        // no global event ⇒ pump_env ~0 ⇒ product ~0 regardless of any local rise.
        float pump_local = pump_env * mask;
        #else
        float pump_local = pump_env;
        #endif
        // Down-gate where growth-mode is already restoring expansion (a fireball
        // triggers both): keeps pump + growth-bypass + spec from stacking the
        // transient peak past the display's ceiling, where the DISPLAY would
        // hard-clip and flatten the hot core. Gradation itself is never at risk
        // from the pump — it's a monotonic multiplier on expansion, so a
        // gradient stays a gradient; this only governs the absolute peak.
        float pump_str = PUMP_STRENGTH * (1.0 - PUMP_GROWTH_DAMP * smoothed_growth_mode);
        pump_gain = min(pump_local * pump_str * pump_w, PUMP_GAIN_CEIL);
        expansion *= 1.0 + pump_gain;
    }
    #endif

    #if DEBUG_SHOW_PUMP && ENABLE_LIGHT_PUMP
    {
        #if ENABLE_SPATIAL_PUMP
        // Red = scalar pump (global amplitude); Green = per-pixel applied gain;
        // Blue = per-cell suppression mask (1 = brightening/kept, 0 = suppressed).
        vec2  pg  = vec2(HOOKED_pos.x * 16.0 - 0.5, HOOKED_pos.y * 9.0 - 0.5);
        vec2  pgf = fract(pg);
        pgf = pgf * pgf * (3.0 - 2.0 * pgf);    // match application: eased C1 sampling
        ivec2 pi0 = clamp(ivec2(floor(pg)),     ivec2(0), ivec2(15, 8));
        ivec2 pi1 = clamp(ivec2(floor(pg)) + 1, ivec2(0), ivec2(15, 8));
        float mask = mix(mix(pump_env_cell[pi0.y*16+pi0.x], pump_env_cell[pi0.y*16+pi1.x], pgf.x),
                         mix(pump_env_cell[pi1.y*16+pi0.x], pump_env_cell[pi1.y*16+pi1.x], pgf.x), pgf.y);
        return vec4(gamma709_to_pq2020(vec3(pump_env, pump_gain, mask)), 1.0);
        #else
        // Red = scene-global trigger; Green = per-pixel applied gain.
        return vec4(gamma709_to_pq2020(vec3(pump_env, pump_gain, 0.0)), 1.0);
        #endif
    }
    #endif

    // -------------------------------------------------------------------------
    // DEBUG: Warm-shift / pale-skin visualization
    // -------------------------------------------------------------------------
    // Self-contained compute so the production path can defer linearize +
    // Oklab until after the early-exit below.
    #if DEBUG_SHOW_WP && (ENABLE_WARM_SHIFT || ENABLE_PALE_SKIN)
    {
        vec3 rl_dbg = eotf_gamma(rgb_gamma);
        vec3 ok_dbg = rgb_to_oklab(rl_dbg);
        float cr_dbg = sqrt(ok_dbg.y * ok_dbg.y + ok_dbg.z * ok_dbg.z);
        float ws_mag = 0.0;
        float ps_mag = 0.0;
        #if ENABLE_WARM_SHIFT
        {
            float inv_c = (cr_dbg > WS_CHROMA_FLOOR) ? (1.0 / cr_dbg) : 0.0;
            float cdh = (ok_dbg.y * WS_HUE_COS + ok_dbg.z * WS_HUE_SIN) * inv_c;
            float hw = pow(max(cdh, 0.0), WS_HUE_POWER);
            float drv = smoothstep(WS_ILLUM_LOW, WS_ILLUM_HIGH, Y_illum);
            float bn = max(ok_dbg.z * inv_c, 0.0);
            ws_mag = WS_STRENGTH * drv * hw * bn * 50.0;
        }
        #endif
        #if ENABLE_PALE_SKIN
        {
            float inv_c = (cr_dbg > 1e-6) ? (1.0 / cr_dbg) : 0.0;
            float cdh = (ok_dbg.y * PS_HUE_COS + ok_dbg.z * PS_HUE_SIN) * inv_c;
            float hw = pow(max(cdh, 0.0), PS_HUE_POWER);
            float gate = smoothstep(PS_BRIGHT_FRAC_LOW, PS_BRIGHT_FRAC_HIGH, smoothed_bright_frac);
            float chw = smoothstep(0.015, 0.06, cr_dbg)
                      * (1.0 - smoothstep(0.04, PS_CHROMA_CEIL + 0.04, cr_dbg));
            #if ENABLE_GRAIN_STABLE
            float Yd = eotf_gamma(color.a * 2.0);
            #else
            float Yd = get_luma(rl_dbg);
            #endif
            float bw = smoothstep(PS_BRIGHT_FLOOR, PS_BRIGHT_FLOOR + 0.15, Yd);
            ps_mag = hw * chw * bw * gate * 10.0;
        }
        #endif
        return vec4(gamma709_to_pq2020(vec3(ps_mag, ws_mag, 0.0)), color.a);
    }
    #endif

    // -------------------------------------------------------------------------
    // EARLY EXIT: non-expanded pixels
    // -------------------------------------------------------------------------
    // Warm-shift rotation and pale-skin chroma boost only apply to expanded
    // pixels, so we defer Oklab + linearize + detection to after the early
    // exit. If PS_COMPRESS > 0 is tuned on, a narrow corner of pixels near
    // expansion ≈ 1.001 will now pay the Oklab cost only to be clamped back
    // below threshold — accepted.
    if (expansion < 1.001) {
        return vec4(gamma709_to_pq2020(color.rgb), 1.0);
    }

    #if DEBUG_SHOW_EXPANSION
    {
        float exp_amount = (expansion - 1.0) / 2.5;
        return vec4(gamma709_to_pq2020(vec3(exp_amount, exp_amount * 0.3, 0.0)), 1.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // LINEARIZE
    // -------------------------------------------------------------------------
    vec3 rgb_linear = eotf_gamma(rgb_gamma);

    // -------------------------------------------------------------------------
    // EXPANSION APPLY — fast path (near-neutrals) vs full Oklab path
    // -------------------------------------------------------------------------
    // For near-neutral pixels (sat_gamma < SAT_BYPASS_THRESH) the Oklab
    // roundtrip degenerates to a uniform linear-RGB scale. All three Oklab
    // manipulations (chroma attenuation, warm shift, pale skin) are exactly
    // zero in that chroma range, so oklab_exp = oklab_orig * cbrt_exp and
    // oklab_to_rgb returns rgb_linear * expansion. Bypass produces the same
    // result without the rgb_to_oklab + manipulations + oklab_to_rgb chain.
    vec3 rgb_expanded;
    #if ENABLE_OKLAB_BYPASS
    if (sat_gamma < SAT_BYPASS_THRESH) {
        rgb_expanded = rgb_linear * expansion;
    } else
    #endif
    {
        vec3 oklab_orig = rgb_to_oklab(rgb_linear);
        float chroma_orig = sqrt(oklab_orig.y * oklab_orig.y + oklab_orig.z * oklab_orig.z);

        // Shared 1/chroma for warm-shift + pale-skin hue detection. Single
        // floor at 1e-6 keeps the divide finite; WS_CHROMA_FLOOR enforced
        // by an explicit branch where it matters.
        float inv_chroma = (chroma_orig > 1e-6) ? (1.0 / chroma_orig) : 0.0;

        // ---- WARM SHIFT DETECTION (Bezold-Brücke hue compensation) ----
        #if ENABLE_WARM_SHIFT
        float ws_angle = 0.0;
        if (chroma_orig > WS_CHROMA_FLOOR) {
            float ws_cos_dh = (oklab_orig.y * WS_HUE_COS + oklab_orig.z * WS_HUE_SIN) * inv_chroma;
            float ws_hue_w = pow(max(ws_cos_dh, 0.0), WS_HUE_POWER);
            float ws_drive = smoothstep(WS_ILLUM_LOW, WS_ILLUM_HIGH, Y_illum);
            float b_norm = max(oklab_orig.z * inv_chroma, 0.0);
            ws_angle = WS_STRENGTH * ws_drive * ws_hue_w * b_norm;
        }
        #endif

        // ---- PALE SKIN PROTECTION ----
        #if ENABLE_PALE_SKIN
            #if ENABLE_GRAIN_STABLE
            float Y_decision = eotf_gamma(color.a * 2.0);
            #else
            float Y_decision = get_luma(rgb_linear);
            #endif
            float ps_cos_dh = (oklab_orig.y * PS_HUE_COS + oklab_orig.z * PS_HUE_SIN) * inv_chroma;
            float ps_hue_w = pow(max(ps_cos_dh, 0.0), PS_HUE_POWER);
            float ps_gate = smoothstep(PS_BRIGHT_FRAC_LOW, PS_BRIGHT_FRAC_HIGH, smoothed_bright_frac);
            float ps_chroma_w = smoothstep(0.015, 0.06, chroma_orig)
                              * (1.0 - smoothstep(0.04, PS_CHROMA_CEIL + 0.04, chroma_orig));
            float ps_bright_w = smoothstep(PS_BRIGHT_FLOOR, PS_BRIGHT_FLOOR + 0.15, Y_decision);
            float ps_w = ps_hue_w * ps_chroma_w * ps_bright_w * ps_gate;
            #if ENABLE_PS_COMPRESS
            expansion = mix(expansion, 1.0, PS_COMPRESS * ps_w);
            #endif
            float ps_sat = PS_SAT_BOOST * ps_w;
        #endif

        // ---- APPLY WARM SHIFT ----
        // Small-angle rotation in Oklab (a,b) plane — clockwise toward red.
        // cos(θ) ≈ 1, sin(θ) ≈ θ. At the current max θ = 0.06 rad, chroma
        // grows by θ²/2 ≈ 0.2% (the small-angle matrix scales by √(1+θ²)).
        #if ENABLE_WARM_SHIFT
        if (ws_angle > 0.0) {
            float a_shifted = oklab_orig.y + oklab_orig.z * ws_angle;
            float b_shifted = oklab_orig.z - oklab_orig.y * ws_angle;
            oklab_orig.y = a_shifted;
            oklab_orig.z = b_shifted;
        }
        #endif

        // ---- APPLY EXPANSION (Oklab Space) ----
        // L scales by cbrt(expansion) (equivalent to linear RGB multiply).
        // Chroma scales by a REDUCED amount for saturated pixels — full cbrt
        // causes saturated colors to gain perceptual brightness from chroma
        // amplification (Helmholtz-Kohlrausch). CHROMA_SCALE controls the
        // attenuation; sat_norm weights toward already-saturated pixels.
        vec3 oklab_exp = oklab_orig;
        float cbrt_exp = fast_cbrt(expansion);
        oklab_exp.x *= cbrt_exp;
        #if ENABLE_CHROMA_ATTEN
        float sat_norm = smoothstep(0.10, 0.25, chroma_orig);
        float chroma_factor = mix(cbrt_exp, mix(1.0, cbrt_exp, CHROMA_SCALE), sat_norm);
        #else
        // H-K attenuation disabled (== CHROMA_SCALE 1.0 exactly): chroma
        // scales by the same cbrt as L — uniform LMS' scale, i.e. plain
        // rgb_linear * expansion for chroma-neutral manipulation paths.
        float chroma_factor = cbrt_exp;
        #endif
        oklab_exp.yz *= chroma_factor;
        #if ENABLE_PALE_SKIN
        oklab_exp.yz *= (1.0 + ps_sat);
        #endif

        rgb_expanded = oklab_to_rgb(oklab_exp);
    }

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

    // Alpha out is a clean 1.0 on every production path — color.a held
    // PASS 1's encoded decision luma, which must not leak past this pass.
    return vec4(rgb_pq, 1.0);
}
