// CelFlare Transport Additive A2 — Illumination-Decomposition SDR→HDR Expansion
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// ADDITIVE DEVELOPMENT TRACK, forked from the verified subtractive checkpoint
// 412f1b8. This file owns separate persistent state; never load it alongside
// CelFlare.glsl or CelFlare-transport.glsl (that double-processes the image and
// collides with the shared transient intermediate namespaces).
// A2 ships the additive-safe opening contract behind cf_additive_pump. Local
// flow refinement, reveal memory, a pump-domain motion veto and persisted proof
// harden every mask opening; cf_additive_pump=0 remains the verified subtractive
// fallback for field comparison.
//
// Design goal: emulate a professional HDR grade of the source — midtones hold
// the SDR grade, highlights expand with natural gradation, speculars get
// grade-realistic pop. Never exaggeration for its own sake. (The "fake-gradient
// hot core" language in the rules below is an anti-squash requirement for
// clipped regions — a defense, not an aesthetic target.)
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
// Quick start: all supported user controls live in the USER TUNING block
// directly below. cf_ref_white MUST match hdr-reference-white in mpv.conf;
// everything else is taste. Every knob can also be set from mpv.conf without
// editing this file, e.g.:
//     glsl-shader-opts=cf_strength=1.2,cf_spec=0.8
// The five sliders respond live during playback; the toggles trigger a quick
// shader recompile on change.
//
// Deep tuning (advanced): search "MAIN TUNING" in the expansion pass. The
// cf_* knobs are neutral overlays on those anchors — deep values stay
// canonical, knobs scale them.

// ---- shampv shader API — machine-readable contract (plain comments to
// libplacebo; read by the shampv tuner script). Declares that this shader
// consumes SDR, emits PQ BT.2020 (the pipeline must retag the frame), and
// that cf_ref_white must track the player's hdr-reference-white.
//@shampv input sdr
//@shampv output trc=pq primaries=bt.2020
//@shampv ref-white-param cf_ref_white

// =============================================================================
//  USER TUNING
// =============================================================================
// Each control is the plain number on the last line of its block. Sliders
// first, then feature toggles (1 = on, 0 = off; changing a toggle recompiles
// the shader). Sliders are DYNAMIC since v5.10: glsl-shader-opts changes
// apply on the next frame, no recompile, scene/pump state preserved.
// Ranges are enforced; defaults = the shipped tune. Anything not
// listed here is internal — tune it in the passes below at your own peril.
// NOTE: no comment lines may sit between a PARAM directive block and the next
// directive — the parser folds them into the value and fails to load.

//!PARAM cf_ref_white
//!DESC SDR white level in nits — MUST match hdr-reference-white in mpv.conf. On Windows = the 'SDR content brightness' slider (README has the slider-to-nits table).
//!TYPE DYNAMIC float
//!MINIMUM 80.0
//!MAXIMUM 480.0
116.0

//!PARAM cf_strength
//!DESC Overall HDR strength. 1 = shipped tune · 0 = plain SDR, no expansion · ↑ 2 = double. Scales base expansion, specular pop and light pump together.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM cf_curve
//!DESC Expansion ramp shape. ↓ <1 = gentle broad lift · ↑ >1 = punch on brightest pixels. Peak unchanged. Default 1.
//!TYPE DYNAMIC float
//!MINIMUM 0.6
//!MAXIMUM 1.8
1.0

//!PARAM cf_shoulder
//!DESC Highlight shoulder — eases how hard expansion hits the brightest pixels. ↑ 1 = smoother, no steepening · ↓ 0 = steepest near-clip pop. 0.7 shipped.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.7

//!PARAM cf_spec
//!DESC Specular pop — extra punch on glints, light sources, clipped highlights. ↑ = punchier · 0 = off. Default 0.6 (shipped, scales internal spec peaks).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.6

//!PARAM cf_pump
//!DESC Light pump — temporary surge on sustained brightening (explosions, tunnel exits, spells). ↑ = stronger surge · 1 = shipped · 0 = off.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM cf_grain_stab
//!DESC Grain stabilization (toggle). 1 = keep film grain filmic after expansion instead of shimmering · 0 = off.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_spec_bonus
//!DESC Master switch for the cf_spec specular-pop slider. 1 = on · 0 = disables cf_spec.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_light_pump
//!DESC Master switch for the cf_pump light-pump slider. 1 = on · 0 = disables cf_pump.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_spatial_pump
//!DESC Pump localization (toggle). 1 = confine the pump to the region actually brightening · 0 = scene-global pump only.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_additive_pump
//!DESC Additive regional pump. 1 = independent per-region amplitude with hardened opening proof · 0 = verified subtractive reference.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_warm_shift
//!DESC Warm-hue correction (toggle). 1 = stop fire, sunsets and skin drifting green as they brighten · 0 = off.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_pale_skin
//!DESC Pale-skin protection (toggle). 1 = keep fair skin from washing out in bright scenes · 0 = off.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_debug
//!DESC Debug views: 0 = off, 1 = bypass, 2 = illumination field, 3 = expansion heat map, 4 = spatial/per-pixel detail, 5 = specular, 6 = pump, 7 = warm-shift/skin, 8 = stats bars, 9 = motion prev-offset, 10 = motion evidence, 11 = motion residual, 12 = additive proof (R established, G emission ratio, B persistence).
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 12
0

//!BUFFER CELFLARE_ADD_STATE
//!VAR float smoothed_bright_frac
//!VAR float smoothed_spec_signal
//!VAR float smoothed_contrast
//!VAR float smoothed_log_avg
//!VAR float smoothed_growth_mode
//!VAR float scene_cut_lockout
//!VAR float smoothed_spec_natural
//!VAR float smoothed_top_frac
//!VAR float pump_fast
//!VAR float pump_slow
//!VAR float pump_env
//!VAR float pump_cover_gate
//!VAR float prev_illum[144]
//!VAR float prev_illum_v[144]
//!VAR float pump_fast_cell[144]
//!VAR float pump_slow_cell[144]
//!VAR float pump_very_slow_cell[144]
//!VAR float pump_open_persist_cell[144]
//!VAR float pump_env_cell[144]
//!VAR float pump_mask_cell[144]
//!VAR float pump_seed_cell[144]
//!VAR float bar_run[8]
//!VAR float motion_state_magic
//!VAR float motion_dom_x
//!VAR float motion_dom_y
//!VAR float motion_dom_support
//!VAR float motion_bad_match_frac
//!VAR float motion_match_coverage
//!VAR float additive_mode_magic
//!VAR float motion_trust_cell[144]
//!VAR float motion_mc_local_cell[144]
//!VAR float motion_mc_effective_cell[144]
//!STORAGE

//!TEXTURE CELFLARE_ADD_MOTION_PREV
//!SIZE 128 72
//!FORMAT rgba16f
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

    // -------- 12 taps: 2 rings x 3 antipodal pairs, one fused loop --------
    // Taps 0-2 are the outer ring (radius R), taps 3-5 the inner (R/2).
    // Each basis direction is rotated once and sampled at +o and -o
    // (6 rotation matmuls saved per pixel vs rotating all 12). The tap
    // chain (sample -> raw diff -> asym -> weight) lives ONCE here; the
    // four formerly hand-synced copies differed only in offset/gradient
    // sign and the per-ring params below. Accumulation order (outer pairs
    // then inner, + before -) is preserved — bit-identical to the old
    // unrolled form. Per original tuning, the inner ring:
    //  - blur weights take INNER_RING_BOOST (inner ring carries 2:1 weight)
    //  - gradient weights use the un-boosted w so the boost only amplifies
    //    blur trust, not gradient estimation (grad_w += w on both rings)
    //  - gradient direction scaled by 2.0 to match outer-ring units
    //    (raw_diff at half radius is half the linear-gradient response;
    //    multiply back to keep gx/gy comparable across rings).
    const vec2 tap_basis[6] = vec2[6](
        vec2( 1.000, 0.000),   // outer ring
        vec2( 0.500, 0.866),
        vec2(-0.500, 0.866),
        vec2( 0.866, 0.500),   // inner ring
        vec2( 0.000, 1.000),
        vec2(-0.866, 0.500)
    );

    // Tile-local center of this thread's pixel.
    vec2 tile_center = vec2(gl_LocalInvocationID.xy) + vec2(HALO);

    float blurred = Y_gamma;
    float total_w = 1.0;
    float gx = 0.0, gy = 0.0, grad_w = 0.0;

    for (int i = 0; i < 6; i++) {
        bool inner   = i >= 3;
        float boost  = inner ? INNER_RING_BOOST : 1.0;
        float gscale = inner ? 2.0 : 1.0;
        vec2 h = tap_basis[i];
        vec2 r = vec2(h.x * ca - h.y * sa, h.x * sa + h.y * ca);
        vec2 o = r * (inner ? float(GRAIN_BLUR_RADIUS) * 0.5
                            : float(GRAIN_BLUR_RADIUS));

        for (int k = 0; k < 2; k++) {
            float sgn = (k == 0) ? 1.0 : -1.0;   // + sample, then antipodal -
            float s  = tile_bilinear(tile_center + o * sgn);
            float rd = s - Y_gamma;
            float a2 = rd < 0.0 ? asym_scale_sq : 1.0;
            float tw = max(0.0, 1.0 - weight_k * rd * rd * a2);
            float w  = tw * tw;
            blurred += s * w * boost;
            total_w += w * boost;
            gx += rd * w * r.x * gscale * sgn;
            gy += rd * w * r.y * gscale * sgn;
            grad_w += w;
        }
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
// The illumination field is blurred at sigma=20 on 1/4 res — grain is
// destroyed by the Gaussian. RGB kept through the blur chain for the V-aware
// pump driver (PASS 5 reads max(R,G,B) of the field); Y extracted downstream
// via dot product.

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
// PASS 4b-4d: PROPER MOTION SENSE (retained-prev-frame block-match) - EXPERIMENTAL
// =============================================================================
// A dedicated optical-flow front end (transport build) so the pump's motion
// guards can key on a TRUE image-motion field instead of the sigma80 band-pass
// proxy - which structurally cannot reach a broad lamp pan (the establishment
// guards need a feature to hold still; a persistently-moving lamp establishes
// nowhere). Three passes:
//   (1) MOTION_CUR  - 128x72 edge-preserving max-channel downsample of MAIN
//                     (8x8 px per 16x9 pump cell).
//   (2) MOTION_FLOW - 16x9 per-cell block-match of MOTION_CUR against the
//                     persistent CELFLARE_ADD_MOTION_PREV (last frame), +-MOT_R px search,
//                     truncated SAD. Output rg = previous-frame source offset
//                     in px, b = mean best SAD, a = current-tile RMS contrast.
//   (3) history     - MOTION_CUR -> CELFLARE_ADD_MOTION_PREV for next frame, AFTER (2) has
//                     read the old prev (file order == execution order).
// PASS 5 consumes MOTION_FLOW: the motion-compensated brightness residual
// |V_now - warp(V_prev, flow)| replaces the per-cell MASK freshness (established-
// level + dipole + LK) with ONE test - small residual = transport (suppress),
// large = emission (pump). SCOPE (paired audit): this governs the mask only; the
// SCALAR onset (loc_on_sum) still uses the legacy established gate. The A0
// subtractive apply uses this as a permissive suppressor. A2 additive adds a
// seven-frame local opening proof, transported persistence, and a second local
// flow route in the pump's own illumination field before the mask may create
// amplitude.
// Frame-0 safety: loop 2 (the MC consumer) is skipped while transient_reset is
// set, so uninitialized CELFLARE_ADD_MOTION_PREV / garbage flow never reaches persistent SSBO
// state - do NOT move the MC block or loop 2 out of that guard.
// Offline 16x9 cell-replica: pan mask-debit 0.87 (vs the affine gate's 0.50),
// growth 0.09 (event pumps). That single-flow result is only the A0 baseline;
// A2's pump-domain route and persistence contract own reveal safety.

//!HOOK MAIN
//!BIND HOOKED
//!SAVE MOTION_CUR
//!WIDTH 128
//!HEIGHT 72
//!DESC CelFlare: Motion analysis downsample
vec4 hook() {
    // Pump-aligned max-channel signature at 128x72 (8x8 px per 16x9 pump cell).
    // The pump driver is V-aware; Rec.709 luma can nearly hide a saturated blue
    // or red light that still drives the pump, making its motion guard blind to
    // the exact feature it must classify. Four corner taps retain modest box AA
    // so a moving edge matches without alias shimmer.
    vec2 o = vec2(0.5 / 128.0, 0.5 / 72.0);
    vec3 c0 = HOOKED_tex(HOOKED_pos + vec2(-o.x, -o.y)).rgb;
    vec3 c1 = HOOKED_tex(HOOKED_pos + vec2( o.x, -o.y)).rgb;
    vec3 c2 = HOOKED_tex(HOOKED_pos + vec2(-o.x,  o.y)).rgb;
    vec3 c3 = HOOKED_tex(HOOKED_pos + vec2( o.x,  o.y)).rgb;
#if cf_additive_pump && cf_spatial_pump
    float v = max(max(c0.r, c0.g), c0.b) + max(max(c1.r, c1.g), c1.b)
            + max(max(c2.r, c2.g), c2.b) + max(max(c3.r, c3.g), c3.b);
    return vec4(v * 0.25, 0.0, 0.0, 1.0);
#else
    // Preserve the A0 subtractive reference bit-for-bit when additive is off.
    const vec3 lc = vec3(0.2126, 0.7152, 0.0722);
    float y = dot(c0, lc) + dot(c1, lc) + dot(c2, lc) + dot(c3, lc);
    return vec4(y * 0.25, 0.0, 0.0, 1.0);
#endif
}

//!HOOK MAIN
//!BIND MOTION_CUR
//!BIND CELFLARE_ADD_MOTION_PREV
//!SAVE MOTION_FLOW
//!WIDTH 16
//!HEIGHT 9
//!COMPUTE 16 9
//!DESC CelFlare: Motion block-match
// One thread per 16x9 pump cell: select a truncated-SAD shift for its 8x8
// MOTION_CUR tile from CELFLARE_ADD_MOTION_PREV over +-MOT_R px. Default is approximate
// coarse-to-fine; the fallback is exhaustive. Output rg = selected (dx,dy) px
// (the PREV offset - where the content came from), b = its photometric SAD / 16
// (MOT_BIAS excluded), a = RMS contrast of the current 4x4 subsample.
// The old adjacent second-best was not valid uniqueness: adjacent candidates
// belong to the same SAD basin, and the final best can move after second is
// recorded. Tile subsampled 2x for cost. CELFLARE_ADD_MOTION_PREV is a STORAGE image
// (imageLoad, integer coords); MOTION_CUR is a normal texture.
#define MOT_R       5      // search radius px (@128x72 ~ +-42 px/frame at 1080p); keep the hard-coded coarse lattice below in sync
#define MOT_TILE    8
#define MOT_SADCAP  0.10   // per-sample truncated SAD (robust to a lone outlier texel)
#define MOT_SADCAP_BRIGHT 0.30 // bright max-channel features are pump evidence, not outliers
#define MOT_SAD_BRIGHT_LO 0.35
#define MOT_SAD_BRIGHT_HI 0.70
#define MOT_BIAS    0.0005 // zero-motion prior: tiny per-shift penalty so a FLAT tile (all shifts equal cost) resolves to (0,0), not the search's first corner (paired audit) — far below any real match difference, so genuine motion is unaffected
#define MOT_TAPS    16
#define MOT_CELLS   144
#define MOT_COARSE_SEARCH 1 // 1 = 5x5 even-offset search + 3x3 refine (<=33 candidates); 0 = exhaustive 11x11 A/B
// One workgroup covers the complete 16x9 flow output. Each lane owns one cell
// and publishes its 16 current-frame taps once; every candidate reuses them.
// Sample-major layout keeps adjacent lanes contiguous for each tap
// (tap*MOT_CELLS + lane), avoiding the strided access of lane-major
// [lane][tap]. Footprint = 2304 floats = 9 KiB.
shared float s_motion_cur[MOT_TAPS * MOT_CELLS];

ivec2 motion_tap_offset(int t) {
    int tx = t & 3, ty = t >> 2;
#if cf_additive_pump && cf_spatial_pump
    // Same 16 samples and same tile coverage as the old even/even lattice, but
    // alternate parity by row/column. This removes period-4 sampling nulls
    // without adding a fetch.
    return ivec2((tx << 1) + (ty & 1), (ty << 1) + (tx & 1));
#else
    return ivec2(tx << 1, ty << 1);
#endif
}

float motion_sad(ivec2 org, ivec2 d, int lane) {
    const ivec2 dims = ivec2(128, 72);
    float sad = 0.0;
    for (int t = 0; t < MOT_TAPS; t++) {
        ivec2 o = motion_tap_offset(t);
        ivec2 p = org + o + d;
        float cv = s_motion_cur[t * MOT_CELLS + lane];
        bool inb = p.x >= 0 && p.y >= 0 && p.x < dims.x && p.y < dims.y;
        float pv = inb ? imageLoad(CELFLARE_ADD_MOTION_PREV, p).r : cv;
#if cf_additive_pump && cf_spatial_pump
        float bright = smoothstep(MOT_SAD_BRIGHT_LO, MOT_SAD_BRIGHT_HI,
                                  max(cv, pv));
        float cap = mix(MOT_SADCAP, MOT_SADCAP_BRIGHT, bright);
#else
        float cap = MOT_SADCAP;
#endif
        float delta = inb ? abs(cv - pv) : cap;
        sad += min(delta, cap);
    }
    return sad;
}

void hook() {
    // WIDTH/HEIGHT exactly match COMPUTE, so the dispatch is one 16x9 group and
    // every lane must reach the barrier below. Do not add an early-return guard.
    ivec2 cell = ivec2(gl_LocalInvocationID.xy);
    int lane = int(gl_LocalInvocationIndex);
    ivec2 org = cell * ivec2(MOT_TILE, MOT_TILE);

    // MOTION_CUR stays at 16 explicit fetches per cell. The coarse search cuts
    // previous-frame imageLoads from 121*16 = 1936 to at most 33*16 = 528;
    // hardware texture caching means neither count is a direct DRAM estimate.
    // The exhaustive fallback retains the original 121-candidate search.
    for (int t = 0; t < MOT_TAPS; t++) {
        ivec2 o = motion_tap_offset(t);
        ivec2 c = org + o;
        s_motion_cur[t * MOT_CELLS + lane] =
            MOTION_CUR_tex((vec2(c) + 0.5) * MOTION_CUR_pt).r;
    }
    barrier();

    float best_rank = 1e9, best_sad = 1e9;
    ivec2 bestd = ivec2(0);
#if MOT_COARSE_SEARCH
    // Span -4..4 with a 5x5 even-offset search, then refine the winning basin
    // by one pixel. The refine can reach +-5 when an edge basin wins. Periodic
    // or repeated texture can select the wrong coarse basin; the exhaustive
    // A/B fallback remains the reference. 25 + at most 8 candidates replaces
    // 121 while preserving integer-pixel output.
    for (int cy = -2; cy <= 2; cy++) {
        for (int cx = -2; cx <= 2; cx++) {
            ivec2 d = ivec2(cx, cy) * 2;
            float sad = motion_sad(org, d, lane);
            float rank_cost = sad + MOT_BIAS * float(abs(d.x) + abs(d.y));
            if (rank_cost < best_rank) {
                best_rank = rank_cost;
                best_sad = sad;
                bestd = d;
            }
        }
    }
    ivec2 coarse_best = bestd;
    for (int ry = -1; ry <= 1; ry++) {
        for (int rx = -1; rx <= 1; rx++) {
            if (rx == 0 && ry == 0) continue;
            ivec2 d = coarse_best + ivec2(rx, ry);
            if (abs(d.x) > MOT_R || abs(d.y) > MOT_R) continue;
            float sad = motion_sad(org, d, lane);
            float rank_cost = sad + MOT_BIAS * float(abs(d.x) + abs(d.y));
            if (rank_cost < best_rank) {
                best_rank = rank_cost;
                best_sad = sad;
                bestd = d;
            }
        }
    }
#else
    for (int dy = -MOT_R; dy <= MOT_R; dy++) {
        for (int dx = -MOT_R; dx <= MOT_R; dx++) {
            float sad = motion_sad(org, ivec2(dx, dy), lane);
            float rank_cost = sad + MOT_BIAS * float(abs(dx) + abs(dy));
            if (rank_cost < best_rank) {
                best_rank = rank_cost;
                best_sad = sad;
                bestd = ivec2(dx, dy);
            }
        }
    }
#endif
    vec2 best_flow = vec2(bestd);
#if cf_additive_pump && cf_spatial_pump
    // Additive A2 needs subpixel flow because the 16x9 V history is sensitive
    // to integer-vector quantization at slow pans. Four cardinal SAD probes fit
    // a bounded quadratic inside the winning basin. This adds 64 previous-frame
    // reads per cell only in additive mode (<=592 vs <=528 on the A0 path).
    if (best_sad > 1e-7 && abs(bestd.x) < MOT_R) {
        float cm = motion_sad(org, bestd + ivec2(-1, 0), lane);
        float cp = motion_sad(org, bestd + ivec2( 1, 0), lane);
        float den = cm - 2.0 * best_sad + cp;
        if (den > 1e-6)
            best_flow.x += clamp(0.5 * (cm - cp) / den, -0.5, 0.5);
    }
    if (best_sad > 1e-7 && abs(bestd.y) < MOT_R) {
        float cm = motion_sad(org, bestd + ivec2(0, -1), lane);
        float cp = motion_sad(org, bestd + ivec2(0,  1), lane);
        float den = cm - 2.0 * best_sad + cp;
        if (den > 1e-6)
            best_flow.y += clamp(0.5 * (cm - cp) / den, -0.5, 0.5);
    }
#endif
    // Keep these metadata accumulators OUT of the large unrolled search's live
    // range. That matters on FXC, especially under the 1936-sample exhaustive
    // fallback; the coarse path evaluates at most 528 samples.
    float tsum = 0.0, tsum2 = 0.0;
    for (int t = 0; t < MOT_TAPS; t++) {
        float cv = s_motion_cur[t * MOT_CELLS + lane];
        tsum += cv;
        tsum2 += cv * cv;
    }
    float tmean = tsum * (1.0 / 16.0);
    float texture_rms = sqrt(max(tsum2 * (1.0 / 16.0) - tmean * tmean, 0.0));
    float best_mean = best_sad * (1.0 / 16.0);
    imageStore(out_image, cell, vec4(best_flow, best_mean, texture_rms));
}

//!HOOK MAIN
//!BIND MOTION_CUR
//!BIND CELFLARE_ADD_MOTION_PREV
//!SAVE MOTION_HIST
//!WIDTH 128
//!HEIGHT 72
//!COMPUTE 16 16
//!DESC CelFlare: Motion history store
// Copy this frame's MOTION_CUR into persistent CELFLARE_ADD_MOTION_PREV for next frame.
// Runs AFTER block-match has read the previous CELFLARE_ADD_MOTION_PREV (file order
// == execution order). The SAVE (MOTION_HIST) is a required dummy dispatch
// target; the real product is the imageStore into CELFLARE_ADD_MOTION_PREV.
void hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= 128 || p.y >= 72) return;
    float y = MOTION_CUR_tex((vec2(p) + 0.5) * MOTION_CUR_pt).r;
    imageStore(CELFLARE_ADD_MOTION_PREV, p, vec4(y, 0.0, 0.0, 1.0));
    imageStore(out_image, p, vec4(y, 0.0, 0.0, 1.0));
}

// =============================================================================
// PASS 5: FRAME STATS (compute, 144-thread parallel reduction)
// =============================================================================
// Samples illumination field on a 16×9 grid. Computes frame-level metrics that
// modulate the expansion curve: average illumination, bright fraction (replaces
// the entire 7-type scene classifier from v3.2), and scene cut detection.
//
// Compute layout: COMPUTE 16 9 dispatches one workgroup of 144 threads
// against the 1×1 output. Each thread owns one grid cell — sampling and the
// per-cell scene-cut delta happen in parallel (each lane also writes its own
// prev_illum slot). Thread 0 performs the final 144-element reduction (tier
// counts derived there from the raw per-lane values) and owns every other
// CELFLARE_ADD_STATE write, including the per-cell pump lanes — single writer, one
// barrier total in this pass.
//
// Scene cut: the previous prev_illum[16] separate 4×4 grid was
// redundant — the 16×9 stats grid already covers the frame at higher density.
// prev_illum was widened to [144] so each lane stores its own slot (no
// cross-lane access -> race-free); change_pct counts over all 144 cells but
// is normalized to the picture-area cells (see the letterbox block).
// SCENE_CUT_PCT semantics are preserved: fraction of cells whose illum moved
// by > ILLUM_CHANGE_THRESH. Spatial density is higher (1/16-w × 1/9-h vs
// 1/4 × 1/4), threshold tuning is unchanged.

//!HOOK MAIN
//!BIND HOOKED
//!BIND CELFLARE_ADD_STATE
//!BIND CELFLARE_ILLUM
//!BIND MOTION_FLOW
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

// LETTERBOX / PILLARBOX BAR EXCLUSION (v5.7). Bar cells used to poison the
// scene statistics: their near-zero illum pinned both contrast extrema at
// ~4-6 stops permanently (the pump's fade-to-white cover gate read 1.0
// forever, DYN_INTENSITY sat at max on all letterboxed content), diluted
// every tier fraction by the bar area, and shrank the scene-cut change_pct
// ceiling (windowboxed content — 4:3 + 2.35 inside 16:9, 60 live cells —
// could NEVER reach SCENE_CUT_PCT 0.50, so cuts went undetected and the
// pump lanes never reset on them). Detection is GEOMETRIC + PERSISTENT,
// deliberately NOT luminance-based: a candidate row (0,1,7,8) or column
// (0,1,14,15) counts as a bar only while EVERY cell in it is invalid
// (source Y <= 0.001) for LB_ENGAGE_FRAMES consecutive frames. Center
// cells are never candidates, so a night scene's true blacks can never be
// misclassified (a fire against black surround keeps its dark cells in the
// cover-gate contrast — a generic "exclude dark cells" rule would mute the
// flagship dark-scene fire). Any content appearing in a bar (aspect change)
// resets its run INSTANTLY. Candidate rows/cols cover 2.35:1 through
// 2.76:1 letterbox and 4:3 pillarbox at cell resolution. Engaged bar cells
// drop out of: both contrast extrema, the four tier sums AND their
// denominator (fractions read as calibrated on the picture area), and the
// scene-cut denominator. The pump p-NORM deliberately stays a mean over
// ALL 144 cells: bars dilute it only ~(144/n_eff)^(1/4) ≈ 3-6%, and keeping
// it fixed avoids a one-time step in the DRIVE path at the engage frame (a
// step in the fracs is EMA-absorbed; a drive step could cross the pump
// onset once). Residuals: dirty bars (level > ~17) never engage
// (conservative, old behavior); hardsubs burned into a bar keep that bar
// un-engaged (the other one still engages); a >5s static shot whose ONLY
// true-blacks form complete edge rows loses them from the extrema —
// harmless unless no other dark cell exists. EMAs make the one-time engage
// step ease in. Per-rendered-frame like every other counter (60fps engages
// ~2.5x sooner — same direction, fine).
#define LB_ENGAGE_FRAMES    120.0   // ~5s @24p of all-black before a row/col is a bar

// Specular detection (source brightness tiers, no illum field)
#define HIGHLIGHT_THRESH    0.75    // Source highlight tier
#define SPECULAR_THRESH     0.92    // Source specular tier
// Top-band tier for the shaped-not-dampened reshape (PASS 6). The reshape's
// analytic payoff zone starts at the body/top crossover Y~0.82: a bright-key
// scene whose content tops out BELOW that (golden-hour faces at Y~0.75-0.8)
// would pay the body-hold with zero separation payoff. This tier measures
// top-band PRESENCE — deliberately between HIGHLIGHT (0.75, fires on the
// no-payoff class itself) and SPECULAR (0.92, misses soft near-clip skies) —
// so the reshape can key on "does this frame actually have a top band"
// rather than scene mean. Raw fraction, NO shutoff/tier gating on purpose:
// unlike the spec-bonus signals, broad near-white fields are exactly where
// the reshape should stay OPEN (that is the flat-bright-anime target class).
#define TOP_BAND_THRESH     0.85
// SOFT tier membership (v5.6 hardening). The tier counters rest on 144
// SINGLE-TEXEL samples; with hard compares a pan/tilt over textured content
// (sparkle clusters, glinting water) slides samples across the thresholds and
// the fracs jitter in 1/144 quanta — right through the shutoff/tier-ratio
// fade bands (clusters sit at spec_frac 0.10-0.16 = exactly the shutoff
// band), which read as scene-wide specular FLICKER. Each intensity compare is
// now a smoothstep over ±this half-band, so a sample sliding across a tier
// contributes continuously instead of popping a whole 1/144 step, and the
// fracs become continuous (also retires the "spec_onset is functionally
// boolean at 1/144" §11 note structurally). Samples farther than the band
// from every threshold count exactly as before — calibration drift exists
// only where the old counts were unstable anyway. Keep > 0: smoothstep with
// equal edges is UNDEFINED in GLSL, so 0.0 is not a valid "hard compare"
// fallback — A/B toward hard behavior with 0.005, or git-revert.
#define TIER_SOFT_HALFBAND  0.02
// (The bright-spec SAT fence below stays a hard compare — near-white
// speculars sit far from 0.30, and softening it would re-tune the pink-carpet
// fence; revisit only if sat-edge flicker is ever observed.)
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
// on anime even though the spatial curve alone was on target. Tightened
// again in v5.15 (0.15/0.40 → 0.10/0.28) for the same class one notch
// further out: sun-facing shots (sun disc + halo + glare on water/faces,
// bs_frac ~0.12-0.30) kept a broad recovery lift that flattened their
// remaining depth. Sparse glints (bs_frac ≤ ~0.08) are unaffected and get
// the v5.15 cf_spec 0.5→0.6 strength raise instead — fewer pixels qualify,
// the ones that do pop slightly more.
#define BRIGHT_SPEC_FRAC_MAX  0.10  // Recovery starts fading at 10% of cells > 0.97
#define BRIGHT_SPEC_FRAC_CEIL 0.28  // Recovery fully off at 28% of cells > 0.97
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
#define BRIGHT_SPEC_SAT_MAX   0.30  // bright-spec tier counts only cells with sat below this

// Saturated-channel spec detection. Count pixels using V = max(R,G,B) rather
// than Y so pure saturated primaries (red LED Y=0.21 but V=1.0) qualify as
// specular tier. V >= Y always, so neutrals behave identically. Scene-level
// gating (tier_ratio, shutoff) still suppresses red-dominated scenes.
// PASS 5-ONLY since 2026-07-02: the PASS 6 per-pixel V escape this used to
// pair with was field-rejected (crushed saturated speculars) and deleted —
// see the PASS 6 spec block. Detection stays on V because every scene-gate
// tuning since f453fc4 was validated against V counting.
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
#define PUMP_ALPHA_FAST     0.18   // fast lane (~5 frame time constant). Sets ATTACK speed — lower = gentler ramp. Tuned 0.12→0.18 2026-07 (onset knee; lands the pump ~4fr earlier, amplitude-neutral; validated day/night/laser/fire)
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
// annihilated. pump_gate takes max(drive, drive_on) — drive_on is drive_loc
// with rises passed through the ESTABLISHED-LEVEL GATE (see the
// PUMP_ESTABLISH_MARGIN block): a rise merely converging to the level its
// neighbourhood already holds (occluder retreat / reveal) contributes
// nothing; falls always count, so a moving light self-cancels. RELEASE: a purely-
// local event never moves the global lanes, so without a local release its
// env would linger ~29s behind closed masks and any later mask opening
// (incl. an occluder wake) would inherit stale amplitude. When drive_loc<0
// (net local fall) the env releases at the FALLING CELLS' OWN frame-to-frame
// fast-lane ratio (fast_new/fast_prev, |d|^p-weighted) — the same frame-to-
// frame, non-recompounding source-change semantics as the global rel and the
// mask's r, dimensionally on the LOCAL axis. (Do NOT divide drive_loc by the
// frame level instead: a cell-delta
// over an absolute frame level guillotines the env in one frame on dark
// frames — audited Rule-2 violation.) A held light has d≈0 → no release;
// a balanced crossing has drive_loc≈0 → no release (conservative hold).
// TRADE: sign cancellation fails on NET-ASYMMETRIC transitions — a large
// dark occluder EXITING frame (reveal with no paired cover), a bright
// object ENTERING (window on a pan). Those fire at a lower size threshold
// than before. Not a new artifact class (photometrically = tunnel exit, a
// designed target: the frame genuinely brightens and the pump eases in and
// out) — judge on real content. If it reads wrong, lower P toward 2
// (drive_loc needs bigger events). p=1 collapses the p-statistics to means,
// but the ESTABLISHED-LEVEL GATE still removes non-fresh rises from the
// onset sum at ANY p — the exact "p=1 == v5.2 frame mean" identity now
// holds only for the RELEASE path (ungated). The mask still can't ADD
// under any p.
#define PUMP_DRIVE_P        4.0    // highlight weighting of BOTH drive statistics. A/B 2.0-6.0
// Spatial pump: per-cell band-pass on the COARSE σ80 CELFLARE_ILLUM V (sampled at
// the cell center — the SAME field the scalar means over the frame) → a [0,1]
// per-cell "is this region brightening" env. How PASS 6 consumes it is that pass's
// SPATIAL_PUMP_ADDITIVE knob (additive = env is the local pump amplitude;
// subtractive = env only suppresses the global scalar). 0 = scalar-only.
// Single-sourced from the top-of-file cf_spatial_pump toggle (a file-scoped
// param injected into every pass). PASS 6 aliases the SAME param, so the old
// PASS5/PASS6 duplicate-define desync (compiled fine, read a garbage mask) is
// structurally impossible now. Don't replace either alias with a literal.
#define ENABLE_SPATIAL_PUMP cf_spatial_pump
// drive_loc rectification/noise floor ONLY (§10): shrinks |d| symmetrically
// before the signed p-mean so idle wobble contributes nothing. It no longer
// touches the mask onset — the old smoothstep(LOW, HIGH, max(0, d - DEADZONE))
// was a pure edge shift, folded into PUMP_CELL_DRIVE_LOW/HIGH below (+0.01
// each), so this knob and the mask thresholds now tune independently.
#define PUMP_CELL_DEADZONE  0.01
// Per-cell drive thresholds (independent of the scalar's PUMP_DRIVE_LOW/HIGH). The
// cell driver is the σ80 V AT the cell vs the scalar's frame p-norm, so a
// LOCALIZED event swings its own cell far more than it moves any frame stat → the
// per-cell drive for a real regional event is comparable-to-larger than the
// scalar's. LOW sits a touch above the scalar onset to reject σ80 neighbour-bleed
// (a rising event bleeding ~half-amplitude into an adjacent cell) and idle wobble;
// HIGH is where a genuine in-cell brightening saturates. Retuned for σ80 (the old
// 0.02/0.18 were for the deleted sharp MID driver). Values absorb the former
// 0.01 dead-zone shift (0.04/0.14 + 0.01 — same mask response, deadzone-free).
// A/B LOW 0.04-0.06, HIGH 0.11-0.17.
#define PUMP_CELL_DRIVE_LOW  0.05   // per-cell band-pass onset (mask starts opening — region is brightening)
#define PUMP_CELL_DRIVE_HIGH 0.15   // per-cell saturation (mask fully open — region clearly brightening)
// ESTABLISHED-LEVEL GATE (2026-07-02, from dissecting the killer reveal clip —
// a BD anime OP, black silhouette shrinking/drifting over a bright background;
// the scene that originally sank the additive spatial pump).
// A rising cell counts as FRESH LIGHT only once its fast lane exceeds the
// highest ESTABLISHED (slow-lane) level among its ring-2 neighbours by this
// margin. Physical meaning: a rise CONVERGING to a level the neighbourhood
// already holds is brightness EXTENDING — an occluder retreating / pan /
// shrinking silhouette re-exposing background — not a light event; only a
// rise EXCEEDING the local established ceiling is new light. Non-fresh rises
// contribute NOTHING to the drive_loc ONSET sum (falls still count, so a
// translating light self-cancels), and (PUMP_MASK_ESTABLISH) may not OPEN
// the cell mask. RELEASE keeps the UNGATED sum — balanced crossings still
// cancel (no false release), dying fires still release.
// Offline-validated on the killer clip (16x9 cell replica of this pass):
// drive_loc onset 0.15→0.000 across the 3 s reveal (was env 0.80 for ~70
// frames = the artifact), mask-open cell-frames 3016→75 (frame-edge cells —
// since closed by the anchored window in the freshness scan below).
// Target events preserved: tunnel exit (uniform rise: slow lags fast
// everywhere → nothing is under a neighbour's ceiling; global drive owns it
// regardless), dark-scene fires (neighbourhood slow is dark), SPREADING
// fires (ignition outruns the 25-frame slow-lane establishment), and the
// σ80 bleed halo around a fire stops opening its own mask (tighter
// localization). Measured trade: a fire igniting within 2 cells of an
// EQUALLY-bright standing region loses drive_loc onset until it exceeds
// that region's level (synthetic: 0.232→0.019) — bounded under-pump.
// MARGIN: 0.02 leaks on the killer clip; 0.03-0.05 fully suppress. Raise
// toward 0.05 if any reveal still breathes; the ring is fixed at 2 (matches
// the σ80 edge smear — a wider ring only widens the standing-mass trade).
#define PUMP_ESTABLISH_MARGIN 0.03
// Mask half of the gate: 1 = a cell mask may only OPEN on a fresh rise (held
// env still max-holds and releases normally — closing is never gated). 0 =
// mask opens on any rise; the scalar onset gate above stays active either way.
// Under PASS 6's SPATIAL_PUMP_ADDITIVE this is the load-bearing reveal
// safety, not an A/B lever — an ungated additive mask is exactly the pre-v5.2
// artifact machine. Keep it 1 while the additive experiment is on.
#define PUMP_MASK_ESTABLISH 1
// TRANSLATION SUPPRESSOR (2026-07-12) — ADDITIVE-MODE motion guard (active when
// SPATIAL_PUMP_ADDITIVE 1; near-inert when the subtractive fallback is selected,
// where the scalar already denies motion pump). The MOTION reveal-safety the
// established-level gate lacks. A bright feature TRANSLATING across the grid
// under a camera pan/tilt (a facial specular, a lamp) reads FRESH in each new
// cell (out-brightens its stale-established neighbours), so the additive mask
// lights it up (Judas Overlord E05 field reports). A translation has a rising
// LEADING edge and a falling TRAILING edge; a real localized event (candle,
// spell, growing fire/explosion) rises with static-or-rising neighbours — no
// matched trailing fall. So: a rising cell whose ring-1 neighbourhood holds a
// falling cell of COMPARABLE magnitude (a rigid translation conserves local
// brightness: |fall|≈rise — the per-cell analog of the scalar's signed-p-sum
// cancellation) is a translation leading edge → debit its mask OPENING. The
// magnitude match is load-bearing (paired design audit, 2026-07-12): the bare
// "any falling neighbour" form suppresses the flagship — a dying multi-fire
// cell kills an adjacent flare, and an expanding bloom's just-peaked interior
// falls behind its rising rim. Match + ring-1 keep those (unmatched: flare rise
// ≫ incidental fall; independent fires ≥2 cells apart). ONE-FRAME, no per-cell
// EMA (an EMA is backwards here — a pan makes a cell a leading edge for ~1
// frame so it never "sustains", while a stationary flickering fire trips it
// every frame → EMA would suppress the fire harder than the pan). Debits only
// fresh_ease∈[0,1] on the OPENING term → pure onset under-pump, rules 0-3 intact
// (can't flatten a gradient or squash an open pump). Residual (directionality
// deferred): two ring-1-adjacent fires with a flare/death of matched magnitude
// read as a translation dipole — validate Symphogear G S02E04 OP first.
// 0 = off (bit-exact prior behaviour).
#define PUMP_TRANS_SUPPRESS   1
#define PUMP_TRANS_RING       1      // fall-scan half-width, documented range {1,2} (the anchored-window clamp below hardcodes 9−TW/16−TW → widening past 4 needs those bounds updated). 1 (3×3) measured best: catches the sharp in-frame facial-specular translation; ring-2 both dilutes the face debit (finds a mismatched further faller) AND barely helps the broad bar-lamp glow (a REVEAL from off-frame has no in-frame trailing fall — that is the border-seed's domain, not this test's).
#define PUMP_TRANS_FALL_LO    0.02   // neighbour fall (illum-V) below: no debit (idle-wobble floor, > deadzone)
#define PUMP_TRANS_FALL_HI    0.08   // neighbour fall above: full trailing-edge weight (magnitude-match then scales it)
#define PUMP_TRANS_MATCH      0.5    // balance floor = min(rise,|fall|)/max(rise,|fall|); below → unmatched (bloom/flare rise≫fall, or independent dying neighbour fall≫rise) → no debit; →1 = a conserved translation dipole → full debit
// TRANSPORT-RESIDUAL MOTION GATE (2026-07-12) — the GLOBAL-motion analog of the
// dipole suppressor above, and its complement. The dipole catches in-frame
// LOCAL object motion (a facial specular tilt — its trailing fall is on-screen,
// ring-1-close), but by its own PUMP_TRANS_RING note it "barely helps the broad
// bar-lamp glow" of a camera PAN, whose matching fall is off-frame or cells
// away. That global-pan false-positive (celflare-pump-motion-falsepos) is the
// residual this gate closes — FIRST PRINCIPLES, not another proxy. Brightness
// constancy: TRANSPORTED light satisfies dI/dt = ∇I·v (a temporal change is a
// spatial gradient displaced by the image velocity). So fit the single global v
// that best explains every cell's temporal band-pass di from its spatial
// gradient ∇I (Lucas–Kanade, PASS 5 reducer), then debit a rising cell's mask
// OPENING by the fraction of its rise that v reproduces. The separation is
// STRUCTURAL: a translating blob has di of both signs across it (leading +,
// trailing −) matching ∇I's sign flip under ONE v → explained → debited; an
// EMITTING/GROWING blob has di>0 on both edges where ∇I flips sign → NO single v
// explains it → residual survives → it pumps, even under a simultaneous pan
// (the event's di is not the background's ∇I·v). This is the motion-compensated
// inter-frame residual a codec keys on, localized to the pump grid. Debit is
// ONSET-ONLY (fresh_ease), so rules 0-3 hold exactly as for the dipole (can't
// flatten a gradient or squash an already-open pump). Runs independently of
// PUMP_TRANS_SUPPRESS — both multiply fresh_ease; A/B each alone or together.
// CONSERVATION GATE (the piece a single-v fit lacks): one global v ALSO fits a
// PROPAGATING EMISSION FRONT — a sweeping spell, an advancing fire wall, a light
// wipe — because its leading edge is kinematically a translating bright edge, so
// texp→1 and the pump onset would be wrongly killed (paired design audit,
// 2026-07-12; offline sim: an unguarded front debited 0.51, as much as a real
// pan). The separator the fit misses: transport CONSERVES brightness (a bounded
// feature — every leading rise has a matched trailing fall → the scalar's signed
// p-sum loc_sum ≈ 0), while emission GENERATES it (a front has no trailing fall
// → bright-mass grows → loc_sum > 0). So the debit fades out as the net signed
// drive goes positive: FULL when loc_sum≈0 (a conserved pan — the target), OFF
// once a real event is present. Transport-suppressor and emission-detector thus
// partition the signed-drive axis. This also stops a growing+translating hero
// object (a flying fireball) from tripping BOTH this and the dipole — its growth
// makes loc_sum>0, so this gate steps aside (no multiplicative over-debit).
// 0 = off (bit-exact prior behaviour).
#define PUMP_TRANSPORT_RESIDUAL   1
#define PUMP_TRANSPORT_STRENGTH   1.0    // max mask-opening debit for fully-explained transport (0 = inert, 1 = full)
#define PUMP_TRANSPORT_VLO        0.10   // |v| below (σ80-field units/band-pass): no coherent motion → gate inert (a uniform fade/growth has di≫0 but v≈0). Offline 16×9 cell-replica: a rigid pan measures |v|≈1.5, pure growth |v|=0.000 — these bracket that, off the noise floor
#define PUMP_TRANSPORT_VHI        0.50   // |v| above: full motion confidence
#define PUMP_TRANSPORT_R2_MIN     0.18   // coherence FLOOR (raised 0.12→0.18, audit finding #2: above the ~0.14 pan+emission / multiplane-parallax collapse so an incoherent fit can't nibble a real event). Fraction of the field's temporal energy the single global v must explain to engage. A GATE (smoothstep over [R2_MIN, R2_MIN+0.15]), not a linear discount — a clean rigid pan measures R²≈0.35 (band-pass di caps it well below 1), so it clears the floor full-strength while parallax/pan+event (R²≈0.14) is blocked
#define PUMP_TRANSPORT_EMIT_LO    0.02   // CONSERVATION GATE (see block above): net drive_loc (signed p-sum; a conserved pan ≈0) below this → brightness conserved → transport gate FULL
#define PUMP_TRANSPORT_EMIT_HI    0.08   // net drive_loc above this → emission present (front/event, bright-mass growing) → gate OFF. A small localized fire measures drive_loc~0.1 > this (protected); a conserved pan ≈0 (suppressed)
#define PUMP_TRANSPORT_RIDGE      0.05   // Tikhonov ridge as a fraction of gradient energy — aperture-problem stabilizer (collinear ∇I → rank-1 normal matrix)
#define PUMP_TRANSPORT_ROBUST     1      // IRLS reweight iterations (0 = single gradient-weighted fit; 1 = one emission-outlier rejection pass so an event can't bias the global v)
#define PUMP_TRANSPORT_ROBUST_C   4.0    // robust residual scale² = C·mean(di²); larger = softer outlier rejection
// === UNIFIED MOTION-COMPENSATED RESIDUAL GATE (approach A, 2026-07-12) ===
// Replaces the mask freshness — the established-level gate + ring-1 dipole + LK
// transport fit (all still computed above but their fresh_ease is OVERRIDDEN in
// loop 2) — with ONE test on the real block-match flow (MOTION_FLOW, PASS 4b-d):
// the mask opens only where the current cell V exceeds the PREVIOUS frame warped
// by the flow, i.e. NEW light the motion cannot explain. A panning lamp warps
// its own prior level into the cell -> residual ~0 -> stays shut (this IS the
// motion-compensated established level — the level travels WITH the lamp). An
// off-frame warp (content entering from beyond the border) presumes influx ->
// shut (subsumes edge-establish). Offline block-match cell-replica: pan mask-
// debit 0.87, growth 0.09 (event pumps). A wrong flow yields a large residual,
// which is permissive in A0; the additive branch below therefore does not use
// this one-frame residual as opening authority.
// 0 = legacy freshness path (established-level + dipole + LK), for A/B.
#define MC_RESIDUAL_GATE  1
#define MC_RES_LO         0.02   // motion-compensated residual (new light) below this: transport -> mask shut
#define MC_RES_HI         0.08   // above: clearly new light -> mask opens
#define SPATIAL_PUMP_ADDITIVE cf_additive_pump
#define ADDITIVE_OPEN_GUARD  (SPATIAL_PUMP_ADDITIVE * MC_RESIDUAL_GATE * ENABLE_SPATIAL_PUMP)
#define ADDITIVE_STATE_EPOCH (3 * ADDITIVE_OPEN_GUARD)
#if cf_additive_pump && !MC_RESIDUAL_GATE
#error Additive apply requires the hardened motion-residual opener
#endif
// Additive A2 opening proof. Established fast-vs-neighbour memory still owns
// WHERE an opening may happen. Motion then asks what fraction of the same-cell
// rise survives a cubic sample at the refined previous offset: transport loses
// most of its fast-lane rise; emission retains it. Using the pump's fast lane
// instead of one-frame raw V lets a slow/held source keep proving itself while
// its EMA catches up. The proof is deliberately seven
// COMPLETE routed frames (2-6-frame flow-error bursts stay shut) and its credit
// follows a moving event through the grid. No frame-global fast lane exists.
#define ADD_VSLOW_ALPHA          0.01
#define ADD_RATIO_RAW_FLOOR      0.001
#define ADD_RATIO_LO             0.20
#define ADD_RATIO_HI             0.55
#define ADD_PERSIST_BASE         7.0
#define ADD_PERSIST_ROUTE_MIN    0.50
#define ADD_MAINT_FULL           8.0
#define ADD_MAINT_STEP           (1.0 / 48.0)
#define ADD_MAINT_EXPIRE         (ADD_PERSIST_BASE + 0.5 * ADD_MAINT_STEP)
#define ADD_MAINT_TRANSPORT_MIN  0.875
#define ADD_MAINT_ENV_LO         0.02
#define ADD_MAINT_ENV_HI         0.10
#define ADD_ATTACK_STEP          0.25
#define ADD_VFLOW_COST_MAX       0.03
#define ADD_VFLOW_SAD_CAP        0.10
#define ADD_VFLOW_BIAS           0.0001
// Raw 128x72 flow owns the primary route. A second additive-only matcher works
// directly on the already-resident 16x9 illumination-V history: the raw source
// can live in one compact analysis tile while its sigma80 pump tail opens the
// next cell, so no amount of same-tile ambiguity bookkeeping recovers that
// identity. A demeaned 3x3 shape SAD selects local translation without treating
// a level change as texture; the selected warp is still applied to ABSOLUTE
// frozen fast history, leaving actual amplitude growth as the opening residual.
// Proof credit follows ONLY the raw primary trajectory. Both motion routes are
// veto-only and cannot lend mature state or frame-global event permission.
// Research A/B for coherent-motion hardening of the SUBTRACTIVE MC mask. The
// permissive local residual remains the opening authority; supported frame
// motion may only DEBIT it by re-testing at the dominant prev-offset. Disabled:
// a deterministic pan + independent ignition control found no further debit on
// the already-correct transported lamp, but did debit the real ignition. Debug
// 10/11 retain the evidence/borrowed-residual views for future experiments.
// No pass, texture, or persistent bandwidth is added while this stays 0.
#define MOTION_COHERENT_VETO     0
// Motion observability / history guards. The cost and texture thresholds let
// evidence-bearing tiles vote a discontinuity without flat tiles declaring a
// false zero-flow match. cf_debug=10/11 expose the underlying evidence/trust.
// The reset is deliberately frame-global and conservative: an evidence-majority
// bad match with at least 10% qualified coverage re-pins all pump lanes rather
// than letting stale motion history manufacture a transient. Sampled pan/reveal/
// fireball checks leave it shut; mismatched cuts open it. A false reset cannot
// add gain, but can discard a held event's onset by re-pinning at its new level,
// so the high bad-fraction knee and debug monitoring are load-bearing.
// MOTION_STATE_EPOCH is an exactly-representable schema token, not a reload or
// seek detector. Bump it whenever MOTION_FLOW format/resolution/sign semantics
// change. Same-schema reload/seek continuity is still not guaranteed: the cost
// reset catches broad textured discontinuities, but no cut detector is universal.
#define MOTION_STATE_EPOCH       51705.0
#define MOTION_COST_RESET        1
#define MOTION_COST_GOOD         0.025   // mean winning SAD: confidently matched below
#define MOTION_COST_BAD          0.075   // mean winning SAD: suspect above
#define MOTION_TEXTURE_LO        0.004   // tile RMS contrast: flat below
#define MOTION_TEXTURE_HI        0.025   // tile RMS contrast: reliable evidence above
#define MOTION_BAD_FRAC_RESET    0.65    // texture-qualified bad-match fraction
#define MOTION_BAD_COVER_MIN     0.10    // qualified evidence mass / 144 required to reset
#define MOTION_DOM_COVER_LO      0.05    // reliable evidence coverage: trust stays off below
#define MOTION_DOM_COVER_HI      0.20    // enough frame support for full coherent trust
#define MOTION_DOM_SPREAD_LO     0.10    // min normalized x/y evidence stddev: clustered below
#define MOTION_DOM_SPREAD_HI     0.22    // evidence distributed across the frame above
#define MOTION_DOM_MAG_LO        0.35    // dominant prev-offset px/frame: static below
#define MOTION_DOM_MAG_HI        1.25    // coherent motion fully armed above
#define MOTION_DOM_AGREE_LO      0.75    // local-vs-dominant distance px: agrees below
#define MOTION_DOM_AGREE_HI      1.75    // disagrees above
#define MOTION_DEBUG_VIEWS       (cf_debug == 10 || cf_debug == 11)
#define MOTION_COHERENT_ROUTE     (MOTION_COHERENT_VETO && MC_RESIDUAL_GATE && ENABLE_SPATIAL_PUMP)
#define MOTION_FRAME_ANALYSIS     (MOTION_COHERENT_ROUTE || MOTION_DEBUG_VIEWS)
// Legacy LK residual notes (MC_RESIDUAL_GATE=0 only; not additive safety claims):
//  · ~50% pan-debit ceiling — band-pass di is an imperfect dI/dt AND a single
//    global translation misfits a curved σ80 profile (mean texp≈0.5 on a rigid
//    pan). A stored 1-frame V history (cleaner dI/dt) or a local flow fit lifts
//    it — follow-up, not this pass.
//  · Fast pans: the fast-EMA smears → |v| drops below VLO → gate disengages
//    (= pre-feature behaviour, the false-positive returns; NOT new over-pump).
//  · Letterbox bar rows adjacent to the picture inject high-∇I / di≈0 cells that
//    mildly DAMP |v| (weaker suppression on scope content). Excluding lb-dead
//    cells from the three fit loops is a clean follow-up (deferred — the predict
//    loop must still zero s_flow_pred for skipped cells; wants its own audit).
//  · Thresholds are per-rendered-frame, 24p-tuned like every EMA here (shader-wide
//    caveat, see TEMPORAL_ALPHA block) — 60p rescales di/|v| ~2.5×.
// OFF-SCREEN / EDGE ESTABLISHMENT (2026-07-07) — the reveal-safety analog for
// content ENTERING through the frame border. The established-level gate above
// only knows ON-SCREEN history: a steady-bright object sliding in from off
// frame has none, and it outruns the 25-frame slow lane, so every cell it
// crosses reads FRESH and — under the ADDITIVE mask, whose only reveal-safety
// IS this gate (the signed-cancellation safety lives in the scalar pump_env,
// which additive does not consume for amplitude) — pumps a glow that tracks the
// object. This is the documented residual of the signed-cancellation approach
// (see the PUMP_DRIVE_P "TRADE" note: an object ENTERING has its matching fall
// off-screen, so the on-screen onset looks unmatched). Three parts, all in
// PASS 5:
//  (1) FAST-ESTABLISH — a rising NON-fresh cell settles its slow lane to its
//      fast lane THIS frame instead of over ~25, so the "established" verdict
//      propagates inward with the moving front at up to ring-2/frame and keeps
//      pace with the object (≈240 px/frame at 1080p). CRUCIALLY it fires only
//      when the rise is gated by an INFLUX ANCHOR — a per-cell seed-origin
//      marker (pump_seed_cell) that traces back to a border seed AND sits a
//      STEP above this cell's fast (PUMP_EDGE_STEP_MARGIN). Two guards:
//        - marker 0 (a genuine established sky/lamp/standing fire) never
//          propagates -> a cell gated by it keeps the slow EMA + the pre-
//          existing bounded-rim gate. Without this the chain runs through ANY
//          connected region within PUMP_ESTABLISH_MARGIN and gates a fire under
//          a brighter sky / a dim co-fire to ~0 (audit finding: "an event is
//          fresh so it can't self-gate" is FALSE — an event dimmer than an
//          adjacent established region is non-fresh at its boundary).
//        - the STEP requirement means the anchor is genuinely HIGHER (this cell
//          is catching up to a front / reveal). A uniform fade has every
//          neighbour's slow lane BELOW this cell's fast, so no step is met and
//          the marker cannot chain through a smooth full-frame rise — the
//          self-latching gate (settle pins di -> stays non-fresh -> sustains
//          the chain) the second audit pass found is structurally impossible.
//  (2) BORDER SEED — the outer ring has no more-outward on-screen neighbour to
//      be non-fresh against, so a rising outer-ring cell is presumed influx
//      (off-frame continuity): it holds its mask shut, marks itself (marker =
//      edge_seed), and fast-establishes. This is the ORIGIN that (1) propagates.
//  (3) GLOBAL GATE — (2) must NOT fire on a frame-wide rise (tunnel exit,
//      fade-to-white) or the whole frame would gate inward from every edge (and
//      seed a frame-filling influx chain). The seed scales by 1 - smoothstep of
//      the FRACTION of the deep interior that is rising: a global rise lights
//      most of it (seed off), a localized central event lights only a handful
//      (seed stays armed — fraction, not mean magnitude, so a few hot central
//      cells can't trip it).
// Vignette safety is the BORDER MIRROR in the publish loop: an outer cell
// max-inherits its inward neighbour's amplitude at full rate, so a genuine
// event reaching the edge pumps clean to the edge while a pure influx — whose
// inward neighbour is itself gated — inherits nothing. Accepted residuals
// (both DOCUMENTED, not bugs — offline audit 2026-07-07): (a) a sustained event
// that originates PURELY at the extreme edge with no interior support reads as
// influx and under-pumps (the source is already bright; spec + expansion still
// carry it); (b) the STEP_MARGIN test speed-limits influx protection to
// ~1.3 cells/frame (≈155 px/frame @1080p) — a FASTER uniform pan-reveal glows
// as it did before this feature (a pre-existing residual of the signed-
// cancellation approach, NOT new: the fix is <= pre-fix pump at every speed, so
// it adds no over-pump). To reclaim the fast-pan ceiling later, make the step
// test velocity-aware (relax it in proportion to the border cell's own di) —
// enhancement, not a ship gate. 0 = pre-fix (feature off) behavior.
#define PUMP_EDGE_ESTABLISH        1
#define PUMP_EDGE_ESTABLISH_ALPHA  1.0    // 1 = settle slow->fast in one frame (max propagation reach); lower = gentler
#define PUMP_EDGE_STEP_MARGIN      0.03   // influx marker propagates only across a step this big (neighbour established above this cell's fast) — rejects uniform co-rise (anti-latch)
#define PUMP_EDGE_GLOBAL_EPS       0.02   // per-cell rise above which a deep-interior cell counts as "rising"
#define PUMP_EDGE_GLOBAL_FRAC_LO   0.40   // fraction of the deep interior rising below this: localized -> seed armed
#define PUMP_EDGE_GLOBAL_FRAC_HI   0.75   // above this: frame-global rise -> border seed fully disengaged
// Mask softening (v5.6, widened after a 2026-07 tunnel-reveal field report):
// the per-cell proof can correctly authorize only the hot core of one broad
// white opening, leaving its surrounding source light visibly tiled. The
// PUBLISHED mask (pump_mask_cell, what PASS 6 samples) is therefore
// max(env, a 5×5 weighted max-stencil). Max preserves an
// isolated event's FULL peak. As bright-field coverage rises over 0.10..0.25,
// the old one-cell skirt grows into a rounded two-cell pre-finish skirt
// (full-gate isolated-cell bounds: 0.56 cardinal / 0.38 diagonal at one cell,
// 0.14 cardinal at two) and multiple proved cells merge into one light volume.
// PASS 6's per-pixel pump_w trims that volume back to the source's own bright
// shape. DYNAMICS ARE UNTOUCHED: band-pass state, attack cap, velocity release,
// max-hold, and proof all remain on pump_env_cell; this is presentation-only.
// Overlapping cells cannot sum or amplify one another; overlaps follow the
// strongest local envelope. Coverage chooses presentation WIDTH only; it never
// grants amplitude. Safety remains opening-bounded: no authorized cell means an
// exact-zero stencil,
// unlike a scene-global scalar backstop (tested and rejected: it re-opened the
// moving-lamp class). The subtractive fallback retains the narrower v5.6 3×3
// skirt; spatial-off compiles this out; PUMP_MASK_BLOB5=0 restores 3×3 for
// additive too.
// PUMP_MASK_SOFTEN=0 publishes the raw env.
#define PUMP_MASK_SOFTEN    1
#define PUMP_MASK_BLOB5     1
#define PUMP_MASK_BLOB_GAIN 6.0
#define PUMP_MASK_BLOB_FRAC_LO 0.10
#define PUMP_MASK_BLOB_FRAC_HI 0.25
// A normalized 3x3 binomial finishing pass connects the adaptive skirt without
// quantizing its amplitude. It is coverage-gated, so small/localized events
// retain the proven narrow mask. Raw authorized cores are restored after
// filtering; this smooths support without inventing an event or attenuating
// its peak. At full gate an isolated core's final cardinal tail is 0.459,
// 0.176, 0.029 at radii 1..3 cells; that last tail is only ~1.8% maximum
// expansion before the source-brightness and cover gates. Cost is 1,584
// thread-0 shared reads + 144 shared writes + 144 final SSBO writes per frame.
#define PUMP_MASK_FINISH     1
#define PUMP_MASK_FINISH_MIX 1.0
// SPATIAL MODEL — two apply modes, selected by PASS 6's SPATIAL_PUMP_ADDITIVE:
//  - SUBTRACTIVE (v5.2–v5.4, field-confirmed): pump_local = pump_env × mask.
//    The env is a [0,1] SUPPRESSOR — spatial can only REMOVE the global scalar
//    pump from non-brightening regions, never ADD it, so the reveal/ghost/
//    occluder-trail class dies by construction (no global event → scalar ~0 →
//    product ~0). Amplitude is shared (one scalar, p-norm + drive_loc onset —
//    see PUMP_DRIVE_P block); per-region TIMING is each cell's own env.
//  - ADDITIVE (v5.5 experiment, the §13 "additive door"): pump_local = mask ×
//    cover. Each cell's gated env IS its own pump amplitude, so independent
//    regional events pump at their own strength AND rhythm (per-cell release —
//    the shared-amplitude bleed class dies too), and a localized event no
//    longer needs the frame statistics to fire. What makes an additive mask
//    safe NOW, where the pre-v5.2 stack (novelty gate, habitual-V memory,
//    pan reject, sustain-protect, motion crossfade — all deleted) kept
//    leaking: every mask OPENING passes the ESTABLISHED-LEVEL GATE above plus
//    the frame-edge rule — a rise merely converging to a level its
//    neighbourhood (or the frame) already holds cannot open a cell.
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
// Cover-gate fall rate (v5.7, audit finding M1). The cover multiplies the
// HELD pump_env every frame, so any one-frame drop in contrast_v used to
// yank an active pump in a single frame — a rule-2 step. The concrete
// trigger: the letterbox exclusion engaging (~5s in, absolute frame count)
// steps the V extrema once, and a pump held across that frame dipped
// visibly. Cover now RISES instantly (a gate re-opening can never hurt) but
// FALLS no faster than this per-frame ratio (half-life ~4 frames; 1→0.15 in
// ~12 frames ≈ 0.5s @24p — "own the mistake, release slowly"). A genuine
// fade-to-white collapses contrast over many frames, so the clamp rarely
// binds there; a hard cut still mutes INSTANTLY via transient_reset (env
// and cover both zeroed). This deliberately supersedes the v5.1-era "cover
// is never slewed" rule — that rule predates cover feeding a held additive
// amplitude and the engage step.
#define PUMP_COVER_FALL     0.85
// Optional coverage backstop for the degenerate uniform-but-high-contrast case
// (rare). Kept for A/B; not wired by default — the contrast gate supersedes it.
//#define PUMP_COVER_HIGH     0.62   // bright_frac above which the pump tapers
//#define PUMP_COVER_FULL     0.85   // bright_frac at/above which the pump is fully muted

float get_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Per-lane scratch for the cross-lane reduction. 7 arrays × 144 elements ×
// 4 bytes (all 32-bit) = 4032 bytes — well under any GPU's 16-32 KB
// shared-memory floor. The tier counters (bright/spec/high/bright-spec) are
// NOT stored as per-lane values: the reducer re-derives them from the raw
// values below (soft smoothstep memberships since v5.6 — see
// TIER_SOFT_HALFBAND), keeping each threshold single-sourced next to the
// sum it feeds.
shared float s_illum[144];
shared float s_log_luma[144];
shared uint  s_valid[144];
shared uint  s_change[144];
shared float s_intensity[144];      // spec/highlight tier source: V or Y per ENABLE_SATURATED_SPEC
shared float s_sat[144];            // near-white fence input for the bright-scene recovery counter
shared float s_illum_v[144];        // max(R,G,B) of the illum field — V-aware pump driver/guard
#if MC_RESIDUAL_GATE
// Motion-compensated residual gate (approach A): last frame's per-cell V and the
// block-match flow at each cell, both written per-lane in loop 1, consumed by
// thread 0 in loop 2 (warp + residual). Own-slot writes → race-free pre-barrier.
shared float s_prev_v[144];
#endif
#if MC_RESIDUAL_GATE || MOTION_COST_RESET || MOTION_COHERENT_ROUTE || MOTION_DEBUG_VIEWS
shared vec2 s_flow[144];
#endif
#if ADDITIVE_OPEN_GUARD
shared vec2 s_add_vflow[144];
shared float s_add_vflow_cost[144];
#endif
#if MOTION_COST_RESET || MOTION_COHERENT_ROUTE || MOTION_DEBUG_VIEWS
shared float s_flow_cost[144];
shared float s_flow_texture[144];
#endif
#if ENABLE_SPATIAL_PUMP
// Thread-0-only scratch (written and read after the barrier by thread 0
// exclusively — no synchronization needed): full pre-update snapshot of the
// cell lanes for the established-level neighbour test, which must see every
// cell's PRE-update state while the fused loop updates them in place.
shared float s_pump_snap_f[144];
shared float s_pump_snap_s[144];
#if PUMP_MASK_FINISH && PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
// Thread-0-only intermediate presentation field for the finishing blur.
shared float s_pump_shape[144];
#endif
#if ADDITIVE_OPEN_GUARD
// Frozen very-slow reveal memory. It must not be read directly from the SSBO
// while loop 2 updates cells serially or later indices would see this frame's
// writes from earlier neighbours.
shared float s_pump_snap_vs[144];
shared float s_pump_snap_persist[144];
shared float s_pump_snap_fast_mc[144];
#endif
#if PUMP_TRANS_SUPPRESS && !MC_RESIDUAL_GATE
// Frozen PRE-update per-cell velocity (di = fast−slow) for the translation
// suppressor's trailing-fall scan. MUST be its own array, NOT recomputed from
// s_pump_snap_f/_s in the neighbour scan: loop 2 CLOBBERS s_pump_snap_f[i] in
// place with the post-update env (line ~1648), so a neighbour already processed
// would yield a phantom fall. This stays frozen (written in loop 1 only).
shared float s_pump_snap_di[144];
#endif
#if PUMP_TRANSPORT_RESIDUAL && !MC_RESIDUAL_GATE
// Per-cell global-flow prediction (flow_pred = ∇I·v_global) for the transport-
// residual mask gate. Thread-0 write/read only (like the snaps) — no barrier.
// di is recomputed from snap_f/snap_s, so this decl does NOT depend on
// PUMP_TRANS_SUPPRESS owning s_pump_snap_di.
shared float s_flow_pred[144];
#endif
#if PUMP_EDGE_ESTABLISH
// Per-cell INFLUX-ORIGIN marker snapshot: 1 = this cell's establishment
// traces back to a border seed (content entering off-frame), 0 = it is a
// genuine interior established region (sky, lamp, standing fire). Only a
// marked anchor may propagate the fast-establish gate inward — the decoupling
// that stops the gate from chain-eating a fire under a brighter sky.
shared float s_pump_snap_seed[144];
#endif
#endif

#if MC_RESIDUAL_GATE
// Exact bilinear sample of the frozen previous 16x9 V field. Callers own the
// half-cell in-bounds verdict; clamping here matches the established local-MC
// edge contract while keeping all four shared reads valid.
float sample_prev_v(vec2 warpc) {
    vec2 cc = clamp(warpc, vec2(0.0), vec2(15.0, 8.0));
    int wx0 = int(floor(cc.x)), wy0 = int(floor(cc.y));
    int wx1 = min(wx0 + 1, 15), wy1 = min(wy0 + 1, 8);
    float wfx = cc.x - float(wx0), wfy = cc.y - float(wy0);
    return mix(mix(s_prev_v[wy0 * 16 + wx0], s_prev_v[wy0 * 16 + wx1], wfx),
               mix(s_prev_v[wy1 * 16 + wx0], s_prev_v[wy1 * 16 + wx1], wfx), wfy);
}

#if ADDITIVE_OPEN_GUARD
float additive_v_shape_cost(ivec2 cell, ivec2 d,
                            vec3 c0, vec3 c1, vec3 c2) {
    vec2 p = vec2(cell) + vec2(d) * (1.0 / 8.0);
    vec3 p0 = vec3(sample_prev_v(p + vec2(-1.0, -1.0)),
                   sample_prev_v(p + vec2( 0.0, -1.0)),
                   sample_prev_v(p + vec2( 1.0, -1.0)));
    vec3 p1 = vec3(sample_prev_v(p + vec2(-1.0,  0.0)),
                   sample_prev_v(p),
                   sample_prev_v(p + vec2( 1.0,  0.0)));
    vec3 p2 = vec3(sample_prev_v(p + vec2(-1.0,  1.0)),
                   sample_prev_v(p + vec2( 0.0,  1.0)),
                   sample_prev_v(p + vec2( 1.0,  1.0)));
    float pm = (dot(p0, vec3(1.0)) + dot(p1, vec3(1.0))
              + dot(p2, vec3(1.0))) * (1.0 / 9.0);
    p0 -= pm; p1 -= pm; p2 -= pm;
    vec3 e0 = min(abs(c0 - p0), vec3(ADD_VFLOW_SAD_CAP));
    vec3 e1 = min(abs(c1 - p1), vec3(ADD_VFLOW_SAD_CAP));
    vec3 e2 = min(abs(c2 - p2), vec3(ADD_VFLOW_SAD_CAP));
    return (dot(e0, vec3(1.0)) + dot(e1, vec3(1.0))
          + dot(e2, vec3(1.0))) * (1.0 / 9.0);
}

float cubic_prev_v(float p0, float p1, float p2, float p3, float t) {
    return p1 + 0.5 * t * (p2 - p0
         + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
         + t * (3.0 * (p1 - p2) + p3 - p0)));
}

// Catmull-Rom reconstruction of the frozen previous fast-lane field. Integer
// motion plus bilinear reconstruction left a curvature residual on slow broad
// lamps; the cubic + subpixel pair removes that false "emission" without a new
// texture or pass. Clamp only the dangerous downward overshoot: an upward one
// conservatively vetoes a real opening, while a downward one could manufacture
// positive residual and is therefore forbidden.
float sample_prev_fast_cubic(vec2 warpc) {
    vec2 cc = clamp(warpc, vec2(0.0), vec2(15.0, 8.0));
    int bx = int(floor(cc.x)), by = int(floor(cc.y));
    int xm = max(bx - 1, 0), x0 = bx, x1 = min(bx + 1, 15), x2 = min(bx + 2, 15);
    int ym = max(by - 1, 0), y0 = by, y1 = min(by + 1, 8), y2 = min(by + 2, 8);
    float fx = cc.x - float(bx), fy = cc.y - float(by);
    float r0 = cubic_prev_v(s_pump_snap_fast_mc[ym * 16 + xm], s_pump_snap_fast_mc[ym * 16 + x0],
                            s_pump_snap_fast_mc[ym * 16 + x1], s_pump_snap_fast_mc[ym * 16 + x2], fx);
    float r1 = cubic_prev_v(s_pump_snap_fast_mc[y0 * 16 + xm], s_pump_snap_fast_mc[y0 * 16 + x0],
                            s_pump_snap_fast_mc[y0 * 16 + x1], s_pump_snap_fast_mc[y0 * 16 + x2], fx);
    float r2 = cubic_prev_v(s_pump_snap_fast_mc[y1 * 16 + xm], s_pump_snap_fast_mc[y1 * 16 + x0],
                            s_pump_snap_fast_mc[y1 * 16 + x1], s_pump_snap_fast_mc[y1 * 16 + x2], fx);
    float r3 = cubic_prev_v(s_pump_snap_fast_mc[y2 * 16 + xm], s_pump_snap_fast_mc[y2 * 16 + x0],
                            s_pump_snap_fast_mc[y2 * 16 + x1], s_pump_snap_fast_mc[y2 * 16 + x2], fx);
    float v = cubic_prev_v(r0, r1, r2, r3, fy);
    float c00 = s_pump_snap_fast_mc[y0 * 16 + x0], c10 = s_pump_snap_fast_mc[y0 * 16 + x1];
    float c01 = s_pump_snap_fast_mc[y1 * 16 + x0], c11 = s_pump_snap_fast_mc[y1 * 16 + x1];
    float vlo = min(min(c00, c10), min(c01, c11));
    return max(v, vlo);
}

float sample_prev_fast_linear(vec2 warpc) {
    vec2 cc = clamp(warpc, vec2(0.0), vec2(15.0, 8.0));
    int x0 = int(floor(cc.x)), y0 = int(floor(cc.y));
    int x1 = min(x0 + 1, 15), y1 = min(y0 + 1, 8);
    float fx = cc.x - float(x0), fy = cc.y - float(y0);
    return mix(mix(s_pump_snap_fast_mc[y0 * 16 + x0],
                   s_pump_snap_fast_mc[y0 * 16 + x1], fx),
               mix(s_pump_snap_fast_mc[y1 * 16 + x0],
                   s_pump_snap_fast_mc[y1 * 16 + x1], fx), fy);
}

// Split bilinear persistence into three independent quantities:
//   x = genuine pre-proof credit (mature corners contribute ZERO),
//   y = mature-corner bilinear support, z = support-weighted mature TTL.
// Mixing both schemas as one scalar lets a diluted mature token masquerade as
// 2-6 frames of partial proof in a neighbour. The split preserves the old
// seven-frame count while still allowing a strongly-supported mature trajectory.
vec3 sample_prev_persist_split(vec2 warpc) {
    vec2 cc = clamp(warpc, vec2(0.0), vec2(15.0, 8.0));
    int wx0 = int(floor(cc.x)), wy0 = int(floor(cc.y));
    int wx1 = min(wx0 + 1, 15), wy1 = min(wy0 + 1, 8);
    float wfx = cc.x - float(wx0), wfy = cc.y - float(wy0);
    float v00 = s_pump_snap_persist[wy0 * 16 + wx0];
    float v10 = s_pump_snap_persist[wy0 * 16 + wx1];
    float v01 = s_pump_snap_persist[wy1 * 16 + wx0];
    float v11 = s_pump_snap_persist[wy1 * 16 + wx1];
    vec3 p00 = (v00 > ADD_PERSIST_BASE) ? vec3(0.0, 1.0, v00) : vec3(v00, 0.0, 0.0);
    vec3 p10 = (v10 > ADD_PERSIST_BASE) ? vec3(0.0, 1.0, v10) : vec3(v10, 0.0, 0.0);
    vec3 p01 = (v01 > ADD_PERSIST_BASE) ? vec3(0.0, 1.0, v01) : vec3(v01, 0.0, 0.0);
    vec3 p11 = (v11 > ADD_PERSIST_BASE) ? vec3(0.0, 1.0, v11) : vec3(v11, 0.0, 0.0);
    return mix(mix(p00, p10, wfx), mix(p01, p11, wfx), wfy);
}
#endif
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

    bool valid = Y_src > 0.001;
    // Near-white fence input for the bright-scene recovery counter (see
    // BRIGHT_SPEC_SAT_MAX). Saturation is computed from V/min of rgb_src
    // regardless of ENABLE_SATURATED_SPEC (under !SAT_SPEC the gate is a
    // structural no-op anyway: Y_src > 0.97 forces near-neutral RGB).
    float v_src   = max(max(rgb_src.r, rgb_src.g), rgb_src.b);
    float sat_src = (v_src > 1e-6)
        ? (v_src - min(min(rgb_src.r, rgb_src.g), rgb_src.b)) / v_src
        : 0.0;
    #if ENABLE_SATURATED_SPEC
    float intensity_src = v_src;
    #else
    float intensity_src = Y_src;
    #endif
    s_illum[lid]     = Y_ill;
    s_illum_v[lid]   = V_ill;
    s_log_luma[lid]  = valid ? log(max(Y_src, 1e-6)) : 0.0;
    s_valid[lid]     = valid ? 1u : 0u;
    s_intensity[lid] = intensity_src;
    s_sat[lid]       = sat_src;
#if MC_RESIDUAL_GATE || MOTION_COST_RESET || MOTION_COHERENT_ROUTE || MOTION_DEBUG_VIEWS
    // Motion gate inputs (own-slot, race-free): stash last frame's cell V when
    // the MC consumer is compiled, keep history advancing in the legacy A/B so
    // a state-preserving re-enable cannot see a stale frame, and sample the
    // block-match flow at this cell. spos is this cell's center; MOTION_FLOW is
    // 16×9 so the sample is the cell's own (dx,dy) px vector.
    vec4 motion_sample = MOTION_FLOW_tex(spos);
    #if MC_RESIDUAL_GATE
    s_prev_v[lid]     = prev_illum_v[lid];
    #endif
    prev_illum_v[lid] = s_illum_v[lid];
    s_flow[lid]       = motion_sample.xy;
    #if MOTION_COST_RESET || MOTION_COHERENT_ROUTE || MOTION_DEBUG_VIEWS
    s_flow_cost[lid]    = motion_sample.z;
    s_flow_texture[lid] = motion_sample.w;
    #endif
#else
    // Even a fully-disabled motion A/B keeps the history contract warm so a
    // state-preserving re-enable cannot consume an arbitrarily old cell field.
    prev_illum_v[lid] = s_illum_v[lid];
#endif

    // Scene-cut delta. Each lane reads + writes its own prev_illum slot —
    // no cross-lane SSBO traffic, so race-free even though the buffer is
    // declared coherent. The barrier below still gates s_change visibility
    // for the reducer.
    s_change[lid]   = (frame > 0 &&
                      abs(Y_ill - prev_illum[lid]) > ILLUM_CHANGE_THRESH)
                      ? 1u : 0u;
    prev_illum[lid] = Y_ill;

    barrier();

#if ADDITIVE_OPEN_GUARD
    // Parallel local matcher in the pump's own sigma80/V domain. Gate the work
    // with the same post-update band-pass test that will reach loop 2; idle
    // cells publish an implausible cost but every lane still reaches the second
    // barrier. All samples are shared-memory reads.
    ivec2 vcell = ivec2(int(ix), int(iy));
    float vf0 = mix(pump_fast_cell[lid], s_illum_v[lid], PUMP_ALPHA_FAST);
    float vs0 = mix(pump_slow_cell[lid], s_illum_v[lid], PUMP_ALPHA_SLOW);
    bool vwork = smoothstep(PUMP_CELL_DRIVE_LOW, PUMP_CELL_DRIVE_HIGH,
                            vf0 - vs0) > 0.0;
    vec2 vbest_flow = vec2(0.0);
    float vbest_cost = 1.0;
    if (vwork) {
        int xm = max(vcell.x - 1, 0), xp = min(vcell.x + 1, 15);
        int ym = max(vcell.y - 1, 0), yp = min(vcell.y + 1, 8);
        vec3 vc0 = vec3(s_illum_v[ym * 16 + xm],
                        s_illum_v[ym * 16 + vcell.x],
                        s_illum_v[ym * 16 + xp]);
        vec3 vc1 = vec3(s_illum_v[vcell.y * 16 + xm],
                        s_illum_v[vcell.y * 16 + vcell.x],
                        s_illum_v[vcell.y * 16 + xp]);
        vec3 vc2 = vec3(s_illum_v[yp * 16 + xm],
                        s_illum_v[yp * 16 + vcell.x],
                        s_illum_v[yp * 16 + xp]);
        float cur_mean = (dot(vc0, vec3(1.0)) + dot(vc1, vec3(1.0))
                        + dot(vc2, vec3(1.0))) * (1.0 / 9.0);
        vc0 -= cur_mean; vc1 -= cur_mean; vc2 -= cur_mean;
        float best_rank = 1e9;
        ivec2 bestd = ivec2(0);
        for (int dy = -4; dy <= 4; dy += 2) {
            for (int dx = -4; dx <= 4; dx += 2) {
                ivec2 d = ivec2(dx, dy);
                float cost = additive_v_shape_cost(vcell, d, vc0, vc1, vc2);
                float rank = cost + ADD_VFLOW_BIAS * float(abs(dx) + abs(dy));
                if (rank < best_rank) {
                    best_rank = rank;
                    vbest_cost = cost;
                    bestd = d;
                }
            }
        }
        ivec2 coarse_best = bestd;
        for (int ry = -1; ry <= 1; ry++) {
            for (int rx = -1; rx <= 1; rx++) {
                if (rx == 0 && ry == 0) continue;
                ivec2 d = coarse_best + ivec2(rx, ry);
                if (abs(d.x) > 5 || abs(d.y) > 5) continue;
                float cost = additive_v_shape_cost(vcell, d, vc0, vc1, vc2);
                float rank = cost + ADD_VFLOW_BIAS
                           * float(abs(d.x) + abs(d.y));
                if (rank < best_rank) {
                    best_rank = rank;
                    vbest_cost = cost;
                    bestd = d;
                }
            }
        }
        vbest_flow = vec2(bestd);
    }
    s_add_vflow[lid] = vbest_flow;
    s_add_vflow_cost[lid] = vbest_cost;
    barrier();
#endif

    // -------- thread-0 reduction + SSBO update --------
    // 144 serial adds on ALU registers is trivially cheap compared to the
    // 144 texture fetches we just parallelized away. Could be replaced with
    // a subgroupAdd ladder for sub-microsecond savings if profiling demands.
    if (lid == 0u) {
        // Schema prime only: this catches fresh/reformatted persistent state,
        // not a same-schema reload or every seek. The block-match has already
        // read CELFLARE_ADD_MOTION_PREV this frame; transient_reset below discards that
        // result while the history-store pass has primed next frame's image.
        bool motion_uninit = motion_state_magic != MOTION_STATE_EPOCH;

        #if MOTION_COST_RESET || MOTION_COHERENT_ROUTE || MOTION_DEBUG_VIEWS
        // Texture-qualified match validity. Flat tiles are deliberately absent
        // from the denominator: MOT_BIAS resolves them to zero flow but they
        // contain no evidence that zero is correct. Cost reset is default-on
        // after real-content threshold validation; the debug views keep the
        // evidence and threshold behavior observable.
        float motion_tex_w = 0.0, motion_bad_w = 0.0;
        for (uint i = 0u; i < 144u; i++) {
            float tc = smoothstep(MOTION_TEXTURE_LO, MOTION_TEXTURE_HI,
                                  s_flow_texture[i]);
            motion_tex_w += tc;
            motion_bad_w += tc * ((abs(s_flow_cost[i]) >= MOTION_COST_BAD) ? 1.0 : 0.0);
        }
        motion_match_coverage = motion_tex_w * (1.0 / 144.0);
        motion_bad_match_frac = (motion_match_coverage >= MOTION_BAD_COVER_MIN
                              && motion_tex_w > 1e-5)
            ? motion_bad_w / motion_tex_w : 0.0;

        #if MOTION_FRAME_ANALYSIS
        // Robust dominant PREV-OFFSET from the block-match vectors themselves.
        // The retired LK vector is in a different (sigma80 band-pass) domain and
        // cannot be used for this warp. A uniform 4x9 evidence lattice (every
        // fourth column, phase shifted by row) keeps both-axis coverage without
        // locking to one vertical stripe phase, while avoiding a second 144-cell
        // serial analysis inside the already-large stats pass.
        // One agreement-reweighted mean iteration is conservative on split/
        // multiplane fields (support collapses instead of choosing an arbitrary
        // mode) and avoids a 121-bin FXC-unroll hazard.
        float motion_wsum = 0.0;
        vec2 dom_sum = vec2(0.0);
        vec2 evidence_pos_sum = vec2(0.0), evidence_pos2_sum = vec2(0.0);
        for (uint ey = 0u; ey < 9u; ey++)
        for (uint ex = 0u; ex < 4u; ex++) {
            uint i = ey * 16u + ex * 4u + (ey & 3u);
            float cost_c = 1.0 - smoothstep(MOTION_COST_GOOD, MOTION_COST_BAD,
                                            abs(s_flow_cost[i]));
            float tex_c = smoothstep(MOTION_TEXTURE_LO, MOTION_TEXTURE_HI,
                                     s_flow_texture[i]);
            float w = cost_c * tex_c;
            motion_wsum += w;
            dom_sum += w * s_flow[i];
            vec2 ep = vec2((float(int(i) & 15) + 0.5) * (1.0 / 16.0),
                           (float(int(i) >> 4) + 0.5) * (1.0 / 9.0));
            evidence_pos_sum += w * ep;
            evidence_pos2_sum += w * ep * ep;
        }
        vec2 dom = vec2(0.0);
        float consensus_w = 0.0;
        if (!motion_uninit && motion_wsum > 1e-5) {
            dom = dom_sum / motion_wsum;
            for (int it = 0; it < 1; it++) {
                vec2 refine_sum = vec2(0.0);
                float refine_w = 0.0;
                for (uint ey = 0u; ey < 9u; ey++)
                for (uint ex = 0u; ex < 4u; ex++) {
                    uint i = ey * 16u + ex * 4u + (ey & 3u);
                    float cost_c = 1.0 - smoothstep(MOTION_COST_GOOD, MOTION_COST_BAD,
                                                    abs(s_flow_cost[i]));
                    float tex_c = smoothstep(MOTION_TEXTURE_LO, MOTION_TEXTURE_HI,
                                             s_flow_texture[i]);
                    float agree = 1.0 - smoothstep(MOTION_DOM_AGREE_LO,
                                                   MOTION_DOM_AGREE_HI,
                                                   length(s_flow[i] - dom));
                    float w = cost_c * tex_c * agree;
                    refine_sum += w * s_flow[i];
                    refine_w += w;
                }
                if (refine_w > 1e-5)
                    dom = refine_sum / refine_w;
                consensus_w = refine_w;
            }
        }
        float consensus_purity = (motion_wsum > 1e-5)
            ? clamp(consensus_w / motion_wsum, 0.0, 1.0) : 0.0;
        float evidence_cover = smoothstep(MOTION_DOM_COVER_LO, MOTION_DOM_COVER_HI,
                                          motion_wsum * (1.0 / 36.0));
        vec2 evidence_std = vec2(0.0);
        if (motion_wsum > 1e-5) {
            vec2 evidence_mean = evidence_pos_sum / motion_wsum;
            evidence_std = sqrt(max(evidence_pos2_sum / motion_wsum
                                  - evidence_mean * evidence_mean, vec2(0.0)));
        }
        // Absolute mass is not enough: a large moving character can own 20% of
        // cells while remaining a local object. Require evidence to span both
        // frame axes before flat cells elsewhere may borrow the dominant flow.
        float evidence_spread = smoothstep(MOTION_DOM_SPREAD_LO, MOTION_DOM_SPREAD_HI,
                                           min(evidence_std.x, evidence_std.y));
        float dom_support = consensus_purity * evidence_cover * evidence_spread;
        float dom_motion = smoothstep(MOTION_DOM_MAG_LO, MOTION_DOM_MAG_HI,
                                      length(dom));
        #if MOTION_DEBUG_VIEWS
        motion_dom_x = dom.x;
        motion_dom_y = dom.y;
        motion_dom_support = dom_support;
        #endif
        #if MOTION_DEBUG_VIEWS
        for (uint i = 0u; i < 144u; i++) {
            // Borrow the dominant vector only to the extent that the local
            // block match lacks evidence. A credible local disagreement keeps
            // authority: it may be an independently-moving/growing event, and
            // under-pumping that is worse than leaving a rare sharp alias open.
            float local_reliable =
                (1.0 - smoothstep(MOTION_COST_GOOD, MOTION_COST_BAD, abs(s_flow_cost[i])))
                * smoothstep(MOTION_TEXTURE_LO, MOTION_TEXTURE_HI, s_flow_texture[i]);
            float local_agree = 1.0 - smoothstep(MOTION_DOM_AGREE_LO,
                                                 MOTION_DOM_AGREE_HI,
                                                 length(s_flow[i] - dom));
            vec2 effective_flow = mix(dom, s_flow[i], local_reliable);
            // Flat/weak tiles inherit frame coherence; as local evidence becomes
            // trustworthy, local-vs-dominant agreement becomes mandatory.
            float trust_agree = mix(1.0, local_agree, local_reliable);
            float motion_trust = motion_uninit ? 0.0
                : dom_support * dom_motion * trust_agree;
            #if MC_RESIDUAL_GATE
            // Diagnostic path mirrors production: exact local and effective
            // bilinear samples of the frozen previous V field.
            int cxm = int(i) & 15, cym = int(i) >> 4;
            vec2 local_warpc = vec2(float(cxm), float(cym)) + s_flow[i] * (1.0 / 8.0);
            vec2 effective_warpc = vec2(float(cxm), float(cym)) + effective_flow * (1.0 / 8.0);
            bool local_inb = local_warpc.x > -0.5 && local_warpc.x < 15.5
                          && local_warpc.y > -0.5 && local_warpc.y < 8.5;
            bool effective_inb = effective_warpc.x > -0.5 && effective_warpc.x < 15.5
                              && effective_warpc.y > -0.5 && effective_warpc.y < 8.5;
            float mc_prev = sample_prev_v(local_warpc);
            float mc_prev_effective = sample_prev_v(effective_warpc);
            float mc_local = local_inb
                ? smoothstep(MC_RES_LO, MC_RES_HI,
                             max(s_illum_v[i] - mc_prev, 0.0)) : 0.0;
            float mc_effective = effective_inb
                ? smoothstep(MC_RES_LO, MC_RES_HI,
                             max(s_illum_v[i] - mc_prev_effective, 0.0)) : 0.0;
            motion_mc_local_cell[i] = mc_local;
            motion_mc_effective_cell[i] = mc_effective;
            motion_trust_cell[i] = motion_trust;
            #else
            motion_mc_local_cell[i] = 0.0;
            motion_mc_effective_cell[i] = 0.0;
            motion_trust_cell[i] = motion_trust;
            #endif
        }
        #endif
        #endif
        #endif

        float illum_sum         = 0.0;
        float log_luma_sum      = 0.0;
        uint  valid_luma        = 0u;
        float bright_sum        = 0.0;
        float spec_sum          = 0.0;
        float high_sum          = 0.0;
        uint  change_count      = 0u;
        float bright_spec_sum   = 0.0;
        float top_sum           = 0.0;
        float illum_min         = 1.0;
        float illum_max         = 0.0;
        float illum_v_psum      = 0.0;
        float illum_v_min       = 1.0;
        float illum_v_max       = 0.0;
        uint  n_bar             = 0u;

        // -------- letterbox / pillarbox bar detection (see LB_ENGAGE_FRAMES) --------
        // Thread-0 serial over the candidate edge lines; state = 8 run
        // counters in the SSBO (rows 0,1,7,8 then cols 0,1,14,15 — thread-0
        // is the single writer). A run survives cuts by design (persistence
        // through content changes is the evidence FOR barness); content in a
        // bar resets it the same frame. Engaged lines become bit masks the
        // stats loop tests per cell.
        uint lb_row_mask = 0u;
        uint lb_col_mask = 0u;
        {
            const int lb_rows[4] = int[4](0, 1, 7, 8);
            const int lb_cols[4] = int[4](0, 1, 14, 15);
            for (int k = 0; k < 4; k++) {
                bool all_dark = true;
                for (int x = 0; x < 16; x++)
                    all_dark = all_dark && (s_valid[lb_rows[k] * 16 + x] == 0u);
                float run = (frame == 0 || !all_dark)
                    ? 0.0 : min(bar_run[k] + 1.0, LB_ENGAGE_FRAMES);
                bar_run[k] = run;
                if (run >= LB_ENGAGE_FRAMES) lb_row_mask |= 1u << uint(lb_rows[k]);
            }
            for (int k = 0; k < 4; k++) {
                bool all_dark = true;
                for (int y = 0; y < 9; y++)
                    all_dark = all_dark && (s_valid[y * 16 + lb_cols[k]] == 0u);
                float run = (frame == 0 || !all_dark)
                    ? 0.0 : min(bar_run[k + 4] + 1.0, LB_ENGAGE_FRAMES);
                bar_run[k + 4] = run;
                if (run >= LB_ENGAGE_FRAMES) lb_col_mask |= 1u << uint(lb_cols[k]);
            }
        }

        for (uint i = 0u; i < 144u; i++) {
            float yi           = s_illum[i];
            illum_sum         += yi;
            // Floor at 0: upstream ringing can undershoot slightly negative and
            // GLSL pow() is undefined for x<0 — a single NaN here would persist
            // in pump_fast/pump_slow (SSBO) until the next scene cut.
            float vi           = max(s_illum_v[i], 0.0);
            // p-norm over ALL 144 cells including bars — deliberate, see the
            // LB_ENGAGE_FRAMES block (drive path must not step at engage).
            illum_v_psum      += pow(vi, PUMP_DRIVE_P);
            log_luma_sum      += s_log_luma[i];
            valid_luma        += s_valid[i];
            // Engaged bar cells are excluded from everything below: contrast
            // extrema (both axes), the tier sums + their denominator, AND the
            // scene-cut change count. The change count must sit below the
            // exclusion (audit M1): s_change tests the σ80 ILLUM field, which
            // bleeds across the bar boundary while the bar's SOURCE stays
            // black — counting bleed-swung bar cells over the n_eff
            // denominator would let a large bright event's halo manufacture
            // a FALSE cut on letterboxed content (lockout → pump reset
            // mid-event). Numerator and denominator now cover the same
            // picture-area population. (i >> 4 = row, i & 15 = col.)
            bool lb_dead = (((lb_row_mask >> (i >> 4u)) & 1u) == 1u)
                        || (((lb_col_mask >> (i & 15u)) & 1u) == 1u);
            if (lb_dead) { n_bar++; continue; }
            change_count      += s_change[i];
            illum_min          = min(illum_min, yi);
            illum_max          = max(illum_max, yi);
            illum_v_min        = min(illum_v_min, vi);
            illum_v_max        = max(illum_v_max, vi);
            // Tier sums, re-derived from the raw per-lane values (single-
            // sourced thresholds next to the sums they feed). SOFT membership
            // (see TIER_SOFT_HALFBAND): each compare is a smoothstep over the
            // ±band so single-texel samples sliding across a tier on a pan
            // contribute continuously — no 1/144 count pops.
            float ii           = s_intensity[i];
            bright_sum        += smoothstep(BRIGHT_STAT_THRESH - TIER_SOFT_HALFBAND,
                                            BRIGHT_STAT_THRESH + TIER_SOFT_HALFBAND, yi);
            spec_sum          += smoothstep(SPECULAR_THRESH - TIER_SOFT_HALFBAND,
                                            SPECULAR_THRESH + TIER_SOFT_HALFBAND, ii);
            high_sum          += smoothstep(HIGHLIGHT_THRESH - TIER_SOFT_HALFBAND,
                                            HIGHLIGHT_THRESH + TIER_SOFT_HALFBAND, ii);
            bright_spec_sum   += (s_sat[i] < BRIGHT_SPEC_SAT_MAX)
                                 ? smoothstep(BRIGHT_SPEC_THRESH - TIER_SOFT_HALFBAND,
                                              BRIGHT_SPEC_THRESH + TIER_SOFT_HALFBAND, ii)
                                 : 0.0;
            top_sum           += smoothstep(TOP_BAND_THRESH - TIER_SOFT_HALFBAND,
                                            TOP_BAND_THRESH + TIER_SOFT_HALFBAND, ii);
        }

        const float N_SAMPLES = 144.0;
        // Effective picture-area cell count. Worst case (2.76:1 letterbox +
        // 4:3 pillarbox simultaneously) excludes 64+36−16 = 84 cells →
        // n_eff ≥ 60, never near zero. avg_illum stays /144 (it is only the
        // log_avg fallback for near-black frames, where bars are moot).
        float n_eff       = N_SAMPLES - float(n_bar);
        float avg_illum   = illum_sum / N_SAMPLES;
        float bright_frac = bright_sum / n_eff;
        float top_frac    = top_sum / n_eff;

        // Contrast: dynamic range from illumination field (stable, noise-free).
        // max/min >= 1 by construction: both are running extrema of one sample set.
        float contrast = (illum_min > 0.001)
            ? log2(illum_max / illum_min)
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
        // Cover-gate contrast, low-tail anchored (min). The old `illum_v_min > 0.001
        // ? log2(max/min) : 0.0` was wrong-signed: one near-black cell forced the
        // ratio to 0.0, muting the pump on every localized bright event that sits on
        // a darker surround — an explosion in a lit scene, a spell on a night sky —
        // AND every ordinary shadowed shot. Floor the denominator instead of dropping
        // to 0, keeping the min's sensitivity to a hot region against ANY darker
        // surround (this is what pumps a real fire-in-a-scene, which a mean anchor
        // muted because the fire's area + scene mid-tones lift the mean above the
        // gate). Numerator floored too (log2(0) is UB; pump_cover_gate is persistent
        // SSBO state — a stray NaN would smear across frames). TRADE (accepted): a
        // structured fade-to-white with a persistent dark region also opens cover
        // here — a mild, multiplicative fade pump. The clean fade/event separator is
        // TEMPORAL (is the dark anchor RISING = fade, or STABLE = event), not spatial;
        // this gate leans on the drive band-pass as the first line of defense.
        float contrast_v    = max(0.0, log2(max(illum_v_max, 1e-6) / max(illum_v_min, 0.001)));

        // Log-average: perceptual brightness key from source pixels
        float log_avg = (valid_luma > 4u)
            ? exp(log_luma_sum / float(valid_luma))
            : avg_illum;

        // Specular signal: present when small fraction at specular tier
        float spec_frac          = spec_sum / n_eff;
        float highlight_frac_src = high_sum / n_eff;
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
        float bs_frac     = bright_spec_sum / n_eff;
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
        // The lockout counter alone encodes the whole cut-window state: a cut
        // sets it to LOCKOUT_FRAMES, so "cut this frame" == lockout at its max
        // and every consumer below keys off lockout > 0 (needs LOCKOUT_FRAMES > 0).
        // Denominator = picture-area cells: bar cells never change, so they
        // only diluted this. Windowboxed content (60 live cells) could never
        // reach SCENE_CUT_PCT 0.50 over /144 — cuts went UNDETECTED and the
        // pump lanes never reset on them. Over n_eff the 0.50 threshold means
        // "majority of the PICTURE moved" at any aspect ratio.
        float change_pct  = float(change_count) / n_eff;
        scene_cut_lockout = max(scene_cut_lockout - 1.0, 0.0);
        if (change_pct > SCENE_CUT_PCT && scene_cut_lockout <= 0.0)
            scene_cut_lockout = LOCKOUT_FRAMES;

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
        // impact frame) couple ~1:1 into every EMA as a visible pulse. On the
        // cut frame itself lockout/LOCKOUT_FRAMES == 1, so the mix lands on
        // TEMPORAL_ALPHA_FAST exactly — no dedicated cut branch needed.
        float alpha = (scene_cut_lockout > 0.0)
            ? mix(TEMPORAL_ALPHA_MID, TEMPORAL_ALPHA_FAST,
                  scene_cut_lockout / LOCKOUT_FRAMES)
            : base_alpha;

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
        // Transient reset window: frame 0 or anywhere inside the cut lockout
        // (the cut frame included — it just set lockout to its max). ONE bool
        // serves growth-mode, the scalar pump, and the per-cell pump: they
        // must reset on the SAME frames or their baselines desync.
        #if MOTION_COST_RESET
        bool motion_match_reset = motion_bad_match_frac >= MOTION_BAD_FRAC_RESET;
        #else
        bool motion_match_reset = false;
        #endif
        bool additive_mode_reset = additive_mode_magic != float(ADDITIVE_STATE_EPOCH);
        bool transient_reset = (frame == 0) || (scene_cut_lockout > 0.0)
                            || motion_uninit || motion_match_reset || additive_mode_reset;
        motion_state_magic = MOTION_STATE_EPOCH;
        additive_mode_magic = float(ADDITIVE_STATE_EPOCH);
        smoothed_growth_mode = transient_reset
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
        if (transient_reset) {
            #if ENABLE_SPATIAL_PUMP
            // Cell lanes re-pin coherently with the scalar lanes so a cut
            // transient can't manufacture per-cell drive either.
            for (uint i = 0u; i < 144u; i++) {
                float v = s_illum_v[i];
                pump_fast_cell[i] = v;
                pump_slow_cell[i] = v;
                pump_very_slow_cell[i] = v;
                pump_open_persist_cell[i] = 0.0;
                pump_env_cell[i]  = 0.0;
                pump_mask_cell[i] = 0.0;
                #if PUMP_EDGE_ESTABLISH
                pump_seed_cell[i] = 0.0;
                #endif
            }
            #endif
            pump_fast = pnorm_illum_v;
            pump_slow = pnorm_illum_v;
            pump_env  = 0.0;
            pump_cover_gate = 0.0;
        } else {
            // Locals: the SSBO is coherent, so re-reading a lane just written
            // is a real memory round-trip — compute once, store once.
            float pf_prev = pump_fast;
            float pf = mix(pf_prev, pnorm_illum_v, PUMP_ALPHA_FAST);
            float ps = mix(pump_slow, pnorm_illum_v, PUMP_ALPHA_SLOW);
            pump_fast = pf;
            pump_slow = ps;
            float drive      = pf - ps;                                  // SIGNED velocity
            // True frame-to-frame source fall. The old pf/ps ratio measured the
            // SAME fast-vs-slow band-pass deficit on every frame, so a brief
            // flicker compounded repeatedly (0.8^N) and irreversibly erased a
            // held pump. pf/pf_prev telescopes to the actual fast-lane level
            // change across the fall. A rebound while still below ps no longer
            // keeps charging the old dip.
            float global_fall_ratio = (drive < 0.0 && pf < pf_prev && pf_prev > 1e-3)
                ? clamp(pf / pf_prev, 0.0, 1.0) : 1.0;
            // Onset candidate — no max(0,·) clamp: smoothstep's onset edge
            // (PUMP_DRIVE_LOW > 0) already maps any x <= 0 to exactly 0.
            float drive_eff  = drive;
            #if ENABLE_SPATIAL_PUMP
            // -------- per-cell band-pass (SUBTRACTIVE mask) + localized onset --------
            // Two serial thread-0 loops do all of the spatial machinery —
            // thread-0 owns every pump SSBO write, so there is no second
            // barrier, no broadcast flag, and no cross-phase ordering to
            // audit (the split-phase version needed all three):
            //  (1) drive_loc terms, read from each cell's PRE-update lanes —
            //      the same previous-frame timing the split design had (1
            //      frame of lag on 8/25-frame EMAs, immaterial). Signed
            //      p-mean AGGREGATE of the per-cell drives (see PUMP_DRIVE_P
            //      block). Deadzone shrinks |d| symmetrically (both signs),
            //      so cancellation is preserved. Sign is the reveal safety:
            //      rise+fall of a crossing occluder/pan cancels; only
            //      net-asymmetric brightening survives. ONSET ONLY — release
            //      stays on the global lanes + per-cell mask closure.
            //  (2) cell EMA update + mask env: pump_env_cell[144] = per-cell
            //      "is this region BRIGHTENING" ∈[0,1]; PASS 6 bilinear-
            //      upsamples it and MULTIPLIES the scalar pump by it.
            //      SUBTRACTIVE: it can only SUPPRESS the scalar in non-
            //      brightening regions, never ADD pump — a reveal / pan /
            //      occluder-wake can't manufacture pump (no global event ⇒
            //      scalar ~0 ⇒ product ~0 for any local rise). Driver =
            //      coarse σ80 CELFLARE_ILLUM V at the cell center
            //      (s_illum_v, stored by the lanes before the barrier) — the
            //      SAME field the scalar p-norms over the frame. V = max
            //      channel → colored blooms register.
            // Loop 1: full pre-update snapshot + the UNGATED aggregates. The
            // snapshot (thread-0 scratch, see decl) is required because the
            // established-level test in loop 2 compares each riser against
            // its neighbours' PRE-update slow lanes while the lanes are being
            // rewritten in place. loc_sum (ungated) is the RELEASE statistic:
            // balanced crossings cancel here, dying fires go net-negative.
            float loc_sum = 0.0, loc_on_sum = 0.0;
            float fall_w = 0.0, fall_ratio = 0.0;
            #if PUMP_EDGE_ESTABLISH
            float edge_interior_n = 0.0;   // count of deep-interior cells that are RISING
            #endif
            for (uint i = 0u; i < 144u; i++) {
                float cf = pump_fast_cell[i];
                float cs = pump_slow_cell[i];
                s_pump_snap_f[i] = cf;
                s_pump_snap_s[i] = cs;
                #if ADDITIVE_OPEN_GUARD
                s_pump_snap_vs[i] = pump_very_slow_cell[i];
                s_pump_snap_persist[i] = pump_open_persist_cell[i];
                s_pump_snap_fast_mc[i] = cf;
                #endif
                #if PUMP_EDGE_ESTABLISH
                s_pump_snap_seed[i] = pump_seed_cell[i];   // prev-frame influx-origin marker
                #endif
                float di = cf - cs;
                #if PUMP_TRANS_SUPPRESS && !MC_RESIDUAL_GATE
                s_pump_snap_di[i] = di;   // frozen neighbour velocity for the trailing-fall scan
                #endif
                float m  = max(abs(di) - PUMP_CELL_DEADZONE, 0.0);
                float w  = pow(m, PUMP_DRIVE_P);
                loc_sum += sign(di) * w;
                #if PUMP_EDGE_ESTABLISH
                // Border-seed global gate: the FRACTION of the deep interior
                // (cols 3-12 x rows 2-6, 50 cells clear of the outer ring AND the
                // letterbox candidate lines 0,1,7,8 / 0,1,14,15) that is rising.
                // A frame-global brightening lights most of them; a LOCALIZED
                // central event (a fire, an explosion) lights only a handful, so
                // fraction — not mean magnitude — keeps the seed armed through
                // localized events while still disengaging on a true global rise
                // (the mean-magnitude form let a few hot central cells trip it).
                {
                    int gy = int(i) / 16, gx = int(i) % 16;
                    if (gx >= 3 && gx <= 12 && gy >= 2 && gy <= 6 && di > PUMP_EDGE_GLOBAL_EPS)
                        edge_interior_n += 1.0;
                }
                #endif
            }
            #if PUMP_EDGE_ESTABLISH
            // >LO fraction rising: broad enough to be coherent; >HI: clearly
            // frame-global -> seed fully disengaged so borders pump like
            // everything else.
            float global_gate = smoothstep(PUMP_EDGE_GLOBAL_FRAC_LO, PUMP_EDGE_GLOBAL_FRAC_HI,
                                           edge_interior_n * (1.0 / 50.0));
            #endif
            #if PUMP_TRANSPORT_RESIDUAL && !MC_RESIDUAL_GATE
            // ---- Global image-motion fit (Lucas–Kanade on the σ80 cell field) ----
            // Fit the single global image velocity v that best explains the whole
            // field's temporal change by brightness constancy di = ∇I·v (see the
            // PUMP_TRANSPORT_RESIDUAL knob block). di = fast−slow (band-pass ∝ the
            // field's recent dI/dt; positive scale folds into |v|); ∇I = central
            // differences of the SAME fast lane (s_pump_snap_f) so di and ∇I are
            // derivatives of one field. Reads only the pre-update snapshots that
            // loop 1 just froze — bit-independent of loop 2's in-place rewrite.
            // Convention: di = ∇I·v (residual r = di − ∇I·v), so flow_pred shares
            // di's sign at coherent-transport cells and texp = flow_pred/di ∈[0,1]
            // is meaningful. Whole block is thread-0-only; s_flow_pred needs no
            // barrier (written and read below on this lane exclusively).
            float flow_conf = 0.0;
            {
                float Sxx = 0.0, Sxy = 0.0, Syy = 0.0, Sxt = 0.0, Syt = 0.0, Sdt2 = 0.0;
                for (uint i = 0u; i < 144u; i++) {
                    int cyv = int(i) >> 4, cxv = int(i) & 15;
                    int xl = max(cxv - 1, 0), xr = min(cxv + 1, 15);
                    int yt = max(cyv - 1, 0), yb = min(cyv + 1, 8);
                    float gxv = 0.5 * (s_pump_snap_f[cyv * 16 + xr] - s_pump_snap_f[cyv * 16 + xl]);
                    float gyv = 0.5 * (s_pump_snap_f[yb * 16 + cxv] - s_pump_snap_f[yt * 16 + cxv]);
                    float dtv = s_pump_snap_f[i] - s_pump_snap_s[i];
                    Sxx += gxv * gxv; Sxy += gxv * gyv; Syy += gyv * gyv;
                    Sxt += gxv * dtv; Syt += gyv * dtv; Sdt2 += dtv * dtv;
                }
                // Ridge (Tikhonov) stabilizes the 2×2 solve under the aperture
                // problem (collinear gradients → rank-1 normal matrix). flow_pred
                // uses only the observable normal-flow component, so the
                // ridge-biased null direction never reaches the per-cell test.
                float ridge = PUMP_TRANSPORT_RIDGE * (Sxx + Syy + 1e-6);
                float vx = 0.0, vy = 0.0;
                {
                    float a = Sxx + ridge, d = Syy + ridge, b = Sxy;
                    float det = a * d - b * b;
                    if (det > 1e-12) { vx = ( d * Sxt - b * Syt) / det;
                                       vy = (-b * Sxt + a * Syt) / det; }
                }
                #if PUMP_TRANSPORT_ROBUST
                // IRLS: down-weight cells whose rise the current v does NOT explain
                // (a coexisting emission) so an event can't drag the global fit
                // toward explaining itself. Robust scale² = C·mean(di²).
                float sigma2 = PUMP_TRANSPORT_ROBUST_C * (Sdt2 / 144.0) + 1e-8;
                for (int it = 0; it < PUMP_TRANSPORT_ROBUST; it++) {
                    float wSxx = 0.0, wSxy = 0.0, wSyy = 0.0, wSxt = 0.0, wSyt = 0.0;
                    for (uint i = 0u; i < 144u; i++) {
                        int cyv = int(i) >> 4, cxv = int(i) & 15;
                        int xl = max(cxv - 1, 0), xr = min(cxv + 1, 15);
                        int yt = max(cyv - 1, 0), yb = min(cyv + 1, 8);
                        float gxv = 0.5 * (s_pump_snap_f[cyv * 16 + xr] - s_pump_snap_f[cyv * 16 + xl]);
                        float gyv = 0.5 * (s_pump_snap_f[yb * 16 + cxv] - s_pump_snap_f[yt * 16 + cxv]);
                        float dtv = s_pump_snap_f[i] - s_pump_snap_s[i];
                        float r   = dtv - (gxv * vx + gyv * vy);
                        float w   = sigma2 / (sigma2 + r * r);
                        wSxx += w * gxv * gxv; wSxy += w * gxv * gyv; wSyy += w * gyv * gyv;
                        wSxt += w * gxv * dtv; wSyt += w * gyv * dtv;
                    }
                    float rr = PUMP_TRANSPORT_RIDGE * (wSxx + wSyy + 1e-6);
                    float a = wSxx + rr, d = wSyy + rr, b = wSxy;
                    float det = a * d - b * b;
                    if (det > 1e-12) { vx = ( d * wSxt - b * wSyt) / det;
                                       vy = (-b * wSxt + a * wSyt) / det; }
                }
                #endif
                // Per-cell flow prediction (stored for loop 2) + global confidence.
                // conf = R² of the fit (fraction of the field's temporal energy the
                // single v explains — high only for COHERENT global motion) × a
                // real-motion floor on |v| (a uniform fade has di≫0 but v≈0 → not
                // suppressed) × strength. The den>eps guard forecloses a 0/0 NaN on
                // a near-static frame (flow_conf → fresh_ease → pump_env_cell SSBO).
                float num = 0.0, den = 0.0;
                for (uint i = 0u; i < 144u; i++) {
                    int cyv = int(i) >> 4, cxv = int(i) & 15;
                    int xl = max(cxv - 1, 0), xr = min(cxv + 1, 15);
                    int yt = max(cyv - 1, 0), yb = min(cyv + 1, 8);
                    float gxv = 0.5 * (s_pump_snap_f[cyv * 16 + xr] - s_pump_snap_f[cyv * 16 + xl]);
                    float gyv = 0.5 * (s_pump_snap_f[yb * 16 + cxv] - s_pump_snap_f[yt * 16 + cxv]);
                    float dtv = s_pump_snap_f[i] - s_pump_snap_s[i];
                    float fp  = gxv * vx + gyv * vy;
                    s_flow_pred[i] = fp;
                    float r = dtv - fp;
                    num += r * r; den += dtv * dtv;
                }
                float r2    = (den > 1e-8) ? clamp(1.0 - num / den, 0.0, 1.0) : 0.0;
                float vmag  = length(vec2(vx, vy));
                float mgate = smoothstep(PUMP_TRANSPORT_VLO, PUMP_TRANSPORT_VHI, vmag);
                // Coherence GATE, not a linear R² discount: the band-pass di caps
                // a clean rigid pan's R² well below 1 (≈0.35 offline), so a linear
                // r2 factor would needlessly halve a real pan's debit. A floor lets
                // any decently-coherent fit engage at full strength and only blocks
                // an incoherent one (a coexisting emission drags R² down → back off
                // globally to protect the event — the conservative failure).
                float coh   = smoothstep(PUMP_TRANSPORT_R2_MIN,
                                         PUMP_TRANSPORT_R2_MIN + 0.15, r2);
                // Conservation gate (see knob block): a single global v also fits
                // a PROPAGATING emission front, so require the motion to be
                // brightness-CONSERVING — fade the debit out as the scalar's net
                // signed drive goes positive. loc_sum (from loop 1) ≈ 0 for a
                // bounded conserved pan (rises balanced by falls), > 0 for a
                // front/event (bright-mass grows). Only the positive half gates;
                // loc_sum>0 guarantees a positive pow() base (no NaN).
                float dl_pos = (loc_sum > 0.0)
                    ? pow(loc_sum / N_SAMPLES, 1.0 / PUMP_DRIVE_P) : 0.0;
                float emit  = 1.0 - smoothstep(PUMP_TRANSPORT_EMIT_LO,
                                               PUMP_TRANSPORT_EMIT_HI, dl_pos);
                flow_conf   = coh * mgate * emit * PUMP_TRANSPORT_STRENGTH;
            }
            #endif
            #if MC_RESIDUAL_GATE
            // Frame-invariant conservation signal for the subtractive A0 floor.
            // Hoisted out of loop 2: pow() is paid once, never per active cell.
            float motion_dl_pos = (loc_sum > 0.0)
                ? pow(loc_sum / N_SAMPLES, 1.0 / PUMP_DRIVE_P) : 0.0;
            float motion_emit_signal = smoothstep(PUMP_TRANSPORT_EMIT_LO,
                                                   PUMP_TRANSPORT_EMIT_HI,
                                                   motion_dl_pos);
            #endif
            // Loop 2: established-level gate → gated ONSET aggregate, then the
            // cell EMA update + mask env. A rise is FRESH only if the cell's
            // pre-update fast exceeds every ring-2 neighbour's pre-update slow
            // by PUMP_ESTABLISH_MARGIN (see knob block). Falls always enter
            // the onset sum, so a light MOVING across cells self-cancels
            // (fresh rise at the new position vs fall at the old); only
            // NET-NEW light onsets.
            for (uint i = 0u; i < 144u; i++) {
                float cf = s_pump_snap_f[i];
                float cs = s_pump_snap_s[i];
                float di = cf - cs;
                float m  = max(abs(di) - PUMP_CELL_DEADZONE, 0.0);
                float w  = pow(m, PUMP_DRIVE_P);
                bool fresh = false;
                float fresh_ease = 0.0;
                #if cf_debug == 12
                motion_trust_cell[i] = 0.0;
                motion_mc_local_cell[i] = 0.0;
                motion_mc_effective_cell[i] = 0.0;
                #endif
                #if ADDITIVE_OPEN_GUARD
                float add_established = 0.0;
                #endif
                int cy = int(i) / 16, cx = int(i) % 16;
                #if PUMP_EDGE_ESTABLISH
                // Border seed (off-screen establishment): a rising outer-ring
                // cell has no more-outward on-screen neighbour to be non-fresh
                // against, so it is presumed influx — brightness continuing in
                // from off frame — except during a frame-global rise, where
                // global_gate disengages it. edge_seed in [0,1] debits the onset
                // and holds the mask shut below, and seeds the fast-establish so
                // the verdict propagates inward with the entering front.
                bool edge_cell = (cx == 0 || cx == 15 || cy == 0 || cy == 8);
                float edge_seed = (edge_cell && di > 0.0) ? (1.0 - global_gate) : 0.0;
                // Max influx-origin marker among the ring-2 neighbours that
                // actually gate this cell (their established level reaches it) —
                // set in the di>0 scan below. This is what scopes the
                // fast-establish to influx: a genuine (unmarked) sky/fire anchor
                // gives nb_seed 0, so the gate never chains through it.
                float nb_seed = 0.0;
                #endif
                if (di > 0.0) {
                    // ANCHORED establishment window (frame-edge rule, the
                    // additive-door prerequisite): the 5×5 ring-2 window is
                    // clamped to stay FULLY inside the grid, so at a frame
                    // edge it shifts inward instead of truncating. The old
                    // truncated scan made freshness EASIER at the border
                    // (killer clip: a bottom-edge cell held its mask ~55
                    // frames — inert under subtractive, a pump under
                    // additive); the shifted window answers with the nearest
                    // full block of in-frame context instead, which is a
                    // strict SUPERSET of the truncated scan — border
                    // freshness can only get harder, never easier. Interior
                    // cells are bit-identical to the old ring-2.
                    // Measured alternatives, both rejected (cell battery):
                    //  - out-of-frame = established-bright (1.0, the §13
                    //    sketch): border cells can never be fresh → a 2-cell
                    //    unpumped VIGNETTE ring on every global event;
                    //  - ring 3: in a 9-row grid a 7-row window can never
                    //    exclude a 3-row sky, so the standing-mass trade goes
                    //    frame-global vertically — a fire under any brighter
                    //    sky never localizes (kills the multi-fire target).
                    //    Ring stays 2; the ≥5-cell-tall occluder wake this
                    //    leaves open is the documented additive residual.
                    // cy/cx hoisted to the loop top (shared with the border
                    // seed + fast-establish); the ring-2 window is unchanged.
                    int ny0 = clamp(cy - 2, 0, 4);
                    int nx0 = clamp(cx - 2, 0, 11);
                    float nb_est = 0.0;
                    #if ADDITIVE_OPEN_GUARD
                    float nb_est_add = 0.0;
                    #endif
                    for (int ny = ny0; ny < ny0 + 5; ny++)
                        for (int nx = nx0; nx < nx0 + 5; nx++)
                            if (ny != cy || nx != cx) {
                                float snb = s_pump_snap_s[ny * 16 + nx];
                                nb_est = max(nb_est, snb);
                                #if ADDITIVE_OPEN_GUARD
                                nb_est_add = max(nb_est_add,
                                    max(snb, s_pump_snap_vs[ny * 16 + nx]));
                                #endif
                                #if PUMP_EDGE_ESTABLISH
                                // Carry the max INFLUX marker among neighbours
                                // established a STEP above this cell's fast
                                // (snb >= cf + STEP_MARGIN). Two guards in one:
                                //  - marker 0 (genuine sky/fire) never propagates
                                //    -> the fire-under-sky decoupling;
                                //  - the STEP requirement means the neighbour is a
                                //    genuinely-higher established anchor this cell
                                //    is CATCHING UP to (an influx front / reveal),
                                //    NOT a co-rising equal neighbour. On a uniform
                                //    fade every neighbour's slow lane sits BELOW
                                //    this cell's fast (snb = cf - di < cf), so the
                                //    step is never met -> the marker cannot chain
                                //    through a smooth full-frame rise (the
                                //    self-latching gate the design audit found).
                                if (snb >= cf + PUMP_EDGE_STEP_MARGIN)
                                    nb_seed = max(nb_seed, s_pump_snap_seed[ny * 16 + nx]);
                                #endif
                            }
                    fresh = cf - PUMP_ESTABLISH_MARGIN > nb_est;
                    #if ADDITIVE_OPEN_GUARD
                    add_established = smoothstep(PUMP_ESTABLISH_MARGIN,
                                                 2.0 * PUMP_ESTABLISH_MARGIN,
                                                 cf - nb_est_add);
                    #endif
                    #if PUMP_EDGE_ESTABLISH
                    // Influx-seeded edge cells contribute a debited onset so an
                    // object entering off-frame cannot fire the global scalar
                    // either (its off-screen fall is invisible to loc_on_sum).
                    if (fresh) loc_on_sum += w * (1.0 - edge_seed);
                    #else
                    if (fresh) loc_on_sum += w;
                    #endif
                    // Mask half consumes an EASED freshness: the band starts
                    // AT the margin, so everything the boolean suppresses
                    // stays exactly 0 (a reveal that noise-overshoots its
                    // neighbour by ~0.01 still cannot open) — the ease only
                    // softens the snap-open of a qualifying rise (the §13
                    // mask-hole seam, which additive amplitude would expose).
                    #if !MC_RESIDUAL_GATE
                    fresh_ease = smoothstep(PUMP_ESTABLISH_MARGIN,
                                            2.0 * PUMP_ESTABLISH_MARGIN,
                                            cf - nb_est);
                    #if PUMP_TRANS_SUPPRESS
                    // TRANSLATION SUPPRESSOR (see knob block): debit the mask
                    // opening when this rising cell is the LEADING edge of a
                    // camera-motion translation — i.e. a ring-1 neighbour is
                    // FALLING with magnitude COMPARABLE to this cell's rise (a
                    // rigid translation conserves local brightness). Ring-1 (a
                    // 3×3 anchored to stay in-grid) keeps the catch tight:
                    // unrelated fallers ≥2 cells away (separated multi-fire, an
                    // independent occluder) never enter it. The magnitude match
                    // is what protects the flagship — an igniting/flaring fire or
                    // an expanding bloom's rim has a rise far LARGER than any
                    // incidental neighbour fall, so bal→0 and it is NOT debited.
                    // Reads FROZEN neighbour di (s_pump_snap_di) — NOT recomputed
                    // from snap_f, which loop 2 clobbers in place below.
                    if (fresh_ease > 0.0) {
                        const int TR = PUMP_TRANS_RING, TW = 2 * PUMP_TRANS_RING + 1;
                        int fy0 = clamp(cy - TR, 0, 9 - TW);
                        int fx0 = clamp(cx - TR, 0, 16 - TW);
                        float nb_fall = 0.0;   // 0 = no faller / all-rising ring → no debit
                        for (int ny = fy0; ny < fy0 + TW; ny++)
                            for (int nx = fx0; nx < fx0 + TW; nx++)
                                if (ny != cy || nx != cx)
                                    nb_fall = min(nb_fall, s_pump_snap_di[ny * 16 + nx]);
                        float fall_amt = -nb_fall;                      // ≥ 0
                        float trans = smoothstep(PUMP_TRANS_FALL_LO, PUMP_TRANS_FALL_HI, fall_amt);
                        float bal = min(di, fall_amt) / max(max(di, fall_amt), 1e-6);
                        float match_w = smoothstep(PUMP_TRANS_MATCH, 1.0, bal);
                        fresh_ease *= 1.0 - trans * match_w;
                    }
                    #endif
                    #if PUMP_TRANSPORT_RESIDUAL
                    // GLOBAL-motion debit (see the knob block): fraction of THIS
                    // cell's rise that the single global image velocity explains.
                    // s_flow_pred[i] = ∇I·v (same di-basis as this loop's di), so
                    // texp≈1 when the rise IS the background sliding (debited) and
                    // ≈0 when it is emission the flow can't reproduce (pumps) —
                    // structurally separate from the dipole's magnitude match, and
                    // valid even under a simultaneous pan. Onset-only; scaled by
                    // flow_conf so it is inert unless a coherent global motion
                    // actually fits the field.
                    if (fresh_ease > 0.0) {
                        float texp = clamp(s_flow_pred[i] / max(di, 1e-4), 0.0, 1.0);
                        fresh_ease *= 1.0 - texp * flow_conf;
                    }
                    #endif
                    #endif
                } else {
                    loc_on_sum -= w;    // sign(di)*w with di <= 0
                }
                // Cell EMA update + mask env (post-update d, as before).
                float v = s_illum_v[i];
                float f = mix(cf, v, PUMP_ALPHA_FAST);
                float s = mix(cs, v, PUMP_ALPHA_SLOW);
                // Weight only cells whose fast lane is actually falling THIS
                // frame. di<0 alone merely says fast remains below slow; using
                // cf/cs there re-applied one old dip on every subsequent frame.
                if (di < 0.0 && f < cf && cf > 1e-3) {
                    fall_w     += w;
                    fall_ratio += w * clamp(f / cf, 0.0, 1.0);
                }
                #if PUMP_EDGE_ESTABLISH
                // Fast-establish, SCOPED TO INFLUX (2026-07-07 rework, after the
                // design audit): a rising cell settles its slow lane to its fast
                // lane THIS frame — instead of over ~25 — ONLY when its rise is
                // gated by an influx anchor (edge_seed for a border cell, else
                // nb_seed inherited from a seed-marked neighbour). That makes the
                // established verdict ride inward with an entering front at up to
                // ring-2/frame WITHOUT chaining through a genuine established
                // region: a non-fresh cell gated by an unmarked sky/fire keeps
                // the slow EMA (nb_seed 0 -> settle 0), so a fire under a brighter
                // sky / a dim co-fire keeps the pre-fix bounded-rim behaviour
                // instead of being chain-gated to zero. `settle` doubles as this
                // cell's new influx marker: it is nonzero only for influx risers,
                // so a fresh event (settle 0) and a fall (di<0) both clear it.
                // Blended (not branched) to avoid a threshold pop. Writes only
                // the SSBO slow lane, NOT the frozen snapshot, so the wave stays
                // symmetric (ring-2/frame) and scan-order-independent.
                float settle = max((di > 0.0 && !fresh) ? nb_seed : 0.0, edge_seed);
                s = mix(s, mix(cs, f, PUMP_EDGE_ESTABLISH_ALPHA), settle);
                pump_seed_cell[i] = settle;
                #endif
                pump_fast_cell[i] = f;
                pump_slow_cell[i] = s;
                #if ADDITIVE_OPEN_GUARD
                pump_very_slow_cell[i] = mix(s_pump_snap_vs[i], v, ADD_VSLOW_ALPHA);
                #endif
                float d = f - s;
                // Idle-wobble cells read true zero: any d below the onset edge
                // maps to exactly 0 (the former dead-zone shift is folded into
                // PUMP_CELL_DRIVE_LOW/HIGH — see the knob block).
                float a = smoothstep(PUMP_CELL_DRIVE_LOW, PUMP_CELL_DRIVE_HIGH, d);
#if MC_RESIDUAL_GATE
                // === MOTION-COMPENSATED RESIDUAL GATE (subtractive) ===
                // The mask opens only where current V exceeds the previous field
                // warped by motion: light the flow cannot explain. Evaluate only
                // for an actually-opening cell (a>0): fresh_ease is multiplied
                // into a immediately below, so idle-cell warps were dead work.
                {
                    fresh_ease = 0.0;
                    if (a > 0.0) {
                        int cxm = int(i) & 15, cym = int(i) >> 4;
                        vec2 local_flow = s_flow[i];
                        vec2 warpc = vec2(float(cxm), float(cym)) + local_flow * (1.0 / 8.0);
                        bool inb = warpc.x > -0.5 && warpc.x < 15.5
                                && warpc.y > -0.5 && warpc.y < 8.5;
                        #if ADDITIVE_OPEN_GUARD
                        // Additive A2: opening authority stays local. Compare the
                        // motion-compensated fast-lane rise with its same-cell
                        // rise so the decision is about EXPLAINED FRACTION, not
                        // an absolute one-frame delta: a slow grow keeps ratio~1,
                        // a carried lamp~0, and a held source retains evidence
                        // while the fast lane is still catching up.
                        // Cubic history and subpixel flow remove the coarse-grid
                        // curvature residual that otherwise poisons slow pans.
                        float mc_prev = f;
                        if (inb)
                            mc_prev = sample_prev_fast_cubic(warpc);
                        float mc_rise = inb ? max(f - mc_prev, 0.0) : 0.0;
                        float raw_rise = max(f - cf, 0.0);
                        float emit_ratio = clamp(mc_rise
                            / max(raw_rise, ADD_RATIO_RAW_FLOOR), 0.0, 1.0);
                        float ratio_route = smoothstep(ADD_RATIO_LO, ADD_RATIO_HI,
                                                       emit_ratio)
                                          * ((raw_rise > 1e-6) ? 1.0 : 0.0);
                        float effective_ratio_route = ratio_route;
                        // A compact source can sit in one raw-analysis tile
                        // while its broad illumination tail opens this one. The
                        // local V-flow route supplies that missing transport
                        // explanation. Selection used demeaned shape; the warp
                        // below uses absolute fast history, so real amplitude
                        // growth survives as residual instead of becoming flow.
                        if (s_add_vflow_cost[i] <= ADD_VFLOW_COST_MAX) {
                            vec2 vwarpc = vec2(float(cxm), float(cym))
                                        + s_add_vflow[i] * (1.0 / 8.0);
                            bool vin = vwarpc.x > -0.5 && vwarpc.x < 15.5
                                    && vwarpc.y > -0.5 && vwarpc.y < 8.5;
                            if (vin) {
                                float vprev = sample_prev_fast_linear(vwarpc);
                                float vrise = max(f - vprev, 0.0);
                                float vratio = clamp(vrise
                                    / max(raw_rise, ADD_RATIO_RAW_FLOOR), 0.0, 1.0);
                                float vroute = smoothstep(ADD_RATIO_LO, ADD_RATIO_HI,
                                                          vratio)
                                             * ((raw_rise > 1e-6) ? 1.0 : 0.0);
                                effective_ratio_route = min(effective_ratio_route,
                                                            vroute);
                            }
                        }
                        float routed_open = inb
                            ? add_established * effective_ratio_route : 0.0;
                        #if PUMP_EDGE_ESTABLISH
                        // Edge/influx authority is part of the persisted route,
                        // not a later output debit. Hidden edge credit therefore
                        // cannot mature while final amplitude is held at zero.
                        routed_open *= 1.0 - edge_seed;
                        #endif
                        // Credit follows one selected physical trajectory only:
                        // current + raw primary. The pump-domain route is an
                        // opening veto, never a persistence donor; lending its
                        // state could bypass the seven-frame proof beside an
                        // unrelated established event. An unproved transported
                        // lamp has routed_open=0 and loses pre-mature credit;
                        // only a previously-proved source may spend maintenance.
                        float prior_credit = s_pump_snap_persist[i];
                        // Once a cell has completed the seven-frame routed
                        // proof, that SAME cell/trajectory may maintain its
                        // opening authority through a short source flicker.
                        // Before maturity, one failed route still resets the
                        // count to zero. Maturity is encoded fractionally in
                        // (7,8]: failed frames spend exactly 1/48, and a valid
                        // route refreshes 8. Keeping the original 0..8 range is
                        // load-bearing: bilinear trajectory transport still
                        // needs >7/8 donor support instead of growing a skirt.
                        // This fixes the "opened, flickered, never restored"
                        // failure without granting a scene-global statistic or
                        // an unproven cell any opening authority. The credit is
                        // carried only by the already-selected raw trajectory.
                        float maintenance_env = smoothstep(ADD_MAINT_ENV_LO,
                                                           ADD_MAINT_ENV_HI,
                                                           pump_env_cell[i]);
                        if (inb) {
                            vec3 carried = sample_prev_persist_split(warpc);
                            prior_credit = max(prior_credit, carried.x);
                            if (carried.y > ADD_MAINT_TRANSPORT_MIN
                                    && maintenance_env > 0.0) {
                                float carried_ttl = carried.z / carried.y;
                                prior_credit = max(prior_credit, carried_ttl);
                            }
                        }
                        // A mature token with no surviving local amplitude is
                        // stale, including one bilinearly arriving at an empty
                        // neighbour. It must start proof from zero, not refresh.
                        if (prior_credit > ADD_PERSIST_BASE && maintenance_env <= 0.0)
                            prior_credit = 0.0;
                        bool prior_authorized = prior_credit > ADD_PERSIST_BASE;
                        float persist = 0.0;
                        if (routed_open >= ADD_PERSIST_ROUTE_MIN) {
                            persist = prior_authorized
                                ? ADD_MAINT_FULL
                                : min(prior_credit + 1.0, ADD_PERSIST_BASE);
                            if (persist >= ADD_PERSIST_BASE)
                                persist = ADD_MAINT_FULL;
                        } else if (prior_authorized) {
                            float spent = prior_credit - ADD_MAINT_STEP;
                            persist = (spent > ADD_MAINT_EXPIRE) ? spent : 0.0;
                        }
                        pump_open_persist_cell[i] = persist;
                        float persist_gate = smoothstep(ADD_PERSIST_BASE - 1.0,
                                                        ADD_PERSIST_BASE, persist);
                        float proved_open = routed_open * persist_gate;
                        // Maintenance preserves a still-live established mask;
                        // spent credit cannot resurrect a cell whose amplitude
                        // already released to zero, even if another object rises
                        // through the same grid position within the hold window.
                        float maintained_open = prior_authorized
                            ? maintenance_env : 0.0;
                        fresh_ease = max(proved_open, maintained_open);
                        #if cf_debug == 12
                        motion_trust_cell[i] = add_established;
                        motion_mc_local_cell[i] = effective_ratio_route;
                        motion_mc_effective_cell[i] = persist_gate;
                        #endif
                        #else
                        float mc_prev = sample_prev_v(warpc);
                        float mc_res = s_illum_v[i] - mc_prev;
                        float mc_fresh = inb
                            ? smoothstep(MC_RES_LO, MC_RES_HI, max(mc_res, 0.0)) : 0.0;
                        #if MOTION_COHERENT_ROUTE
                        // Supported dominant motion gets a one-way veto only on
                        // this live opening. Weak local matches borrow dom; a
                        // credible disagreement protects an independent event.
                        float local_reliable =
                            (1.0 - smoothstep(MOTION_COST_GOOD, MOTION_COST_BAD,
                                              abs(s_flow_cost[i])))
                            * smoothstep(MOTION_TEXTURE_LO, MOTION_TEXTURE_HI,
                                         s_flow_texture[i]);
                        float local_agree = 1.0 - smoothstep(
                            MOTION_DOM_AGREE_LO, MOTION_DOM_AGREE_HI,
                            length(local_flow - dom));
                        float trust_agree = mix(1.0, local_agree, local_reliable);
                        float motion_trust = motion_uninit ? 0.0
                            : dom_support * dom_motion * trust_agree;
                        if (mc_fresh > 0.0 && motion_trust > 0.0) {
                            vec2 effective_flow = mix(dom, local_flow, local_reliable);
                            vec2 effective_warpc = vec2(float(cxm), float(cym))
                                                 + effective_flow * (1.0 / 8.0);
                            bool effective_inb = effective_warpc.x > -0.5
                                              && effective_warpc.x < 15.5
                                              && effective_warpc.y > -0.5
                                              && effective_warpc.y < 8.5;
                            // Exact second bilinear sample. Shared history keeps
                            // this bandwidth on-chip; the active/trusted-opening
                            // guards above keep it off the idle-cell path.
                            float mc_prev_effective = sample_prev_v(effective_warpc);
                            float mc_effective = effective_inb
                                ? smoothstep(MC_RES_LO, MC_RES_HI,
                                    max(s_illum_v[i] - mc_prev_effective, 0.0))
                                : 0.0;
                            mc_fresh *= mix(1.0, mc_effective, motion_trust);
                        }
                        #endif
                        // A0 subtractive conservation floor: broad generation may
                        // restore the permissive mask because scalar pump_env still
                        // owns amplitude. The additive branch above deliberately
                        // does not inherit this frame-global opening floor.
                        fresh_ease = max(mc_fresh, motion_emit_signal);
                        #endif
                    }
                    #if ADDITIVE_OPEN_GUARD
                    else {
                        // A zero-opening frame used to erase even mature credit,
                        // reproducing the reported flicker dropout before the
                        // next rebound could use it. Pre-mature partial proof
                        // still resets immediately. A mature, still-live env
                        // spends the same bounded fractional credit while idle;
                        // once the local amplitude is gone, stale authority is
                        // discarded rather than lent to a later object.
                        float idle_credit = s_pump_snap_persist[i];
                        float idle_live = smoothstep(ADD_MAINT_ENV_LO,
                                                     ADD_MAINT_ENV_HI,
                                                     pump_env_cell[i]);
                        if (idle_credit > ADD_PERSIST_BASE && idle_live > 0.0) {
                            float spent = idle_credit - ADD_MAINT_STEP;
                            pump_open_persist_cell[i] =
                                (spent > ADD_MAINT_EXPIRE) ? spent : 0.0;
                        } else {
                            pump_open_persist_cell[i] = 0.0;
                        }
                    }
                    #endif
                }
#endif
                #if PUMP_MASK_ESTABLISH
                // Mask half of the gate: only a FRESH rise may OPEN the mask,
                // eased over [MARGIN, 2·MARGIN] above the neighbour ceiling
                // (see fresh_ease above). An already-open cell max-holds and
                // releases exactly as before — closing is never gated.
                a *= fresh_ease;
                #endif
                #if PUMP_EDGE_ESTABLISH && !ADDITIVE_OPEN_GUARD
                // Edge cells open only on a coherent (global) rise; a pure
                // influx (edge_seed->1, so 1-edge_seed->0) is held shut. A
                // genuine event that reaches the edge is restored at full rate
                // by the border mirror in the publish loop.
                a *= (1.0 - edge_seed);
                #endif
                // Additive proof completes after the source has already spent
                // seven rising frames. Publishing its current `a` in one step
                // made that delayed authorization look like a pop and exposed
                // cell-to-cell proof timing as patches. Cap only the RISING
                // authorized target to +0.25 env/frame: full-scale takes four
                // frames, while a moderate event is not proportionally
                // damped by an EMA. Pre-proof stays exact zero, while source
                // fall/release below remains unslewed.
                #if ADDITIVE_OPEN_GUARD
                a = min(a, pump_env_cell[i] + ADD_ATTACK_STEP);
                #endif
                // Velocity-matched release: when the negative band-pass region
                // is still falling THIS frame, follow its frame-to-frame fast
                // level. The ratios telescope across a real fade instead of
                // repeatedly compounding the same fast-vs-slow deficit. A
                // rebound below the slow lane holds rather than continuing to
                // erase the mask. Max-held + adapt floor.
                float r = (d < 0.0 && f < cf && cf > 1e-3)
                    ? clamp(f / cf, 0.0, 1.0) : 1.0;
                float e = max(pump_env_cell[i] * r * PUMP_ADAPT_FLOOR, a);
                pump_env_cell[i] = e;
                // Post-update env stash for the softening pass below. Reusing
                // s_pump_snap_f is safe: this iteration already consumed its
                // own cf, and no iteration reads another cell's snap_f (the
                // neighbour scan reads snap_s only).
                s_pump_snap_f[i] = e;
            }
            #if PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
            // Frame-global presentation-width selector: compute once on the
            // reducer lane, not once per output cell (coherent SSBO read).
            float pump_blob_gate = smoothstep(PUMP_MASK_BLOB_FRAC_LO,
                                              PUMP_MASK_BLOB_FRAC_HI,
                                              max(bright_frac, smoothed_bright_frac));
            #if PUMP_MASK_FINISH
            float pump_finish_mix = clamp(PUMP_MASK_FINISH_MIX * pump_blob_gate,
                                          0.0, 1.0);
            #endif
            #endif
            // Publish the PRESENTATION mask (see PUMP_MASK_SOFTEN): the raw
            // env stays the dynamics state; PASS 6 samples pump_mask_cell.
            for (uint i = 0u; i < 144u; i++) {
                float published;
                #if PUMP_MASK_SOFTEN
                int cy = int(i) / 16, cx = int(i) % 16;
                float b = 0.0;
                #if PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
                for (int dy = -2; dy <= 2; dy++)
                    for (int dx = -2; dx <= 2; dx++) {
                        int ny = clamp(cy + dy, 0, 8);
                        int nx = clamp(cx + dx, 0, 15);
                        int ax = abs(dx), ay = abs(dy);
                        float wx = (ax == 0) ? 6.0 : (ax == 1) ? 4.0 : 1.0;
                        float wy = (ay == 0) ? 6.0 : (ay == 1) ? 4.0 : 1.0;
                        b = max(b, s_pump_snap_f[ny * 16 + nx] * wx * wy
                                 * (PUMP_MASK_BLOB_GAIN / 256.0));
                    }
                float bn = 0.0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = clamp(cy + dy, 0, 8);
                        int nx = clamp(cx + dx, 0, 15);
                        bn += s_pump_snap_f[ny * 16 + nx]
                            * float((2 - abs(dy)) * (2 - abs(dx)));
                    }
                bn *= 1.0 / 16.0;
                float shaped = mix(bn, max(bn, b), pump_blob_gate);
                published = max(s_pump_snap_f[i], shaped);
                #else
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = clamp(cy + dy, 0, 8);
                        int nx = clamp(cx + dx, 0, 15);
                        // (1,2,1)⊗(1,2,1)/16 binomial; edge cells replicate
                        // (index clamp), keeping border amplitude full.
                        b += s_pump_snap_f[ny * 16 + nx]
                           * float((2 - abs(dy)) * (2 - abs(dx)));
                    }
                published = max(s_pump_snap_f[i], b * (1.0 / 16.0));
                #endif
                #else
                published = s_pump_snap_f[i];
                #endif
                #if PUMP_EDGE_ESTABLISH
                // Border mirror (vignette safety): an outer-ring cell inherits
                // its inward neighbour's amplitude at FULL rate (max, not the
                // ~1/4 soften skirt). A genuine event reaching the frame edge
                // pumps clean to the edge; a pure influx — whose inward
                // neighbour is itself gated — inherits 0, so no glow. Reads the
                // post-update env stashed in s_pump_snap_f by loop 2.
                int mcy = int(i) / 16, mcx = int(i) % 16;
                float mir = 0.0;
                if (mcx == 0)  mir = max(mir, s_pump_snap_f[mcy * 16 + 1]);
                if (mcx == 15) mir = max(mir, s_pump_snap_f[mcy * 16 + 14]);
                if (mcy == 0)  mir = max(mir, s_pump_snap_f[16 + mcx]);
                if (mcy == 8)  mir = max(mir, s_pump_snap_f[7 * 16 + mcx]);
                // Corner cells: both orthogonal inward neighbours are THEMSELVES
                // edge cells (gated by the same global_gate), so also reach the
                // inward DIAGONAL — the nearest genuinely-interior cell — or a
                // real corner event leaves a 1-cell notch at the very corner.
                int dcy = (mcy == 0) ? 1 : (mcy == 8) ? 7 : mcy;
                int dcx = (mcx == 0) ? 1 : (mcx == 15) ? 14 : mcx;
                if ((mcy == 0 || mcy == 8) && (mcx == 0 || mcx == 15))
                    mir = max(mir, s_pump_snap_f[dcy * 16 + dcx]);
                published = max(published, mir);
                #endif
                #if PUMP_MASK_FINISH && PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
                s_pump_shape[i] = published;
                #else
                pump_mask_cell[i] = published;
                #endif
            }
            #if PUMP_MASK_FINISH && PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
            // Continuous finishing blur over the already-authorized
            // presentation field. The broad-field selector controls only the
            // blend toward this smoother shape. Restore raw env afterwards so
            // no proved core can be reduced by the filter.
            for (uint i = 0u; i < 144u; i++) {
                int cy = int(i) / 16, cx = int(i) % 16;
                float bf = 0.0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = clamp(cy + dy, 0, 8);
                        int nx = clamp(cx + dx, 0, 15);
                        bf += s_pump_shape[ny * 16 + nx]
                            * float((2 - abs(dy)) * (2 - abs(dx)));
                    }
                bf *= 1.0 / 16.0;
                float finished = mix(s_pump_shape[i], bf, pump_finish_mix);
                pump_mask_cell[i] = max(s_pump_snap_f[i], finished);
            }
            #endif
            // ONSET takes the gated aggregate; the release path below keeps
            // the ungated drive_loc (sign + fall ratio), so reveal
            // suppression can never manufacture a release.
            float drive_loc = sign(loc_sum) * pow(abs(loc_sum) / N_SAMPLES, 1.0 / PUMP_DRIVE_P);
            float drive_on  = sign(loc_on_sum) * pow(abs(loc_on_sum) / N_SAMPLES, 1.0 / PUMP_DRIVE_P);
            drive_eff = max(drive_eff, drive_on);
            #endif
            float pump_gate  = smoothstep(PUMP_DRIVE_LOW, PUMP_DRIVE_HIGH, drive_eff);
            // Contrast-retention fade guard on the SAME V axis as the driver
            // (else a colored event would be driven but muted): high while the
            // frame keeps a hot core vs dark surround (explosion, spell),
            // collapses toward 0 as the field goes uniform (fade-to-white or
            // fade-to-colour). Events are never muted; fades ease out.
            float cover_raw = smoothstep(PUMP_CONTRAST_LOW, PUMP_CONTRAST_HIGH, contrast_v);
            // Asymmetric cover envelope (see PUMP_COVER_FALL): instant rise,
            // rate-clamped fall. pump_cover_gate doubles as the previous
            // frame's effective cover (thread-0 read-then-write, single
            // writer). transient_reset writes 0.0, so a cut still mutes
            // instantly and the post-cut re-open is an instant rise.
            float cover_gate = (cover_raw >= pump_cover_gate)
                ? cover_raw
                : max(cover_raw, pump_cover_gate * PUMP_COVER_FALL);
            // Published for PASS 6's ADDITIVE apply: the per-cell mask carries
            // no scene guard of its own, so the additive path multiplies this
            // in PASS 6 — the same enveloped value the scalar bakes into
            // pump_env (a genuine event holds contrast, so it never bites one).
            pump_cover_gate = cover_gate;
            // Velocity-matched release. The NEGATIVE half of the band-pass arms
            // release, but amplitude follows the source's TRUE frame-to-frame
            // fast-lane change (global_fall_ratio), not fast/slow separation.
            // These ratios telescope across a fade; they cannot charge the same
            // flicker deficit repeatedly. A steady source or rebound gives 1 →
            // no release. PUMP_ADAPT_FLOOR is the only clock left: it relaxes an
            // indefinitely-held light imperceptibly slowly (eye-adapting).
            float rel = global_fall_ratio;
            #if ENABLE_SPATIAL_PUMP
            // Local release — velocity-matched on the LOCAL axis. When the net
            // local drive is a fall (a dying fire), release at the falling
            // cells' own frame-to-frame fast ratio, |d|^p-weighted — the same
            // non-recompounding semantics as the global rel and mask r. (The
            // aggregate itself is not strictly telescoping because its faller
            // population and weights can change from frame to frame.)
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
            smoothed_top_frac     = 0.0;
            smoothed_contrast     = contrast;
            smoothed_log_avg      = log_avg;
            scene_cut_lockout     = 0.0;
        } else {
            // Spec-gate hardening (v5.6): the APPLIED spec gate gets its own
            // alpha that does NOT speed up with vel_mag. During a pan/tilt
            // vel_mag holds base_alpha at MID (~8-frame) for the whole move,
            // so the gate used to TRACK the residual tier jitter of textured
            // content (clustered speculars breathing scene-wide). Pan jitter
            // is zero-mean — a SLOW EMA reads its mean and the ripple dies
            // (~±1% vs ~±20%); cuts still lock on via the lockout alpha, and
            // a slower ease-in on genuine scene changes is the house
            // philosophy (eyes adjusting). smoothed_spec_natural deliberately
            // KEEPS the shared alpha: spec_vel (growth-mode driver) is
            // calibrated against it — do not slow that one. If a fireball's
            // spec bonus ever feels lagged, the lever is
            // mix(TEMPORAL_ALPHA_SLOW, base_alpha, smoothed_growth_mode).
            float alpha_spec = (scene_cut_lockout > 0.0) ? alpha : TEMPORAL_ALPHA_SLOW;
            smoothed_bright_frac  = mix(smoothed_bright_frac, bright_frac, alpha);
            smoothed_top_frac     = mix(smoothed_top_frac, top_frac, alpha);
            smoothed_spec_signal  = mix(smoothed_spec_signal, spec_raw, alpha_spec);
            smoothed_spec_natural = mix(smoothed_spec_natural, spec_raw_natural, alpha);
            smoothed_contrast     = mix(smoothed_contrast, contrast, alpha);
            smoothed_log_avg      = mix(smoothed_log_avg, log_avg, alpha);
        }

        // Dummy write satisfies the 1×1 SAVE target; SSBO above is the
        // real product. Other lanes are out-of-bounds for the image and
        // would be no-ops, but the guard avoids 143 redundant store ops.
        imageStore(out_image, ivec2(0), vec4(0));
    }
    // (The per-cell pump band-pass used to run here as a second phase on all
    // 144 lanes, behind a barrier + an s_pump_reset broadcast flag. Folded
    // into the thread-0 reducer 2026-07-02 — see the SUBTRACTIVE mask block
    // there. Same math, single writer, one barrier total in this pass.)
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
// the stats themselves arrive through the CELFLARE_ADD_STATE SSBO. Do not remove
// the bind without verifying pass ordering/visibility on every backend.

//!HOOK MAIN
//!BIND HOOKED
//!BIND CELFLARE_ADD_STATE
//!BIND CELFLARE_STATS
//!BIND CELFLARE_ILLUM
//!BIND MOTION_FLOW
//!DESC CelFlare Additive A2 (guarded additive default)

// =============================================
//  MAIN TUNING — deep anchors. The supported user surface is the cf_* block
//  at the top of the file; those knobs scale the values below (neutral at 1).
// =============================================
#define INTENSITY       cf_strength  // Global scaling knob (top of file). Also scales
                                     // spec + pump at their apply sites, so 0 = SDR.
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
// Bright-scene SHAPE control ("bright where it matters", author 2026-07-09):
// bright anime under the amplitude-dampened curve still ran the whole scene
// body 30-50% over SDR — sustained high APL that tires the eyes shot after
// shot. HDR is not about being bright; it is about being bright where it
// matters. Three pieces, all riding the illum-weighted cool_w (see
// COOL_ILLUM_LO/HI — bright FIELDS cool, faces/warm objects at mid illum
// keep the ship curve; dark and mid-key scenes at apl_t <= 0.5 are
// bit-identical to ship):
//  - SHAPE COOLING (this define): steepen the ramp gamma — the field's
//    body pulls back toward the SDR grade (expansion >= 1 always, so
//    "cooled" means closer to SDR, never below).
//  - FIELD LEVEL (APL_BRIGHT_COOL, STEP 4): amplitude pulldown so broad
//    near-white fields settle at ~170-185 nits — the calm level the
//    speculars and the light pump read against.
//  - TOP-BAND SPEND (APL_BRIGHT_RELAX, STEP 4, default 0): optional
//    amplitude return gated on top-band presence (relax_w = cool_w x the
//    smoothed_top_frac gate below). With the field pulled down, separation
//    comes from the sunken field against the held top — contrast, not
//    more light.
#define GAMMA_APL_BOOST 1.8     // local_gamma multiplier at cool_w=1
// Top-band presence gate for the RELAX only. The analytic body/top crossover
// sits at Y~0.82: HIGHLIGHT_THRESH 0.75 fires on golden-hour faces (the
// class that should stay cooled), SPECULAR 0.92 misses soft near-clip skies,
// so PASS 5 counts a dedicated tier at 0.85 (smoothed_top_frac). LO..HI in
// 16x9-cell picture fraction: ~3 cells opens, ~11 cells (8%) fully open.
#define TOP_FRAC_LO     0.02
#define TOP_FRAC_HI     0.08
// Illumination-field weight on the cooling: cool the FIELDS, not the faces.
// The scene-key cooling alone manufactures a lightness-ratio percept on
// mid-luminance warm content — shaded skin's own level drops while its
// surround gains, and orange at lowered relative luminance reads BROWN
// (Bezold-Brucke and simultaneous-contrast territory; the expansion itself
// is chromaticity-preserving, so this is purely a luminance-relationship
// effect and only the curve can own it). The APL fatigue the cooling
// targets lives in broad bright fields (sky, sea glare, white walls) —
// high Y_illum; the psychovisual victims (faces, warm objects) sit at mid
// Y_illum. Weighting the WHOLE reshape pair by the sigma-80 field keeps
// mid-luminance regions on the exact ship curve (their SDR-grade
// impression anchored) while the fields still cool and separate. Same
// spatial-modulation channel the curve already uses: smooth field, both
// endpoint curves monotone, so the per-pixel mix stays contour-free.
#define COOL_ILLUM_LO   0.55    // Y_illum below: reshape fully off (ship curve)
#define COOL_ILLUM_HI   0.80    // Y_illum above: full cooling + top spend

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
// from stabilized Y toward stabilized V on the ramp input. This survives
// where the spec path's V escape (v_drive — field-rejected twice, deleted
// 2026-07-02, see the spec block) did not, because its exposure is a
// fraction of that one's: the base ramp's slope is ~10x gentler than the
// spec ramp and the credit halves the coupling, so WEB-grade 4:2:0 chroma
// noise in V works out to ~1-2 nits of wobble (vs the full-ramp speckle and
// gradient crush that killed V spec drivers), and the bounded mix cannot go
// flat where V clips. Near-neutrals are bit-identical (sat gate = 0).
// The Y floor fade keeps the EARLY_EXIT_GAMMA boundary conservative-EXACT:
// at/below BASE_V_Y_LO the credit is 0, so any pixel the early exit skips
// would have computed expansion = 1.0 anyway — no contour at the boundary.
// Consequence: dim saturated emissives (red LED Y~0.21) get no BASE credit
// and — with the spec V escape gone — no per-pixel V path at all: they keep
// their SDR level BY DESIGN (the twice-field-rejected trade).
#define ENABLE_BASE_V_CREDIT 1
#define BASE_V_CREDIT        0.75   // fraction of the Y->V gap credited at full gate
#define BASE_V_SAT_LO        0.10   // sat_gamma gate (the band the deleted spec v_drive used)
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
// Bright-field level ("150-180 nits is fine for bright scenes", author
// 2026-07-09): on top of the GAMMA_APL_BOOST shape cooling, bright FIELDS
// get an amplitude pull on the illum-weighted cool_w so broad near-white
// areas settle at ~170-185 nits instead of ship's ~200+ — the calm level
// the speculars (cf_spec, unchanged) and the light pump read AGAINST.
// Effective bright-field endpoint 0.65 - 0.20 = 0.45. Rides cool_w — NOT a
// raw APL_DAMPEN_BRIGHT retune, which would leak ~-10% into mid-key scenes
// through the linear apl_t mix (measured, round one of this design) — and
// cool_w carries the Y_illum weight, so faces/warm objects at mid illum
// keep ship amplitude (the brown-skin guard applies to this pulldown too).
#define APL_BRIGHT_COOL     0.20    // apl_factor pulldown at cool_w=1
// Top-band spend: optional amplitude RETURN on relax_w (cooling weight x
// top-band presence gate) for bright scenes with a real top band. Default 0
// under the 150-180 field target: separation comes from the sunken field
// against the held top — contrast, not more light — and sparse speculars
// ride above via the spec path. Raise (e.g. 0.05-0.10) to let top-banded
// bright scenes carry a slightly hotter field than topless ones. At 0.20
// it fully cancels APL_BRIGHT_COOL where the gate opens (measured +2.4%
// FALL over ship on blazing-beach content — the fatigue goal inverted).
#define APL_BRIGHT_RELAX    0.0     // apl_factor addback at relax_w=1
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
#define ENABLE_LIGHT_PUMP   cf_light_pump   // top-of-file toggle
#define PUMP_STRENGTH       0.6      // gain per unit pump_env at full pixel weight. At = CEIL the response is
                                   // PROPORTIONAL (peak reserved for full-detection events, not roof-pinned);
                                   // > CEIL slams moderate events to the roof (aggressive); subtle ≈ 0.4
#define PUMP_Y_LOW          0.35   // per-pixel weight onset — low/broad so the whole bright region lifts (not a pinpoint)
#define PUMP_GAIN_CEIL      1.5    // hard cap on the pump multiplier (safety against runaway expansion)
#define PUMP_GROWTH_DAMP    0.6    // down-gate pump where growth-mode already lifts expansion (anti double-stack on fireballs)
// Spatial pump. Single-sourced from the top-of-file cf_spatial_pump toggle —
// PASS 5 aliases the SAME param, so the historical PASS5/PASS6 duplicate-define
// desync (garbage mask) can no longer happen. Don't replace with a literal.
// PASS 5 produces the per-cell brightening mask (pump_mask_cell[144], 16×9 —
// the softened presentation of pump_env_cell); this pass bilinear-samples it.
// 0 = scalar-only.
#define ENABLE_SPATIAL_PUMP cf_spatial_pump
// Apply mode (PASS 6-only knob — PASS 5 needs no copy). 1 = ADDITIVE (v5.5
// experiment, the HANDOFF §13 "additive door"): pump_local = mask ×
// pump_cover_gate — the bilinear per-cell env IS the local pump amplitude, so
// each region pumps at its own strength and rhythm and a localized event no
// longer needs the frame statistics to fire (small-event amplitude back).
// In this A2 track PASS 5 additionally requires a seven-frame local emission
// proof across both the raw-source and pump-domain motion routes. The
// established-level and frame-edge rules remain the first opening authority.
// 0 = SUBTRACTIVE (v5.2–v5.4 shipping behavior): pump_local = pump_env × mask
// — the mask only suppresses the scalar; instant fallback if the experiment
// misbehaves in the field.
// Amplitude note: the additive path saturates at PUMP_CELL_DRIVE_HIGH (0.15,
// PASS 5) per cell vs the scalar's PUMP_DRIVE_HIGH (0.20), so moderate events
// run a touch hotter than v5.4 — that PASS 5 knob is the amplitude-reserve
// lever. Mask fractional coords are smoothstep-eased (C1) to kill bilinear
// seams.
// v5.17 (2026-07-12): the first transport experiment reverted to subtractive
// after field reports on camera motion (Judas Overlord E05: a spot lit up on a
// talking face during a
// tilt-up; the bar-lamp scene flickered on a pan). The additive door removed
// the scalar's two motion-safety properties — (a) requiring net-new GLOBAL
// light, (b) temporal smoothing via pump_env — and a bright feature (facial
// specular, a lamp) TRANSLATING across cells under a moving camera reads
// "fresh" in each new cell (out-brightens its darker neighbours), which the
// established-level gate can't reject without motion vectors. Subtractive
// restores both properties: an in-frame highlight self-cancels the scalar
// (pump_env≈0 → no pump), and the revealed-lamp pump is temporally smoothed
// (measured: p999 peak +145→+45 nits over base, frame-to-frame flicker ~halved).
// The cost was the per-cell additive amplitude on genuine localized dark-scene
// events (multi-fire). A2 is the separate response: pump-domain motion veto,
// longer-baseline establishment, and persisted multi-frame proof. It defaults to
// additive after synthetic and paired design/compute review; 0 remains the
// verified subtractive fallback.
#define SPATIAL_PUMP_ADDITIVE cf_additive_pump
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
#define ENABLE_SPECULAR_BONUS cf_spec_bonus   // top-of-file toggle
// (ENABLE_SATURATED_SPEC is PASS 5-only now. The per-pixel V escape it once
// gated HERE was field-rejected twice and DELETED 2026-07-02: restoring it
// crushed saturated speculars — in a saturated region the peak channel clips
// before luma, so a V driver feeds the ramp a flat near-1.0 signal across a
// core that still has luma gradient; the uniform spec add compresses the
// gradient (rule-1 direction). PASS 5 keeps counting tiers on V — every
// scene-gate tuning since f453fc4 was validated against that. History:
// HANDOFF §12 + git. Two general lessons from this define's life, kept for
// the next cross-pass feature: each HOOK block is a SEPARATE compilation
// unit, so a define tested in two passes must be duplicated in both — and an
// undefined identifier inside #if silently evaluates to 0, which is how this
// path once compiled out for three versions without an error. Also: do not
// write the literal directive prefix in prose anywhere in this file — the
// libplacebo parser splits sections on it even mid-comment.)
#define SPEC_Y_LOW          0.80    // Ramp onset — widened to soften "crunchy" clipped speculars.
                                    // Builds the spec as a gradient INTO the clipped core from
                                    // below; peak at Y=1.0 is unchanged, so no attenuation of the
                                    // source clip. Lower = wider/gentler phase-in (floor ~0.75
                                    // before spec starts catching bright-but-not-specular pixels).
#define SPEC_Y_LOW_MID_BUMP 0.05    // Parabolic bump pushes onset to 0.93 at apl_t=0.5,
                                    // which corresponds to smoothed_log_avg ≈ 0.16 —
                                    // normally-lit interiors, dusk exteriors, mid-key
                                    // cinematic lighting. Adds selectivity in scenes
                                    // with abundant mid-bright surfaces (lampshades,
                                    // faces, fabrics, TV screens) by requiring V > 0.93
                                    // before spec fires. Endpoints (dark/bright APL)
                                    // are unaffected — bump is zero at apl_t=0 and 1.
#define SPEC_PEAK_DARK      1.2     // Specular boost in dark scenes (highlight pop)
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
// (SPEC_APL_LOW/HIGH deleted 2026-07-02 — the pair was numerically identical
// to APL_KEY_DARK/BRIGHT since introduction and never diverged. The spec
// params now read the shared apl_t scene axis; retune via APL_KEY_DARK/BRIGHT,
// or re-split deliberately if specular ever needs its own APL window.)
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

// (Clip diffusion DELETED 2026-07-02. Zeroed since v4.4 — its one field result
// was darkening small light cores against a dark illum field, i.e. attenuating
// source clipping / inverting gradients: a structural rule-1 violation no
// tuning can fix. History in git if it's ever reconsidered.)

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
#define ENABLE_WARM_SHIFT    cf_warm_shift   // top-of-file toggle
#define WS_HUE_COS          0.3420  // cos(70°) — center of warm range in Oklab
#define WS_HUE_SIN          0.9397  // sin(70°)
#define WS_HUE_POWER        1.2     // Hue window width (lower = wider, ~50° each side)
#define WS_STRENGTH          0.06   // Max rotation in radians (~3.4° at full drive)
#define WS_ILLUM_LOW         0.35   // Y_illum below: no shift (dark region)
#define WS_ILLUM_HIGH        0.80   // Y_illum above: full shift
#define WS_CHROMA_FLOOR      0.015  // Skip near-neutrals (unstable hue)

#define ENABLE_PALE_SKIN    cf_pale_skin   // top-of-file toggle
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
// Skin lift — Hunt-effect compensation for the v5.16 field cooling. With
// bright FIELDS pulled to ~170-185 nits while skin holds its (elevated)
// level, the eye adapts to a dimmer surround: the same skin luminance
// reads brighter AND more colorful, and more-colorful skin reads more TAN
// vs the SDR grade (author observation, direct-A/B-only magnitude). A
// small real lift toward pale is the appearance counter — same
// compensation class as the warm-shift hue rotation. Rides the SCENE
// cooling weight (smoothstep of apl_t), NOT the pixel cool_w: skin is
// illum-exempted from cooling, so its own cool_w is ~0 by design — the
// compensation keys on "this scene's fields are cooled". Chroma window is
// WIDER than the pale-skin sat boost's (tan skin carries more chroma than
// pale); hue window and bright/scene gates are shared with PS. Zero at
// apl_t <= 0.5 — dark/mid-key stay bit-exact.
#define PS_LIFT             0.10    // linear-light lift at full gate (~3.2% Oklab L)
#define PS_LIFT_CHROMA_HI   0.09    // lift chroma falloff start (pale band ends ~0.07)
#define PS_LIFT_CHROMA_CEIL 0.14    // lift fully off — deep-saturated warm colors excluded

// =============================================
//  OUTPUT — encoding
// =============================================
#define REFERENCE_WHITE cf_ref_white   // top-of-file knob — match hdr-reference-white
#define PQ_FAST_APPROX  1
#define EOTF_GAMMA      2.4
#define ENABLE_GRAIN_STABLE cf_grain_stab   // top-of-file toggle
// Early-exit luma bound. == KNEE is exact for the base curve: PASS 1 writes
// RAW luma (alpha encode) below its GRAIN_EARLY_EXIT (0.30), so for
// Y_gamma < KNEE the decision luma equals Y_gamma, t = 0, and expansion is
// exactly 1.0 through dynamic/APL (both scale expansion-1). Must stay
// <= PASS 1's GRAIN_EARLY_EXIT or stabilized decisions could cross KNEE.
#define EARLY_EXIT_GAMMA    KNEE

// =============================================
//  DEBUG
// =============================================
// All views are driven by the single top-of-file cf_debug selector (0 = off),
// so they're switchable from mpv.conf / a keybind without editing this file.
#define DEBUG_BYPASS         (cf_debug == 1)
#define DEBUG_SHOW_ILLUM     (cf_debug == 2)   // Illumination field as grayscale
#define DEBUG_SHOW_EXPANSION (cf_debug == 3)   // Expansion amount as heat map
#define DEBUG_SHOW_DETAIL    (cf_debug == 4)   // Spatial vs per-pixel: green=spatial, red=per-pixel fallback
#define DEBUG_SHOW_SPECULAR  (cf_debug == 5)   // Specular bonus: cyan = spec strength
#define DEBUG_SHOW_PUMP      (cf_debug == 6)   // Light pump: red = scene pump_env, green = per-pixel applied gain
#define DEBUG_SHOW_WP        (cf_debug == 7)   // Warm shift + pale skin
#define DEBUG_SHOW_STATS     (cf_debug == 8)   // avg_illum + bright_frac + contrast + log_avg bars

// ---------------------------------------------------------------------------
// Debug legend overlay — title + color key, bottom-left panel
// ---------------------------------------------------------------------------
// Every active debug view draws a small self-describing panel: a title line
// ("6 LIGHT PUMP") plus one swatch+label row per channel, so a screenshot or
// a mid-session eyeball needs no trip back to the cf_debug DESC. Swatches
// repeat the exact colors the view emits. Feature-gated views (specular,
// pump, warm/skin) advertise DISABLED when their toggle is off — the view
// body falls through to the production render in that case and the panel is
// the only tell. The whole overlay, font included, is preprocessed out at
// cf_debug == 0; production pays nothing.
#if cf_debug != 0

// 5x6 bitmap font. Bit index = y*5 + x (x=0 left, y=0 top), bit set = pixel
// on. Glyph order: A-Z, 0-9, then - . / = ( ). Generated and round-trip
// verified by dev/gen-debug-font.py (gitignored) — regenerate there, don't
// hand-edit hex.
const uint DBG_FONT[42] = uint[42](
    0x231fc62eu, 0x1f18be2fu, 0x3c10843eu, 0x1f18c62fu, 0x3e10bc3fu, 0x0210bc3fu,   // A B C D E F
    0x3d18e43eu, 0x2318fe31u, 0x3e42109fu, 0x1d184210u, 0x23149d31u, 0x3e108421u,   // G H I J K L
    0x2318d771u, 0x231cd671u, 0x1d18c62eu, 0x0210be2fu, 0x2c9ac62eu, 0x2314be2fu,   // M N O P Q R
    0x1f08383eu, 0x0842109fu, 0x1d18c631u, 0x08a8c631u, 0x23bac631u, 0x22a21151u,   // S T U V W X
    0x08421151u, 0x3e11111fu, 0x1d19d72eu, 0x3e4210c4u, 0x3e22222eu, 0x1f08320fu,   // Y Z 0 1 2 3
    0x108fa988u, 0x1f083c3fu, 0x1d18bc2eu, 0x0842221fu, 0x1d18ba2eu, 0x1d087a2eu,   // 4 5 6 7 8 9
    0x00007c00u, 0x08400000u, 0x02221110u, 0x000f83e0u, 0x08210844u, 0x08842104u    // - . / = ( )
);

// Labels are packed 6 bits/char, 5 chars per uint, LSB first — one uvec4
// holds up to 20 chars. Char codes: 0 = space, 1-26 = A-Z, 27-36 = 0-9,
// 37-42 = - . / = ( ). The plain-text string rides in a comment beside each
// constant; the generator script emits both.
//
// Coverage of one text pixel at glyph-space p (one unit = one font pixel;
// caller divides by its pixel scale AFTER clamping negatives out — GLSL int
// division truncates toward zero, so -1/sc would alias onto column 0).
float dbg_line(ivec2 p, uvec4 txt, int len) {
    if (p.x < 0 || p.y < 0 || p.y >= 6) return 0.0;
    int ci = p.x / 6;                    // 5px glyph + 1px advance
    int gx = p.x - ci * 6;
    if (ci >= len || gx > 4) return 0.0;
    uint ch = (txt[ci / 5] >> uint((ci % 5) * 6)) & 63u;
    if (ch == 0u || ch > 42u) return 0.0;
    return float((DBG_FONT[ch - 1u] >> uint(p.y * 5 + gx)) & 1u);
}

// Per-view panel content. DBG_TITLE_CH / DBG_NROWS / DBG_ROW_MAXCH size the
// panel; dbg_row() below supplies swatch color + label per key row.
#if DEBUG_BYPASS
    #define DBG_TITLE     uvec4(0x1064201cu, 0x000134c1u, 0u, 0u)               // 1 BYPASS
    #define DBG_TITLE_CH  8
    #define DBG_NROWS     0
    #define DBG_ROW_MAXCH 0
#elif DEBUG_SHOW_ILLUM
    #define DBG_TITLE     uvec4(0x0c30901du, 0x09180355u, 0x00004305u, 0u)      // 2 ILLUM FIELD
    #define DBG_TITLE_CH  13
    #define DBG_NROWS     1
    #define DBG_ROW_MAXCH 19
#elif DEBUG_SHOW_EXPANSION
    #define DBG_TITLE     uvec4(0x1060501eu, 0x0f253381u, 0x0000000eu, 0u)      // 3 EXPANSION
    #define DBG_TITLE_CH  11
    #define DBG_NROWS     1
    #define DBG_ROW_MAXCH 19
#elif DEBUG_SHOW_DETAIL
    #define DBG_TITLE     uvec4(0x1414401fu, 0x0000c241u, 0u, 0u)               // 4 DETAIL
    #define DBG_TITLE_CH  8
    #define DBG_NROWS     1
    #define DBG_ROW_MAXCH 16
#elif DEBUG_SHOW_SPECULAR
    #define DBG_TITLE     uvec4(0x05413020u, 0x1204c543u, 0u, 0u)               // 5 SPECULAR
    #define DBG_TITLE_CH  10
    #define DBG_NROWS     1
    #define DBG_ROW_MAXCH 18
#elif DEBUG_SHOW_PUMP
    #define DBG_TITLE     uvec4(0x0724c021u, 0x15400508u, 0x0000040du, 0u)      // 6 LIGHT PUMP
    #define DBG_TITLE_CH  12
    #if !ENABLE_LIGHT_PUMP
        #define DBG_NROWS     1
        #define DBG_ROW_MAXCH 8
    #elif ENABLE_SPATIAL_PUMP
        #define DBG_NROWS     3
        #define DBG_ROW_MAXCH 14
    #else
        #define DBG_NROWS     2
        #define DBG_ROW_MAXCH 14
    #endif
#elif DEBUG_SHOW_WP
    #define DBG_TITLE     uvec4(0x12057022u, 0x092d39cdu, 0x0000000eu, 0u)      // 7 WARM/SKIN
    #define DBG_TITLE_CH  11
    #if !ENABLE_WARM_SHIFT && !ENABLE_PALE_SKIN
        #define DBG_NROWS     1
        #define DBG_ROW_MAXCH 8
    #else
        #define DBG_NROWS     (ENABLE_PALE_SKIN + ENABLE_WARM_SHIFT)
        #define DBG_ROW_MAXCH 16
    #endif
#else  // DEBUG_SHOW_STATS
    #define DBG_TITLE     uvec4(0x01513023u, 0x000004d4u, 0u, 0u)               // 8 STATS
    #define DBG_TITLE_CH  7
    #define DBG_NROWS     4
    #define DBG_ROW_MAXCH 11
#endif

// Swatch color + packed label for key row i of the active view. Swatch
// values repeat what the view actually writes (channel primaries, the stats
// bar colors), so the key doubles as a sanity check on the view itself.
void dbg_row(int i, out vec3 col, out uvec4 txt, out int len) {
    col = vec3(0.4); txt = uvec4(0u); len = 0;
#if DEBUG_SHOW_ILLUM
    if (i == 0) { col = vec3(0.7);
        txt = uvec4(0x28641487u, 0x0d54c309u, 0x0d1c94c0u, 0x006e3001u); len = 19; } // GRAY=ILLUM SIGMA 80
#elif DEBUG_SHOW_EXPANSION
    if (i == 0) { col = vec3(1.0, 0.3, 0.0);
        txt = uvec4(0x18169a12u, 0x094ce050u, 0x2a72538fu, 0x00826767u); len = 19; } // R=(EXPANSION-1)/2.5
#elif DEBUG_SHOW_DETAIL
    if (i == 0) { col = vec3(0.0, 1.0, 0.0);
        txt = uvec4(0x010a9a07u, 0x18140153u, 0x18a9c950u, 0x0000001du); len = 16; } // G=(BASE EXP-1)X2
#elif DEBUG_SHOW_SPECULAR
    #if ENABLE_SPECULAR_BONUS
    if (i == 0) { col = vec3(0.0, 1.0, 1.0);
        txt = uvec4(0x28381643u, 0x000c5413u, 0x0e152513u, 0x00008507u); len = 18; } // CYAN=SPEC STRENGTH
    #else
    if (i == 0) { txt = uvec4(0x02053244u, 0x0000414cu, 0u, 0u); len = 8; }          // DISABLED
    #endif
#elif DEBUG_SHOW_PUMP
    #if ENABLE_LIGHT_PUMP
    if (i == 0) { col = vec3(1.0, 0.0, 0.0);
        txt = uvec4(0x010d3a12u, 0x0501204cu, 0x0000058eu, 0u); len = 12; }          // R=SCALAR ENV
    else if (i == 1) { col = vec3(0.0, 1.0, 0.0);
        txt = uvec4(0x10401a07u, 0x0010524cu, 0x00389047u, 0u); len = 14; }          // G=APPLIED GAIN
    #if ENABLE_SPATIAL_PUMP
    else if (i == 2) { col = vec3(0.0, 0.0, 1.0);
        txt = uvec4(0x0c143a02u, 0x1304d00cu, 0x0000000bu, 0u); len = 11; }          // B=CELL MASK
    #endif
    #else
    if (i == 0) { txt = uvec4(0x02053244u, 0x0000414cu, 0u, 0u); len = 8; }          // DISABLED
    #endif
#elif DEBUG_SHOW_WP
    int r = i;
    #if ENABLE_PALE_SKIN
    if (r == 0) { col = vec3(1.0, 0.0, 0.0);
        txt = uvec4(0x0c050a12u, 0x092d3005u, 0x1b71800eu, 0u); len = 15; return; }  // R=PALE SKIN X10
    r--;
    #endif
    #if ENABLE_WARM_SHIFT
    if (r == 0) { col = vec3(0.0, 1.0, 0.0);
        txt = uvec4(0x12057a07u, 0x0921300du, 0x20600506u, 0x0000001bu); len = 16; } // G=WARM SHIFT X50
    #endif
    #if !ENABLE_PALE_SKIN && !ENABLE_WARM_SHIFT
    if (r == 0) { txt = uvec4(0x02053244u, 0x0000414cu, 0u, 0u); len = 8; }          // DISABLED
    #endif
#elif DEBUG_SHOW_STATS
    if (i == 0) { col = vec3(0.6, 0.6, 0.0);
        txt = uvec4(0x081c9482u, 0x01486014u, 0x00000003u, 0u); len = 11; }          // BRIGHT FRAC
    else if (i == 1) { col = vec3(0.7, 0.4, 0.0);
        txt = uvec4(0x1250e3c3u, 0x239d44c1u, 0u, 0u); len = 10; }                   // CONTRAST/8
    else if (i == 2) { col = vec3(0.0, 0.6, 0.0);
        txt = uvec4(0x010073ccu, 0x000001d6u, 0u, 0u); len = 7; }                    // LOG AVG
    else if (i == 3) { col = vec3(0.0, 0.6, 0.6);
        txt = uvec4(0x000c5413u, 0x01387253u, 0x0000000cu, 0u); len = 11; }          // SPEC SIGNAL
#endif
}

#endif  // cf_debug != 0

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
// The illumination field is a sigma~100px Gaussian — extremely smooth, so
// hardware bilinear (1 fetch) is sufficient. (A C2 cubic B-spline variant
// lived behind BSPLINE_UPSAMPLE from v4.0 but was never once enabled —
// deleted 2026-07-02; it's in git if gradients ever visibly kink.)

vec3 upsample_illum_rgb() {
    return CELFLARE_ILLUM_tex(CELFLARE_ILLUM_pos).rgb;
}

// =============================================================================
// MAIN PROCESSING
// =============================================================================

vec4 hook() {
#if cf_debug == 9
    // Motion-flow debug view: the block-match field (MOTION_FLOW, 16x9)
    // bilinear-upsampled. Grey (0.5,0.5) = still; red/green encode +x/+y
    // PREVIOUS-FRAME SOURCE OFFSET (where current content came from), so a
    // right/down image move is negative. Blue = |offset| / MOT_R. A coherent
    // pan shows a uniform tint; a broken block-match shows speckle.
    vec2 mf = MOTION_FLOW_tex(HOOKED_pos).xy;
    return vec4(clamp(0.5 + mf.x * 0.1, 0.0, 1.0),
                clamp(0.5 + mf.y * 0.1, 0.0, 1.0),
                clamp(length(mf) * (1.0 / 5.0), 0.0, 1.0), 1.0);
#endif
#if cf_debug == 10
    // Motion evidence view. Red = mean winning SAD (0.10 reaches full red),
    // green = current-tile RMS contrast (0.05 reaches full green), blue = the
    // research coherent-veto trust derived from robust block-flow consensus.
    ivec2 mc = clamp(ivec2(HOOKED_pos * vec2(16.0, 9.0)),
                      ivec2(0), ivec2(15, 8));
    vec2 mpos = (vec2(mc) + 0.5) / vec2(16.0, 9.0);
    vec4 md = MOTION_FLOW_tex(mpos);
    float trust = motion_trust_cell[mc.y * 16 + mc.x];
    return vec4(clamp(abs(md.z) * 10.0, 0.0, 1.0),
                clamp(md.w * 20.0, 0.0, 1.0),
                clamp(trust, 0.0, 1.0), 1.0);
#endif
#if cf_debug == 11
    // Residual-contract view, nearest-cell throughout: red is the mc_emit used
    // by today's local block-flow warp; green is the same residual after an
    // unreliable local tile borrows the supported dominant prev-offset; blue
    // is the research router's prospective coherent-flow trust.
    ivec2 mc = clamp(ivec2(HOOKED_pos * vec2(16.0, 9.0)),
                      ivec2(0), ivec2(15, 8));
    int mi = mc.y * 16 + mc.x;
    return vec4(clamp(motion_mc_local_cell[mi], 0.0, 1.0),
                clamp(motion_mc_effective_cell[mi], 0.0, 1.0),
                clamp(motion_trust_cell[mi], 0.0, 1.0), 1.0);
#endif
#if cf_debug == 12
    // Additive opening proof, nearest-cell: red = established fast level above
    // slow+very-slow ring memory; green = motion-unexplained fast-rise ratio;
    // blue = seven-frame carried persistence. White is eligible to open.
    ivec2 mc = clamp(ivec2(HOOKED_pos * vec2(16.0, 9.0)),
                      ivec2(0), ivec2(15, 8));
    int mi = mc.y * 16 + mc.x;
    return vec4(clamp(motion_trust_cell[mi], 0.0, 1.0),
                clamp(motion_mc_local_cell[mi], 0.0, 1.0),
                clamp(motion_mc_effective_cell[mi], 0.0, 1.0), 1.0);
#endif
    vec4 color = HOOKED_texOff(0);
    vec3 rgb_gamma = color.rgb;

    // -------------------------------------------------------------------------
    // DEBUG: legend panel (bottom-left) — title + color key for the active view
    // -------------------------------------------------------------------------
    // Drawn before every debug return path (bypass included) so each view is
    // labeled. Opaque panel, early return — nothing downstream sees it.
    #if cf_debug != 0
    {
        int sc = max(1, int(HOOKED_size.y) / 540);   // 12px text at 1080p, 24px at 4K
        int pad = 4 * sc;
        int lh  = 8 * sc;                            // line advance: 6px glyph + 2 gap
        int key_w = DBG_NROWS > 0 ? 8 * sc : 0;      // swatch column incl. gap
        int box_w = 2 * pad + max(DBG_TITLE_CH * 6 * sc, key_w + DBG_ROW_MAXCH * 6 * sc);
        int box_h = 2 * pad + lh * (1 + DBG_NROWS);
        ivec2 q = ivec2(HOOKED_pos * HOOKED_size)
                - ivec2(2 * pad, int(HOOKED_size.y) - 2 * pad - box_h);
        if (q.x >= 0 && q.y >= 0 && q.x < box_w && q.y < box_h) {
            vec3 leg = vec3(0.04);                   // panel ground
            int line = (q.y - pad) / lh;             // 0 = title, 1.. = key rows
            ivec2 tp = ivec2(q.x - pad, (q.y - pad) - line * lh - sc);
            if (q.y >= pad && tp.x >= 0 && tp.y >= 0) {
                if (line == 0) {
                    leg = mix(leg, vec3(0.85), dbg_line(tp / sc, DBG_TITLE, DBG_TITLE_CH));
                } else if (line <= DBG_NROWS) {
                    vec3 rc; uvec4 rt; int rl;
                    dbg_row(line - 1, rc, rt, rl);
                    if (tp.x < 6 * sc && tp.y < 6 * sc) leg = rc;   // swatch block
                    else if (tp.x >= 8 * sc)
                        leg = mix(leg, vec3(0.85),
                                  dbg_line(ivec2(tp.x - 8 * sc, tp.y) / sc, rt, rl));
                }
            }
            return vec4(gamma709_to_pq2020(leg), 1.0);
        }
    }
    #endif

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
    // V_gamma (peak channel): feeds the spec block's emissive carve, the
    // stabilized V for the base-ramp credit, and sat_gamma below.
    // sat_gamma (V − min): gamma-space saturation. Reused by the spec
    // sat gate and as the gate for the Oklab fast path on near-neutrals.
    float V_gamma   = max(max(rgb_gamma.r, rgb_gamma.g), rgb_gamma.b);
    float min_gamma = min(min(rgb_gamma.r, rgb_gamma.g), rgb_gamma.b);
    float sat_gamma = V_gamma - min_gamma;

    // -------------------------------------------------------------------------
    // ILLUMINATION FIELD
    // -------------------------------------------------------------------------
    // Production fetch sits AFTER the early exit so dark pixels skip it; the
    // debug view needs it for every pixel and is self-contained here.
    #if DEBUG_SHOW_ILLUM
    return vec4(gamma709_to_pq2020(upsample_illum_rgb()), 1.0);
    #endif

    // Early exit: no base expansion possible below the knee. (A V-aware
    // variant that kept dim saturated emissives alive for the spec V escape
    // was deleted with the escape — see the spec block.)
    if (Y_gamma < EARLY_EXIT_GAMMA) return vec4(gamma709_to_pq2020(color.rgb), 1.0);

    vec3 illum_rgb = upsample_illum_rgb();
    float Y_illum = get_luma(illum_rgb);

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

    // Grain-stabilized peak channel: the luma stabilizer's own measured
    // correction transplanted onto V (cancels achromatic grain exactly;
    // residual chroma noise is unfixable here — PASS 1 has no chroma
    // decision). Sole consumer since the spec V escape was deleted
    // (2026-07-02): the base-ramp V credit below.
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
    // Scene APL axis (0 = dark key, 1 = bright key) — hoisted above the curve
    // (was declared at STEP 4) so the shape can read it: shared by the gamma
    // boost here, the APL modulation at STEP 4, and the specular-bonus params.
    // ONE axis BY DESIGN (merged 2026-07-02, see STEP 4). bright_w is the
    // shaped-not-dampened weight: OFF through mid-key, phasing in only for
    // genuinely bright scene keys (both smoothsteps of the temporally
    // smoothed log_avg, so shape transitions inherit the EMA's smoothness).
    float apl_t = smoothstep(APL_KEY_DARK, APL_KEY_BRIGHT, smoothed_log_avg);
    // cool_w — bright-key scenes cool toward the SDR grade, weighted by the
    // illumination field so the cooling lands on broad bright FIELDS while
    // mid-luminance regions (faces, warm objects) keep the ship curve — see
    // COOL_ILLUM_LO/HI for the psychovisual rationale. relax_w — amplitude
    // returns only where a top band exists to spend it on. relax_w derives
    // from the weighted cool_w ON PURPOSE: the pair must move as a unit
    // per-pixel (relax without the gamma boost is a bare +31% amplitude
    // lift — exactly the uniform-lift class the design audit rejected).
    float cool_w  = smoothstep(0.5, 1.0, apl_t)
                  * smoothstep(COOL_ILLUM_LO, COOL_ILLUM_HI, Y_illum);
    float relax_w = cool_w * smoothstep(TOP_FRAC_LO, TOP_FRAC_HI, smoothed_top_frac);
    // Growth-mode bypass on the SHAPE, mirroring the amplitude bypass at
    // STEP 4: an expanding bright event should restore the full pre-reshape
    // curve (bloom through the midtones), not just its amplitude — without
    // this the steepened gamma would keep holding the event's mid-rim down
    // at full growth_mode (audit-caught half-bypass).
    #if ENABLE_GROWTH_BYPASS
    float bw_gamma = cool_w * (1.0 - GROWTH_APL_BYPASS * smoothed_growth_mode);
    #else
    float bw_gamma = cool_w;
    #endif
    // cf_curve (top-of-file knob) scales the exponent: <1 broadens the ramp
    // (gentle lift deeper into the highlights), >1 concentrates it against
    // Y=1 (harsher pop). Peak is invariant (t=1 → pow=1 for any gamma). The
    // max(1.0) floor preserves the monotonically-increasing-derivative
    // invariant the curve comment above relies on — never let gamma < 1.
    // GAMMA_APL_BOOST steepens the exponent in bright scenes (shaped-not-
    // dampened, see the define block): multiplicative on the same exponent,
    // so the >= 1 floor and the monotone-derivative proof carry over.
    float local_gamma = max(1.0, mix(GAMMA_DARK, GAMMA_BRIGHT, spatial_t) * cf_curve
                                 * mix(1.0, GAMMA_APL_BOOST, bw_gamma));

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
    // cf_shoulder (top-of-file knob): one-sided cubic shoulder bump — softens
    // the TOP-END derivative for sources whose highlights are already
    // harsh/hard-clipped. Slope at t=1 scales as (1 - cf_shoulder): 0 =
    // shipped look (steepest near-clip differentiation), 1 = expansion
    // arrives at peak flat (clipped regions read uniformly ~peak× brighter;
    // source gradation preserved at amplitude scale, not exaggerated). The cubic t²(1-t) — NOT the symmetric
    // SPEC_Y_LOW_MID_BUMP parabola — is deliberate: its perturbation peaks at
    // t=2/3 (Y≈0.81-0.88) and has quadratic contact at the knee, so faces are
    // untouched and the composite curve's inflection stays at Y≥0.82 at
    // default gamma (worst corner s=1 + cf_curve=0.6: Y≈0.66, band edge; the
    // symmetric bump planted it at Y≈0.47, mid-face — audit-caught).
    // Monotonicity: d/dt [t + s·t²·(1-t)] = 1 + s(2t - 3t²) >= 1 - s >= 0 —
    // the PARAM's MAXIMUM 1.0 IS the proof bound, don't widen it. Peak is
    // invariant (bump is 0 at t=1). The multiplier stays monotonic at every
    // setting, so rule #1 holds: output gradient never drops below
    // peak × source gradient.
    t += cf_shoulder * t * t * (1.0 - t);
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
    // apl_t (the shared scene APL axis) is declared above STEP 2 — the
    // shaped-not-dampened gamma boost reads it first. ONE axis BY DESIGN
    // (merged 2026-07-02): the former SPEC_APL_LOW/HIGH knob pair was
    // numerically identical to APL_KEY_DARK/BRIGHT and never diverged.
    #if ENABLE_APL_MOD
    {
        float apl_factor = mix(APL_BOOST_DARK, APL_DAMPEN_BRIGHT, apl_t);
        // Mid-scene notch: parabolic dampener at apl_t=0.5. Trims the APL
        // multiplier on mid-key interiors / dusk exteriors specifically;
        // endpoints unaffected. Applied BEFORE growth bypass so a fireball
        // in a mid-key scene can still drive apl_factor back to 1.0.
        float mid_notch = MID_APL_DAMPEN * apl_t * (1.0 - apl_t) * 4.0;
        apl_factor *= 1.0 - mid_notch;
        // Bright-field pulldown + optional top-band return (see the
        // APL_BRIGHT_COOL / APL_BRIGHT_RELAX block): the pulldown rides the
        // illum-weighted cool_w (bright FIELDS settle at ~170-185 nits;
        // faces at mid illum keep ship amplitude), the relax rides relax_w
        // (top-banded scenes may carry a hotter field, default off). Both
        // are post-notch ADDs — notch, pulldown and relax are independent
        // terms, order-correct for any notch value in the transition band.
        // Placed before the growth bypass: the mix toward 1.0 overrides
        // both during an event (no stacking), matching the shape-side
        // bw_gamma bypass.
        apl_factor += APL_BRIGHT_RELAX * relax_w - APL_BRIGHT_COOL * cool_w;
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
        // but no spatial edge bias within a single bright region. apl_t is
        // the shared scene axis hoisted above STEP 4.
        float spec_peak = mix(SPEC_PEAK_DARK, SPEC_PEAK_BRIGHT, apl_t);
        float spec_gamma = mix(SPEC_GAMMA_DARK, SPEC_GAMMA_BRIGHT, apl_t);
        // Drive signal: grain-stabilized luma, Y ONLY — by design, twice
        // field-rejected otherwise. A saturation-gated peak-channel (V)
        // escape for dim saturated emissives (red LED Y=0.21, blue laser
        // Y=0.07, magenta neon Y=0.29) lived here from v5.1 (v_drive +
        // SPEC_V_ESCAPE luma fade 0.45-0.60) and was DELETED 2026-07-02
        // after its restore crushed saturated speculars in the field.
        // Mechanism: in a saturated region the peak channel clips before
        // luma, so a V driver feeds this ramp a flat near-1.0 signal across
        // a core that still has luma gradient — the uniform spec add (with
        // the emissive carve holding the sat gate open at high V) compresses
        // the gradient toward flat: rule-1 direction. The v5.1.1 pink-carpet
        // speckle came from the same driver (Cr chroma noise in V is
        // unstabilizable by the luma transplant). Dim saturated emissives
        // therefore keep their SDR level BY DESIGN — do not re-add a V
        // driver to THIS ramp. The surviving V path is the bounded
        // BASE_V_CREDIT on the ~10x-gentler base ramp (see its block for
        // why that one is safe where this one wasn't).
        float spec_driver = Y_decision_gamma;
        // APL-tiered onset: parabolic bump pushes the ramp threshold up in
        // mid-bright scenes where lots of pixels are 0.88–0.93 but aren't
        // genuine specular. Endpoints stay at SPEC_Y_LOW. Bump peaks at
        // apl_t=0.5 with value SPEC_Y_LOW_MID_BUMP.
        float spec_y_low = SPEC_Y_LOW
                         + SPEC_Y_LOW_MID_BUMP * apl_t * (1.0 - apl_t) * 4.0;
        float spec_t = smoothstep(spec_y_low, 1.0, spec_driver);
        // Super-white overshoot: upscaler can produce signal > 1.0. Treat
        // it as evidence that the source was SDR-clipped — add linear bonus
        // on top of the saturated ramp, scene-signal-gated as everything
        // else. Hard-capped so extreme overshoot can't blow the nit budget.
        float overshoot = max(spec_driver - 1.0, 0.0);
        float spec_ramp = min(pow(spec_t, spec_gamma) + overshoot * SPEC_OVERSHOOT_GAIN,
                              SPEC_RAMP_CEIL);
        // User knobs: cf_spec scales spec pop alone; cf_strength rides along
        // so strength 0 is a true no-op (spec is additive — it wouldn't die
        // with the base curve on its own). Scaling here (not on the PEAK
        // defines) keeps the dark:bright ratio and the sat/emissive gates
        // untouched at any knob setting.
        spec_strength = spec_peak * spec_ramp * smoothed_spec_signal
                      * cf_spec * cf_strength;

        // Saturation gate. Genuine specular is near-white. Bright colored
        // objects (red shirts under sun, blonde hair, saturated sky) shouldn't
        // get a "specular pop". Pairs with bright-scene recovery upstream:
        // recovery turns spec back on in daylight, sat-gate keeps it from
        // firing on the red car under the sun. Emissive carve-out preserves
        // light sources at V ≥ 0.92 (tungsten/fire/sodium) from being gated.
        // V_gamma + sat_gamma already computed at PASS 6 entry (shared with
        // the Oklab fast-path bypass).
        float emissive_carve = smoothstep(SPEC_SAT_EMISSIVE_LOW,
                                          SPEC_SAT_EMISSIVE_HIGH, V_gamma);
        float sat_atten = mix(SPEC_SAT_ATTEN_DARK, SPEC_SAT_ATTEN_BRIGHT, apl_t)
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
    #if ENABLE_SPATIAL_PUMP
    // Hoisted (same pattern as pump_gain/spec_strength) so DEBUG_SHOW_PUMP
    // shows the exact mask the apply used — no hand-synced debug resample.
    float pump_mask = 1.0;
    #endif
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
        ivec2 pib = ivec2(floor(pg));
        ivec2 pi0 = clamp(pib,     ivec2(0), ivec2(15, 8));
        ivec2 pi1 = clamp(pib + 1, ivec2(0), ivec2(15, 8));
        // pump_mask_cell = the PRESENTATION mask (additive: max-preserving
        // 5×5 weighted stencil; subtractive fallback: 3×3 binomial soften —
        // see PASS 5's PUMP_MASK_SOFTEN). It rounds the bilinear diamond and
        // fills a large event's under-driven bright body. The raw env
        // (pump_env_cell) stays the dynamics state.
        float m00 = pump_mask_cell[pi0.y * 16 + pi0.x];
        float m10 = pump_mask_cell[pi0.y * 16 + pi1.x];
        float m01 = pump_mask_cell[pi1.y * 16 + pi0.x];
        float m11 = pump_mask_cell[pi1.y * 16 + pi1.x];
        pump_mask = mix(mix(m00, m10, pgf.x), mix(m01, m11, pgf.x), pgf.y);
        #if SPATIAL_PUMP_ADDITIVE
        // ADDITIVE: the mask (per-cell established-gated brightening env) is the
        // local pump amplitude; pump_cover_gate is the scene fade-to-white guard
        // the scalar bakes into pump_env — an asymmetric envelope since v5.7
        // (instant rise, rate-clamped fall ~0.5s; see PASS 5's PUMP_COVER_FALL;
        // a hard cut still zeroes it instantly via the reset). pump_env itself
        // is not consumed: no scene-global amplitude can change a local
        // light's strength or rhythm. The guarded local amplitude is mask ×
        // cover. A cell's bounded post-proof maintenance credit is already
        // folded into mask.
        float pump_local = pump_mask * pump_cover_gate;
        #else
        // SUBTRACTIVE: the mask (per-cell "is this region brightening" ∈[0,1]) only
        // SUPPRESSES the global scalar pump — it can't add pump. pump_env already
        // carries the scalar's amplitude, contrast/cover guard, and velocity release.
        // A brightening region has mask→1 (keeps the scalar); a static/darkening one
        // decays to 0 (suppressed). A reveal/pan/occluder-wake can't manufacture pump:
        // no global event ⇒ pump_env ~0 ⇒ product ~0 regardless of any local rise.
        float pump_local = pump_env * pump_mask;
        #endif
        #else
        float pump_local = pump_env;
        #endif
        // Down-gate where growth-mode is already restoring expansion (a fireball
        // triggers both): keeps pump + growth-bypass + spec from stacking the
        // transient peak past the display's ceiling, where the DISPLAY would
        // hard-clip and flatten the hot core. Gradation itself is never at risk
        // from the pump — it's a monotonic multiplier on expansion, so a
        // gradient stays a gradient; this only governs the absolute peak.
        // User knobs: cf_pump scales the pump alone; cf_strength rides along
        // so strength 0 is a true no-op (the pump multiplies expansion — at
        // expansion 1.0 it would still lift without this). PUMP_GAIN_CEIL is
        // NOT scaled: it's the safety roof, not a taste knob.
        float pump_str = PUMP_STRENGTH * cf_pump * cf_strength
                       * (1.0 - PUMP_GROWTH_DAMP * smoothed_growth_mode);
        pump_gain = min(pump_local * pump_str * pump_w, PUMP_GAIN_CEIL);
        expansion *= 1.0 + pump_gain;
    }
    #endif

    #if DEBUG_SHOW_PUMP && ENABLE_LIGHT_PUMP
    {
        #if ENABLE_SPATIAL_PUMP
        // Red = scalar pump (under ADDITIVE this is the reference — what the
        // scalar alone would do; the applied amplitude is blue × cover);
        // Green = per-pixel applied gain; Blue = per-cell mask.
        // pump_mask is the hoisted value the apply block actually used.
        return vec4(gamma709_to_pq2020(vec3(pump_env, pump_gain, pump_mask)), 1.0);
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
            // Y_decision_gamma == color.a * 2.0 on this path — reuse it so the
            // alpha-protocol decode lives at one production site (the debug WP
            // block keeps its own copy by design; it is self-contained).
            float Y_decision = eotf_gamma(Y_decision_gamma);
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
            // Skin lift (see PS_LIFT block): wider chroma window than the
            // sat boost, scene-cooling weighted. Multiplicative on
            // expansion and its gates rise with Y (ps_bright_w) or vary in
            // chroma, not Y — composite curve stays monotone per pixel.
            float psl_chroma_w = smoothstep(0.015, 0.05, chroma_orig)
                               * (1.0 - smoothstep(PS_LIFT_CHROMA_HI, PS_LIFT_CHROMA_CEIL, chroma_orig));
            float psl_w = ps_hue_w * psl_chroma_w * ps_bright_w * ps_gate
                        * smoothstep(0.5, 1.0, apl_t);
            expansion *= 1.0 + PS_LIFT * psl_w;
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
