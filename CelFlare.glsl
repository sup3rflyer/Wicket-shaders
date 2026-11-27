// CelFlare v7.0 — SDR-to-HDR highlight expansion for mpv (libplacebo)
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// Turns an SDR picture into PQ BT.2020 HDR by expanding its highlights.
// Mission: an authentic HDR grade of the source, not a brightening filter.
// Shadows and mid-tones keep the SDR grade: nothing changes below signal 0.5.
// Above it the expansion grows smoothly with brightness: a few percent at
// 0.5-0.6, about x1.15 at 0.7-0.8 and x1.3-1.4 on bright diffuse tones
// (0.85-0.95) in a bright scene, more in a dark one, and most on highlights
// above the scene's own white. Gentle over flashy. The shader does no tone
// mapping: the display owns the final mapping.
//
// Rules (cited in comments as rule 1-3):
//  1. Never attenuate source clipping; gradation first. The expansion is a
//     monotone per-pixel multiplier: tonal order is kept, no gradient
//     inverts, nothing clamps or divides a region flat. A clipped region gets
//     a gradient into a hot core, not a flat squash (an anti-squash defense,
//     not a look). Local contrast IS scaled, by the curve's slope (~1.5-3.8x
//     in the 0.7-0.95 band).
//  2. Controlled, slow release: transients ease out like eye adaptation,
//     never a 2-3 frame snap.
//  3. The display owns the ceiling; PUMP_GAIN_CEIL is a safety roof, not a
//     tone curve.
//
// Signal path (pass 9, per pixel): smooth base of the luma (a guided filter
// at 1/4 resolution, so texture and grain ride on the expansion 1:1) -> base
// curve whose shape follows the illumination field (the regional brightness)
// -> x dynamic intensity (scene contrast) -> x APL modulation (scene key) ->
// x top lift above the scene's diffuse white on compact highlights
// (cf_highlight) -> x (1 +
// light-pump gain) -> applied in Oklab at constant chromaticity (+ warm-hue
// and pale-skin corrections) -> PQ BT.2020 -> dither.
//
// Passes (all hook MAIN, in this order; the numbers are the names used in
// the comments, not the pass count):
//   1 Grain pre-filter      COMPUTE 16x16; decision luma into alpha as Y*0.5
//                           (read by the pump's pixel weight)
//   2 Downsample 1/4        -> CELFLARE_DS (alpha: near-white membership)
//     Highlight base        3 passes at 1/4: luma, statistics -> CELFLARE_GA
//   3 Illumination blur H   -> CELFLARE_BLUR_H
//   4 Illumination blur V   -> CELFLARE_ILLUM (sigma 80 px at 1080p, scales
//                              with the picture)
//     Skin weight           4b-4c and the surface spreads, the smoothing,
//                           the fit (CELFLARE_SKSM, CELFLARE_SKF)
//     Highlight base        coefficients and the coring reference -> CELFLARE_GF
//     Highlight size        seed + 6 spread passes on a 68-row grid (after pass 8): object coverage -> CELFLARE_OBJ
//   5 Motion downsample     -> MOTION_CUR 128x72
//   6 Motion block-match    COMPUTE 16x9 -> MOTION_FLOW, one vector per cell
//   7 Motion history store  -> CELFLARE_ADD_MOTION_PREV for the next frame
//   8 Frame stats           COMPUTE 16x9: scene statistics, cut detection,
//                           diffuse-white anchor, light-pump state
//   9 Expansion apply       COMPUTE 16x16, full resolution, the output
// Frame state lives in the CELFLARE_ADD_STATE buffer. A "cell" is one of the
// 16x9 = 144 grid regions used by pass 8 and the pump.
//
// Controls (USER TUNING below; all settable via glsl-shader-opts):
//   cf_ref_white   SDR white in nits; MUST equal mpv's hdr-reference-white
//   cf_strength    overall strength (0 = plain SDR)   cf_curve   ramp shape
//   cf_highlight   extra lift of clipped white        cf_pump    light pump
//   cf_key_hold    scene-key hold, s                  cf_skin_steady  skin exemption
//   toggles (recompile): cf_grain_stab, cf_additive_pump, cf_warm_shift,
//   cf_pale_skin, cf_debug. Sliders act on the next frame.
//   Deep anchors: search "MAIN TUNING" (the knobs scale them).
// cf_debug views: 1 bypass, 2 illumination field, 3 expansion, 4 detail (base
// expansion before the scene terms), 5 top lift, 6 light pump, 7 warm/skin,
// 8 scene stats, 9 motion offset, 10 motion evidence (11 = alias of 10),
// 12 additive opening proof.
//
// Load ONE CelFlare file; never together with CelFlare-transport.glsl (the
// picture would be processed twice and the two share state names). The
// output is PQ BT.2020, so the player must retag the frame (see the README).
// Version history is in the git log.

// shampv tuner API: plain comments to libplacebo, read by the shampv script.
// Declares SDR in, PQ BT.2020 out (the pipeline must retag the frame), and
// that cf_ref_white tracks the player's hdr-reference-white.
//@shampv input sdr
//@shampv output trc=pq primaries=bt.2020
//@shampv ref-white-param cf_ref_white
//@shampv choice cf_debug off bypass illum expand detail toplift pump warm-skin stats mv-offset mv-evidence mv-evidence-alias additive-proof

// =============================================================================
//  USER TUNING
// =============================================================================
// Each control is the plain number on the last line of its block: sliders
// first, then toggles (1 = on, 0 = off). Sliders are DYNAMIC (next frame, no
// recompile, scene/pump state kept); toggles recompile. Ranges are enforced;
// the numbers here are the defaults. Everything else is internal.
// Parser rules (break either and the shader fails to load): no comment line
// inside a PARAM/BUFFER/TEXTURE block or between its value and the next
// directive; and never write the directive prefix (two slashes and a bang)
// in prose anywhere: the parser splits the file on it, even mid-comment.

//!PARAM cf_ref_white
//!DESC SDR white level in nits — MUST match hdr-reference-white in mpv.conf. On Windows = the 'SDR content brightness' slider (README has the slider-to-nits table).
//!TYPE DYNAMIC float
//!MINIMUM 80.0
//!MAXIMUM 480.0
116.0

//!PARAM cf_strength
//!DESC Overall HDR strength. 0 = plain SDR, no expansion · 1 = the full internal tune (default) · ↑ 2 = double. Scales base expansion, the top lift and the light pump together.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM cf_curve
//!DESC Expansion ramp shape. ↓ <1 = gentle broad lift · ↑ >1 = lift concentrated on the brightest pixels, midtones closer to the SDR grade. Peak unchanged. 1.6 default.
//!TYPE DYNAMIC float
//!MINIMUM 0.599
//!MAXIMUM 2.0
1.6

//!PARAM cf_pump
//!DESC Light pump — temporary surge on sustained brightening (explosions, tunnel exits, spells). ↑ = stronger surge · 1 = shipped · 0 = off.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM cf_key_hold
//!DESC Scene-key hold, seconds. After a brighter shot, a darker shot keeps the brighter shot's calmer scene adaptation and eases to its own over about this long, so faces and outfits stop stepping brighter or darker at every cut inside one scene. A cut to a brighter shot still adapts at once, so nothing ever runs hotter than with the hold off. ↑ = steadier, dark-scene boost arrives later · 0 = off, every shot adapts on its own.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 64.0
10.0

//!PARAM cf_skin_steady
//!DESC Skin steadiness. Bright-scene cooling is meant for broad bright fields, but a large, brightly painted face is one, so a close-up was cooled and the same face in a wider shot was not. ↑ 1 = skin-hued pixels are exempt from the cooling and keep one curve in every framing (default) · 0 = off.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM cf_highlight
//!DESC Highlight pop: the extra factor a compact highlight (a glint, a lamp, a small reflection) gets at clipped white, on top of the curve, in a scene whose whites are not near clip. 1 = none · 1.5 = default. Painted whites, shirts and eye whites (the scene's own diffuse white) stay put, broad near-white areas (clouds, bands, garments, glare) get none of it, and it shrinks by itself as the scene's white nears clip. The factor is raised to the power cf_strength.
//!TYPE DYNAMIC float
//!MINIMUM 1.0
//!MAXIMUM 2.0
1.5

//!PARAM cf_grain_stab
//!DESC Grain stabilization (toggle). Feeds only the light pump's per-pixel weight (the expansion reads the smooth base and does not need it). 1 = on · 0 = off.
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
//!DESC Pale-skin protection (toggle). 1 = restore skin saturation lost to expansion AND a small deliberate brightening in bright cooled scenes (Hunt-effect counter — lifted skin reads PALER against the dimmed field, not tanner) · 0 = off.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM cf_debug
//!DESC Debug view selector (recompiles). Cycles: off, bypass, illum, expand, detail (base expansion), top lift, pump, warm/skin, stats, mv-offset, mv-evidence, 11 = mv-evidence, additive-proof.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 12
0

//!BUFFER CELFLARE_ADD_STATE
//!VAR float smoothed_bright_frac
//!VAR float smoothed_contrast
//!VAR float smoothed_log_avg
//!VAR float smoothed_growth_mode
//!VAR float scene_cut_lockout
//!VAR float smoothed_spec_natural
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
//!VAR float motion_bad_match_frac
//!VAR float motion_match_coverage
//!VAR float additive_mode_magic
//!VAR float dbg_cell_r[144]
//!VAR float dbg_cell_g[144]
//!VAR float dbg_cell_b[144]
//!VAR float cut_rate
//!VAR float pump_drive_prev
//!VAR float held_log_avg
//!VAR float spec_anchor_held
//!VAR float spec_anchor_last
//!STORAGE

//!TEXTURE CELFLARE_ADD_MOTION_PREV
//!SIZE 128 72
//!FORMAT r16f
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!COMPUTE 16 16
//!DESC CelFlare: Grain Pre-filter

// Role: stabilize the expansion DECISION across grain, not clean the image.
// Pixels sharing an underlying luma must get the same expansion, so grain
// passes through the multiplicative curve intact (SDR grain, scaled).
// A 12-tap bilateral (2 rings x 3 antipodal pairs, 9 / 18 px) keeps the
// decision inside one surface; larger radii average decisions across
// unrelated features and step the expansion at their boundaries. The blur
// weight is bright-asymmetric (darker taps count less).
// Layout: 16x16 workgroup; a 54x54 luma tile is loaded into shared memory and
// each thread does 12 manual bilinear reads. A workgroup vote skips the tile
// when no pixel is in range. HALO = radius + 1 (the bilinear floor(p)+1 read).
// ALPHA PROTOCOL: the decision luma goes into alpha as Y * 0.5 (stabilized,
// or raw on the exit paths). No sentinel value exists, so every stabilized
// luma is used as is; the 0.5 leaves headroom to Y 2.0 even on a unorm
// intermediate. Pass 9 decodes with * 2.0 and writes alpha 1.0.
// No raw-luma leak-through (the pixel keeps only its 1/19 center weight in
// the blur): grain left in the decision is amplified by the expansion ramp.
// GRAIN_THRESHOLD 0.35: going back to 0.28 was tested and rejected (it
// removed only 22 % of the edge halo vs 51 % for the edge gate, and gave
// back ~15-25 % of the grain win).
#define GRAIN_THRESHOLD     0.35
#define GRAIN_BLUR_RADIUS   18     // 9 px inner / 18 px outer — stays inside one patch
#define GRAIN_RANGE_MIN     0.35
#define GRAIN_RANGE_MAX     0.95   // established upper fade: signed stabilization releases to raw by clip
#define GRAIN_EDGE_LOW      0.05
#define GRAIN_EDGE_HIGH     0.15
// Matches the lower edge of range_mask's smoothstep; below this range_mask
// is identically 0 so stabilization would no-op anyway.
#define GRAIN_EARLY_EXIT    0.30
#define BILATERAL_SHARPNESS 6.0
#define INNER_RING_BOOST    2.0    // Inner carries 2:1 weight over outer (inner 12 : outer 6)
// Pair weights (GRAIN_PAIR_W 1). The per-tap weights are bright-asymmetric
// (a darker tap counts up to 7x less) so the anti-aliased band along a dark
// outline is held at its surround's level; per tap, that asymmetry also
// pulled every pixel on a smooth slope toward the brighter side, and the
// flank of a white cloud on a blue sky got a bright band of extra expansion
// (owner, 2026-11-23: "ring bands"). Now each antipodal pair contributes its
// mean at the weaker of its two weights (a straight slope is an exact
// identity), except where the pixel itself sits on a steep local step (the
// outline's AA band: about 0.25 over 4 px at 4K; a soft cloud flank 0.035),
// where the stronger tap keeps its full weight as before.
#define GRAIN_PAIR_W        1
#define GRAIN_STEP_R        2.0     // local step: tile samples this far either side, px
#define GRAIN_STEP_LO       0.10    // luma step over 2 x GRAIN_STEP_R where the old rule starts to return
#define GRAIN_STEP_HI       0.18    // ... and fully applies
// Edge gate (GRAIN_EDGE_LOW/HIGH). gx/gy use the SYMMETRIC weight: with the
// asymmetric one the dark-side taps of an edge get ~0 weight and the gate
// never fires. LOW/HIGH are raw estimator units, calibrated at 4K:
// halo-damaged shoulders p50 0.061, heavy grain (sigma 0.10) p95 0.055,
// smooth ramps <= 0.012. This cuts the worst halo (> 25 nits) by 51 % at
// negligible grain cost; grain and soft edges cannot be fully separated at
// this disc size. Decision bleed along smooth gradients is intended (bleed
// into ramps, hold edge contrast).

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
    // Manual bilinear in tile-local pixel coords: fract(pos) gives the same
    // weights as hardware sampling at that coordinate.
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

    // This thread's own pixel (the tile holds luma only).
    vec2 g_uv = (vec2(g_pixel) + 0.5) * HOOKED_pt;
    vec4 original = HOOKED_tex(g_uv);
    float Y_gamma = dot(original.rgb, luma_coeff);

    // Disabled stabilizer: a compile-time fast path that keeps the alpha
    // protocol and skips the tile. Every lane returns before the first
    // barrier, so control flow stays uniform.
    #if !cf_grain_stab
    imageStore(out_image, g_pixel, vec4(original.rgb, Y_gamma * 0.5));
    return;
    #endif

    // -------- workgroup vote: skip the tile when nothing is in range --------
    // All three barriers sit in workgroup-uniform control flow (the vote is
    // uniform), so the conditional return is FXC safe.
    if (lid == 0u) wg_active = 0u;
    barrier();
    if (Y_gamma >= GRAIN_EARLY_EXIT && Y_gamma < GRAIN_RANGE_MAX + 0.05)
        atomicOr(wg_active, 1u);
    barrier();
    if (wg_active == 0u) {
        imageStore(out_image, g_pixel, vec4(original.rgb, Y_gamma * 0.5));
        return;
    }

    // -------- cooperative luma tile load --------
    // Taps outside the image rely on the texture's clamp mode.
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
    // Outside the stabilizer's range the decision is raw; the upper exit
    // starts after the 0.95->1.00 fade, keeping super-white evidence.
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

    // -------- per-pixel rotation angle (static hash of the position) --------
    vec2 pixel_f = vec2(g_pixel);
    float angle = fract(sin(dot(pixel_f, vec2(12.9898, 78.233))) * 43758.5453) * 6.2832;
    float ca = cos(angle);
    float sa = sin(angle);

    // -------- bilateral parameters (division-free form) --------
    float asym_scale = mix(1.0, 7.0, smoothstep(0.55, 0.98, Y_gamma));
    float effective_sharpness = BILATERAL_SHARPNESS * mix(1.0, 0.82, smoothstep(0.60, 1.0, Y_gamma));
    float weight_k = 0.25 * effective_sharpness / (GRAIN_THRESHOLD * GRAIN_THRESHOLD);
    float asym_scale_sq = asym_scale * asym_scale;

    // -------- 12 taps: 2 rings x 3 antipodal pairs --------
    // Taps 0-2 outer ring (R), 3-5 inner (R/2), each at +o and -o. The inner
    // ring gets INNER_RING_BOOST in the blur and a x2 gradient scale (half
    // radius, half response); the gradient uses the un-boosted symmetric ws.
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
    #if GRAIN_PAIR_W
    // Steep local step at this pixel (see GRAIN_PAIR_W): axes and diagonals.
    float lstep = max(max(abs(tile_bilinear(tile_center + vec2(GRAIN_STEP_R, 0.0))
                            - tile_bilinear(tile_center - vec2(GRAIN_STEP_R, 0.0))),
                          abs(tile_bilinear(tile_center + vec2(0.0, GRAIN_STEP_R))
                            - tile_bilinear(tile_center - vec2(0.0, GRAIN_STEP_R)))),
                      max(abs(tile_bilinear(tile_center + vec2(GRAIN_STEP_R, GRAIN_STEP_R))
                            - tile_bilinear(tile_center - vec2(GRAIN_STEP_R, GRAIN_STEP_R))),
                          abs(tile_bilinear(tile_center + vec2(GRAIN_STEP_R, -GRAIN_STEP_R))
                            - tile_bilinear(tile_center - vec2(GRAIN_STEP_R, -GRAIN_STEP_R)))));
    float old_rule = smoothstep(GRAIN_STEP_LO, GRAIN_STEP_HI, lstep);
    #endif

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

        vec2 pr_s = vec2(0.0);    // the pair's samples and asymmetric weights
        vec2 pr_w = vec2(0.0);
        for (int k = 0; k < 2; k++) {
            float sgn = (k == 0) ? 1.0 : -1.0;   // + sample, then antipodal -
            float s  = tile_bilinear(tile_center + o * sgn);
            float rd = s - Y_gamma;
            float a2 = rd < 0.0 ? asym_scale_sq : 1.0;
            float tw = max(0.0, 1.0 - weight_k * rd * rd * a2);
            float w  = tw * tw;
            float tws = max(0.0, 1.0 - weight_k * rd * rd);
            float ws  = tws * tws;
            #if GRAIN_PAIR_W
            if (k == 0) { pr_s.x = s; pr_w.x = w; }
            else        { pr_s.y = s; pr_w.y = w; }
            #else
            blurred += s * w * boost;
            total_w += w * boost;
            #endif
            gx += rd * ws * r.x * gscale * sgn;
            gy += rd * ws * r.y * gscale * sgn;
            grad_w += ws;
        }
        #if GRAIN_PAIR_W
        {
            // Pair rule, plus the stronger tap's excess where the old rule applies.
            float pw     = min(pr_w.x, pr_w.y);
            float excess = (max(pr_w.x, pr_w.y) - pw) * old_rule;
            float s_str  = (pr_w.x >= pr_w.y) ? pr_s.x : pr_s.y;
            blurred += ((pr_s.x + pr_s.y) * pw + s_str * excess) * boost;
            total_w += (2.0 * pw + excess) * boost;
        }
        #endif
    }

    float inv_grad_w = grad_w > 0.0 ? 1.0 / grad_w : 0.0;
    gx *= inv_grad_w;
    gy *= inv_grad_w;

    blurred /= total_w;

    float edge = sqrt(gx * gx + gy * gy);
    float edge_mask = smoothstep(GRAIN_EDGE_LOW, GRAIN_EDGE_HIGH, edge);

    // The decision feeds the light pump's pixel weight in pass 9 (the base
    // curve reads the highlight base instead). It fades to raw from 0.95 to
    // clip, so a hot core is never decision-squashed.
    float stab = (1.0 - edge_mask) * range_mask;
    float Y_decision = Y_gamma + (blurred - Y_gamma) * stab;

    // Alpha = Y * 0.5 (see the alpha protocol above).
    imageStore(out_image, g_pixel, vec4(original.rgb, Y_decision * 0.5));
}

// =============================================================================
// PASS 2: DOWNSAMPLE 1/4
// =============================================================================
// Four bilinear taps = an exact 4x4 box of the RAW pixel RGB. RGB is kept
// through the blur so pass 8 can read max(R,G,B) of the field (the V-aware
// pump driver). CELFLARE_DS is an aliased box, not a Gaussian: never
// point-sample it as if it were smooth.

//!HOOK MAIN
//!BIND HOOKED
//!SAVE CELFLARE_DS
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC CelFlare: Downsample 1/4

vec4 hook() {
    vec2 pt = HOOKED_pt;
    vec3 c0 = HOOKED_tex(HOOKED_pos + vec2(-pt.x, -pt.y)).rgb;
    vec3 c1 = HOOKED_tex(HOOKED_pos + vec2( pt.x, -pt.y)).rgb;
    vec3 c2 = HOOKED_tex(HOOKED_pos + vec2(-pt.x,  pt.y)).rgb;
    vec3 c3 = HOOKED_tex(HOOKED_pos + vec2( pt.x,  pt.y)).rgb;
    vec3 rgb  = c0;
    rgb      += c1;
    rgb      += c2;
    rgb      += c3;
    rgb *= 0.25;
    // Alpha carries a soft near-white membership (the diffuse-white anchor,
    // pass 8). The two blur passes below carry it along, so CELFLARE_ILLUM.a
    // is the share of the sigma-80 neighbourhood that is near-white: the
    // "bright coverage".
    const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
    float m = smoothstep(0.86, 0.92, dot(c0, lw)) + smoothstep(0.86, 0.92, dot(c1, lw))
            + smoothstep(0.86, 0.92, dot(c2, lw)) + smoothstep(0.86, 0.92, dot(c3, lw));
    return vec4(rgb, m * 0.25);
}

// =============================================================================
// HIGHLIGHT BASE: fast guided filter at 1/4 resolution
// =============================================================================
// The curve reads a smooth base of the picture instead of each pixel's own
// luma, so texture, grain and compression noise ride on the expansion 1:1
// (a steep curve on per-pixel luma magnified them: salt and pepper on
// textured whites, beads in thin outlines, contours in soft glare). Self-
// guided fast guided filter (He & Sun 2015): per 1/4 texel, mean m and
// variance v of the decision luma over 5x5 texels, a = v / (v + eps),
// b = (1 - a) m; a and b averaged with a 5x5 tent; pass 9: B = a Y_dec + b.
// A linear ramp of any height is reproduced exactly (no staircase), an edge
// with v >> eps is kept, detail with v << eps is removed. Variance is
// computed from the samples in 32-bit, never as E[q^2] - m^2 from stored
// half floats.

//!HOOK MAIN
//!BIND HOOKED
//!SAVE CELFLARE_GQ
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC CelFlare: Highlight base luma 1/4

#define HL_RAW_LUMA 1   // keep equal to pass 9

vec4 hook() {
    // Area mean of 4x4 of the luma: the picture's own (HL_RAW_LUMA), or pass
    // 1's decision luma (alpha protocol: Y * 0.5). Pass 1's wide range kernel
    // pulls the darker side of a step toward the brighter side over its
    // 18 px reach (a glow along sclera, white shirts); the guided filter
    // already keeps grain out of the base, so the plain luma is the default.
    vec2 pt = HOOKED_pt;
    vec4 c0 = HOOKED_tex(HOOKED_pos + vec2(-pt.x, -pt.y));
    vec4 c1 = HOOKED_tex(HOOKED_pos + vec2( pt.x, -pt.y));
    vec4 c2 = HOOKED_tex(HOOKED_pos + vec2(-pt.x,  pt.y));
    vec4 c3 = HOOKED_tex(HOOKED_pos + vec2( pt.x,  pt.y));
#if HL_RAW_LUMA
    const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
    float q = (dot(c0.rgb, lw) + dot(c1.rgb, lw) + dot(c2.rgb, lw) + dot(c3.rgb, lw)) * 0.25;
#else
    float q = (c0.a + c1.a + c2.a + c3.a) * 0.5;
#endif
    return vec4(q, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_GQ
//!SAVE CELFLARE_GA
//!WIDTH CELFLARE_GQ.w
//!HEIGHT CELFLARE_GQ.h
//!DESC CelFlare: Highlight base statistics

#define HLB_EPS 0.0016   // 0.04^2, gamma-signal units

vec4 hook() {
    float s = 0.0;
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++)
            s += CELFLARE_GQ_tex(CELFLARE_GQ_pos + vec2(float(i), float(j)) * CELFLARE_GQ_pt).r;
    float m = s * (1.0 / 25.0);
    float v = 0.0;
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++) {
            float d = CELFLARE_GQ_tex(CELFLARE_GQ_pos + vec2(float(i), float(j)) * CELFLARE_GQ_pt).r - m;
            v += d * d;
        }
    v *= 1.0 / 25.0;
    float a = v / (v + HLB_EPS);
    return vec4(a, (1.0 - a) * m, 0.0, 1.0);
}

// =============================================================================
// PASSES 3-4: ILLUMINATION FIELD (separable Gaussian blur at 1/4 resolution)
// =============================================================================
// The illumination field is the regional brightness every expansion decision
// reads. sigma = 20 DS texels = 80 px on a 1920x1080 picture, and it SCALES
// WITH THE PICTURE: the larger dimension relative to 1920x1080 sets the
// scale, so a 4K MAIN (e.g. a 2x upscale) gets 160 px, while a 1920x800 scope
// or a 1440x1080 4:3 encode keeps the 1080p kernel. (A fixed 80 px made the
// look more local at 4K and made the 16x9 pump cells alias.) At the reference
// geometry the precomputed table runs (bit-identical to earlier 1080p
// builds); other sizes compute the same merged-pair construction out to
// 2 sigma. The weights are data-independent on purpose: a bright bias would
// make the blur non-separable and draw outlines at bright/dark boundaries.
// Keep ILLUM_* identical in both blur passes (separate translation units).

//!HOOK MAIN
//!BIND CELFLARE_DS
//!SAVE CELFLARE_BLUR_H
//!WIDTH CELFLARE_DS.w
//!HEIGHT CELFLARE_DS.h
//!DESC CelFlare: Illumination Blur H

#define ILLUM_SIGMA_DS  20.0    // sigma in DS texels at the reference geometry (= 80 px)
#define ILLUM_REF_W     480.0   // reference DS size: a 1920x1080 MAIN
#define ILLUM_REF_H     270.0

vec4 illum_blur(vec2 pos, vec2 dir) {
    vec2 ds = CELFLARE_DS_size;
    vec4 sum = CELFLARE_DS_tex(pos);
    // Reference geometry, tested on the exact integer texture size (a
    // float-quotient compare can miss 1.0 by an ULP on some drivers).
    if ((ds.x == ILLUM_REF_W && ds.y <= ILLUM_REF_H)
     || (ds.y == ILLUM_REF_H && ds.x <= ILLUM_REF_W)) {
        // Reference geometry: sigma 20, radius 40, 20 merged pairs (41 fetches).
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
        for (int i = 0; i < 20; i++) {
            vec4 sp = CELFLARE_DS_tex(pos + go[i] * dir);
            vec4 sn = CELFLARE_DS_tex(pos - go[i] * dir);
            sum += (sp + sn) * gw[i];
        }
        return sum * 0.02084002;   // 1 / (1 + 2 * sum(gw)) = 1 / 47.98460032
    }
    // Any other geometry: same construction, computed. Pair i merges the
    // texels at 2i+1 and 2i+2 into one bilinear tap at their weighted centre.
    float sigma = ILLUM_SIGMA_DS * max(ds.x / ILLUM_REF_W, ds.y / ILLUM_REF_H);
    int pairs = int(ceil(sigma));
    float k = -0.5 / (sigma * sigma);
    float wsum = 1.0;
    for (int i = 0; i < pairs; i++) {
        float x1 = float(2 * i + 1);
        float x2 = x1 + 1.0;
        float w1 = exp(k * x1 * x1);
        float w2 = exp(k * x2 * x2);
        float w = w1 + w2;
        float o = (w1 * x1 + w2 * x2) / w;
        sum += (CELFLARE_DS_tex(pos + o * dir) + CELFLARE_DS_tex(pos - o * dir)) * w;
        wsum += 2.0 * w;
    }
    return sum / wsum;
}

vec4 hook() {
    return illum_blur(CELFLARE_DS_pos, vec2(CELFLARE_DS_pt.x, 0.0));
}

//!HOOK MAIN
//!BIND CELFLARE_DS
//!BIND CELFLARE_BLUR_H
//!SAVE CELFLARE_ILLUM
//!WIDTH CELFLARE_BLUR_H.w
//!HEIGHT CELFLARE_BLUR_H.h
//!DESC CelFlare: Illumination Blur V

#define ILLUM_SIGMA_DS  20.0    // keep identical to the H pass
#define ILLUM_REF_W     480.0
#define ILLUM_REF_H     270.0

vec4 illum_blur(vec2 pos, vec2 dir) {
    vec2 ds = CELFLARE_DS_size;
    vec4 sum = CELFLARE_BLUR_H_tex(pos);
    // Reference geometry, tested on the exact integer texture size (a
    // float-quotient compare can miss 1.0 by an ULP on some drivers).
    if ((ds.x == ILLUM_REF_W && ds.y <= ILLUM_REF_H)
     || (ds.y == ILLUM_REF_H && ds.x <= ILLUM_REF_W)) {
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
        for (int i = 0; i < 20; i++) {
            vec4 sp = CELFLARE_BLUR_H_tex(pos + go[i] * dir);
            vec4 sn = CELFLARE_BLUR_H_tex(pos - go[i] * dir);
            sum += (sp + sn) * gw[i];
        }
        return sum * 0.02084002;
    }
    float sigma = ILLUM_SIGMA_DS * max(ds.x / ILLUM_REF_W, ds.y / ILLUM_REF_H);
    int pairs = int(ceil(sigma));
    float k = -0.5 / (sigma * sigma);
    float wsum = 1.0;
    for (int i = 0; i < pairs; i++) {
        float x1 = float(2 * i + 1);
        float x2 = x1 + 1.0;
        float w1 = exp(k * x1 * x1);
        float w2 = exp(k * x2 * x2);
        float w = w1 + w2;
        float o = (w1 * x1 + w2 * x2) / w;
        sum += (CELFLARE_BLUR_H_tex(pos + o * dir) + CELFLARE_BLUR_H_tex(pos - o * dir)) * w;
        wsum += 2.0 * w;
    }
    return sum / wsum;
}

vec4 hook() {
    return illum_blur(CELFLARE_BLUR_H_pos, vec2(0.0, CELFLARE_BLUR_H_pt.y));
}

// =============================================================================
// PASSES 4b-4c: SKIN WEIGHT (cf_skin_steady)
// =============================================================================
// The cooling exemption needs "is this skin-coloured paint" per pixel. Read
// from the raw pixel, hue and chroma carry the source's compression noise,
// and the window's edges are steep: inside a soft warm gradient (defocused
// stained glass) the weight switched on along a ragged line that crawled every
// frame, with the whole cooling difference (up to 17 %) across it.
// These passes take the decision off the raw pixel, the same way the spec
// stabiliser takes its drive off raw luma (a same-surface reference):
//   4b  1/8 resolution: Oklab and window weight of each block's mean colour.
//   4c  7x7 bilateral average of the weight over blocks of LIKE colour: it
//       averages along a surface and across a gentle gradient, and stops at a
//       drawn outline (two flat colours a step apart do not mix).
//   4c2-4c5  four more such averages with the taps 16, 8, 4 and 2 blocks
//       apart (at 2160 lines; scaled with the picture), weight only, over
//       blocks of the SAME colour (likeness sigma 0.015): the weight becomes
//       a property of the surface, not of the patch. Coarse to fine, so the
//       later passes remove the lattice of the sparse ones. The tight
//       likeness keeps a small face from taking in its surroundings: at
//       0.03 the many taps beyond an outline outweighed the few inside.
//       A surface that sits on one of the window's ramps (pink skin and
//       blush on the lower hue ramp) had its weight moved by 0.2-0.4 by the
//       encode's chroma mottle, in blobs several blocks wide that are new
//       every frame: blotches of a few percent that boil in a pan.
// The main pass reads 4c through a lookup weighted by likeness to the pixel's
// own colour. Where no block resembles the pixel (an anti-aliased or soft
// edge pixel: a MIX of its two sides) it takes the plain bilinear weight, so
// a mix of two sides gets the mix of their weights. It never reads the
// window from the pixel's own colour: a mix of a warm and a cool colour
// passes through skin colour, which drew a bright fringe on such edges.

//!HOOK MAIN
//!BIND HOOKED
//!SAVE CELFLARE_SKQ
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC CelFlare: Skin texel purity

// Per 4x4 cell: gamma rgb mean and a purity confidence from its four 2x2
// sub-means. A cell an outline runs through holds two colours; the skin
// window is not monotone along such a mix (chroma ceiling, onset, hue band),
// and which cells hold mixes depends on where the grid falls, so the weight
// jumped as content moved (owner, 2026-11-24: warm hearts "ragged ... in
// motion"). Impure cells are left out of the window's decision.
#define SKQ_PURITY 0.04   // gamma rgb distance where purity falls to 1/e

vec4 hook() {
    vec2 pt = HOOKED_pt;
    vec3 c0 = HOOKED_tex(HOOKED_pos + vec2(-pt.x, -pt.y)).rgb;
    vec3 c1 = HOOKED_tex(HOOKED_pos + vec2( pt.x, -pt.y)).rgb;
    vec3 c2 = HOOKED_tex(HOOKED_pos + vec2(-pt.x,  pt.y)).rgb;
    vec3 c3 = HOOKED_tex(HOOKED_pos + vec2( pt.x,  pt.y)).rgb;
    vec3 m = (c0 + c1 + c2 + c3) * 0.25;
    float d = max(max(length(c0 - m), length(c1 - m)), max(length(c2 - m), length(c3 - m)));
    float c = exp(-(d / SKQ_PURITY) * (d / SKQ_PURITY));
    return vec4(m, c);
}

//!HOOK MAIN
//!BIND CELFLARE_DS
//!BIND CELFLARE_SKQ
//!SAVE CELFLARE_SKW
//!WIDTH CELFLARE_DS.w 2 /
//!HEIGHT CELFLARE_DS.h 2 /
//!DESC CelFlare: Skin weight 1/8

// Skin window. Lit skin measured at Oklab hue 34-82 deg across eight
// productions: pink-leaning paint at 34-45, neutral at 54-62, and a
// warm-filtered show at 71-82 (rgb 246 220 182), so no narrower window is
// general. Flat top within 27 deg of the centre (30-84), zero from 45 deg out
// (12 and 102). Chroma onset, two ends blended by luma: near-white (Y above
// 0.95) on from 0.012-0.035, so warm white fields keep the cooling; below
// Y 0.90 on from 0.006-0.018 (very pale skin sits at 0.015-0.027). Off
// through 0.09-0.14 (deep-saturated warm colours). Hue cannot tell skin from
// a cream wall, sand or blonde hair of the same hue: those are exempt too.
#define SKIN_HUE_COS        0.5446  // cos(57 deg)
#define SKIN_HUE_SIN        0.8387  // sin(57 deg)
#define SKIN_HUE_FULL       0.8910  // cos(27 deg): full weight inside
#define SKIN_HUE_ZERO       0.7071  // cos(45 deg): no weight outside
#define SKIN_CHROMA_LO      0.012   // near-white end
#define SKIN_CHROMA_ON      0.035
#define SKIN_CHROMA_LO_PALE 0.006   // below the luma cap
#define SKIN_CHROMA_ON_PALE 0.018
#define SKIN_PALE_Y_LO      0.90
#define SKIN_PALE_Y_HI      0.95
#define SKIN_CHROMA_HI      0.09
#define SKIN_CHROMA_CEIL    0.14

#define SKW_FIT_POOL 1   // keep equal to SKW_FIT in pass 9 (0 = plain block mean, as before)

float skw_window(vec3 c) {
    float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 lin = pow(c, vec3(2.4));
    float l = pow(0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b, 1.0 / 3.0);
    float m = pow(0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b, 1.0 / 3.0);
    float s = pow(0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b, 1.0 / 3.0);
    float oa = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    float ob = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
    float ch = sqrt(oa * oa + ob * ob);
    float hh = (ch > 1e-6) ? (oa * SKIN_HUE_COS + ob * SKIN_HUE_SIN) / ch : 0.0;
    float nw = smoothstep(SKIN_PALE_Y_LO, SKIN_PALE_Y_HI, y);
    return smoothstep(SKIN_HUE_ZERO, SKIN_HUE_FULL, hh)
         * smoothstep(mix(SKIN_CHROMA_LO_PALE, SKIN_CHROMA_LO, nw),
                      mix(SKIN_CHROMA_ON_PALE, SKIN_CHROMA_ON, nw), ch)
         * (1.0 - smoothstep(SKIN_CHROMA_HI, SKIN_CHROMA_CEIL, ch));
}

vec4 hook() {
    vec2 h = 0.5 * CELFLARE_DS_pt;
    vec3 c = CELFLARE_DS_tex(CELFLARE_DS_pos + vec2(-h.x, -h.y)).rgb;
    c     += CELFLARE_DS_tex(CELFLARE_DS_pos + vec2( h.x, -h.y)).rgb;
    c     += CELFLARE_DS_tex(CELFLARE_DS_pos + vec2(-h.x,  h.y)).rgb;
    c     += CELFLARE_DS_tex(CELFLARE_DS_pos + vec2( h.x,  h.y)).rgb;
    c = max(c * 0.25, 0.0);
#if SKW_FIT_POOL
    // Confident pooling: colour and window from pure cells only (plain mean
    // where none is pure).
    vec2 hq = 0.5 * CELFLARE_SKQ_pt;
    vec4 q0 = CELFLARE_SKQ_tex(CELFLARE_SKQ_pos + vec2(-hq.x, -hq.y));
    vec4 q1 = CELFLARE_SKQ_tex(CELFLARE_SKQ_pos + vec2( hq.x, -hq.y));
    vec4 q2 = CELFLARE_SKQ_tex(CELFLARE_SKQ_pos + vec2(-hq.x,  hq.y));
    vec4 q3 = CELFLARE_SKQ_tex(CELFLARE_SKQ_pos + vec2( hq.x,  hq.y));
    float cs = q0.a + q1.a + q2.a + q3.a;
    vec3 cc = (q0.rgb * q0.a + q1.rgb * q1.a + q2.rgb * q2.a + q3.rgb * q3.a) / max(cs, 1e-4);
    float wc = (skw_window(max(q0.rgb, 0.0)) * q0.a + skw_window(max(q1.rgb, 0.0)) * q1.a
              + skw_window(max(q2.rgb, 0.0)) * q2.a + skw_window(max(q3.rgb, 0.0)) * q3.a) / max(cs, 1e-4);
    float fpure = smoothstep(0.0, 0.05, cs);
    // The block colour stays the plain mean: other features look it up per
    // pixel (colour credit, decision chroma), and pure colours there made a
    // mixed edge pixel match blocks by grid position.
    float w_pure = wc;
#endif
    float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 lin = pow(c, vec3(2.4));
    float l = pow(0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b, 1.0 / 3.0);
    float m = pow(0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b, 1.0 / 3.0);
    float s = pow(0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b, 1.0 / 3.0);
    float oL = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
    float oa = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    float ob = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
    float ch = sqrt(oa * oa + ob * ob);
    float hh = (ch > 1e-6) ? (oa * SKIN_HUE_COS + ob * SKIN_HUE_SIN) / ch : 0.0;
    float nw = smoothstep(SKIN_PALE_Y_LO, SKIN_PALE_Y_HI, y);
    float w = smoothstep(SKIN_HUE_ZERO, SKIN_HUE_FULL, hh)
            * smoothstep(mix(SKIN_CHROMA_LO_PALE, SKIN_CHROMA_LO, nw),
                         mix(SKIN_CHROMA_ON_PALE, SKIN_CHROMA_ON, nw), ch)
            * (1.0 - smoothstep(SKIN_CHROMA_HI, SKIN_CHROMA_CEIL, ch));
#if SKW_FIT_POOL
    w = mix(w, w_pure, fpure);
#endif
    // a and b are stored with a 0.5 offset (safe on a unorm intermediate).
    return vec4(oL, oa + 0.5, ob + 0.5, w);
}

//!HOOK MAIN
//!BIND CELFLARE_SKW
//!SAVE CELFLARE_SKC
//!WIDTH CELFLARE_SKW.w
//!HEIGHT CELFLARE_SKW.h
//!DESC CelFlare: Skin weight same-surface average

// Likeness is plain Oklab distance. 0.03 sits between the steps of a soft
// gradient (a defocused edge moves about 0.02-0.035 per block) and the step
// across a drawn outline (pale skin to white 0.07, skin to white 0.12).
#define SKS_RANGE_K   555.6   // 1 / (2 * 0.03^2)
#define SKS_SPACE_K   0.2222  // 1 / (2 * 1.5^2): spatial sigma 1.5 blocks, radius 3

vec4 hook() {
    vec4 c = CELFLARE_SKW_tex(CELFLARE_SKW_pos);
    float acc = 0.0;
    float wsum = 0.0;
    vec3 lab = vec3(0.0);
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            vec4 s = CELFLARE_SKW_tex(CELFLARE_SKW_pos + vec2(float(i), float(j)) * CELFLARE_SKW_pt);
            vec3 d = s.rgb - c.rgb;
            float w = exp(-SKS_SPACE_K * float(i * i + j * j) - SKS_RANGE_K * dot(d, d));
            acc += w * s.a;
            lab += w * s.rgb;
            wsum += w;
        }
    }
    // rgb = the same average of the block colour (the decision chroma's source).
    return vec4(lab / wsum, acc / wsum);
}

//!HOOK MAIN
//!BIND CELFLARE_SKC
//!SAVE CELFLARE_SK1
//!WIDTH CELFLARE_SKC.w
//!HEIGHT CELFLARE_SKC.h
//!DESC CelFlare: Skin weight surface spread 1

// Passes 4c2-4c5. Guide = pass 4c's block colour, carried through unchanged
// in rgb (the main pass reads it as the decision chroma); only the weight in
// alpha is averaged. Tap spacing in blocks at 2160 lines; 0 = this pass off.
#define SKF_STRIDE_1  16.0
#define SKF_RANGE_K   2222.2  // likeness sigma 0.015: same colour only
#define SKF_SPACE_K   0.2222  // sigma 1.5 taps, radius 3

vec4 hook() {
    vec4 c = CELFLARE_SKC_tex(CELFLARE_SKC_pos);
    vec2 st = SKF_STRIDE_1 * (CELFLARE_SKC_size.y / 270.0) * CELFLARE_SKC_pt;
    float acc = 0.0;
    float wsum = 0.0;
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            vec4 s = CELFLARE_SKC_tex(CELFLARE_SKC_pos + vec2(float(i), float(j)) * st);
            vec3 d = s.rgb - c.rgb;
            float w = exp(-SKF_SPACE_K * float(i * i + j * j) - SKF_RANGE_K * dot(d, d));
            acc += w * s.a;
            wsum += w;
        }
    }
    return vec4(c.rgb, acc / wsum);
}

//!HOOK MAIN
//!BIND CELFLARE_SK1
//!SAVE CELFLARE_SK2
//!WIDTH CELFLARE_SK1.w
//!HEIGHT CELFLARE_SK1.h
//!DESC CelFlare: Skin weight surface spread 2

#define SKF_STRIDE_2  8.0
#define SKF_RANGE_K   2222.2  // likeness sigma 0.015: same colour only
#define SKF_SPACE_K   0.2222  // sigma 1.5 taps, radius 3

vec4 hook() {
    vec4 c = CELFLARE_SK1_tex(CELFLARE_SK1_pos);
    vec2 st = SKF_STRIDE_2 * (CELFLARE_SK1_size.y / 270.0) * CELFLARE_SK1_pt;
    float acc = 0.0;
    float wsum = 0.0;
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            vec4 s = CELFLARE_SK1_tex(CELFLARE_SK1_pos + vec2(float(i), float(j)) * st);
            vec3 d = s.rgb - c.rgb;
            float w = exp(-SKF_SPACE_K * float(i * i + j * j) - SKF_RANGE_K * dot(d, d));
            acc += w * s.a;
            wsum += w;
        }
    }
    return vec4(c.rgb, acc / wsum);
}

//!HOOK MAIN
//!BIND CELFLARE_SK2
//!SAVE CELFLARE_SK3
//!WIDTH CELFLARE_SK2.w
//!HEIGHT CELFLARE_SK2.h
//!DESC CelFlare: Skin weight surface spread 3

#define SKF_STRIDE_3  4.0
#define SKF_RANGE_K   2222.2  // likeness sigma 0.015: same colour only
#define SKF_SPACE_K   0.2222  // sigma 1.5 taps, radius 3

vec4 hook() {
    vec4 c = CELFLARE_SK2_tex(CELFLARE_SK2_pos);
    vec2 st = SKF_STRIDE_3 * (CELFLARE_SK2_size.y / 270.0) * CELFLARE_SK2_pt;
    float acc = 0.0;
    float wsum = 0.0;
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            vec4 s = CELFLARE_SK2_tex(CELFLARE_SK2_pos + vec2(float(i), float(j)) * st);
            vec3 d = s.rgb - c.rgb;
            float w = exp(-SKF_SPACE_K * float(i * i + j * j) - SKF_RANGE_K * dot(d, d));
            acc += w * s.a;
            wsum += w;
        }
    }
    return vec4(c.rgb, acc / wsum);
}

//!HOOK MAIN
//!BIND CELFLARE_SK3
//!SAVE CELFLARE_SKS
//!WIDTH CELFLARE_SK3.w
//!HEIGHT CELFLARE_SK3.h
//!DESC CelFlare: Skin weight surface spread 4

#define SKF_STRIDE_4  2.0
#define SKF_RANGE_K   2222.2  // likeness sigma 0.015: same colour only
#define SKF_SPACE_K   0.2222  // sigma 1.5 taps, radius 3

vec4 hook() {
    vec4 c = CELFLARE_SK3_tex(CELFLARE_SK3_pos);
    vec2 st = SKF_STRIDE_4 * (CELFLARE_SK3_size.y / 270.0) * CELFLARE_SK3_pt;
    float acc = 0.0;
    float wsum = 0.0;
    for (int j = -3; j <= 3; j++) {
        for (int i = -3; i <= 3; i++) {
            vec4 s = CELFLARE_SK3_tex(CELFLARE_SK3_pos + vec2(float(i), float(j)) * st);
            vec3 d = s.rgb - c.rgb;
            float w = exp(-SKF_SPACE_K * float(i * i + j * j) - SKF_RANGE_K * dot(d, d));
            acc += w * s.a;
            wsum += w;
        }
    }
    return vec4(c.rgb, acc / wsum);
}

//!HOOK MAIN
//!BIND CELFLARE_SKS
//!SAVE CELFLARE_SKSM
//!WIDTH CELFLARE_SKS.w
//!HEIGHT CELFLARE_SKS.h
//!DESC CelFlare: Skin weight smooth

// The same-surface passes keep the weight crisp between blocks of different
// colour, so where a soft warm gradient crosses the skin window's chroma
// onset the weight stepped 0 -> 1 between neighbouring blocks and the
// exemption drew staircase contours around defocused shapes (owner,
// 2026-11-23: "ragged edges on the background"). A small Gaussian of the
// weight, stopped only by strong colour differences, turns those steps into
// a ramp; outlines between clearly different colours still stop it.
#define SKM_SIGMA     1.2     // blocks (8 px at 4K); 0 = copy
#define SKM_RANGE     0.05    // Oklab distance where a neighbour's weight halves-ish

vec4 hook() {
    vec4 c = CELFLARE_SKS_tex(CELFLARE_SKS_pos);
    if (SKM_SIGMA <= 0.0) return c;
    float ks = 0.5 / (SKM_SIGMA * SKM_SIGMA);
    float kr = 0.5 / (SKM_RANGE * SKM_RANGE);
    float acc = 0.0;
    float wsum = 0.0;
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++) {
            vec4 s = CELFLARE_SKS_tex(CELFLARE_SKS_pos + vec2(float(i), float(j)) * CELFLARE_SKS_pt);
            vec3 d = s.rgb - c.rgb;
            float w = exp(-ks * float(i * i + j * j) - kr * dot(d, d));
            acc += w * s.a;
            wsum += w;
        }
    return vec4(c.rgb, acc / wsum);
}

//!HOOK MAIN
//!BIND CELFLARE_GA
//!SAVE CELFLARE_GF0
//!WIDTH CELFLARE_GA.w
//!HEIGHT CELFLARE_GA.h
//!DESC CelFlare: Highlight base coefficients

// 5x5 tent (1 2 3 2 1)^2 / 81 of the highlight base's (a, b) in .rg. The next
// two passes add the base's coring reference in .b (one texture: pass 9 has
// no constant-buffer slot left on d3d11 for another).
vec4 hook() {
    vec4 acc = vec4(0.0);
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++) {
            float w = float((3 - abs(i)) * (3 - abs(j)));
            vec2 o = vec2(float(i), float(j)) * CELFLARE_GA_pt;
            acc += w * vec4(CELFLARE_GA_tex(CELFLARE_GA_pos + o).rg, 0.0, 0.0);
        }
    return acc * (1.0 / 81.0);
}

//!HOOK MAIN
//!BIND CELFLARE_GF0
//!BIND CELFLARE_GQ
//!SAVE CELFLARE_GBH
//!WIDTH CELFLARE_GF0.w
//!HEIGHT CELFLARE_GF0.h
//!DESC CelFlare: Highlight base reference H

// Reference for coring the highlight base (pass 9): a range-weighted Gaussian
// of the base B = a q + b at 1/4 resolution. Same-surface (a dark neighbour
// does not pull it down near an outline); symmetric pairs on a linear ramp
// get equal weights, so a ramp is reproduced exactly. Keep HLC_* equal in
// both reference passes.
#define HLC_SIGMA 6.0   // texels (x4 px)
#define HLC_RANGE 0.03   // gamma units

float hlc_b(vec2 p) {
    vec2 ab = CELFLARE_GF0_tex(p).rg;
    return ab.x * CELFLARE_GQ_tex(p).r + ab.y;
}

vec4 hook() {
    float c = hlc_b(CELFLARE_GF0_pos);
    float acc = c, wacc = 1.0;
    const float kr = 0.5 / (HLC_RANGE * HLC_RANGE);
    const float ks = 0.5 / (HLC_SIGMA * HLC_SIGMA);
    for (int i = 1; i <= int(2.0 * HLC_SIGMA); i++) {
        float fi = float(i);
        float bp = hlc_b(CELFLARE_GF0_pos + vec2(fi, 0.0) * CELFLARE_GF0_pt);
        float bm = hlc_b(CELFLARE_GF0_pos - vec2(fi, 0.0) * CELFLARE_GF0_pt);
        float ws = exp(-ks * fi * fi);
        float dp = bp - c, dm = bm - c;
        // pair weight = the weaker side's: a straight ramp stays exact
        float w = ws * exp(-kr * max(dp * dp, dm * dm));
        acc += w * (bp + bm);
        wacc += 2.0 * w;
    }
    return vec4(acc / wacc, c, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_GBH
//!BIND CELFLARE_GF0
//!SAVE CELFLARE_GF
//!WIDTH CELFLARE_GF0.w
//!HEIGHT CELFLARE_GF0.h
//!DESC CelFlare: Highlight base reference V

#define HLC_SIGMA 6.0
#define HLC_RANGE 0.03

vec4 hook() {
    vec2 h0 = CELFLARE_GBH_tex(CELFLARE_GBH_pos).rg;   // (row-filtered, own B)
    float c = h0.y;
    float acc = h0.x, wacc = 1.0;
    const float kr = 0.5 / (HLC_RANGE * HLC_RANGE);
    const float ks = 0.5 / (HLC_SIGMA * HLC_SIGMA);
    for (int i = 1; i <= int(2.0 * HLC_SIGMA); i++) {
        float fi = float(i);
        vec2 hp = CELFLARE_GBH_tex(CELFLARE_GBH_pos + vec2(0.0, fi) * CELFLARE_GBH_pt).rg;
        vec2 hm = CELFLARE_GBH_tex(CELFLARE_GBH_pos - vec2(0.0, fi) * CELFLARE_GBH_pt).rg;
        float ws = exp(-ks * fi * fi);
        float dp = hp.y - c, dm = hm.y - c;
        float w = ws * exp(-kr * max(dp * dp, dm * dm));
        acc += w * (hp.x + hm.x);
        wacc += 2.0 * w;
    }
    vec4 g = CELFLARE_GF0_tex(CELFLARE_GF0_pos);
    return vec4(g.rg, acc / wacc, g.b);
}

//!HOOK MAIN
//!BIND CELFLARE_SKQ
//!BIND CELFLARE_SKSM
//!SAVE CELFLARE_SKT
//!WIDTH CELFLARE_SKQ.w
//!HEIGHT CELFLARE_SKQ.h
//!DESC CelFlare: Skin texel weight

// Each 4x4 cell takes its surface's weight from the four nearest blocks
// (bilinear share x likeness to the cell's colour); its fit weight is its
// purity times how well it found its surface.
#define SKT_LOOK_K 225.0   // 1 / (2 * 0.047^2)
#define SKT_LOOK_L 0.25

vec3 skq_oklab(vec3 c) {
    vec3 lin = pow(max(c, 0.0), vec3(2.4));
    float l = pow(0.4122214708 * lin.r + 0.5363325363 * lin.g + 0.0514459929 * lin.b, 1.0 / 3.0);
    float m = pow(0.2119034982 * lin.r + 0.6806995451 * lin.g + 0.1073969566 * lin.b, 1.0 / 3.0);
    float s = pow(0.0883024619 * lin.r + 0.2817188376 * lin.g + 0.6299787005 * lin.b, 1.0 / 3.0);
    return vec3(0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s);
}

vec4 hook() {
    vec4 q = CELFLARE_SKQ_tex(CELFLARE_SKQ_pos);
    vec3 lab = skq_oklab(q.rgb);
    vec3 lab_px = vec3(lab.x, lab.y + 0.5, lab.z + 0.5);
    vec2 sp = CELFLARE_SKQ_pos * CELFLARE_SKSM_size - 0.5;
    vec2 sb = floor(sp);
    vec2 sf = sp - sb;
    float acc = 0.0, wacc = 0.0;
    for (int k = 0; k < 4; k++) {
        vec2 o = vec2(float(k & 1), float(k >> 1));
        vec4 s = CELFLARE_SKSM_tex((sb + 0.5 + o) * CELFLARE_SKSM_pt);
        vec2 bw2 = mix(1.0 - sf, sf, o);
        vec3 d = (s.rgb - lab_px) * vec3(SKT_LOOK_L, 1.0, 1.0);
        float w = bw2.x * bw2.y * exp(-SKT_LOOK_K * dot(d, d));
        acc += w * s.a;
        wacc += w;
    }
    float wt = acc / max(wacc, 1e-6);
    float om = q.a * (0.05 + 0.95 * clamp(wacc, 0.0, 1.0)) + 0.001;
    return vec4(wt, om, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_SKQ
//!BIND CELFLARE_SKT
//!SAVE CELFLARE_SKFA
//!WIDTH CELFLARE_SKQ.w
//!HEIGHT CELFLARE_SKQ.h
//!DESC CelFlare: Skin fit statistics

// Weighted least-squares fit of the cells' weight on their gamma rgb over
// 5x5 cells: W = A . rgb + B (guided filter with a colour guide). Pass 9
// evaluates it at every pixel with the pixel's own colour, so an
// anti-aliased pixel, whose colour lies between its two sides, gets a weight
// between theirs (no rim), and nothing in the result is tied to the grid.
#define SKF_EPS 0.0002   // gamma rgb units

vec4 hook() {
    float S = 0.0, sW = 0.0;
    vec3 sC = vec3(0.0);
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++) {
            vec2 p = CELFLARE_SKQ_pos + vec2(float(i), float(j)) * CELFLARE_SKQ_pt;
            vec2 t = CELFLARE_SKT_tex(p).rg;
            S += t.y; sW += t.y * t.x; sC += t.y * CELFLARE_SKQ_tex(p).rgb;
        }
    float mW = sW / S;
    vec3 mC = sC / S;
    float c00 = 0.0, c01 = 0.0, c02 = 0.0, c11 = 0.0, c12 = 0.0, c22 = 0.0;
    vec3 cw = vec3(0.0);
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++) {
            vec2 p = CELFLARE_SKQ_pos + vec2(float(i), float(j)) * CELFLARE_SKQ_pt;
            vec2 t = CELFLARE_SKT_tex(p).rg;
            vec3 d = CELFLARE_SKQ_tex(p).rgb - mC;
            float o = t.y;
            c00 += o * d.r * d.r; c01 += o * d.r * d.g; c02 += o * d.r * d.b;
            c11 += o * d.g * d.g; c12 += o * d.g * d.b; c22 += o * d.b * d.b;
            cw += o * d * (t.x - mW);
        }
    float is = 1.0 / S;
    c00 = c00 * is + SKF_EPS; c11 = c11 * is + SKF_EPS; c22 = c22 * is + SKF_EPS;
    c01 *= is; c02 *= is; c12 *= is; cw *= is;
    float k00 = c11 * c22 - c12 * c12;
    float k01 = c02 * c12 - c01 * c22;
    float k02 = c01 * c12 - c02 * c11;
    float k11 = c00 * c22 - c02 * c02;
    float k12 = c01 * c02 - c00 * c12;
    float k22 = c00 * c11 - c01 * c01;
    float det = c00 * k00 + c01 * k01 + c02 * k02;
    vec3 A = vec3(k00 * cw.x + k01 * cw.y + k02 * cw.z,
                  k01 * cw.x + k11 * cw.y + k12 * cw.z,
                  k02 * cw.x + k12 * cw.y + k22 * cw.z) / max(det, 1e-12);
    return vec4(A, mW - dot(A, mC));
}

//!HOOK MAIN
//!BIND CELFLARE_SKFA
//!SAVE CELFLARE_SKF
//!WIDTH CELFLARE_SKFA.w
//!HEIGHT CELFLARE_SKFA.h
//!DESC CelFlare: Skin fit coefficients

vec4 hook() {
    vec4 acc = vec4(0.0);
    for (int j = -2; j <= 2; j++)
        for (int i = -2; i <= 2; i++) {
            float w = float((3 - abs(i)) * (3 - abs(j)));
            acc += w * CELFLARE_SKFA_tex(CELFLARE_SKFA_pos + vec2(float(i), float(j)) * CELFLARE_SKFA_pt);
        }
    return acc * (1.0 / 81.0);
}

// =============================================================================
// PASSES 5-7: MOTION SENSE (block-match against the previous frame)
// =============================================================================
// A small optical-flow front end, so the pump can tell a panning lamp from
// new light (the sigma-80 field alone cannot: a feature that never holds
// still never looks "established").
//   5 MOTION_CUR   128x72 downsample of MAIN (max channel; luma in the
//                  subtractive build). One pump cell = 8x8 texels
//                  (120x120 px at 1080p).
//   6 MOTION_FLOW  per-cell block-match against CELFLARE_ADD_MOTION_PREV.
//   7 history      copies MOTION_CUR into CELFLARE_ADD_MOTION_PREV after
//                  pass 6 has read it (file order = execution order).
// Pass 8 warps last frame's cell values with the flow: light the motion
// cannot explain may open a cell mask, the rest is transport. The structure
// cost also drives the motion-cost reset (a cut detector).
// Init safety: pass 8 runs its per-cell motion block only when no transient
// reset is set, so an unprimed history or garbage flow never reaches
// persistent state; keep that block inside the reset guard.

//!HOOK MAIN
//!BIND HOOKED
//!SAVE MOTION_CUR
//!WIDTH 128
//!HEIGHT 72
//!DESC CelFlare: Motion analysis downsample
vec4 hook() {
    // Max channel, matching the V-aware pump driver: Rec.709 luma can nearly
    // hide a saturated blue or red light that drives the pump. Four corner
    // taps give a modest box AA so moving edges match without shimmer.
    vec2 o = vec2(0.5 / 128.0, 0.5 / 72.0);
    vec3 c0 = HOOKED_tex(HOOKED_pos + vec2(-o.x, -o.y)).rgb;
    vec3 c1 = HOOKED_tex(HOOKED_pos + vec2( o.x, -o.y)).rgb;
    vec3 c2 = HOOKED_tex(HOOKED_pos + vec2(-o.x,  o.y)).rgb;
    vec3 c3 = HOOKED_tex(HOOKED_pos + vec2( o.x,  o.y)).rgb;
#if cf_additive_pump
    float v = max(max(c0.r, c0.g), c0.b) + max(max(c1.r, c1.g), c1.b)
            + max(max(c2.r, c2.g), c2.b) + max(max(c3.r, c3.g), c3.b);
    return vec4(v * 0.25, 0.0, 0.0, 1.0);
#else
    // Luma, keeping the subtractive reference bit-exact.
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
// One thread per 16x9 cell: the truncated-SAD shift of its 8x8 MOTION_CUR
// tile against the previous frame within +-MOT_R texels (coarse 5x5
// even-offset search + 3x3 refine; subpixel in the additive build).
// Output rg = (dx, dy) in MOTION_CUR texels, the PREVIOUS-frame offset (where
// the content came from); b = motion-reset evidence (the winning SAD; stored
// negative where only the brightness changed, so it cannot vote; read by
// pass 8 and debug view 10); a = tile RMS contrast.
#define MOT_R       5      // search radius in MOTION_CUR texels (+-75 px/frame at 1080p); keep the hard-coded coarse lattice below in sync
#define MOT_TILE    8
#define MOT_SADCAP  0.10   // per-sample truncated SAD (robust to a lone outlier texel)
#define MOT_SADCAP_BRIGHT 0.30 // bright max-channel features are pump evidence, not outliers
#define MOT_SAD_BRIGHT_LO 0.35
#define MOT_SAD_BRIGHT_HI 0.70
#define MOT_STRUCT_VETO 0.25 // aligned cost below this x tile RMS contrast: only the brightness changed
#define MOT_BIAS    0.0005 // zero-motion prior: a FLAT tile (all shifts equal) resolves to (0,0), not the first corner searched; far below any real match difference
#define MOT_TAPS    16
#define MOT_CELLS   144
// One workgroup = the whole 16x9 output; each lane stores its cell's 16 taps
// once (tap-major, 9 KiB) and every candidate shift reuses them.
shared float s_motion_cur[MOT_TAPS * MOT_CELLS];

ivec2 motion_tap_offset(int t) {
    int tx = t & 3, ty = t >> 2;
#if cf_additive_pump
    // Parity alternates by row/column: no period-4 sampling nulls.
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
#if cf_additive_pump
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

    // 16 MOTION_CUR taps per cell into shared memory; the search then reads
    // at most 33 x 16 = 528 previous-frame samples per cell.
    for (int t = 0; t < MOT_TAPS; t++) {
        ivec2 o = motion_tap_offset(t);
        ivec2 c = org + o;
        s_motion_cur[t * MOT_CELLS + lane] =
            MOTION_CUR_tex((vec2(c) + 0.5) * MOTION_CUR_pt).r;
    }
    barrier();

    float best_rank = 1e9, best_sad = 1e9;
    ivec2 bestd = ivec2(0);
    // Coarse: -4..4 on even offsets (5x5); refine the winning basin by one
    // texel (can reach +-5). Periodic texture can pick the wrong basin.
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
    vec2 best_flow = vec2(bestd);
#if cf_additive_pump
    // Subpixel flow for A2 (the 16x9 V history is sensitive to integer
    // quantization at slow pans): a bounded quadratic fit on four cardinal
    // SAD probes inside the winning basin.
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
    // Kept out of the search's live range (FXC register pressure).
    float tsum = 0.0, tsum2 = 0.0;
    for (int t = 0; t < MOT_TAPS; t++) {
        float cv = s_motion_cur[t * MOT_CELLS + lane];
        tsum += cv;
        tsum2 += cv * cv;
    }
    float tmean = tsum * (1.0 / 16.0);
    float texture_rms = sqrt(max(tsum2 * (1.0 / 16.0) - tmean * tmean, 0.0));
    // Reset evidence for pass 8. The winning SAD is absolute, so a frame-
    // filling BRIGHTENING reads as a mismatch and would reset the pump on
    // textured fireballs and tunnel exits. The veto: align the previous tile
    // photometrically (mean removed, gain matched and clamped to [0.5, 2]),
    // take the capped SAD at the selected shift and at zero shift (a
    // brightness step biases the selection) and keep the better one. The
    // test is RELATIVE to the tile's contrast: a pure brightness change
    // leaves ~0, two unrelated tiles score ~1.13x the tile's RMS. Below
    // MOT_STRUCT_VETO x RMS the tile is vetoed (stored negative, never votes);
    // otherwise it publishes the larger of the absolute SAD and the aligned
    // cost. Neither an absolute veto (it swallows every low-contrast tile)
    // nor the aligned cost alone (two unrelated tiles score only ~1.1x their
    // contrast) keeps cut recall.
    float struct_cost = 1e9;
    for (int k = 0; k < 2; k++) {
        ivec2 dsel = (k == 0) ? bestd : ivec2(0);
        float pvs[MOT_TAPS];
        float mc = 0.0, mp = 0.0;
        for (int t = 0; t < MOT_TAPS; t++) {
            ivec2 p = org + motion_tap_offset(t) + dsel;
            float cv = s_motion_cur[t * MOT_CELLS + lane];
            bool inb = p.x >= 0 && p.y >= 0 && p.x < 128 && p.y < 72;
            float pv = inb ? imageLoad(CELFLARE_ADD_MOTION_PREV, p).r : cv;
            pvs[t] = pv;
            mc += cv;
            mp += pv;
        }
        mc *= 1.0 / 16.0;
        mp *= 1.0 / 16.0;
        float vc = 0.0, vp = 0.0;
        for (int t = 0; t < MOT_TAPS; t++) {
            float dc = s_motion_cur[t * MOT_CELLS + lane] - mc;
            float dp = pvs[t] - mp;
            vc += dc * dc;
            vp += dp * dp;
        }
        float gain = (vp > 1e-8) ? clamp(sqrt(vc / vp), 0.5, 2.0) : 1.0;
        float s = 0.0;
        for (int t = 0; t < MOT_TAPS; t++) {
            float cv = s_motion_cur[t * MOT_CELLS + lane];
#if cf_additive_pump
            float bright = smoothstep(MOT_SAD_BRIGHT_LO, MOT_SAD_BRIGHT_HI,
                                      max(cv, pvs[t]));
            float cap = mix(MOT_SADCAP, MOT_SADCAP_BRIGHT, bright);
#else
            float cap = MOT_SADCAP;
#endif
            s += min(abs((cv - mc) - gain * (pvs[t] - mp)), cap);
        }
        struct_cost = min(struct_cost, s * (1.0 / 16.0));
    }
    float sad_mean = best_sad * (1.0 / 16.0);
    float reset_cost = (struct_cost < MOT_STRUCT_VETO * texture_rms)
                     ? -sad_mean : max(sad_mean, struct_cost);
    imageStore(out_image, cell, vec4(best_flow, reset_cost, texture_rms));
}

//!HOOK MAIN
//!BIND MOTION_CUR
//!BIND CELFLARE_ADD_MOTION_PREV
//!SAVE MOTION_HIST
//!WIDTH 128
//!HEIGHT 72
//!COMPUTE 16 16
//!DESC CelFlare: Motion history store
// Copies MOTION_CUR into persistent CELFLARE_ADD_MOTION_PREV after the
// block-match has read it. MOTION_HIST is a required dummy SAVE target.
void hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= 128 || p.y >= 72) return;
    float y = MOTION_CUR_tex((vec2(p) + 0.5) * MOTION_CUR_pt).r;
    imageStore(CELFLARE_ADD_MOTION_PREV, p, vec4(y, 0.0, 0.0, 1.0));
    imageStore(out_image, p, vec4(y, 0.0, 0.0, 1.0));
}

// =============================================================================
// PASS 8: FRAME STATS (compute, one 144-lane workgroup)
// =============================================================================
// Samples the illumination field and the source on the 16x9 cell grid and
// keeps all frame-level state in the CELFLARE_ADD_STATE buffer: scene
// statistics, cut detection and the light-pump state machine.
// Layout: COMPUTE 16 9 = one workgroup, one cell per lane, 1x1 dummy output.
//  - Every lane samples its cell, computes its cut delta and snapshots its
//    pump lanes into shared memory (own slots, race-free); in the additive
//    build it also runs the local V-flow matcher.
//  - Thread 0 computes the scene statistics and every frame-global decision
//    and broadcasts them through shared memory.
//  - All lanes run the per-cell pump update, mask publish and finishing blur;
//    cross-cell reads use frozen pre-update snapshots, so order is irrelevant.
//  - Thread 0 sums the per-cell terms IN INDEX ORDER with the same
//    accumulate expressions as a serial cell loop (bit-exact), then runs the
//    scalar pump tail.
// Every barrier sits at top level (no return anywhere in the pass): FXC
// X3663 requires barriers in uniform control flow.
// In the additive build the scalar pump is never applied to the picture; it
// still feeds the cut classifier's "event in progress" arm and debug view 6.

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

// Above pass 9's KNEE (0.30) on purpose: PEAK_ATTEN, BRIGHT_FRAC_REF and the
// growth floor are calibrated against bright_frac at 0.40.
#define BRIGHT_STAT_THRESH  0.40

// LETTERBOX / PILLARBOX BAR EXCLUSION. Bar cells would pin the contrast
// extrema (cover gate and dynamic intensity at max on all letterboxed
// content), dilute the tier fractions, and keep windowboxed content (60 live
// cells) from ever reaching the cut threshold. Detection is geometric and
// persistent: a candidate row (0,1,7,8) or column (0,1,14,15) is a bar only
// while EVERY cell in it has source Y <= 0.001 for LB_ENGAGE_FRAMES frames in
// a row; content in it resets the run. Center cells are never candidates, so
// real night-scene blacks stay in the statistics. Engaged bar cells leave the
// contrast extrema, the tier sums and the cut count (numerators and
// denominators). The pump p-norm stays a mean over all 144 cells (bars
// dilute it ~3-6 %), so the drive never steps at the engage frame. Dirty
// bars or hardsubs in a bar keep that bar un-engaged.
#define LB_ENGAGE_FRAMES    120.0   // ~5s @24p of all-black before a row/col is a bar

// Specular detection (source brightness tiers, no illum field)
#define HIGHLIGHT_THRESH    0.75    // Source highlight tier
#define SPECULAR_THRESH     0.92    // Source specular tier
// Soft tier membership: 144 single-texel samples with hard compares make the
// fractions jump in 1/144 steps on a pan, right through the shutoff bands
// (scene-wide specular flicker). Each compare is a smoothstep over +-this.
// Keep > 0 (smoothstep with equal edges is undefined in GLSL).
#define TIER_SOFT_HALFBAND  0.02
// (The bright-spec sat fence below stays a hard compare.)
#define SPEC_FRAC_MIN       0.007   // ~1/144 noise floor
#define SPEC_FRAC_MAX       0.10    // Fade begins (sparkle clusters can reach 12-15%)
#define SPEC_FRAC_CEIL      0.16    // Full shutoff (large bright skies)
#define SPARSE_SPEC_CEIL    0.02    // Below this spec_frac, "sparse points against dark" fires alongside tier_gate — catches candles/LEDs/stars

// Bright-scene specular recovery: when most of the frame is above
// SPECULAR_THRESH the normal detection collapses. A stricter 0.97 tier
// restores separation (chrome at noon, sun glints, headlights in daylight).
// Keyed on smoothed_log_avg so it adds no step to spec_vel; it can only add.
#define BRIGHT_SPEC_THRESH    0.97  // Super-specular threshold for bright scenes
#define BRIGHT_SCENE_LOW      0.20  // smoothed_log_avg below: normal detection only
#define BRIGHT_SCENE_HIGH     0.35  // smoothed_log_avg above: bright fallback active
// Its own shutoff separates sparse specular (bs_frac <= ~0.10) from broad
// white surfaces (cel shirts and walls, snow, a sun disc with glare:
// ~0.12-0.30), which a wider window lifted 20-40 nits ("too hot").
#define BRIGHT_SPEC_FRAC_MAX  0.10  // Recovery starts fading at 10% of cells > 0.97
#define BRIGHT_SPEC_FRAC_CEIL 0.28  // Recovery fully off at 28% of cells > 0.97
// It counts NEAR-WHITE cells only: with V counting, a large saturated surface
// (a pink carpet, sat 0.40-0.88) would pose as "sparse chrome" and arm
// growth mode on 4:2:0 chroma noise. Saturated emissives in the dark still
// count through the normal tier.
#define BRIGHT_SPEC_SAT_MAX   0.30  // bright-spec tier counts only cells with sat below this

// Spec tiers count V = max(R,G,B), so a saturated primary (red LED: Y 0.21,
// V 1.0) qualifies; neutrals are unchanged (V >= Y). The tiers now feed only
// growth mode (spec_vel), not a picture term of their own.
#define ENABLE_SATURATED_SPEC 1

// Scene diffuse-white anchor, read by pass 9's top lift (the lift starts at
// it). Painted whites (shirts, white hair, eye whites at 0.93-0.97 in most
// productions) must not be lifted, in any framing. The anchor is the level of
// the BROAD near-white fields: a
// power mean (p 32, close to their maximum: a shaded white such as a cloud
// must anchor at its lit top, or the top quarter of every cloud qualifies as
// a highlight) of the source luma over sample cells that sit inside one
// (bright coverage, see pass 2), valid from about two cells. It rises quickly (at once on a cut) and releases slowly,
// so a close-up without white in frame keeps the scene's anchor; a black
// frame releases it. SPEC_ANCHOR_NONE (below the ramp onset) means "no
// diffuse white known": pass 9 then measures the lift from its floor
// (HL_W_MIN 0.90).
#define SPEC_ANCHOR_NONE    0.88
#define SPEC_ANCHOR_COV_LO  0.35
#define SPEC_ANCHOR_COV_HI  0.60
#define SPEC_ANCHOR_ATTACK  0.25    // per frame inside a shot
#define SPEC_ANCHOR_HOLD_S  30.0    // release time constant, seconds at 24 fps (10 let eye whites creep up inside one close-up)
#define SPEC_ANCHOR_HOLD_CONF_S 4.0 // release when the frame has a confident lower diffuse white (a flash or a fade
                                   // through white held the anchor for a minute on the 30 s constant)
#define SPEC_ANCHOR_PERSIST 1       // rise only to what two frames in a row agree on (0 = rise on one frame)

// Velocity-adaptive temporal alpha: still scenes SLOW, quick lighting
// changes MID, cuts FAST. vel_mag = max(|d bright_frac|, |d log_avg|).
// KNOWN LIMITATION: every temporal constant is per rendered video frame,
// tuned at 24p; 60 fps content runs every EMA ~2.5x faster (snappier, never
// artifacts). User shaders get no PTS, and fps estimation is unsound on
// anime (held cels are identical frames).
#define TEMPORAL_ALPHA_SLOW 0.03    // Stable scenes
#define TEMPORAL_ALPHA_MID  0.12    // Quick lighting / brightness shifts
#define TEMPORAL_ALPHA_FAST 0.9     // Scene cut + lockout
#define ADAPT_DELTA_LOW     0.02    // Below: slow alpha
#define ADAPT_DELTA_HIGH    0.10    // Above: mid alpha
#define LOCKOUT_FRAMES      6.0
#define ILLUM_CHANGE_THRESH 0.06
#define SCENE_CUT_PCT       0.50
// Event-vs-cut classifier. The majority-vote cut test also fires on the
// pump's own target (a frame-filling brightening moves most cells > 0.06),
// and the reset would drop the pump at the climax. A nearly all-POSITIVE
// change field while the pump was already presenting or charging (values
// read before their update) is treated as the event continuing. A cut during
// an event is caught by the motion-cost reset instead, when the new shot has
// texture (a cut into a near-white or flat shot is missed for ~8 frames; a
// brightness step that keeps the picture's layout, e.g. lights coming on,
// is treated as the event).
// Residuals: a mid-scene flash's return frame is all-negative and still
// cuts; a slow drift >= ~0.009/s keeps the env arm live while env holds.
#define CUT_EVENT_POS_FRAC  0.80   // >= this fraction of changed cells rising -> "same-sign rise"
#define CUT_EVENT_ENV_MIN   0.10   // prior pump_env above: event presenting (0.05 kept the arm
                                   // live 1-2 min after an event whose light remains)
#define CUT_EVENT_DRIVE_MIN 0.015  // prior drive above (~PUMP_DRIVE_LOW/2) AND rising. The rising
                                   // conjunct is load-bearing: slow ambient drift (push-ins,
                                   // sunrise fades) reaches a bare 0.015 and would suppress
                                   // real cuts; an attack grows frame over frame, drift is flat.
#define CUT_EVENT_DRIVE_EPS 0.001  // minimum frame-over-frame drive growth for "rising"
// Strobe refractory: cut_rate is an EMA of cut fires (+0.30 per fire, -0.02
// per frame ~ 2 s). Isolated cuts never reach STROBE_LO; sustained strobing
// (~0.6+) blends the lockout alpha back toward MID so EMAs are not held at
// the cut alpha. WARNING: strobe_t must read the PRE-update cut_rate; with
// the post-update read the SECOND cut of any pair within ~3 s crosses
// STROBE_LO and ordinary shot-reverse-shot loses its lock-on.
#define CUT_RATE_RISE       0.30
#define CUT_RATE_FALL       0.02
#define CUT_RATE_STROBE_LO  0.35
#define CUT_RATE_STROBE_HI  0.60

// Growth mode: an expanding bright object (fireball, crash-zoom on a backlit
// window) lets pass 9 bypass the bright-scene dampeners. Signature: spec_vel
// rising faster than bright_vel (the hot core saturates first) AND contrast
// climbing (a fade to white loses range); frac_floor rejects title text.
#define GROWTH_SPEC_BIAS       0.4    // weight of bright_vel subtracted from spec_vel
#define GROWTH_SIG_LOW         0.015  // smoothstep onset on (spec_vel - bias*bright_vel)
#define GROWTH_SIG_HIGH        0.06
#define GROWTH_C_GATE_LOW      0.05   // contrast_vel onset (a zero edge let pan ripples
                                      // keep c_gate open and breathe the base curve)
#define GROWTH_C_GATE_HIGH     0.25   // smoothstep saturation on contrast_vel
#define GROWTH_FRAC_FLOOR_LOW  0.04   // smoothed_bright_frac required to activate
#define GROWTH_FRAC_FLOOR_HIGH 0.10
#define GROWTH_SPEC_CELLS_LO   1.0    // corroboration: growth needs multi-cell spec evidence
#define GROWTH_SPEC_CELLS_HI   2.5    // (one glint on one sample point held spec_vel high
                                      // ~90 frames; a real fireball crosses 2-3 cells fast)

// Light-pump detector for sudden SUSTAINED brightening (explosion bloom,
// tunnel exit, spell): a band-pass (fast lane minus slow lane) of the
// frame's illumination-V statistic, positive only during a multi-frame RISE.
// The DRIVE self-releases at plateau; the held ENV relaxes only through
// PUMP_ADAPT_FLOOR and the velocity-matched release on real falls. 2-3 frame
// flashes reverse before the fast lane builds. PUMP_ALPHA_FAST is the
// flash-vs-sustained dial (lower = more rejection, slower attack).
#define PUMP_ALPHA_FAST     0.18   // fast lane (~5-frame time constant): sets attack speed
#define PUMP_ALPHA_SLOW     0.04   // slow baseline lane (~25 frame time constant)
#define PUMP_DRIVE_LOW      0.03   // band-pass onset — below this, no pump
#define PUMP_DRIVE_HIGH     0.20   // band-pass saturation — full pump needs a steep rise (reserves full for violent events)
// The drive statistic is the p-NORM of the 144 cell V samples: highlights
// dominate V^p, so a local event (two cells of fire in a dark scene) moves it
// several times more than the mean, and an occluder covering a fraction f of
// bright content dips it by only 1-(1-f)^(1/p) (~12 % at f=0.4, p=4). A
// uniform rise moves p-norm and mean alike, so DRIVE_LOW/HIGH keep their
// meaning for global events.
// drive_loc: the p-norm is diluted by bright mass already in frame (a 2-cell
// fire moves it ~0.14 in a black frame, ~0.005 next to a sky), so the scalar
// has a second onset source, a signed p-mean of the per-cell drives:
//   drive_loc = sign(S) * (|S|/144)^(1/p),  S = sum sign(d_i) * |d_i|^p
// Static content (d ~ 0) adds nothing. The SIGN is the safety: a moving
// occluder or a pan pairs every rise with a fall and cancels. KEEP IT SIGNED
// (a rectified sum re-opens the crossing-trail class). The onset uses
// drive_on = drive_loc with rises through the established-level gate;
// pump_gate takes max(drive, drive_on).
// Release: when drive_loc < 0 the env follows the falling cells' own
// frame-to-frame fast ratio. Do NOT divide drive_loc by the frame level
// (that guillotines the env on dark frames). Cancellation fails on
// net-asymmetric transitions (a dark occluder leaving, a bright object
// entering on a pan); these pump like tunnel exits. If that reads wrong,
// lower P toward 2.
#define PUMP_DRIVE_P        4.0    // highlight weighting of BOTH drive statistics. A/B 2.0-6.0
// Spatial pump: a per-cell band-pass on the sigma-80 V at each cell center
// gives a [0,1] "is this region brightening" env (additive: the local
// amplitude; subtractive: it only suppresses the scalar). Always on: a
// scalar-only mode false-pumped real camera pans (+27 % peak).
// PUMP_CELL_DEADZONE is the drive_loc noise floor only (shrinks |d| on both
// signs, so cancellation holds); the mask onset is PUMP_CELL_DRIVE_LOW/HIGH.
#define PUMP_CELL_DEADZONE  0.01
// Per-cell drive thresholds. LOW sits a touch above the scalar onset to
// reject sigma-80 neighbour bleed and idle wobble; HIGH is where a real
// in-cell brightening saturates. A ramp of rate rho per frame settles at
// d = rho * ((1-as)/as - (1-af)/af) = 19.4 * rho, so the 0.05 knee is a
// ~0.06 V/s opening floor at 24p. Faces ramping into key light also clear
// it; separating them is the excursion gate's job, not this knee's.
#define PUMP_CELL_DRIVE_LOW  0.05   // per-cell band-pass onset (mask starts opening — region is brightening)
#define PUMP_CELL_DRIVE_HIGH 0.15   // per-cell saturation (mask fully open — region clearly brightening)
// ESTABLISHED-LEVEL GATE. A rising cell is FRESH light only once its fast
// lane exceeds the highest ESTABLISHED (slow-lane) level among its ring-2
// neighbours by this margin. A rise that only converges to a level the
// neighbourhood already holds (an occluder retreating, a pan, a shrinking
// silhouette) is not a light event: it adds nothing to the ONSET sum (falls
// still count) and may not OPEN a mask. Release keeps the ungated sum. On a
// reveal test clip: onset 0.15 -> 0.000, mask-open cell-frames 3016 -> 75.
// Trade: a fire within 2 cells of an equally bright standing region waits
// until it exceeds that level. MARGIN 0.02 leaks on the reveal clip,
// 0.03-0.05 hold. Ring 2 matches the sigma-80 smear; wider only widens the
// trade.
#define PUMP_ESTABLISH_MARGIN 0.03
// 1 = a cell mask may only OPEN on a fresh rise (closing is never gated).
// The additive build applies its whole opening proof through this flag, so
// it is required there (#error below). 0 = open on any rise (subtractive).
#define PUMP_MASK_ESTABLISH 1
// Subtractive floor: the signed local drive sits near 0 for a
// brightness-conserving pan and rises for emission; above MC_EMIT_HI every
// rising cell may open (the scalar owns amplitude in that build).
#define MC_EMIT_LO        0.02
#define MC_EMIT_HI        0.08
// Subtractive motion gate: a mask may open only where V exceeds last frame's
// V warped by the flow (light motion cannot explain). A panning lamp warps
// its own level into the cell and stays shut; an off-frame warp is influx.
// A wrong flow is permissive here, so additive never uses this one-frame
// residual as opening authority.
#define MC_RES_LO         0.02   // motion-compensated residual (new light) below this: transport -> mask shut
#define MC_RES_HI         0.08   // above: clearly new light -> mask opens
#define SPATIAL_PUMP_ADDITIVE cf_additive_pump
#define ADDITIVE_OPEN_GUARD  SPATIAL_PUMP_ADDITIVE
#define ADDITIVE_STATE_EPOCH (3 * ADDITIVE_OPEN_GUARD)
#if ADDITIVE_OPEN_GUARD && !PUMP_MASK_ESTABLISH
#error The additive opening proof is applied through PUMP_MASK_ESTABLISH
#endif
// Additive A2 opening proof. The established-level gate owns WHERE an
// opening may happen. Motion then asks what fraction of the cell's fast-lane
// rise survives a warp to the previous offset (transport loses it, emission
// keeps it). The fast lane also rises 4-5 frames past a source peak, so
// proof frames also need a not-falling SOURCE and an excursion above the
// cell's own very-slow baseline. The proof takes seven COMPLETE routed frames
// (2-6 frame flow-error bursts stay shut); its credit follows a moving event.
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
// Excursion: an opening must stand this far above the cell's OWN very-slow
// baseline (a face ramping into key light is ~0.15 at proof time, an
// ignition >= 0.4). The EFFECTIVE floor is the LO/HI midpoint ~0.31 (the
// route must clear ADD_PERSIST_ROUTE_MIN 0.5): do not lower LO expecting a
// 0.22 floor. A monotone rise of total V delta >= 0.45 still opens at any
// speed (big slow dissolves are the accepted residual).
// ADD_SRC_FALL_DZ gates AMPLITUDE (proved_open), NOT the persist counter.
#define ADD_EXCURSION_LO         0.22
#define ADD_EXCURSION_HI         0.40
#define ADD_SRC_FALL_DZ          0.01
#define ADD_VFLOW_COST_MAX       0.03
#define ADD_VFLOW_SAD_CAP        0.10
#define ADD_VFLOW_BIAS           0.0001
// Second, additive-only route: a local matcher on the 16x9 V history (a
// compact source can sit in one raw tile while its sigma-80 tail opens the
// next cell). A demeaned 3x3 shape SAD picks the translation; the warp is
// applied to the ABSOLUTE fast history, so real growth stays residual. Both
// routes are veto-only; proof credit follows the raw primary trajectory.
// Motion-cost reset: texture-qualified tiles vote a discontinuity (flat
// tiles carry no evidence). A majority bad match with >= 10 % qualified
// coverage is a cut the brightness vote missed: all pump lanes re-pin and
// the cut lockout starts. A false reset cannot add gain but can drop a held
// event, hence the high bad-fraction knee. cf_debug=10 shows the vote.
// MOTION_STATE_EPOCH is an exactly representable schema token and the
// state-init key (not a seek detector). Bump it when the MOTION_FLOW
// format, resolution or sign convention changes.
#define MOTION_STATE_EPOCH       51705.0
#define MOTION_COST_RESET        1
#define MOTION_COST_BAD          0.075   // mean winning SAD after the brightening veto: suspect above
#define MOTION_TEXTURE_LO        0.004   // tile RMS contrast: flat below
#define MOTION_TEXTURE_HI        0.025   // tile RMS contrast: reliable evidence above
#define MOTION_BAD_FRAC_RESET    0.65    // texture-qualified bad-match fraction
#define MOTION_BAD_COVER_MIN     0.10    // qualified evidence mass / 144 required to reset
#define MOTION_DEBUG_VIEWS       (cf_debug == 10 || cf_debug == 11)
// EDGE ESTABLISHMENT: reveal safety for content ENTERING through the picture
// border. A bright object sliding in from off frame has no on-screen history,
// outruns the slow lane and would read FRESH in every cell it crosses (its
// matching fall is off-screen). Three parts:
//  (1) BORDER SEED: a rising outer-ring cell of the PICTURE (bar lines
//      excluded) is presumed influx: its mask stays shut, it marks itself
//      (pump_seed_cell) and fast-establishes.
//  (2) FAST-ESTABLISH: a rising non-fresh cell whose gating neighbour carries
//      the influx marker settles its slow lane to its fast lane THIS frame, so
//      the "established" verdict follows the front inward at up to ring-2
//      per frame (~240 px/frame at 1080p). Unmarked anchors (a real sky,
//      lamp or fire) never propagate, and the anchor must sit a STEP
//      (PUMP_EDGE_STEP_MARGIN) above this cell's fast lane, so a uniform fade
//      cannot chain or self-latch.
//  (3) GLOBAL GATE: the seed disengages by the FRACTION of the deep interior
//      that is rising (a frame-wide rise must not gate inward from every
//      edge; a few hot central cells cannot trip it).
// The BORDER MIRROR in the publish step gives an outer cell its inward
// neighbour's amplitude, so a real event pumps to the edge while pure influx
// inherits nothing. Residuals: an event starting purely at the edge reads as
// influx and under-pumps; influx faster than ~1.3 cells/frame (~155 px/frame
// at 1080p) glows as it would without this feature (never more).
#define PUMP_EDGE_ESTABLISH        1
#define PUMP_EDGE_ESTABLISH_ALPHA  1.0    // 1 = settle slow->fast in one frame (max propagation reach); lower = gentler
#define PUMP_EDGE_STEP_MARGIN      0.03   // influx marker propagates only across a step this big (neighbour established above this cell's fast) — rejects uniform co-rise (anti-latch)
#define PUMP_EDGE_GLOBAL_EPS       0.02   // per-cell rise above which a deep-interior cell counts as "rising"
#define PUMP_EDGE_GLOBAL_FRAC_LO   0.40   // fraction of the deep interior rising below this: localized -> seed armed
#define PUMP_EDGE_GLOBAL_FRAC_HI   0.75   // above this: frame-global rise -> border seed fully disengaged
// Mask softening (presentation only): the proof can authorize just the hot
// core of one broad opening, leaving its surrounding light tiled. The
// PUBLISHED mask (pump_mask_cell, sampled by pass 9) is max(env, a 5x5
// weighted max-stencil). As smoothed bright coverage rises over 0.10..0.25 the
// one-cell skirt grows to a rounded two-cell skirt (isolated-cell bounds at
// full gate: 0.56 cardinal / 0.38 diagonal at one cell, 0.14 at two) and
// proved neighbours merge into one light volume; pass 9's pump_w trims it to
// the source's bright shape. Max, never a sum; coverage sets width, never
// amplitude; no authorized cell = an exact-zero stencil. Dynamics stay on
// pump_env_cell. Subtractive (or PUMP_MASK_BLOB5=0): a 3x3 binomial skirt.
#define PUMP_MASK_SOFTEN    1
#define PUMP_MASK_BLOB5     1
#define PUMP_MASK_BLOB_GAIN 6.0
#define PUMP_MASK_BLOB_FRAC_LO 0.10
#define PUMP_MASK_BLOB_FRAC_HI 0.25
// Finishing pass: a coverage-gated 3x3 binomial blur connects the skirt; raw
// authorized cores are restored afterwards, so it cannot lower a peak.
// Isolated-core cardinal tail at full gate: 0.459, 0.176, 0.029 at 1..3 cells.
#define PUMP_MASK_FINISH     1
#define PUMP_MASK_FINISH_MIX 1.0
// SPATIAL MODEL, selected by SPATIAL_PUMP_ADDITIVE:
//  - SUBTRACTIVE (cf_additive_pump=0, the verified reference): pump_local =
//    pump_env x mask. The mask can only REMOVE the global scalar from
//    non-brightening regions, so reveal/ghost/trail artifacts cannot appear.
//  - ADDITIVE (default): pump_local = mask x cover. Each cell's gated env IS
//    its amplitude, so regional events pump at their own strength and rhythm
//    without the frame statistics. Safe only because every mask OPENING
//    passes the established-level gate, the edge rules and the A2 proof.
// Release is VELOCITY-MATCHED (it follows the source's own fall), so a held
// light never releases on a timer. PUMP_ADAPT_FLOOR is the only clock: an
// imperceptibly slow relaxation of a held light, like the eye adjusting.
// Half-life = ln(0.5)/ln(F): 0.999 ~ 29 s at 24p, 0.9995 ~ 58 s, 1.0 = hold.
#define PUMP_ADAPT_FLOOR    0.999   // held-light relaxation; raise toward 1.0 = even slower/imperceptible
// Fade guard via CONTRAST RETENTION (log2 range of the field, in stops): a
// real event keeps a hot core against a darker surround, a fade to white or
// any uniform color collapses contrast and mutes the pump gradually.
#define PUMP_CONTRAST_LOW   1.0    // below this (≈uniform): pump fully muted
#define PUMP_CONTRAST_HIGH  2.5    // above this (structured frame): full pump
// Cover fall rate: the cover multiplies the HELD pump, so a one-frame
// contrast drop (e.g. the letterbox exclusion engaging) must not yank it.
// It falls no faster than this ratio per frame (1 -> 0.15 in ~12 frames).
#define PUMP_COVER_FALL     0.85
#define PUMP_RESET_DECAY    0.4    // a transient reset multiplies the PRESENTATION values (pump_env,
                                   // cover, env/mask cells, growth mode) by this per frame
                                   // instead of zeroing: a 2-3 frame ease inside the cut's
                                   // masking window. State lanes hard-pin; init restarts at 0.
#define PUMP_COVER_RISE     0.25   // ADDITIVE only: max cover rise per frame. The mask holds
                                   // near-full amplitude while cover dips, so an instant
                                   // re-open would re-apply the whole gain in one frame.

float get_luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Per-lane scratch. The tier counts are re-derived by thread 0 from these raw
// values, so each threshold sits next to the sum it feeds.
shared float s_illum[144];
shared float s_log_luma[144];
shared uint  s_valid[144];
shared uint  s_change[144];
shared float s_intensity[144];      // spec/highlight tier source: V or Y per ENABLE_SATURATED_SPEC
shared float s_cov[144];            // bright coverage at the cell (CELFLARE_ILLUM.a), for the spec anchor
shared float s_sat[144];            // near-white fence input for the bright-scene recovery counter
shared float s_illum_v[144];        // max(R,G,B) of the illum field — V-aware pump driver/guard
// Motion gate inputs: last frame's cell V and this cell's flow (own slots).
shared float s_prev_v[144];
shared vec2 s_flow[144];
#if ADDITIVE_OPEN_GUARD
shared vec2 s_add_vflow[144];
shared float s_add_vflow_cost[144];
#endif
#if MOTION_COST_RESET || MOTION_DEBUG_VIEWS
shared float s_flow_cost[144];
shared float s_flow_texture[144];
#endif
// Frozen pre-update snapshots of the cell lanes (own slot per lane). Every
// cross-cell read goes through them, so cells update in place in parallel.
shared float s_pump_snap_f[144];
shared float s_pump_snap_s[144];
#if PUMP_MASK_FINISH && PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
// Presentation field for the finishing blur.
shared float s_pump_shape[144];
#endif
#if ADDITIVE_OPEN_GUARD
shared float s_pump_snap_vs[144];
shared float s_pump_snap_persist[144];
#endif
#if PUMP_EDGE_ESTABLISH
// Influx-origin marker: 1 = traces back to a border seed, 0 = a real interior
// established region. Only a marked anchor may propagate fast-establish.
shared float s_pump_snap_seed[144];
#endif
// Per-lane results of the parallel cell update, summed by thread 0 in index
// order with the original accumulate expressions (bit-identical reductions).
shared float s_pump_env_post[144];   // post-update env (publish/finish input)
shared float s_red_w[144];           // |d|^p onset/release weight
#if PUMP_EDGE_ESTABLISH
shared float s_red_edge[144];        // edge_seed debit of a fresh onset
#endif
shared uint  s_red_on[144];          // 1 = fresh onset (+w), 2 = non-rising (-w), 0 = none
shared uint  s_red_fall[144];        // 1 = fast lane falling this frame
shared float s_red_fall_r[144];      // its frame-to-frame fast ratio
// Thread-0 decisions broadcast to the per-cell phases.
shared uint  s_bc_reset;
shared uint  s_bc_init;            // state_init: presentation state restarts from 0
shared float s_bc_global_gate;
shared float s_bc_emit;
shared float s_bc_blob_gate;
shared float s_bc_finish_mix;
shared int   s_bc_lb[4];             // picture border cells: left, right, top, bottom

// Bilinear sample of the frozen previous 16x9 V field (callers own the
// in-bounds verdict; the clamp keeps the reads valid).
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

// Catmull-Rom reconstruction of the frozen previous fast-lane field: removes
// the curvature residual that bilinear left on slow broad lamps (a false
// "emission"). Only the downward overshoot is clamped: an upward one merely
// vetoes, a downward one could manufacture a positive residual.
float sample_prev_fast_cubic(vec2 warpc) {
    vec2 cc = clamp(warpc, vec2(0.0), vec2(15.0, 8.0));
    int bx = int(floor(cc.x)), by = int(floor(cc.y));
    int xm = max(bx - 1, 0), x0 = bx, x1 = min(bx + 1, 15), x2 = min(bx + 2, 15);
    int ym = max(by - 1, 0), y0 = by, y1 = min(by + 1, 8), y2 = min(by + 2, 8);
    float fx = cc.x - float(bx), fy = cc.y - float(by);
    float r0 = cubic_prev_v(s_pump_snap_f[ym * 16 + xm], s_pump_snap_f[ym * 16 + x0],
                            s_pump_snap_f[ym * 16 + x1], s_pump_snap_f[ym * 16 + x2], fx);
    float r1 = cubic_prev_v(s_pump_snap_f[y0 * 16 + xm], s_pump_snap_f[y0 * 16 + x0],
                            s_pump_snap_f[y0 * 16 + x1], s_pump_snap_f[y0 * 16 + x2], fx);
    float r2 = cubic_prev_v(s_pump_snap_f[y1 * 16 + xm], s_pump_snap_f[y1 * 16 + x0],
                            s_pump_snap_f[y1 * 16 + x1], s_pump_snap_f[y1 * 16 + x2], fx);
    float r3 = cubic_prev_v(s_pump_snap_f[y2 * 16 + xm], s_pump_snap_f[y2 * 16 + x0],
                            s_pump_snap_f[y2 * 16 + x1], s_pump_snap_f[y2 * 16 + x2], fx);
    float v = cubic_prev_v(r0, r1, r2, r3, fy);
    float c00 = s_pump_snap_f[y0 * 16 + x0], c10 = s_pump_snap_f[y0 * 16 + x1];
    float c01 = s_pump_snap_f[y1 * 16 + x0], c11 = s_pump_snap_f[y1 * 16 + x1];
    float vlo = min(min(c00, c10), min(c01, c11));
    return max(v, vlo);
}

float sample_prev_fast_linear(vec2 warpc) {
    vec2 cc = clamp(warpc, vec2(0.0), vec2(15.0, 8.0));
    int x0 = int(floor(cc.x)), y0 = int(floor(cc.y));
    int x1 = min(x0 + 1, 15), y1 = min(y0 + 1, 8);
    float fx = cc.x - float(x0), fy = cc.y - float(y0);
    return mix(mix(s_pump_snap_f[y0 * 16 + x0],
                   s_pump_snap_f[y0 * 16 + x1], fx),
               mix(s_pump_snap_f[y1 * 16 + x0],
                   s_pump_snap_f[y1 * 16 + x1], fx), fy);
}

// Bilinear persistence split into x = pre-proof credit (mature corners give
// 0), y = mature-corner support, z = support-weighted mature TTL. As one
// scalar, a diluted mature token could pose as partial proof in a neighbour.
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

void hook() {
    uint lid = gl_LocalInvocationIndex;    // 0..143
    uint ix  = gl_LocalInvocationID.x;     // 0..15
    uint iy  = gl_LocalInvocationID.y;     // 0..8

    // Cell center in normalized texture coordinates.
    vec2 spos = vec2((float(ix) + 0.5) / 16.0,
                     (float(iy) + 0.5) / 9.0);

    // -------- per-cell sampling (parallel) --------
    vec3  illum_rgb = CELFLARE_ILLUM_tex(spos).rgb;
    float Y_ill   = get_luma(illum_rgb);
    float V_ill   = max(max(illum_rgb.r, illum_rgb.g), illum_rgb.b);  // V-aware pump driver/guard
    vec3  rgb_src = HOOKED_tex(spos).rgb;
    float Y_src   = get_luma(rgb_src);

    bool valid = Y_src > 0.001;
    // Near-white fence input for the bright-scene recovery (see
    // BRIGHT_SPEC_SAT_MAX).
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
    s_cov[lid]       = CELFLARE_ILLUM_tex(spos).a;
    // Pre-update snapshot of this cell's pump lanes (own slot).
    s_pump_snap_f[lid] = pump_fast_cell[lid];
    s_pump_snap_s[lid] = pump_slow_cell[lid];
    #if ADDITIVE_OPEN_GUARD
    s_pump_snap_vs[lid] = pump_very_slow_cell[lid];
    s_pump_snap_persist[lid] = pump_open_persist_cell[lid];
    #endif
    #if PUMP_EDGE_ESTABLISH
    s_pump_snap_seed[lid] = pump_seed_cell[lid];   // prev-frame influx-origin marker
    #endif
    // Motion gate inputs (own slot): stash last frame's V, advance the
    // history, and read this cell's flow (MOTION_FLOW is 16x9).
    vec4 motion_sample = MOTION_FLOW_tex(spos);
    s_prev_v[lid]     = prev_illum_v[lid];
    prev_illum_v[lid] = s_illum_v[lid];
    s_flow[lid]       = motion_sample.xy;
    #if MOTION_COST_RESET || MOTION_DEBUG_VIEWS
    s_flow_cost[lid]    = motion_sample.z;
    s_flow_texture[lid] = motion_sample.w;
    #endif

    // Scene-cut delta, own slot. Sign-encoded (0 quiet, 1 rise, 2 fall) for
    // the event-vs-cut classifier.
    {
        float d_ill = Y_ill - prev_illum[lid];
        // `frame` is never 0 in any pass (libplacebo counts executed hook
        // passes), so a fresh buffer is detected by the schema magic; its
        // prev_illum is garbage, so it casts no vote.
        bool lane_init = motion_state_magic != MOTION_STATE_EPOCH;
        s_change[lid] = (!lane_init && abs(d_ill) > ILLUM_CHANGE_THRESH)
                        ? ((d_ill > 0.0) ? 1u : 2u) : 0u;
    }
    prev_illum[lid] = Y_ill;

    barrier();

#if ADDITIVE_OPEN_GUARD
    // Local V-flow matcher (additive only), in shared memory. Only cells that
    // will pass the band-pass test do the work; every lane reaches the barrier.
    ivec2 vcell = ivec2(int(ix), int(iy));
    float vf0 = mix(s_pump_snap_f[lid], s_illum_v[lid], PUMP_ALPHA_FAST);
    float vs0 = mix(s_pump_snap_s[lid], s_illum_v[lid], PUMP_ALPHA_SLOW);
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

    // ================= thread 0: scene statistics + decisions =================
    // Values the thread-0 tail needs after the per-cell phases:
    bool  t0_reset = false;
    float t0_drive_eff = 0.0, t0_global_fall_ratio = 1.0;
    float t0_loc_sum = 0.0, t0_contrast_v = 0.0;
    if (lid == 0u) {
        // Schema prime: catches fresh or reformatted state, not a same-schema
        // reload or a seek. The transient reset below discards this frame's
        // flow (read from an unprimed history); pass 7 primes the next one.
        bool motion_uninit = motion_state_magic != MOTION_STATE_EPOCH;
        // STATE INIT. The state buffer is not zeroed and `frame` is never 0,
        // so fresh state = schema magic mismatch, backed by a range test that
        // also catches garbage or a NaN/Inf in a same-schema buffer (every NaN
        // comparison is false). On init the scene EMAs snap to this frame, the
        // pump lanes pin, and presentation state restarts from exactly 0.
        bool state_sane = pump_env >= 0.0 && pump_env <= 1.0
                       && pump_cover_gate >= 0.0 && pump_cover_gate <= 1.0
                       && cut_rate >= 0.0 && cut_rate <= 1.0
                       && scene_cut_lockout >= 0.0 && scene_cut_lockout <= LOCKOUT_FRAMES
                       && smoothed_bright_frac >= 0.0 && smoothed_bright_frac <= 1.0
                       && smoothed_log_avg >= 0.0 && smoothed_log_avg <= 16.0
                       && smoothed_contrast >= 0.0 && smoothed_contrast <= 64.0
                       && smoothed_growth_mode >= 0.0 && smoothed_growth_mode <= 1.0;
        bool state_init = motion_uninit || !state_sane;

        #if MOTION_COST_RESET || MOTION_DEBUG_VIEWS
        // Texture-qualified match validity: flat tiles resolve to zero flow
        // by MOT_BIAS, but that is no evidence, so they are left out.
        float motion_tex_w = 0.0, motion_bad_w = 0.0;
        for (uint i = 0u; i < 144u; i++) {
            float tc = smoothstep(MOTION_TEXTURE_LO, MOTION_TEXTURE_HI,
                                  s_flow_texture[i]);
            motion_tex_w += tc;
            motion_bad_w += tc * ((s_flow_cost[i] >= MOTION_COST_BAD) ? 1.0 : 0.0);
            #if MOTION_DEBUG_VIEWS
            // cf_debug 10 blue: this tile's weighted vote for the cost reset.
            dbg_cell_b[i] = tc * ((s_flow_cost[i] >= MOTION_COST_BAD) ? 1.0 : 0.0);
            #endif
        }
        motion_match_coverage = motion_tex_w * (1.0 / 144.0);
        motion_bad_match_frac = (motion_match_coverage >= MOTION_BAD_COVER_MIN
                              && motion_tex_w > 1e-5)
            ? motion_bad_w / motion_tex_w : 0.0;

        #endif

        float illum_sum         = 0.0;
        float log_luma_sum      = 0.0;
        uint  valid_luma        = 0u;
        float bright_sum        = 0.0;
        float spec_sum          = 0.0;
        float high_sum          = 0.0;
        uint  change_count      = 0u;
        uint  change_pos_count  = 0u;
        float bright_spec_sum   = 0.0;
        float illum_min         = 1.0;
        float illum_max         = 0.0;
        float illum_v_psum      = 0.0;
        float illum_v_min       = 1.0;
        float illum_v_max       = 0.0;
        uint  n_bar             = 0u;

        // -------- letterbox / pillarbox bar detection (see LB_ENGAGE_FRAMES) --------
        // 8 run counters (rows 0,1,7,8, cols 0,1,14,15). A run survives cuts on
        // purpose; content in a bar resets it at once.
        uint lb_row_mask = 0u;
        uint lb_col_mask = 0u;
        {
            const int lb_rows[4] = int[4](0, 1, 7, 8);
            const int lb_cols[4] = int[4](0, 1, 14, 15);
            for (int k = 0; k < 4; k++) {
                bool all_dark = true;
                for (int x = 0; x < 16; x++)
                    all_dark = all_dark && (s_valid[lb_rows[k] * 16 + x] == 0u);
                float run = (state_init || !all_dark)
                    ? 0.0 : min(bar_run[k] + 1.0, LB_ENGAGE_FRAMES);
                bar_run[k] = run;
                if (run >= LB_ENGAGE_FRAMES) lb_row_mask |= 1u << uint(lb_rows[k]);
            }
            for (int k = 0; k < 4; k++) {
                bool all_dark = true;
                for (int y = 0; y < 9; y++)
                    all_dark = all_dark && (s_valid[y * 16 + lb_cols[k]] == 0u);
                float run = (state_init || !all_dark)
                    ? 0.0 : min(bar_run[k + 4] + 1.0, LB_ENGAGE_FRAMES);
                bar_run[k + 4] = run;
                if (run >= LB_ENGAGE_FRAMES) lb_col_mask |= 1u << uint(lb_cols[k]);
            }
        }
        // PICTURE border lines for the border seed (first live row/col when
        // bars are engaged), computed once and broadcast.
        int lb_top   = ((lb_row_mask & 1u)   != 0u)
                     ? (((lb_row_mask & 2u)   != 0u) ? 2 : 1) : 0;
        int lb_bot   = ((lb_row_mask & 256u) != 0u)
                     ? (((lb_row_mask & 128u) != 0u) ? 6 : 7) : 8;
        int lb_left  = ((lb_col_mask & 1u)   != 0u)
                     ? (((lb_col_mask & 2u)   != 0u) ? 2 : 1) : 0;
        int lb_right = ((lb_col_mask & 32768u) != 0u)
                     ? (((lb_col_mask & 16384u) != 0u) ? 13 : 14) : 15;

        float anc_w = 0.0, anc_p = 0.0;
        for (uint i = 0u; i < 144u; i++) {
            float yi           = s_illum[i];
            illum_sum         += yi;
            // Floor at 0: ringing can undershoot and pow() is undefined for
            // x < 0; one NaN would persist in the pump state until a cut.
            float vi           = max(s_illum_v[i], 0.0);
            // p-norm over ALL 144 cells, bars included (see LB_ENGAGE_FRAMES).
            illum_v_psum      += pow(vi, PUMP_DRIVE_P);
            log_luma_sum      += s_log_luma[i];
            valid_luma        += s_valid[i];
            // Engaged bar cells are excluded from everything below. The change
            // count must sit below the exclusion: s_change tests the sigma-80
            // field, which bleeds across the bar edge, so a bright event's halo
            // could otherwise fake a cut on letterboxed content.
            bool lb_dead = (((lb_row_mask >> (i >> 4u)) & 1u) == 1u)
                        || (((lb_col_mask >> (i & 15u)) & 1u) == 1u);
            if (lb_dead) { n_bar++; continue; }
            // Spec anchor: cells inside a broad near-white field.
            float anc_y = (s_valid[i] != 0u) ? min(exp(s_log_luma[i]), 1.0) : 0.0;
            float anc_m = smoothstep(SPEC_ANCHOR_COV_LO, SPEC_ANCHOR_COV_HI, s_cov[i])
                        * smoothstep(0.84, 0.88, anc_y);
            float anc_y2 = anc_y * anc_y;
            float anc_y4 = anc_y2 * anc_y2;
            anc_w += anc_m;
            float anc_y16 = anc_y4 * anc_y4 * anc_y4 * anc_y4;
            anc_p += anc_m * anc_y16 * anc_y16;
            change_count      += (s_change[i] != 0u) ? 1u : 0u;
            change_pos_count  += (s_change[i] == 1u) ? 1u : 0u;
            illum_min          = min(illum_min, yi);
            illum_max          = max(illum_max, yi);
            illum_v_min        = min(illum_v_min, vi);
            illum_v_max        = max(illum_v_max, vi);
            // Tier sums with SOFT membership (see TIER_SOFT_HALFBAND).
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
        }

        const float N_SAMPLES = 144.0;
        // Picture-area cell count: worst case (2.76:1 letterbox + 4:3
        // pillarbox) leaves n_eff >= 60. avg_illum stays /144 (a fallback only).
        float n_eff       = N_SAMPLES - float(n_bar);
        float avg_illum   = illum_sum / N_SAMPLES;
        float bright_frac = bright_sum / n_eff;

        // Contrast: dynamic range of the illumination field in stops. Floored
        // like contrast_v: a zero fallback read any frame with one near-black
        // cell as FLAT and let invisible black-level changes breathe the
        // dynamic-intensity scale +-8 %.
        float contrast = max(0.0, log2(max(illum_max, 1e-6) / max(illum_min, 0.001)));

        // V-aware pump signals (max(R,G,B) of the field), so saturated colored
        // events both DRIVE the pump and SURVIVE its contrast guard. Growth
        // mode and APL keep the luma-axis avg_illum/contrast.
        float pnorm_illum_v = pow(illum_v_psum / N_SAMPLES, 1.0 / PUMP_DRIVE_P);
        // Cover-gate contrast, anchored on the minimum and floored (not zeroed),
        // so one near-black cell cannot mute the pump on every event over a
        // darker surround. Numerator floored too (log2(0), persistent state).
        // Trade: a fade to white that keeps a dark region also opens cover;
        // the drive band-pass is the first line of defense there.
        float contrast_v    = max(0.0, log2(max(illum_v_max, 1e-6) / max(illum_v_min, 0.001)));

        // Log-average: perceptual brightness key from source pixels. The
        // all-dark fallback is floored: a below-black source must not drive
        // the state negative (that fails the init range test every frame).
        float log_avg = (valid_luma > 4u)
            ? exp(log_luma_sum / float(valid_luma))
            : max(avg_illum, 0.0);

        // Specular signal: present when small fraction at specular tier
        float spec_frac          = spec_sum / n_eff;
        float highlight_frac_src = high_sum / n_eff;
        float spec_onset         = smoothstep(0.0, SPEC_FRAC_MIN, spec_frac);
        float spec_shutoff       = 1.0 - smoothstep(SPEC_FRAC_MAX, SPEC_FRAC_CEIL, spec_frac);
        // Tier separation: specular must be rarer than highlight (fails for
        // points with no sub-specular halo, hence the sparse fallback).
        float tier_ratio   = 1.0 - spec_frac / max(highlight_frac_src, 0.001);
        float tier_gate    = smoothstep(0.3, 0.7, tier_ratio);
        // Sparse-points-against-dark fallback (spec_onset still gates noise).
        float sparse_bonus = 1.0 - smoothstep(0.0, SPARSE_SPEC_CEIL, spec_frac);
        float tier_mode    = max(tier_gate, sparse_bonus);

        // Bright-scene recovery (see BRIGHT_SPEC_THRESH), keyed on the
        // smoothed key so fade-ups cannot step spec_vel.
        float bs_frac     = bright_spec_sum / n_eff;
        float bs_onset    = smoothstep(0.0, SPEC_FRAC_MIN, bs_frac);
        float bs_shutoff  = 1.0 - smoothstep(BRIGHT_SPEC_FRAC_MAX,
                                             BRIGHT_SPEC_FRAC_CEIL, bs_frac);
        float bs_tier_r   = 1.0 - bs_frac / max(highlight_frac_src, 0.001);
        float bs_gate     = smoothstep(0.2, 0.5, bs_tier_r);
        float bs_raw      = bs_onset * bs_shutoff * bs_gate;
        // On init the smoothed key is garbage and is about to snap to log_avg.
        float bright_scene = smoothstep(BRIGHT_SCENE_LOW, BRIGHT_SCENE_HIGH,
                                        state_init ? log_avg : smoothed_log_avg);

        // The spec signal feeds the velocity calc only (growth mode).
        float spec_normal_natural = spec_onset * spec_shutoff * tier_mode;
        float spec_raw_natural = mix(spec_normal_natural,
                                     max(spec_normal_natural, bs_raw), bright_scene);

        // Scene cut: a majority of PICTURE cells moved by more than
        // ILLUM_CHANGE_THRESH. The lockout counter holds the whole cut-window
        // state ("cut this frame" == lockout at max; consumers test > 0).
        float change_pct  = float(change_count) / n_eff;
        // Event-vs-cut classifier (see CUT_EVENT_*); the pump values here are
        // still last frame's.
        float pos_frac = (change_count > 0u)
            ? float(change_pos_count) / float(change_count) : 0.0;
        // prior_drive = last frame's drive; pump_drive_prev = the frame before
        // (stored before the lanes move), so an event cannot arm itself.
        float prior_drive = state_init ? 0.0 : pump_fast - pump_slow;
        float drive_prev_in = state_init ? 0.0 : pump_drive_prev;
        bool drive_rising = prior_drive > drive_prev_in + CUT_EVENT_DRIVE_EPS;
        // The presenting arm also needs a RECENT drive: pump_env holds at
        // ~0.999/frame, and an env-only arm suppressed cuts for 46-96 s after
        // any brightening (a real cut in that tail opened a new held pump).
        // Drive >= half the charging level bounds it to the event + ~3 s.
        bool prior_event = (!state_init && pump_env > CUT_EVENT_ENV_MIN
                            && prior_drive > 0.5 * CUT_EVENT_DRIVE_MIN)
                        || (prior_drive > CUT_EVENT_DRIVE_MIN && drive_rising);
        pump_drive_prev = prior_drive;
        bool event_rise = (pos_frac >= CUT_EVENT_POS_FRAC) && prior_event;
        scene_cut_lockout = state_init ? 0.0 : max(scene_cut_lockout - 1.0, 0.0);
        // Motion-cost reset: a texture-backed STRUCTURE mismatch across most
        // of the frame (the cost is photometric-invariant, see pass 6): a cut
        // the vote missed. It starts the same lockout as a vote cut.
        #if MOTION_COST_RESET
        bool motion_match_reset = !state_init
                               && motion_bad_match_frac >= MOTION_BAD_FRAC_RESET;
        #else
        bool motion_match_reset = false;
        #endif
        bool cut_fired = scene_cut_lockout <= 0.0
                      && ((change_pct > SCENE_CUT_PCT && !event_rise)
                          || motion_match_reset);
        if (cut_fired)
            scene_cut_lockout = LOCKOUT_FRAMES;
        // Strobe pressure (see CUT_RATE_*): the alpha below must see the
        // PRE-update value. One unconditional computed store (no FXC
        // store-in-one-arm + RMW-in-the-other shape).
        float cut_rate_prev = state_init ? 0.0 : cut_rate;
        cut_rate = mix(cut_rate_prev, cut_fired ? 1.0 : 0.0,
                       cut_fired ? CUT_RATE_RISE : CUT_RATE_FALL);

        // ---- Velocity-driven adaptation ----
        // Signed velocities (value minus smoothed value). spec_vel uses the
        // smoothed natural spec signal (a growth-lifted one would quench growth
        // mode before long events finish).
        float bright_vel   = bright_frac - smoothed_bright_frac;
        float spec_vel     = spec_raw_natural - smoothed_spec_natural;
        float contrast_vel = contrast - smoothed_contrast;
        float log_avg_vel  = log_avg - smoothed_log_avg;

        float vel_mag    = max(abs(bright_vel), abs(log_avg_vel));
        float base_alpha = mix(TEMPORAL_ALPHA_SLOW, TEMPORAL_ALPHA_MID,
                               smoothstep(ADAPT_DELTA_LOW, ADAPT_DELTA_HIGH, vel_mag));
        // Post-cut alpha decays FAST -> MID across the lockout (lands exactly
        // on FAST on the cut frame), so a one-frame event inside the window
        // does not couple 1:1 into every EMA. Sustained strobing blends it
        // back toward MID using the PRE-update cut_rate (see CUT_RATE_*).
        float strobe_t = smoothstep(CUT_RATE_STROBE_LO, CUT_RATE_STROBE_HI,
                                    cut_rate_prev);
        float alpha = (scene_cut_lockout > 0.0)
            ? mix(mix(TEMPORAL_ALPHA_MID, TEMPORAL_ALPHA_FAST,
                      scene_cut_lockout / LOCKOUT_FRAMES),
                  TEMPORAL_ALPHA_MID, strobe_t)
            : base_alpha;

        // Growth mode (see GROWTH_*): spec_vel outruns bright_vel, contrast
        // rises, and the bright fraction is above a floor.
        float growth_sig   = spec_vel - GROWTH_SPEC_BIAS * bright_vel;
        float c_gate       = smoothstep(GROWTH_C_GATE_LOW, GROWTH_C_GATE_HIGH,
                                        contrast_vel);
        float frac_floor   = smoothstep(GROWTH_FRAC_FLOOR_LOW, GROWTH_FRAC_FLOOR_HIGH,
                                        smoothed_bright_frac);
        // Needs multi-cell spec corroboration (see GROWTH_SPEC_CELLS).
        float growth_corrob = smoothstep(GROWTH_SPEC_CELLS_LO,
                                         GROWTH_SPEC_CELLS_HI, spec_sum);
        float growth_mode_instant = smoothstep(GROWTH_SIG_LOW, GROWTH_SIG_HIGH, growth_sig)
                                  * c_gate * frac_floor * growth_corrob;

        // Transient reset window: state init, the cut lockout (cut frame
        // included), a motion-history prime, a motion-cost reset, or an
        // additive/subtractive switch. ONE flag serves growth mode, the scalar
        // pump and the cell pump, so their baselines never desync.
        bool additive_mode_reset = additive_mode_magic != float(ADDITIVE_STATE_EPOCH);
        bool transient_reset = state_init || (scene_cut_lockout > 0.0)
                            || motion_uninit || motion_match_reset || additive_mode_reset;
        motion_state_magic = MOTION_STATE_EPOCH;
        additive_mode_magic = float(ADDITIVE_STATE_EPOCH);
        // A reset decays growth like other presentation state (rule 2):
        // zeroing it stepped the base curve ~20 % in one frame.
        smoothed_growth_mode = transient_reset
            ? (state_init ? 0.0 : smoothed_growth_mode * PUMP_RESET_DECAY)
            : mix(smoothed_growth_mode, growth_mode_instant, alpha);


        // ---- Light-pump band-pass (sudden sustained brightening) ----
        // Two EMAs of pnorm_illum_v; their difference is the drive. A transient
        // reset re-pins both lanes so a cut cannot manufacture drive (a hard
        // cut to a brighter scene must NOT pump).
        if (transient_reset) {
            pump_fast = pnorm_illum_v;
            pump_slow = pnorm_illum_v;
            pump_env        = state_init ? 0.0 : pump_env * PUMP_RESET_DECAY;
            pump_cover_gate = state_init ? 0.0 : pump_cover_gate * PUMP_RESET_DECAY;
        } else {
            // Locals: the buffer is coherent (a re-read is a real memory round
            // trip), so compute once, store once.
            float pf_prev = pump_fast;
            float pf = mix(pf_prev, pnorm_illum_v, PUMP_ALPHA_FAST);
            float ps = mix(pump_slow, pnorm_illum_v, PUMP_ALPHA_SLOW);
            pump_fast = pf;
            pump_slow = ps;
            float drive      = pf - ps;                                  // SIGNED velocity
            // True frame-to-frame source fall (pf/pf_prev telescopes across a
            // fade; a pf/ps ratio re-applied the same deficit every frame).
            float global_fall_ratio = (drive < 0.0 && pf < pf_prev && pf_prev > 1e-3)
                ? clamp(pf / pf_prev, 0.0, 1.0) : 1.0;
            // No max(0, x) needed: the smoothstep maps x <= 0 to 0.
            float drive_eff  = drive;
            // -------- ungated local aggregate (frozen snapshot) --------
            // loc_sum: the signed p-mean aggregate of the cell drives, the
            // RELEASE statistic (balanced crossings cancel). The gated ONSET
            // sum is built after the per-cell phase.
            float loc_sum = 0.0, loc_on_sum = 0.0;
            float fall_w = 0.0, fall_ratio = 0.0;
            #if PUMP_EDGE_ESTABLISH
            float edge_interior_n = 0.0;   // count of deep-interior cells that are RISING
            #endif
            for (uint i = 0u; i < 144u; i++) {
                float cf = s_pump_snap_f[i];
                float cs = s_pump_snap_s[i];
                float di = cf - cs;
                float m  = max(abs(di) - PUMP_CELL_DEADZONE, 0.0);
                float w  = pow(m, PUMP_DRIVE_P);
                loc_sum += sign(di) * w;
                #if PUMP_EDGE_ESTABLISH
                // Border-seed global gate: the FRACTION of the deep interior
                // (cols 3-12 x rows 2-6, 50 cells clear of the outer ring and
                // the bar candidates) that is rising.
                {
                    int gy = int(i) / 16, gx = int(i) % 16;
                    if (gx >= 3 && gx <= 12 && gy >= 2 && gy <= 6 && di > PUMP_EDGE_GLOBAL_EPS)
                        edge_interior_n += 1.0;
                }
                #endif
            }
            #if PUMP_EDGE_ESTABLISH
            float global_gate = smoothstep(PUMP_EDGE_GLOBAL_FRAC_LO, PUMP_EDGE_GLOBAL_FRAC_HI,
                                           edge_interior_n * (1.0 / 50.0));
            #endif
            #if !ADDITIVE_OPEN_GUARD
            // Conservation signal for the subtractive floor (once per frame).
            float motion_dl_pos = (loc_sum > 0.0)
                ? pow(loc_sum / N_SAMPLES, 1.0 / PUMP_DRIVE_P) : 0.0;
            float motion_emit_signal = smoothstep(MC_EMIT_LO, MC_EMIT_HI,
                                                   motion_dl_pos);
            #endif
            #if PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
            // Skirt-width selector: SMOOTHED bright coverage only (an
            // instantaneous term let a flickering light flip an open pump's
            // skirt every frame).
            float pump_blob_gate = smoothstep(PUMP_MASK_BLOB_FRAC_LO,
                                              PUMP_MASK_BLOB_FRAC_HI,
                                              smoothed_bright_frac);
            #if PUMP_MASK_FINISH
            float pump_finish_mix = clamp(PUMP_MASK_FINISH_MIX * pump_blob_gate,
                                          0.0, 1.0);
            #endif
            #endif
            t0_loc_sum = loc_sum;
            #if PUMP_EDGE_ESTABLISH
            s_bc_global_gate = global_gate;
            #endif
            #if !ADDITIVE_OPEN_GUARD
            s_bc_emit = motion_emit_signal;
            #endif
            #if PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
            s_bc_blob_gate = pump_blob_gate;
            #if PUMP_MASK_FINISH
            s_bc_finish_mix = pump_finish_mix;
            #endif
            #endif
            t0_drive_eff = drive_eff;
            t0_global_fall_ratio = global_fall_ratio;
        }
        t0_reset = transient_reset;
        t0_contrast_v = contrast_v;
        s_bc_reset = transient_reset ? 1u : 0u;
        s_bc_init  = state_init ? 1u : 0u;
        s_bc_lb[0] = lb_left;  s_bc_lb[1] = lb_right;
        s_bc_lb[2] = lb_top;   s_bc_lb[3] = lb_bot;

        // Scene EMAs; on init they snap (select form, FXC-safe).
        // smoothed_spec_natural keeps the shared alpha: spec_vel is calibrated
        // against it.
        smoothed_bright_frac   = state_init ? bright_frac
                               : mix(smoothed_bright_frac, bright_frac, alpha);
        smoothed_spec_natural  = state_init ? spec_raw_natural
                               : mix(smoothed_spec_natural, spec_raw_natural, alpha);
        smoothed_contrast      = state_init ? contrast
                               : mix(smoothed_contrast, contrast, alpha);
        smoothed_log_avg       = state_init ? log_avg
                               : mix(smoothed_log_avg, log_avg, alpha);

        // Scene-key hold (cf_key_hold). The key above snaps at every cut, so
        // inside ONE scene a close-up, a wide shot and a reverse angle over a
        // dark wall each get their own APL factor: the same painted skin tone
        // measured 119 -> 158 nits across a single cut (key 0.54 -> 0.20).
        // Framing is not lighting. The held key rises with the key at once
        // (a brighter shot is dampened immediately, so output is never hotter
        // than with the hold off) and releases toward a darker key in the log
        // domain with time constant cf_key_hold seconds (24 fps assumed, like
        // every temporal constant here). Only pass 9's APL axis reads it; the
        // velocities keep smoothed_log_avg. A black frame
        // (fade through black, title card) releases it outright: that is a
        // scene boundary, and nothing is expanded on it. Select form
        // throughout (FXC-safe); own range test, since the state magic does
        // not cover a buffer that grew by this field.
        {
            float key_cur   = max(smoothed_log_avg, 1e-4);
            bool  held_ok   = held_log_avg >= 1e-4 && held_log_avg <= 16.0;
            float key_prev  = (state_init || !held_ok) ? key_cur : held_log_avg;
            float key_rel   = (valid_luma > 4u)
                            ? 1.0 - exp(-1.0 / (max(cf_key_hold, 1e-3) * 24.0))
                            : 1.0;
            float key_eased = exp(mix(log(key_prev), log(key_cur), key_rel));
            held_log_avg    = max(key_cur, key_eased);
        }
        // Scene diffuse-white anchor (see SPEC_ANCHOR_*). Select form, own
        // range test (the buffer grew by this field).
        {
            float anc_conf = smoothstep(1.0, 3.0, anc_w);
            float anc_lvl  = pow(anc_p / max(anc_w, 1e-4), 0.03125);
            float anc_cur  = mix(SPEC_ANCHOR_NONE, clamp(anc_lvl, SPEC_ANCHOR_NONE, 1.0), anc_conf);
            bool  anc_ok   = spec_anchor_held >= 0.5 && spec_anchor_held <= 1.2;
            float anc_prev = (state_init || !anc_ok) ? anc_cur : spec_anchor_held;
            #if SPEC_ANCHOR_PERSIST
            // Rise only to what this frame and the last agree on: a one-frame
            // flash does not move it.
            bool  anc_lok  = spec_anchor_last >= 0.5 && spec_anchor_last <= 1.2;
            float anc_last = (state_init || !anc_lok) ? anc_cur : spec_anchor_last;
            float anc_up   = min(anc_cur, anc_last);
            #else
            float anc_up   = anc_cur;
            #endif
            spec_anchor_last = anc_cur;
            float anc_att  = (scene_cut_lockout > 0.0) ? 1.0 : SPEC_ANCHOR_ATTACK;
            // Release: slow when the frame has no diffuse white of its own (a
            // close-up keeps the scene's), quick when it has a confident lower
            // one (after a flash or a fade through white).
            float anc_hold = mix(SPEC_ANCHOR_HOLD_S, SPEC_ANCHOR_HOLD_CONF_S, anc_conf);
            float anc_rel  = (valid_luma > 4u)
                           ? 1.0 - exp(-1.0 / (anc_hold * 24.0)) : 1.0;
            spec_anchor_held = (anc_up > anc_prev) ? mix(anc_prev, anc_up, anc_att)
                                                   : mix(anc_prev, min(anc_cur, anc_prev), anc_rel);
        }

    }

    // ================= per-cell pump update (one cell per lane) =================
    // Every barrier in this pass sits at top level (no return anywhere): FXC X3663.
    barrier();
    // Each lane updates its own cell: established-level freshness (see
    // PUMP_ESTABLISH_MARGIN), the cell EMAs, the A2 proof and the mask env.
    {
        uint i = lid;
        if (s_bc_reset != 0u) {
            // Cell lanes re-pin with the scalar lanes.
            float v = s_illum_v[i];
            pump_fast_cell[i] = v;
            pump_slow_cell[i] = v;
            pump_very_slow_cell[i] = v;
            pump_open_persist_cell[i] = 0.0;
            // Presentation values DECAY on reset (PUMP_RESET_DECAY) and restart
            // from exactly 0 on init; state (lanes, proof, seed) hard-resets.
            // Both arms compute the stored value (no FXC const-store/RMW pair).
            pump_env_cell[i]  = (s_bc_init != 0u) ? 0.0 : pump_env_cell[i] * PUMP_RESET_DECAY;
            pump_mask_cell[i] = (s_bc_init != 0u) ? 0.0 : pump_mask_cell[i] * PUMP_RESET_DECAY;
            #if PUMP_EDGE_ESTABLISH
            pump_seed_cell[i] = 0.0;
            #endif
        } else {
            uint on_mode = 0u, fall_flag = 0u;
            float fall_r = 0.0;
            float cf = s_pump_snap_f[i];
            float cs = s_pump_snap_s[i];
            float di = cf - cs;
            float m  = max(abs(di) - PUMP_CELL_DEADZONE, 0.0);
            float w  = pow(m, PUMP_DRIVE_P);
            bool fresh = false;
            float fresh_ease = 0.0;
            #if cf_debug == 12
            dbg_cell_r[i] = 0.0;
            dbg_cell_g[i] = 0.0;
            dbg_cell_b[i] = 0.0;
            #endif
            #if ADDITIVE_OPEN_GUARD
            float add_established = 0.0;
            #endif
            int cy = int(i) / 16, cx = int(i) % 16;
            #if PUMP_EDGE_ESTABLISH
            // Border seed (see EDGE ESTABLISHMENT): a rising outer-ring cell
            // of the PICTURE (s_bc_lb excludes bar lines; inside bars it would
            // never arm) is presumed influx unless the rise is frame-global.
            bool edge_cell = (cx == s_bc_lb[0] || cx == s_bc_lb[1]
                           || cy == s_bc_lb[2]  || cy == s_bc_lb[3]);
            float edge_seed = (edge_cell && di > 0.0) ? (1.0 - s_bc_global_gate) : 0.0;
            // Max influx marker among the ring-2 neighbours that gate this cell.
            float nb_seed = 0.0;
            #endif
            if (di > 0.0) {
                // ANCHORED window: the 5x5 ring-2 window is clamped inside the
                // grid, so at an edge it shifts inward instead of truncating (a
                // truncated scan made border freshness EASIER). Interior cells
                // are unchanged. Rejected: out-of-frame = bright (an unpumped
                // vignette ring on every global event); ring 3 (cannot exclude a
                // 3-row sky in a 9-row grid, so a fire under a brighter sky never
                // localizes).
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
                            // Influx marker only from a neighbour a STEP above
                            // this cell's fast lane; on a uniform fade every
                            // neighbour's slow lane sits below it (snb = cf - di).
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
                // The ordered tail debits edge_seed cells, so off-frame influx
                // cannot fire the scalar either.
                if (fresh) on_mode = 1u;
                #else
                if (fresh) on_mode = 1u;
                #endif
                // Additive eases freshness over [MARGIN, 2*MARGIN]: everything
                // the boolean suppresses stays exactly 0.
            } else {
                on_mode = 2u;    // sign(di)*w with di <= 0
            }
            // Cell EMAs, then the mask env from the post-update drive d.
            float v = s_illum_v[i];
            float f = mix(cf, v, PUMP_ALPHA_FAST);
            float s = mix(cs, v, PUMP_ALPHA_SLOW);
            // Fall weight only if the fast lane falls THIS frame.
            if (di < 0.0 && f < cf && cf > 1e-3) {
                fall_flag = 1u;
                fall_r = clamp(f / cf, 0.0, 1.0);
            }
            #if PUMP_EDGE_ESTABLISH
            // Fast-establish, scoped to influx (see EDGE ESTABLISHMENT): settle
            // is nonzero only for a rise gated by an influx anchor and doubles
            // as this cell's new marker. Blended, not branched. Writes the SSBO
            // slow lane, not the frozen snapshot, so the wave stays symmetric.
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
            // Idle wobble maps to exactly 0 (PUMP_CELL_DRIVE_LOW/HIGH include
            // the dead-zone offset).
            float a = smoothstep(PUMP_CELL_DRIVE_LOW, PUMP_CELL_DRIVE_HIGH, d);
            #if ADDITIVE_OPEN_GUARD
            // 1 only on a route-authorized frame: maintenance credit may
            // sustain amplitude, never grow it (growth on it pumped decays).
            float add_attack_gate = 0.0;
            #endif
            // === MOTION-COMPENSATED OPENING GATE (only for a > 0) ===
            {
                fresh_ease = 0.0;
                if (a > 0.0) {
                    int cxm = int(i) & 15, cym = int(i) >> 4;
                    vec2 local_flow = s_flow[i];
                    vec2 warpc = vec2(float(cxm), float(cym)) + local_flow * (1.0 / 8.0);
                    bool inb = warpc.x > -0.5 && warpc.x < 15.5
                            && warpc.y > -0.5 && warpc.y < 8.5;
                    #if ADDITIVE_OPEN_GUARD
                    // A2: compare the motion-compensated fast-lane rise with
                    // the same-cell rise (an EXPLAINED FRACTION, not a one-frame
                    // delta): a slow grow keeps ratio ~1, a carried lamp ~0.
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
                    // Local V-flow route: explains a broad tail opening a
                    // neighbour of a compact source. Absolute fast history is
                    // warped, so real growth survives as residual.
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
                    // Excursion floor (see ADD_EXCURSION_LO/HI): the established
                    // gate is neighbour-relative and the drive band admits any
                    // rise above ~0.06 V/s. The ~0.31 floor also sets a minimum
                    // emitter size (~1.3 cells: a sub-160 px ignition stays
                    // shut, accepted).
                    float exc_gate = smoothstep(ADD_EXCURSION_LO,
                                                ADD_EXCURSION_HI,
                                                f - s_pump_snap_vs[i]);
                    routed_open *= exc_gate;
                    // Source-velocity anchor: no authorized AMPLITUDE while the
                    // source falls (a short flash would otherwise pump its own
                    // decay). NOT folded into routed_open: the persist counter
                    // resets below ROUTE_MIN, so a velocity term there demands 7
                    // near-monotone frames and flame flicker (+-0.01-0.06 V)
                    // never matures. Noisy vetoes gate proved_open, NOT the
                    // persist counter.
                    float src_rise_gate = 1.0 - smoothstep(0.0,
                        ADD_SRC_FALL_DZ,
                        -(s_illum_v[i] - s_prev_v[i]));
                    #if PUMP_EDGE_ESTABLISH
                    // Influx authority is part of the persisted route.
                    routed_open *= 1.0 - edge_seed;
                    #endif
                    // Credit follows one trajectory (this cell + raw primary
                    // flow); the V route is a veto, never a donor, or it could
                    // bypass the seven-frame proof beside an unrelated event.
                    float prior_credit = s_pump_snap_persist[i];
                    // Maintenance: a proved cell may hold its opening through a
                    // short flicker. Maturity is encoded in (7, 8]: a valid route
                    // refreshes to 8, a failed frame spends 1/48; before maturity
                    // one failed route resets to 0. The 0..8 range is
                    // load-bearing: bilinear transport needs > 7/8 donor
                    // support, so credit cannot grow a skirt.
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
                    // A mature token without local amplitude is stale.
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
                    float proved_open = routed_open * persist_gate
                                      * src_rise_gate;
                    // Growth only on a fully route-authorized, non-falling frame.
                    add_attack_gate =
                        (routed_open >= ADD_PERSIST_ROUTE_MIN
                         && proved_open > 0.0) ? 1.0 : 0.0;
                    // Spent credit cannot resurrect a released cell.
                    float maintained_open = prior_authorized
                        ? maintenance_env : 0.0;
                    fresh_ease = max(proved_open, maintained_open);
                    #if cf_debug == 12
                    // G = the full applied route product.
                    dbg_cell_r[i] = add_established;
                    dbg_cell_g[i] = effective_ratio_route
                                            * exc_gate * src_rise_gate;
                    dbg_cell_b[i] = persist_gate;
                    #endif
                    #else
                    float mc_prev = sample_prev_v(warpc);
                    float mc_res = s_illum_v[i] - mc_prev;
                    float mc_fresh = inb
                        ? smoothstep(MC_RES_LO, MC_RES_HI, max(mc_res, 0.0)) : 0.0;
                    // Subtractive floor (the scalar owns amplitude there).
                    fresh_ease = max(mc_fresh, s_bc_emit);
                    #endif
                }
                #if ADDITIVE_OPEN_GUARD
                else {
                    // Idle frame: partial proof resets; a mature, live cell
                    // spends credit so a flicker dropout can recover.
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
            #if PUMP_MASK_ESTABLISH
            // Only a FRESH (proved) rise may OPEN the mask; closing is never
            // gated.
            a *= fresh_ease;
            #endif
            #if PUMP_EDGE_ESTABLISH && !ADDITIVE_OPEN_GUARD
            // Subtractive: pure influx is held shut; the border mirror restores
            // real events at the edge.
            a *= (1.0 - edge_seed);
            #endif
            // Attack step: the RISING authorized target is capped at +0.25 env
            // per frame (a delayed proof would otherwise pop). Pre-proof stays
            // exactly 0; falls are not slewed.
            #if ADDITIVE_OPEN_GUARD
            a = min(a, pump_env_cell[i] + ADD_ATTACK_STEP * add_attack_gate);
            #endif
            // Velocity-matched release: follow the fast lane's frame-to-frame
            // fall (ratios telescope across a fade; a rebound gives r = 1).
            // The arms differ in ARMING only.
            #if ADDITIVE_OPEN_GUARD
            // Additive arms on the fast lane's own turnover (waiting for d < 0
            // held a mistimed opening ~10 frames while its source died).
            float r = (f < cf && cf > 1e-3)
                ? clamp(f / cf, 0.0, 1.0) : 1.0;
            #else
            float r = (d < 0.0 && f < cf && cf > 1e-3)
                ? clamp(f / cf, 0.0, 1.0) : 1.0;
            #endif
            float e = max(pump_env_cell[i] * r * PUMP_ADAPT_FLOOR, a);
            pump_env_cell[i] = e;
            // Post-update env for the publish/finish phases.
            s_pump_env_post[i] = e;
            // Reduction operands for thread 0's ordered sums (see s_red_*).
            s_red_w[i]      = w;
            #if PUMP_EDGE_ESTABLISH
            s_red_edge[i]   = edge_seed;
            #endif
            s_red_on[i]     = on_mode;
            s_red_fall[i]   = fall_flag;
            s_red_fall_r[i] = fall_r;
        }
    }
    barrier();
    // Publish the PRESENTATION mask (see PUMP_MASK_SOFTEN).
    if (s_bc_reset == 0u) {
        uint i = lid;
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
                b = max(b, s_pump_env_post[ny * 16 + nx] * wx * wy
                         * (PUMP_MASK_BLOB_GAIN / 256.0));
            }
        float bn = 0.0;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int ny = clamp(cy + dy, 0, 8);
                int nx = clamp(cx + dx, 0, 15);
                bn += s_pump_env_post[ny * 16 + nx]
                    * float((2 - abs(dy)) * (2 - abs(dx)));
            }
        bn *= 1.0 / 16.0;
        float shaped = mix(bn, max(bn, b), s_bc_blob_gate);
        published = max(s_pump_env_post[i], shaped);
        #else
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int ny = clamp(cy + dy, 0, 8);
                int nx = clamp(cx + dx, 0, 15);
                // (1,2,1)⊗(1,2,1)/16 binomial; edge cells replicate
                // (index clamp), keeping border amplitude full.
                b += s_pump_env_post[ny * 16 + nx]
                   * float((2 - abs(dy)) * (2 - abs(dx)));
            }
        published = max(s_pump_env_post[i], b * (1.0 / 16.0));
        #endif
        #else
        published = s_pump_env_post[i];
        #endif
        #if PUMP_EDGE_ESTABLISH
        // Border mirror: an outer cell takes its inward neighbour's amplitude
        // at full rate; pure influx (gated neighbour) inherits 0.
        int mcy = int(i) / 16, mcx = int(i) % 16;
        float mir = 0.0;
        if (mcx == 0)  mir = max(mir, s_pump_env_post[mcy * 16 + 1]);
        if (mcx == 15) mir = max(mir, s_pump_env_post[mcy * 16 + 14]);
        if (mcy == 0)  mir = max(mir, s_pump_env_post[16 + mcx]);
        if (mcy == 8)  mir = max(mir, s_pump_env_post[7 * 16 + mcx]);
        // Corners also reach the inward diagonal (no 1-cell notch).
        int dcy = (mcy == 0) ? 1 : (mcy == 8) ? 7 : mcy;
        int dcx = (mcx == 0) ? 1 : (mcx == 15) ? 14 : mcx;
        if ((mcy == 0 || mcy == 8) && (mcx == 0 || mcx == 15))
            mir = max(mir, s_pump_env_post[dcy * 16 + dcx]);
        published = max(published, mir);
        #endif
        #if PUMP_MASK_FINISH && PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
        s_pump_shape[i] = published;
        #else
        pump_mask_cell[i] = published;
        #endif
    }
    #if PUMP_MASK_FINISH && PUMP_MASK_SOFTEN && PUMP_MASK_BLOB5 && ADDITIVE_OPEN_GUARD
    barrier();
    // Finishing blur (see PUMP_MASK_FINISH); raw env restored afterwards.
    if (s_bc_reset == 0u) {
        uint i = lid;
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
        float finished = mix(s_pump_shape[i], bf, s_bc_finish_mix);
        pump_mask_cell[i] = max(s_pump_env_post[i], finished);
    }
    #endif

    // ================= thread 0: ordered reductions + scalar pump tail =================
    if (lid == 0u) {
        if (!t0_reset) {
            float drive_eff = t0_drive_eff;
            float global_fall_ratio = t0_global_fall_ratio;
            float contrast_v = t0_contrast_v;
            const float N_SAMPLES = 144.0;
            float loc_sum = t0_loc_sum;
            float loc_on_sum = 0.0, fall_w = 0.0, fall_ratio = 0.0;
            for (uint i = 0u; i < 144u; i++) {
                #if PUMP_EDGE_ESTABLISH
                if (s_red_on[i] == 1u) loc_on_sum += s_red_w[i] * (1.0 - s_red_edge[i]);
                #else
                if (s_red_on[i] == 1u) loc_on_sum += s_red_w[i];
                #endif
                if (s_red_on[i] == 2u) loc_on_sum -= s_red_w[i];
                if (s_red_fall[i] != 0u) {
                    fall_w     += s_red_w[i];
                    fall_ratio += s_red_w[i] * s_red_fall_r[i];
                }
            }
            // ONSET uses the gated aggregate; release keeps the ungated
            // drive_loc, so reveal suppression cannot cause a release.
            float drive_loc = sign(loc_sum) * pow(abs(loc_sum) / N_SAMPLES, 1.0 / PUMP_DRIVE_P);
            float drive_on  = sign(loc_on_sum) * pow(abs(loc_on_sum) / N_SAMPLES, 1.0 / PUMP_DRIVE_P);
            drive_eff = max(drive_eff, drive_on);
            float pump_gate  = smoothstep(PUMP_DRIVE_LOW, PUMP_DRIVE_HIGH, drive_eff);
            // Contrast-retention fade guard on the same V axis as the driver.
            float cover_raw = smoothstep(PUMP_CONTRAST_LOW, PUMP_CONTRAST_HIGH, contrast_v);
            // Cover envelope: instant rise (subtractive) or PUMP_COVER_RISE
            // (additive), fall limited by PUMP_COVER_FALL. pump_cover_gate
            // holds last frame's cover.
            float cover_gate = (cover_raw >= pump_cover_gate)
            #if SPATIAL_PUMP_ADDITIVE
                // rate-clamped rise under additive
                ? min(cover_raw, pump_cover_gate + PUMP_COVER_RISE)
            #else
                ? cover_raw
            #endif
                : max(cover_raw, pump_cover_gate * PUMP_COVER_FALL);
            // Pass 9's additive apply multiplies this in (the mask has no
            // scene guard of its own).
            pump_cover_gate = cover_gate;
            // Velocity-matched release on the source's true frame-to-frame
            // fall (global_fall_ratio); a steady source or a rebound gives 1.
            float rel = global_fall_ratio;
            // Local release (see PUMP_DRIVE_P): a net local fall releases at
            // the falling cells' own fast ratio, so a purely local event's env
            // does not linger behind closed masks. The fall_w floor is
            // normal-range: it forecloses a flush-to-zero 0/0 NaN.
            if (drive_loc < 0.0 && fall_w > 1e-8)
                rel = min(rel, clamp(fall_ratio / fall_w, 0.0, 1.0));
            pump_env = max(pump_env * rel * PUMP_ADAPT_FLOOR, pump_gate) * cover_gate;
        }
        // Dummy write satisfies the 1x1 SAVE target; the SSBO is the real product.
        imageStore(out_image, ivec2(0), vec4(0));
    }
}

// =============================================================================
// PASS 9: EXPANSION APPLY (full resolution)
// =============================================================================
// The expansion is a monotone curve of the smooth highlight base of the
// pixel's luma; the illumination field (the sigma-80 regional brightness, a
// symmetric Gaussian) sets the curve's SHAPE, so a region shares one curve.
// Tonal order is kept; local contrast is scaled by the curve's slope. Scene
// adaptation is continuous (bright fraction, contrast, log-average key).
// CELFLARE_STATS is bound only as the data dependency on pass 8 (the stats
// arrive through the state buffer). Do not remove the bind without checking
// pass ordering on every backend.

//!HOOK MAIN
//!BIND CELFLARE_DS
//!BIND CELFLARE_ILLUM
//!SAVE CELFLARE_OB0
//!WIDTH CELFLARE_DS.w CELFLARE_DS.h / 68 *
//!HEIGHT 68
//!DESC CelFlare: Highlight size seed

// HIGHLIGHT SIZE (these passes run after pass 8, on a grid of 68 rows whatever
// the picture size: cells of 16 px at 1080p). The top lift is meant for compact
// highlights: glints, lamps, small reflections. A broad near-white field (a
// cloud bank, a painted band, a white garment, a soft glare over a screen)
// given the same steep lift magnified its texture and compression mottle and
// drew a cliff in its soft gradients. The bright coverage around a pixel
// (CELFLARE_ILLUM.a) is 0.5 or less on the border of any white object however
// large, so one object would get more lift along its border than inside.
// These passes give every cell of one white object the largest coverage found
// in that object: each pass takes the maximum over the cell and its 8
// neighbours, and a cell passes its value on only in proportion to how much
// near-white it holds (OBJ_M_LO to OBJ_M_HI of its area). A cell with almost
// no near-white passes nothing on, so a glint beside a cloud keeps its own
// small reading once the gap between them holds such a cell (16 to 32 px at
// 1080p, depending on where the grid falls); a glint touching the cloud is
// part of it. 6 passes reach about 96 px at 1080p along the axes, more
// diagonally: enough to carry an object's centre to its border while the
// border itself still reads below HL_SIZE_HI (larger objects read above it
// everywhere). Pass 9 fades the lift out between HL_SIZE_LO and HL_SIZE_HI of
// the result. The seed's 16 taps cover a cell fully up to 4K pictures.
#define OBJ_ROWS  68.0
#define OBJ_M_LO  0.02
#define OBJ_M_HI  0.10

vec4 hook() {
    // near-white share of the cell: 4x4 bilinear taps spread over its area
    vec2 cell = (CELFLARE_DS_size.y / OBJ_ROWS) * CELFLARE_DS_pt;
    float m = 0.0;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.375, -0.375)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.125, -0.375)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.125, -0.375)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.375, -0.375)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.375, -0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.125, -0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.125, -0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.375, -0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.375,  0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.125,  0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.125,  0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.375,  0.125)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.375,  0.375)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2(-0.125,  0.375)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.125,  0.375)).a;
    m += CELFLARE_DS_tex(CELFLARE_DS_pos + cell * vec2( 0.375,  0.375)).a;
    m *= 0.0625;
    float cov = CELFLARE_ILLUM_tex(CELFLARE_ILLUM_pos).a;
    // r: the cell's value so far, g: how much of it the cell passes on, b: its own coverage
    return vec4(cov, smoothstep(OBJ_M_LO, OBJ_M_HI, m), cov, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_OB0
//!SAVE CELFLARE_OB1
//!WIDTH CELFLARE_OB0.w
//!HEIGHT CELFLARE_OB0.h
//!DESC CelFlare: Highlight size spread 1

#define OBJ_NB(dx, dy) { vec4 n = CELFLARE_OB0_tex(CELFLARE_OB0_pos + vec2(dx, dy) * CELFLARE_OB0_pt); u = max(u, n.g * n.r); }

vec4 hook() {
    vec4 s = CELFLARE_OB0_tex(CELFLARE_OB0_pos);
    float u = s.b;
    OBJ_NB(-1.0, -1.0) OBJ_NB(0.0, -1.0) OBJ_NB(1.0, -1.0)
    OBJ_NB(-1.0,  0.0) OBJ_NB(0.0,  0.0) OBJ_NB(1.0,  0.0)
    OBJ_NB(-1.0,  1.0) OBJ_NB(0.0,  1.0) OBJ_NB(1.0,  1.0)
    return vec4(u, s.g, s.b, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_OB1
//!SAVE CELFLARE_OB2
//!WIDTH CELFLARE_OB1.w
//!HEIGHT CELFLARE_OB1.h
//!DESC CelFlare: Highlight size spread 2

#define OBJ_NB(dx, dy) { vec4 n = CELFLARE_OB1_tex(CELFLARE_OB1_pos + vec2(dx, dy) * CELFLARE_OB1_pt); u = max(u, n.g * n.r); }

vec4 hook() {
    vec4 s = CELFLARE_OB1_tex(CELFLARE_OB1_pos);
    float u = s.b;
    OBJ_NB(-1.0, -1.0) OBJ_NB(0.0, -1.0) OBJ_NB(1.0, -1.0)
    OBJ_NB(-1.0,  0.0) OBJ_NB(0.0,  0.0) OBJ_NB(1.0,  0.0)
    OBJ_NB(-1.0,  1.0) OBJ_NB(0.0,  1.0) OBJ_NB(1.0,  1.0)
    return vec4(u, s.g, s.b, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_OB2
//!SAVE CELFLARE_OB3
//!WIDTH CELFLARE_OB2.w
//!HEIGHT CELFLARE_OB2.h
//!DESC CelFlare: Highlight size spread 3

#define OBJ_NB(dx, dy) { vec4 n = CELFLARE_OB2_tex(CELFLARE_OB2_pos + vec2(dx, dy) * CELFLARE_OB2_pt); u = max(u, n.g * n.r); }

vec4 hook() {
    vec4 s = CELFLARE_OB2_tex(CELFLARE_OB2_pos);
    float u = s.b;
    OBJ_NB(-1.0, -1.0) OBJ_NB(0.0, -1.0) OBJ_NB(1.0, -1.0)
    OBJ_NB(-1.0,  0.0) OBJ_NB(0.0,  0.0) OBJ_NB(1.0,  0.0)
    OBJ_NB(-1.0,  1.0) OBJ_NB(0.0,  1.0) OBJ_NB(1.0,  1.0)
    return vec4(u, s.g, s.b, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_OB3
//!SAVE CELFLARE_OB4
//!WIDTH CELFLARE_OB3.w
//!HEIGHT CELFLARE_OB3.h
//!DESC CelFlare: Highlight size spread 4

#define OBJ_NB(dx, dy) { vec4 n = CELFLARE_OB3_tex(CELFLARE_OB3_pos + vec2(dx, dy) * CELFLARE_OB3_pt); u = max(u, n.g * n.r); }

vec4 hook() {
    vec4 s = CELFLARE_OB3_tex(CELFLARE_OB3_pos);
    float u = s.b;
    OBJ_NB(-1.0, -1.0) OBJ_NB(0.0, -1.0) OBJ_NB(1.0, -1.0)
    OBJ_NB(-1.0,  0.0) OBJ_NB(0.0,  0.0) OBJ_NB(1.0,  0.0)
    OBJ_NB(-1.0,  1.0) OBJ_NB(0.0,  1.0) OBJ_NB(1.0,  1.0)
    return vec4(u, s.g, s.b, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_OB4
//!SAVE CELFLARE_OB5
//!WIDTH CELFLARE_OB4.w
//!HEIGHT CELFLARE_OB4.h
//!DESC CelFlare: Highlight size spread 5

#define OBJ_NB(dx, dy) { vec4 n = CELFLARE_OB4_tex(CELFLARE_OB4_pos + vec2(dx, dy) * CELFLARE_OB4_pt); u = max(u, n.g * n.r); }

vec4 hook() {
    vec4 s = CELFLARE_OB4_tex(CELFLARE_OB4_pos);
    float u = s.b;
    OBJ_NB(-1.0, -1.0) OBJ_NB(0.0, -1.0) OBJ_NB(1.0, -1.0)
    OBJ_NB(-1.0,  0.0) OBJ_NB(0.0,  0.0) OBJ_NB(1.0,  0.0)
    OBJ_NB(-1.0,  1.0) OBJ_NB(0.0,  1.0) OBJ_NB(1.0,  1.0)
    return vec4(u, s.g, s.b, 1.0);
}

//!HOOK MAIN
//!BIND CELFLARE_OB5
//!SAVE CELFLARE_OBJ
//!WIDTH CELFLARE_OB5.w
//!HEIGHT CELFLARE_OB5.h
//!DESC CelFlare: Highlight size spread 6

#define OBJ_NB(dx, dy) { vec4 n = CELFLARE_OB5_tex(CELFLARE_OB5_pos + vec2(dx, dy) * CELFLARE_OB5_pt); u = max(u, n.g * n.r); }

vec4 hook() {
    vec4 s = CELFLARE_OB5_tex(CELFLARE_OB5_pos);
    float u = s.b;
    OBJ_NB(-1.0, -1.0) OBJ_NB(0.0, -1.0) OBJ_NB(1.0, -1.0)
    OBJ_NB(-1.0,  0.0) OBJ_NB(0.0,  0.0) OBJ_NB(1.0,  0.0)
    OBJ_NB(-1.0,  1.0) OBJ_NB(0.0,  1.0) OBJ_NB(1.0,  1.0)
    return vec4(s.g * u, s.g, s.b, 1.0);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND CELFLARE_ADD_STATE
//!BIND CELFLARE_STATS
//!BIND CELFLARE_ILLUM
//!BIND MOTION_FLOW
//!BIND CELFLARE_SKSM
//!BIND CELFLARE_SKF
//!BIND CELFLARE_GF
//!BIND CELFLARE_OBJ
//!COMPUTE 16 16
//!DESC CelFlare v7.0 (motion-aware additive A2 + gradient-preserving highlights)

// =============================================
//  MAIN TUNING — deep anchors. The supported user surface is the cf_* block
//  at the top of the file; those knobs scale the values below (neutral at 1).
// =============================================
#define INTENSITY       cf_strength  // Global scaling knob (top of file). Also scales
                                     // the top lift + pump at their apply sites, so 0 = SDR.
#define KNEE            0.30    // Expansion onset — midtones below this stay near SDR


// =============================================
//  SPATIALLY-MODULATED CURVE — regional adaptation
// =============================================
// expansion = 1 + (peak - 1) * t, t = pow(ramp above KNEE, gamma). Y_illum
// sets peak and gamma: bright regions get gentle, broad curves (highlight
// gradients kept), dark regions steep ones (pop). With gamma >= 1 the
// derivative only increases: no inflection in the face range. The APL and
// dynamic-intensity terms are gentle (~10 %) scene adjustments on top.
// Nits at cf_ref_white 116, cf_strength 1, cf_curve 1, curve
// only: peak (Y 1.00) ~278-313, highlights (Y 0.90-0.95) 180-250, reference
// white (Y 0.85) 145-155, midtones (Y <= 0.50) near SDR.
#define PEAK_BRIGHT     2.4     // Expansion peak for bright regions (~278 nits pre-APL)
#define PEAK_DARK       2.7     // Expansion peak for dark regions (~313 nits pre-APL)
#define GAMMA_BRIGHT    2.1     // Gentler ramp through 0.85–0.95 — peak preserved at Y=1.0
#define GAMMA_DARK      2.3     // Matching gradualness in dark scenes
// Bright-scene SHAPE control ("bright where it matters"): bright anime still
// ran whole scene bodies 30-50 % over SDR, a high APL that tires the eyes.
// Two pieces ride the illum-weighted cool_w (bright FIELDS cool; faces and
// warm objects at mid illumination keep the normal curve; scenes with
// apl_t <= 0.5 are unchanged):
//  - SHAPE (this define): a steeper ramp gamma pulls the field's body back
//    toward the SDR grade (expansion stays >= 1).
//  - LEVEL (APL_BRIGHT_COOL): broad near-white fields settle at ~170-185 nits.
#define GAMMA_APL_BOOST 1.8     // local_gamma multiplier at cool_w=1
// Cool the FIELDS, not the faces: scene-key cooling alone drops shaded
// skin's level while its surround gains, and warm orange at lower relative
// luminance reads BROWN. APL fatigue lives in broad bright fields (high
// Y_illum); faces sit at mid Y_illum. The field is smooth and both endpoint
// curves are monotone, so the per-pixel mix stays contour-free.
#define COOL_ILLUM_LO   0.55    // Y_illum below: reshape fully off (normal curve)
#define COOL_ILLUM_HI   0.80    // Y_illum above: full cooling

// Saturated-brightness credit on the base ramp. BT.709 luma under-credits
// saturated R/B-dominant colors, so a bright saturated accent in a bright
// field lags the convex ramp and reads as a dark stain (cheek blush: V 0.952
// vs skin 0.994 but Y 0.735 vs 0.942, x1.23 vs x1.84: a purple bruise).
// Perception follows V more than Y (Helmholtz-Kohlrausch), so saturated
// pixels get a BOUNDED credit from stabilized Y toward stabilized V. Safe
// where a V spec driver was not: this ramp is ~10x gentler and the credit
// halves the coupling (4:2:0 chroma noise ~1-2 nits), and the bounded mix
// cannot go flat where V clips. Near-neutrals are unchanged; the Y floor
// keeps the early exit exact; dim saturated emissives keep their SDR level.
// BASE_V_CREDIT 0.50 is a measured parity point: blush strokes and the AA
// band along dark outlines land within ~1 % of their surround (0.75 made a
// hot outline halo; 0.0 is the bruise, -9 %). Calibrated at cf_curve 2; at
// cf_curve 1 the ideal is ~0.40-0.46. Content-dependent: re-sweep on a
// saturated-accent capture before changing it.
#define ENABLE_BASE_V_CREDIT 1
#define BASE_V_CREDIT        0.50   // fraction of the Y->V gap credited at full gate
#define BASE_V_SAT_LO        0.10   // sat_gamma gate band
#define BASE_V_SAT_HI        0.30
#define BASE_V_Y_LO          0.32   // luma floor fade-in: 0 at/below the early exit (KNEE)
#define BASE_V_Y_HI          0.48
#define PEAK_ATTEN      0.12    // Gentle bright_frac dampening (spatial curve adapts)
#define BRIGHT_FRAC_REF 0.40    // Bright fraction where scene adaptation plateaus

// =============================================
//  DYNAMIC INTENSITY — contrast-driven expansion scaling
// =============================================
// Flat or pastel scenes slightly softer, dramatic ones slightly punchier.
#define ENABLE_DYNAMIC_INTENSITY 1
#define DYN_CONTRAST_LOW    2.5     // Below this: flat scene, minimum intensity
#define DYN_CONTRAST_HIGH   5.5     // Above this: dramatic scene, maximum intensity
#define DYN_INTENSITY_LOW   0.90    // Multiplier for flat scenes (gentle)
#define DYN_INTENSITY_HIGH  1.15    // Multiplier for dramatic scenes (gentle)

// =============================================
//  APL MODULATION — brightness-driven expansion scaling
// =============================================
// Dark scenes get extra headroom (GAMMA_DARK still holds the midtones);
// bright scenes are gently reduced.
#define ENABLE_APL_MOD      1
#define APL_KEY_DARK        0.03    // Below this: dark scene multiplier
#define APL_KEY_BRIGHT      0.30    // Above this: bright scene multiplier
#define APL_BOOST_DARK      1.25    // Dark-scene factor (extra headroom)
#define APL_DAMPEN_BRIGHT   0.65    // Bright-scene factor: full white ~206 nits at ref white 116, cf_strength 1 (Y_illum 0.7, bf 1, before APL_BRIGHT_COOL)
// Bright-field level: an amplitude pull on the illum-weighted cool_w, so
// broad near-white fields settle at ~170-185 nits (bright endpoint 0.65 -
// 0.20 = 0.45). A raw APL_DAMPEN_BRIGHT retune would leak ~-10 % into
// mid-key scenes and would not spare faces.
#define APL_BRIGHT_COOL     0.20    // apl_factor pulldown at cool_w=1
// Mid-scene notch: a parabolic dampener peaking at apl_t 0.5 (normally lit
// interiors) so skin, fabric and hair do not look lit up. Applied before the
// growth bypass.
#define MID_APL_DAMPEN      0.08    // Peak reduction at apl_t=0.5 (~8% off expansion)

// =============================================
//  VELOCITY-GATED DAMPENER BYPASS — expanding-object HDR pop
// =============================================
// Pass 8's smoothed_growth_mode (an expanding bright object) pulls the
// bright-scene dampeners back toward neutral. Bypass strength = fraction of
// each dampener removed at full growth mode.
#define ENABLE_GROWTH_BYPASS    1
#define GROWTH_PEAK_ATTEN_BYPASS 0.7   // PEAK_ATTEN scaling: (1 - this * growth_mode)
#define GROWTH_APL_BYPASS        0.8   // APL factor → mix toward 1.0 by (this * growth_mode)

// =============================================
//  LIGHT PUMP — augment sudden sustained brightening
// =============================================
// Pass 8 publishes the per-cell mask pump_mask_cell (16x9, bilinear-sampled
// here), the cover gate and the scalar pump_env (see its SPATIAL MODEL). The
// pump multiplies the finished expansion (base, APL, top lift) by a
// brightness-weighted gain on the rising EDGE of an event; the growth bypass
// instead REMOVES dampening on a sustained object. A fireball triggers
// both, so the pump is down-gated by growth mode.
#define PUMP_STRENGTH       0.6      // gain per unit mask/env at full pixel weight (x cf_pump x
                                   // cf_strength). Proportional while the product stays <= CEIL.
#define PUMP_Y_LOW          0.62   // per-pixel weight onset: midtones hold the SDR grade (a
                                   // lit face at Y 0.70 gets 0.11, a near-clip body 0.95).
                                   // A saturated event (blue spell, Y ~0.45) scores 0. If one
                                   // regresses, do NOT switch to plain max(Y,V): V reads skin
                                   // ~0.16 hotter and re-admits the face leak wholesale. The
                                   // bridge is chroma-qualified V (high V AND high saturation).
                                   // Do not lower this back.
#define PUMP_GAIN_CEIL      1.5    // hard cap on the pump multiplier (safety against runaway expansion)
#define PUMP_GROWTH_DAMP    0.6    // down-gate pump where growth-mode already lifts expansion (anti double-stack on fireballs)
// Apply mode (= cf_additive_pump; pass 8 aliases the same PARAM).
// 1 = ADDITIVE: pump_local = mask x pump_cover_gate; each region pumps at
// its own strength and rhythm. 0 = SUBTRACTIVE: pump_local = pump_env x mask,
// the mask only suppresses the scalar (the verified reference). The first
// additive build lost the scalar's motion safety (a bright feature crossing
// cells under a moving camera read "fresh" in each new cell); the A2 motion
// veto and persisted proof in pass 8 answer that. The additive path saturates
// at PUMP_CELL_DRIVE_HIGH (0.15) per cell vs the scalar's 0.20, so moderate
// events run a little hotter (that pass-8 knob is the amplitude lever).
#define SPATIAL_PUMP_ADDITIVE cf_additive_pump

// Cross-pass rules: each HOOK block is a SEPARATE compilation unit (a define
// used in two passes must exist in both), and an undefined identifier in an
// #if silently evaluates to 0.

// (No clip diffusion: darkening small light cores against a dark field
// attenuates source clipping, a rule-1 violation no tuning can fix.)

// =============================================
//  CHROMA — expansion color behavior
// =============================================
// Chroma scales with cbrt(expansion) like L: constant chromaticity in linear
// light (the BT.2446-style choice). No chroma attenuation.

// Near-neutral fast path: for desaturated pixels the Oklab round trip is a
// uniform linear-RGB scale (warm shift and pale skin are gated by chroma >
// 0.015). Not exact at sat_gamma 0.04 (Oklab chroma up to 0.0237): the worst
// seam is a hue displacement <= 0.0014 in (a,b), sub-JND.
#define ENABLE_OKLAB_BYPASS 1
#define SAT_BYPASS_THRESH   0.04

// --- Warm shift: Bezold-Brucke hue compensation ---
// Rotates warm hues (yellow-green to near-red) toward red in Oklab to offset
// the perceived green shift at higher luminance. Driven by the illumination
// field (the effect is regional). b_norm scaling stops overshoot: near-red
// barely rotates, yellow fully.
#define ENABLE_WARM_SHIFT    cf_warm_shift   // top-of-file toggle
#define WS_HUE_COS          0.3420  // cos(70°) — center of warm range in Oklab
#define WS_HUE_SIN          0.9397  // sin(70°)
#define WS_HUE_POWER        1.2     // Hue window width (lower = wider, ~50° each side)
#define WS_STRENGTH          0.06   // Max rotation in radians (~3.4° at full drive)
#define WS_ILLUM_LOW         0.35   // Y_illum below: no shift (dark region)
#define WS_ILLUM_HIGH        0.80   // Y_illum above: full shift
#define WS_CHROMA_FLOOR      0.015  // Skip near-neutrals (unstable hue)

#define ENABLE_PALE_SKIN    cf_pale_skin   // top-of-file toggle
#define PS_HUE_COS          0.7317  // cos(43°) — warm hue center for skin detection
#define PS_HUE_SIN          0.6816  // sin(43°)
#define PS_HUE_POWER        2.0     // Sharpness of hue window
#define PS_BRIGHT_FRAC_LOW  0.05
#define PS_BRIGHT_FRAC_HIGH 0.30
#define PS_SAT_BOOST        0.20
#define PS_BRIGHT_FLOOR     0.50
#define PS_CHROMA_CEIL      0.03
// Skin lift: a Hunt-effect counter to the bright-field cooling. With fields
// pulled down and skin held, skin reads brighter and more colorful, i.e.
// more TAN than the SDR grade; a small lift toward pale counters it. Keyed on
// the SCENE cooling weight (skin itself is exempt from cool_w). Wider chroma
// window than the sat boost; zero at apl_t <= 0.5.
#define PS_LIFT             0.10    // linear-light lift at full gate (~3.2% Oklab L)
#define PS_LIFT_CHROMA_HI   0.09    // lift chroma falloff start (pale band ends ~0.07)
#define PS_LIFT_CHROMA_CEIL 0.14    // lift fully off — deep-saturated warm colors excluded
// cf_skin_steady (cooling exemption): the skin window itself is in pass 4b.
#define SKIN_ORDER_SOFT     0.04    // order guard's smooth-minimum width, expansion units
// Lookup of the pass-4c weight: likeness of a block to the pixel's own colour
// (Oklab distance, sigma 0.015: half the averaging range, so a block that
// straddles an outline does not speak for the pixels on either side), and the
// summed likeness below which the plain bilinear weight takes over.
#define SKS_LOOK_K          2222.2  // 1 / (2 * 0.015^2)
#define SKS_CONF_LO         0.02
#define SKS_CONF_HI         0.20
// Decision chroma (pass 4c) feeds these gates' weights instead of the raw pixel.
#define DEC_CHROMA_PS       1       // pale-skin lift and saturation boost windows
#define DEC_CHROMA_VC       1       // saturated-colour credit on the base ramp
#define VC_LOCAL            1       // credit's decision chroma from the pixel's 5x5 neighbourhood (0 = block lookup)
#define VC_LOCAL_R          4.0     // tap spacing, px (4: chroma noise averaged a little better than 2, edges equal)
#define VC_LOCAL_PS         1       // pale-skin gates read the same neighbourhood chroma
#define VC_LOCAL_LIKE       1       // taps weighted by likeness to the pixel (0 = plain mean)
#define VC_LOCAL_LIKE_K     25.0   // 1 / (2 sigma^2) on gamma rgb
#define VC_LOCAL_LIKE_FLOOR 0.0   // least weight of a tap (keeps an anti-aliased edge pixel's decision smooth in motion)
// The colour credit's ratio reads its own likeness (DEC_OWN_LOOK). The skin
// weight's likeness is sharp (sigma 0.015) and counts lightness fully; on a
// soft white-to-blue slope (a cloud's flank) the lightness changes by about
// 0.03 from block to block, so the read hopped from block to block as the
// pixel's lightness passed each block's level, the ratio stepped, and the
// credit drew thin rings along the contours (owner, 2026-11-23: "ring
// bands"). Its own likeness is mostly colour and wider; the pale-skin gates
// keep ab_dec.
#define DEC_OWN_LOOK        1
// The skin exemption's weight reads its own likeness too (SKW_OWN_LOOK): with
// the skin likeness (sigma 0.015, lightness counted fully) the read hopped
// from block to block along soft warm gradients as the lightness passed each
// block's level, and drew staircase contours around defocused shapes (owner,
// 2026-11-23, testing the highlight model: "ragged edges on the background").
#define SKW_OWN_LOOK        1
#define SKW_FIT             1       // skin weight = per-pixel fit linear in rgb (keep equal to SKW_FIT_POOL)
#define PS_DEC_WIDE         1       // pale-skin gates read the wide decision chroma
#define SKW_LOOK_K          225.0   // 1 / (2 * 0.047^2)
#define SKW_LOOK_L          0.25    // weight of the lightness difference in it
#define DEC_LOOK_K          225.0   // 1 / (2 * 0.047^2)
#define DEC_LOOK_L          0.25    // weight of the lightness difference in it
#define VC_DEC_DRIVE        1       // 1 = the credit's drive gap scaled by the ratio too; 0 = gate only
#define VC_DEC_LO           0.5     // clamp of the ratio
#define VC_DEC_HI           2.0
// =============================================
//  OUTPUT — encoding
// =============================================
#define REFERENCE_WHITE cf_ref_white   // top-of-file knob — match hdr-reference-white
#define PQ_FAST_APPROX  1
#define EOTF_GAMMA      2.4
#define ENABLE_GRAIN_STABLE cf_grain_stab   // top-of-file toggle
// == KNEE is exact: pass 1 writes raw luma below its GRAIN_EARLY_EXIT (0.30),
// so below KNEE t = 0 and expansion is exactly 1.0. Must stay <= pass 1's
// GRAIN_EARLY_EXIT, or stabilized decisions could cross KNEE.
#define EARLY_EXIT_GAMMA    KNEE
#define HL_W_MIN            0.90    // lowest diffuse white the top lift is measured from
// Span of the top lift in log luminance from the lowest white to clip,
// 2.4 ln(1 / HL_W_MIN) = 2.4 ln(10 / 9) = 0.2528652. cf_highlight is the lift
// at clip over this full span; a higher scene white gets the same lift per
// unit of span, so less in total. Keep equal to 2.4 ln(1 / HL_W_MIN).
#define HL_SPAN_REF         0.2528652
#define HL_RAW_LUMA         1       // base from the plain luma, not pass 1's (keep equal to the base pass)
#define HL_CORE             1       // core the base's small variations (0 = off)
#define HL_CORE_T           0.006   // coring threshold, gamma units
// Top-lift amount by highlight size (passes 4g-4i): full cf_highlight on a
// compact highlight (object coverage up to HL_SIZE_LO), none on a broad
// near-white field (from HL_SIZE_HI). Set by measurement: these bounds take a
// 60 px painted band at 1080p out of the lift (0.15-0.50 left it half lifted),
// while glints and small lamps keep the full lift.
#define HL_SIZE_LO          0.05
#define HL_SIZE_HI          0.30
// Shoulder: in a cooled bright field the curve is steeper (GAMMA_APL_BOOST)
// and arrives at its peak hard; there it arrives softly instead:
// t += HL_SHOULDER x cool_w x t^HL_SHOULDER_K (1 - t). Monotone for
// HL_SHOULDER <= 1 (dt'/dt >= 1 - HL_SHOULDER x cool_w >= 0), peak unchanged.
// Skin (exempt from the cooling) and every uncooled pixel keep the plain
// curve. Without it a fade from white left a quarter of the picture outside
// the range of its two ends while the cooling relaxed.
#define HL_SHOULDER         1.0
#define HL_SHOULDER_K       2.0

// =============================================
//  DEBUG
// =============================================
// All views are selected by cf_debug (0 = off).
#define DEBUG_BYPASS         (cf_debug == 1)
#define DEBUG_SHOW_ILLUM     (cf_debug == 2)   // Illumination field as grayscale
#define DEBUG_SHOW_EXPANSION (cf_debug == 3)   // Expansion amount as heat map
#define DEBUG_SHOW_DETAIL    (cf_debug == 4)   // Base expansion only (before scene terms): green = (expansion - 1) x 2
#define DEBUG_SHOW_TOPLIFT   (cf_debug == 5)   // Top lift: cyan = (lift - 1) x 4
#define DEBUG_SHOW_PUMP      (cf_debug == 6)   // Light pump: red = scalar pump_env, green = applied gain, blue = cell mask
#define DEBUG_SHOW_WP        (cf_debug == 7)   // Warm shift + pale skin
#define DEBUG_SHOW_STATS     (cf_debug == 8)   // bright_frac, contrast, log_avg and anchor bars

// ---------------------------------------------------------------------------
// Debug legend overlay — title + color key, bottom-left panel
// ---------------------------------------------------------------------------
// Each view draws a title and one swatch + label row per channel, in the
// colors it emits. The warm/skin view says DISABLED when both its toggles are
// off. Views 9-12 have no panel. The overlay compiles out at cf_debug == 0.
#if cf_debug != 0

// 5x6 bitmap font: bit y*5 + x = pixel on. Glyphs A-Z, 0-9, - . / = ( ).
// Machine-generated: regenerate rather than hand-edit the hex.
const uint DBG_FONT[42] = uint[42](
    0x231fc62eu, 0x1f18be2fu, 0x3c10843eu, 0x1f18c62fu, 0x3e10bc3fu, 0x0210bc3fu,   // A B C D E F
    0x3d18e43eu, 0x2318fe31u, 0x3e42109fu, 0x1d184210u, 0x23149d31u, 0x3e108421u,   // G H I J K L
    0x2318d771u, 0x231cd671u, 0x1d18c62eu, 0x0210be2fu, 0x2c9ac62eu, 0x2314be2fu,   // M N O P Q R
    0x1f08383eu, 0x0842109fu, 0x1d18c631u, 0x08a8c631u, 0x23bac631u, 0x22a21151u,   // S T U V W X
    0x08421151u, 0x3e11111fu, 0x1d19d72eu, 0x3e4210c4u, 0x3e22222eu, 0x1f08320fu,   // Y Z 0 1 2 3
    0x108fa988u, 0x1f083c3fu, 0x1d18bc2eu, 0x0842221fu, 0x1d18ba2eu, 0x1d087a2eu,   // 4 5 6 7 8 9
    0x00007c00u, 0x08400000u, 0x02221110u, 0x000f83e0u, 0x08210844u, 0x08842104u    // - . / = ( )
);

// Labels: 6 bits/char, 5 chars per uint, LSB first (codes 0 space, 1-26 A-Z,
// 27-36 0-9, 37-42 - . / = ( )); the text rides in a comment beside each.
// The caller divides p by its pixel scale AFTER clamping negatives out (int
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
#elif DEBUG_SHOW_TOPLIFT
    #define DBG_TITLE     uvec4(0x103d4020u, 0x14189300u, 0u, 0u)               // 5 TOP LIFT
    #define DBG_TITLE_CH  10
    #define DBG_NROWS     1
    #define DBG_ROW_MAXCH 15
#elif DEBUG_SHOW_PUMP
    #define DBG_TITLE     uvec4(0x0724c021u, 0x15400508u, 0x0000040du, 0u)      // 6 LIGHT PUMP
    #define DBG_TITLE_CH  12
        #define DBG_NROWS     3
        #define DBG_ROW_MAXCH 14
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
    #define DBG_ROW_MAXCH 12
#endif

// Swatch color + packed label for key row i of the active view.
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
#elif DEBUG_SHOW_TOPLIFT
    if (i == 0) { col = vec3(0.0, 1.0, 1.0);
        txt = uvec4(0x28381643u, 0x14189329u, 0x1f62a725u, 0u); len = 15; } // CYAN=(LIFT-1)X4
#elif DEBUG_SHOW_PUMP
    if (i == 0) { col = vec3(1.0, 0.0, 0.0);
        txt = uvec4(0x010d3a12u, 0x0501204cu, 0x0000058eu, 0u); len = 12; }          // R=SCALAR ENV
    else if (i == 1) { col = vec3(0.0, 1.0, 0.0);
        txt = uvec4(0x10401a07u, 0x0010524cu, 0x00389047u, 0u); len = 14; }          // G=APPLIED GAIN
    else if (i == 2) { col = vec3(0.0, 0.0, 1.0);
        txt = uvec4(0x0c143a02u, 0x1304d00cu, 0x0000000bu, 0u); len = 11; }          // B=CELL MASK
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
        txt = uvec4(0x0f203381u, 0x238e6012u, 0x00000725u, 0u); len = 12; }          // ANCHOR .88-1
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
    // Passthrough and early-exit path: ALWAYS the exact OETF (the fast
    // polynomial decodes L=0 as ~0.12 nits and would lift bars and blacks).
    vec3 linear = eotf_gamma(rgb_gamma);
    vec3 bt2020 = max(bt709_to_bt2020(linear), 0.0);
    return pq_oetf(bt2020 * (REFERENCE_WHITE / 10000.0));
}

// Fast-polynomial repairs. Below ~30 nits its error explodes (a ~0.2-0.35 nit
// black floor per channel), so small channels (e.g. the blue of a red
// emissive) blend to the exact OETF.
#define PQ_EXACT_LOW    0.0015   // L normalized (≈15 nits): fully exact below
#define PQ_EXACT_HIGH   0.0030   // L normalized (≈30 nits): fully fast above
// It is fitted only up to L ~0.20: past ~2000 nits per channel it runs away
// (2500 nits -> ~4000, >= ~2900 -> 10000). Reachable at high cf_ref_white,
// cf_strength or cf_highlight, so channels above 1800 nits blend to the exact OETF.
#define PQ_FAST_MAX_LO  0.18     // L normalized (1800 nits): exact blend starts
#define PQ_FAST_MAX_HI  0.20     // L normalized (2000 nits): fully exact above

vec3 linear709_to_pq2020(vec3 rgb_linear) {
    vec3 bt2020 = max(bt709_to_bt2020(rgb_linear), 0.0);
    vec3 L = bt2020 * (REFERENCE_WHITE / 10000.0);
    #if PQ_FAST_APPROX
        vec3 pq = pq_oetf_fast(L);
        if (min(min(L.r, L.g), L.b) < PQ_EXACT_HIGH) {
            vec3 w = smoothstep(PQ_EXACT_LOW, PQ_EXACT_HIGH, L);
            pq = mix(pq_oetf(L), pq, w);
        }
        if (max(max(L.r, L.g), L.b) > PQ_FAST_MAX_LO) {
            vec3 wh = smoothstep(PQ_FAST_MAX_LO, PQ_FAST_MAX_HI, L);
            pq = mix(pq, pq_oetf(L), wh);
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
// Sigma-80 px (at 1080p) field: one bilinear fetch is enough.

vec3 upsample_illum_rgb() {
    return CELFLARE_ILLUM_tex(CELFLARE_ILLUM_pos).rgb;
}

// =============================================================================
// MAIN PROCESSING
// =============================================================================

// ---------------------------------------------------------------------------
// WORKGROUP STATE SNAPSHOT — NV Vulkan uncached SSBO reads
// ---------------------------------------------------------------------------
// On NVIDIA Vulkan, per-pixel loads from a BUFFER-directive SSBO bypass the
// cache: this pass's ~10 scalar + 4 cell reads per pixel measured
// 0.153 ms/frame at 1080p vs 0.024 ms on d3d11. So the pass is COMPUTE and
// lane 0 copies every scalar the live path reads into shared memory (the 16x9
// mask cooperatively), published by one barrier FIRST in hook(), before any
// data-dependent return (FXC X3663: barriers only in uniform control flow;
// no other barrier in this TU). Value-identical: pass 8 finished the state
// before this pass started. Debug views read the buffer directly.
// COMPUTE hooks write out_image themselves, so the pixel body is cf_shade()
// and hook() owns the snapshot, the barrier and the guarded store.
shared float sh_bright_frac;
shared float sh_growth_mode;
shared float sh_log_avg;
shared float sh_contrast;
shared float sh_spec_anchor;
shared float sh_pump_env;
shared float sh_pump_cover_gate;
shared float sh_pump_mask_cell[144];

vec4 cf_shade() {
#if cf_debug == 9
    // Motion offset: grey = still; red/green = +x/+y PREVIOUS-frame source
    // offset (a right/down move is negative); blue = |offset| / MOT_R.
    vec2 mf = MOTION_FLOW_tex(HOOKED_pos).xy;
    return vec4(clamp(0.5 + mf.x * 0.1, 0.0, 1.0),
                clamp(0.5 + mf.y * 0.1, 0.0, 1.0),
                clamp(length(mf) * (1.0 / 5.0), 0.0, 1.0), 1.0);
#endif
#if cf_debug == 10 || cf_debug == 11
    // Motion evidence (11 = alias), per cell: red = winning SAD (full at
    // 0.10; also shown on vetoed tiles), green = tile RMS contrast (full at
    // 0.05), blue = the tile's vote for the reset (0 on vetoed tiles).
    ivec2 mc = clamp(ivec2(HOOKED_pos * vec2(16.0, 9.0)),
                      ivec2(0), ivec2(15, 8));
    vec2 mpos = (vec2(mc) + 0.5) / vec2(16.0, 9.0);
    vec4 md = MOTION_FLOW_tex(mpos);
    return vec4(clamp(abs(md.z) * 10.0, 0.0, 1.0),
                clamp(md.w * 20.0, 0.0, 1.0),
                clamp(dbg_cell_b[mc.y * 16 + mc.x], 0.0, 1.0), 1.0);
#endif
#if cf_debug == 12
    // Additive opening proof, per cell: red = established level, green =
    // motion-unexplained rise (after the excursion/source gates), blue =
    // seven-frame persistence. White can open.
    ivec2 mc = clamp(ivec2(HOOKED_pos * vec2(16.0, 9.0)),
                      ivec2(0), ivec2(15, 8));
    int mi = mc.y * 16 + mc.x;
    return vec4(clamp(dbg_cell_r[mi], 0.0, 1.0),
                clamp(dbg_cell_g[mi], 0.0, 1.0),
                clamp(dbg_cell_b[mi], 0.0, 1.0), 1.0);
#endif
    vec4 color = HOOKED_texOff(0);
    vec3 rgb_gamma = color.rgb;

    // -------------------------------------------------------------------------
    // DEBUG: legend panel (bottom-left) — title + color key for the active view
    // -------------------------------------------------------------------------
    // Drawn before every debug return, bypass included.
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
                // Row 4: the held diffuse-white anchor, 0.88 (none) to 1.0 (cyan)
                if (bar_x < (spec_anchor_held - 0.88) * (1.0 / 0.12)) dbg = vec3(0.0, 0.6, 0.6);
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
    // V_gamma feeds the base-ramp V credit; sat_gamma (V - min) the credit's
    // gate and the Oklab fast-path test.
    float V_gamma   = max(max(rgb_gamma.r, rgb_gamma.g), rgb_gamma.b);
    float min_gamma = min(min(rgb_gamma.r, rgb_gamma.g), rgb_gamma.b);
    float sat_gamma = V_gamma - min_gamma;

    // -------------------------------------------------------------------------
    // ILLUMINATION FIELD
    // -------------------------------------------------------------------------
    // The production fetch comes after the early exit; the debug view needs
    // every pixel.
    #if DEBUG_SHOW_ILLUM
    return vec4(gamma709_to_pq2020(upsample_illum_rgb()), 1.0);
    #endif

    // Early exit: no base expansion is possible below the knee.
    // Highlight base: B = a Y_dec + b from the 1/4-resolution fit.
    float Yb_gamma = Y_gamma;
    {
        vec3 gfab = CELFLARE_GF_tex(CELFLARE_GF_pos).rgb;
    #if HL_RAW_LUMA
        Yb_gamma = clamp(gfab.x * Y_gamma + gfab.y, 0.0, 1.0);
    #else
        Yb_gamma = clamp(gfab.x * (color.a * 2.0) + gfab.y, 0.0, 1.0);
    #endif
    #if HL_CORE
        // Core the base's small variations against its same-surface
        // reference: the curve and the top lift multiplied every change of
        // the base, the source's frame-to-frame compression mottle on a flat
        // near-white surface included (boiling, owner 2026-11-24). Changes
        // below HL_CORE_T are flattened in the expansion only; shading,
        // highlights and edges pass.
        {
            float hd = Yb_gamma - gfab.z;
            Yb_gamma = gfab.z + hd * hd * hd / (hd * hd + HL_CORE_T * HL_CORE_T);
        }
    #endif
    }
    if (max(Y_gamma, Yb_gamma) < EARLY_EXIT_GAMMA)
        return vec4(gamma709_to_pq2020(color.rgb), 1.0);

    vec3 illum_rgb = upsample_illum_rgb();
    float Y_illum = get_luma(illum_rgb);
    // Highlight size: the largest coverage in the white object this pixel
    // belongs to (highlight size passes), read from the cells that hold
    // near-white only; the pixel's own coverage where none does.
    float hl_cov = CELFLARE_ILLUM_tex(CELFLARE_ILLUM_pos).a;
    {
        vec2 ob = CELFLARE_OBJ_tex(CELFLARE_OBJ_pos).rg;
        float cov_o = ob.r / max(ob.g, 1e-4);
        hl_cov = mix(hl_cov, max(hl_cov, cov_o), smoothstep(0.05, 0.30, ob.g));
    }

    // -------------------------------------------------------------------------
    // GRAIN STABILIZATION
    // -------------------------------------------------------------------------
    float Y_decision_gamma = Y_gamma;
    #if ENABLE_GRAIN_STABLE
        // Alpha protocol: decision luma stored as Y * 0.5.
        Y_decision_gamma = color.a * 2.0;
    #endif
    // The curve's argument: the smooth base.
    float Y_curve_gamma = Yb_gamma;

    // Stabilized peak channel: the luma stabilizer's correction moved onto V
    // (cancels achromatic grain; chroma noise stays). Used only by the
    // base-ramp V credit.
    float V_stable = V_gamma + (Y_decision_gamma - Y_gamma);

    // -------------------------------------------------------------------------
    // SPATIALLY-MODULATED PER-PIXEL EXPANSION
    // -------------------------------------------------------------------------
    // f(Y_pixel), a monotone remap; Y_illum sets the curve's parameters.

    // Scene-level adaptation (from the state snapshot, uniform per frame)
    float bf = smoothstep(0.0, BRIGHT_FRAC_REF, sh_bright_frac);

    // Regional adaptation (from illumination field — varies per pixel)
    float spatial_t = Y_illum;
    float local_peak = mix(PEAK_DARK, PEAK_BRIGHT, spatial_t);
    // Growth bypass: an expanding hot region must not lose peak as it grows.
    #if ENABLE_GROWTH_BYPASS
    float peak_atten_eff = PEAK_ATTEN * (1.0 - GROWTH_PEAK_ATTEN_BYPASS * sh_growth_mode);
    #else
    float peak_atten_eff = PEAK_ATTEN;
    #endif
    local_peak *= (1.0 - peak_atten_eff * bf);  // scene-level dampening on top
    // Scene APL axis (0 dark key, 1 bright key), shared by the gamma boost,
    // the APL step and the spec params: ONE axis, from the smoothed key.
    float apl_t = smoothstep(APL_KEY_DARK, APL_KEY_BRIGHT, sh_log_avg);
    // cool_w: bright-key cooling, weighted to bright FIELDS (see COOL_ILLUM).
    float cool_w  = smoothstep(0.5, 1.0, apl_t)
                  * smoothstep(COOL_ILLUM_LO, COOL_ILLUM_HI, Y_illum);
    // Skin exemption (cf_skin_steady). The illum gate spares a face only
    // while it is small or mid-toned: a production that paints lit skin near
    // Y 0.87 fills the sigma-80 field with it in a close-up, so the same
    // paint was cooled in one framing and not the next (115 vs 132 nits
    // across one cut). A flat-topped Oklab hue window (pass 4b) and a
    // chroma band pick the colour. The weight is NOT taken from the raw
    // pixel: its hue and chroma carry compression noise and the window's
    // edges are steep, so inside a soft warm gradient the weight switched on
    // along a ragged line that crawled every frame. It is read from the
    // same-surface average of passes 4b-4c instead (crisp at outlines,
    // smooth along gradients, a mix pixel gets the mix of its sides). No
    // saturation cut-off in front: that drew its own line. Hue cannot tell
    // skin from a cream wall, sand or blonde hair of the same hue (see pass
    // 4b), so those keep the normal curve too:
    // the exemption is for pale warm tones, and the cooling keeps working on
    // white, grey, blue and green fields.
    // Only the cooling is exempted. The top lift and the pump keep their
    // own gates: exempting skin hues there (from the old specular bonus)
    // bought under 1 % steadiness and would thin pale fire and sunset cores.
    float cool_w0 = cool_w;   // un-exempted weight: what a neutral pixel here gets
    float skin_ex = 0.0;      // share of the cooling this pixel was let off
    // Decision chroma: the same-surface chroma of this pixel (pass 4c), for
    // gates that key on hue or chroma. Equal to the pixel's own where no block
    // resembles it. dec_ratio = decision chroma / pixel chroma, for gates
    // written on gamma-domain saturation.
    vec2 ab_dec = vec2(0.0);
    vec2 ab_dec_w = vec2(0.0);   // decision chroma from the wide likeness (DEC_OWN_LOOK)
    float dec_ratio = 1.0;
    bool want_skin = cf_skin_steady > 0.0 && cool_w > 0.0;
    if (want_skin || sat_gamma >= SAT_BYPASS_THRESH) {
        vec3 ok_sk = rgb_to_oklab(eotf_gamma(rgb_gamma));
        float skin_w;
        // Same-surface weight (passes 4b-4c): the four nearest blocks, each
        // counted by its bilinear share times its likeness to this pixel.
        {
            vec2 sp = CELFLARE_SKSM_pos * CELFLARE_SKSM_size - 0.5;
            vec2 sb = floor(sp);
            vec2 sf = sp - sb;
            vec3 lab_px = vec3(ok_sk.x, ok_sk.y + 0.5, ok_sk.z + 0.5);
            float acc = 0.0;
            float wacc = 0.0;
            float accb = 0.0;
            vec2 ab_acc = vec2(0.0);
            vec2 abv_acc = vec2(0.0);
            float wdacc = 0.0;
            float sacc = 0.0;
            float swacc = 0.0;
            for (int k = 0; k < 4; k++) {
                vec2 o = vec2(float(k & 1), float(k >> 1));
                vec4 s = CELFLARE_SKSM_tex((sb + 0.5 + o) * CELFLARE_SKSM_pt);
                vec2 bw2 = mix(1.0 - sf, sf, o);
                vec3 d = s.rgb - lab_px;
                float w = bw2.x * bw2.y * exp(-SKS_LOOK_K * dot(d, d));
                accb += bw2.x * bw2.y * s.a;
                acc += w * s.a;
                ab_acc += w * s.gb;
                wacc += w;
                #if SKW_OWN_LOOK
                {
                    vec3 ds = d * vec3(SKW_LOOK_L, 1.0, 1.0);
                    float wsk = bw2.x * bw2.y * exp(-SKW_LOOK_K * dot(ds, ds));
                    sacc += wsk * s.a;
                    swacc += wsk;
                }
                #endif
                #if DEC_OWN_LOOK
                vec3 dd = d * vec3(DEC_LOOK_L, 1.0, 1.0);
                float wd = bw2.x * bw2.y * exp(-DEC_LOOK_K * dot(dd, dd));
                abv_acc += wd * s.gb;
                wdacc += wd;
                #endif
            }
            float sks_conf = smoothstep(SKS_CONF_LO, SKS_CONF_HI, wacc);
            #if SKW_OWN_LOOK
            skin_w = mix(accb, sacc / max(swacc, 1e-6), smoothstep(SKS_CONF_LO, SKS_CONF_HI, swacc));
            #else
            skin_w = mix(accb, acc / max(wacc, 1e-6), sks_conf);
            #endif
            ab_dec = mix(ok_sk.yz, ab_acc / max(wacc, 1e-6) - 0.5, sks_conf);
            #if DEC_OWN_LOOK
            vec2 ab_vc = mix(ok_sk.yz, abv_acc / max(wdacc, 1e-6) - 0.5,
                             smoothstep(SKS_CONF_LO, SKS_CONF_HI, wdacc));
            ab_dec_w = ab_vc;
            #if VC_LOCAL
            // Decision chroma for the colour credit from the pixel's own 9x9
            // neighbourhood (9 taps VC_LOCAL_R px apart): averages chroma noise like the
            // block lookup did, but moves with the content. The lookup depended
            // on where the 8 px grid fell at a moving outline: the credit
            // shimmered along the edges of moving warm objects (owner,
            // 2026-11-24).
            {
                vec3 lm = vec3(0.0);
                float lw = 0.0;
                for (int j = -1; j <= 1; j++)
                    for (int i = -1; i <= 1; i++) {
                        vec3 lc = HOOKED_texOff(vec2(float(i), float(j)) * VC_LOCAL_R).rgb;
                        #if VC_LOCAL_LIKE
                        // likeness to this pixel: the far side of an outline does
                        // not dilute the chroma (a plain mean drew a 4 px band of
                        // lower credit inside saturated fills, audit 2026-11-24)
                        vec3 ld = lc - rgb_gamma;
                        float lwk = mix(VC_LOCAL_LIKE_FLOOR, 1.0, exp(-VC_LOCAL_LIKE_K * dot(ld, ld)));
                        #else
                        float lwk = 1.0;
                        #endif
                        lm += lwk * lc;
                        lw += lwk;
                    }
                vec3 ok_lm = rgb_to_oklab(eotf_gamma(max(lm / max(lw, 1e-6), 0.0)));
                dec_ratio = clamp(length(ok_lm.yz) / max(length(ok_sk.yz), 1e-4), VC_DEC_LO, VC_DEC_HI);
                #if VC_LOCAL_PS
                ab_dec_w = ok_lm.yz;   // the pale-skin gates read the same neighbourhood chroma
                #endif
            }
            #else
            dec_ratio = clamp(length(ab_vc) / max(length(ok_sk.yz), 1e-4), VC_DEC_LO, VC_DEC_HI);
            #endif
            #else
            dec_ratio = clamp(length(ab_dec) / max(length(ok_sk.yz), 1e-4), VC_DEC_LO, VC_DEC_HI);
            #endif
        }
        #if SKW_FIT
        // Skin weight = the cells' fit evaluated with this pixel's own colour
        // (see "Skin fit statistics"): no grid in it, an anti-aliased pixel
        // between its sides' weights.
        {
            vec4 skf = CELFLARE_SKF_tex(CELFLARE_SKF_pos);
            skin_w = clamp(dot(skf.rgb, rgb_gamma) + skf.a, 0.0, 1.0);
        }
        #endif
        if (want_skin) {
            skin_ex = cf_skin_steady * skin_w;
            cool_w *= 1.0 - skin_ex;
        }
    }
    // Growth bypass on the SHAPE too (full pre-reshape curve during an event).
    #if ENABLE_GROWTH_BYPASS
    float bw_gamma = cool_w * (1.0 - GROWTH_APL_BYPASS * sh_growth_mode);
    #else
    float bw_gamma = cool_w;
    #endif
    // cf_curve scales the exponent (< 1 broader lift, > 1 concentrated near
    // Y=1); peak is unchanged. The max(1.0) floor keeps gamma >= 1, i.e. an
    // increasing derivative; GAMMA_APL_BOOST multiplies the same exponent.
    float local_gamma = max(1.0, mix(GAMMA_DARK, GAMMA_BRIGHT, spatial_t) * cf_curve
                                 * mix(1.0, GAMMA_APL_BOOST, bw_gamma));

    // Ramp input: stabilized luma plus the bounded V credit (lift-only via
    // max(); all gates are smooth, so the drive stays contour-free).
    #if ENABLE_BASE_V_CREDIT
    float vcredit_w = BASE_V_CREDIT
    #if DEC_CHROMA_VC
                    * smoothstep(BASE_V_SAT_LO, BASE_V_SAT_HI, sat_gamma * dec_ratio)
    #else
                    * smoothstep(BASE_V_SAT_LO, BASE_V_SAT_HI, sat_gamma)
    #endif
                    * smoothstep(BASE_V_Y_LO, BASE_V_Y_HI, Y_curve_gamma);
    float base_drive = mix(Y_curve_gamma,
    #if DEC_CHROMA_VC
                           max(Y_curve_gamma, Y_curve_gamma + (V_stable - Y_decision_gamma) * mix(1.0, dec_ratio, float(VC_DEC_DRIVE))), vcredit_w);
    #else
                           max(Y_curve_gamma, Y_curve_gamma + (V_stable - Y_decision_gamma)), vcredit_w);
    #endif
    #else
    float base_drive = Y_curve_gamma;
    #endif
    float t = max(base_drive - KNEE, 0.0) / (1.0 - KNEE);
    t = pow(min(t, 1.0), local_gamma);  // clamp for upscaler super-whites
    // Shoulder in cooled bright fields (see HL_SHOULDER).
    t += HL_SHOULDER * cool_w * pow(t, HL_SHOULDER_K) * (1.0 - t);
    float expansion = 1.0 + (local_peak - 1.0) * t * INTENSITY;
    // Expansion a COOLED white (t = 1) gets at this spot; carried through the
    // scene steps for the exemption's guard.
    float e_white = 1.0 + (local_peak - 1.0) * INTENSITY;

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
        float dyn_factor = smoothstep(DYN_CONTRAST_LOW, DYN_CONTRAST_HIGH, sh_contrast);
        float dyn_intensity = mix(DYN_INTENSITY_LOW, DYN_INTENSITY_HIGH, dyn_factor);
        expansion = 1.0 + (expansion - 1.0) * dyn_intensity;
        e_white = 1.0 + (e_white - 1.0) * dyn_intensity;
    }
    #endif

    // -------------------------------------------------------------------------
    // STEP 4: APL MODULATION (brightness-driven scaling)
    // -------------------------------------------------------------------------
    // apl_t is the shared scene axis declared above the curve.
    #if ENABLE_APL_MOD
    {
        float apl_factor = mix(APL_BOOST_DARK, APL_DAMPEN_BRIGHT, apl_t);
        // Mid-scene notch (before the growth bypass, which can undo it).
        float mid_notch = MID_APL_DAMPEN * apl_t * (1.0 - apl_t) * 4.0;
        apl_factor *= 1.0 - mid_notch;
        // Bright-field pulldown (see APL_BRIGHT_COOL), also before the bypass
        // so an event overrides it instead of stacking.
        float apl_white = apl_factor - APL_BRIGHT_COOL * cool_w0;
        apl_factor -= APL_BRIGHT_COOL * cool_w;
        // Growth bypass: pull apl_factor toward 1.0 during an event (dark
        // scenes lose a little boost too; the spatial curve covers it).
        #if ENABLE_GROWTH_BYPASS
        apl_factor = mix(apl_factor, 1.0, GROWTH_APL_BYPASS * sh_growth_mode);
        apl_white  = mix(apl_white, 1.0, GROWTH_APL_BYPASS * sh_growth_mode);
        #endif
        expansion = 1.0 + (expansion - 1.0) * apl_factor;
        e_white = 1.0 + (e_white - 1.0) * apl_white;
    }
    #endif

    // Order guard for the skin exemption. An exempt pixel runs the uncooled
    // curve beside neutral pixels that are cooled, and the gate is per pixel,
    // so without a bound it reorders tones inside one object: blonde hair
    // fill (Y 0.948) reached 161 nits against its own highlights at 153-170
    // (source 102 vs 112-114), cheek blush strokes rose from 0.84 to 0.91 of
    // the skin around them (source 0.80), and anti-aliased pixels along dark
    // line art drew a bright fringe. Bound: an exempt pixel's expansion
    // factor never exceeds the one cooled white gets at the same spot, so it
    // keeps its source contrast to that white. The bound sits above the
    // cooled curve everywhere (that curve peaks at white), so it only takes
    // back part of the exemption, and both branches rise with luma, so the
    // result does too. Polynomial smooth minimum (no crease where they meet);
    // the soft term fades in with the exemption so a pixel at the edge of the
    // skin window takes no step. Measured cost: pairs of same-skin shots
    // stepping over 10 % 7 % -> 11 % on one production, unchanged on three.
    if (skin_ex > 0.0) {
        float sm_h = max(SKIN_ORDER_SOFT - abs(expansion - e_white), 0.0) / SKIN_ORDER_SOFT;
        expansion = min(expansion, e_white)
                  - sm_h * sm_h * SKIN_ORDER_SOFT * 0.25 * min(skin_ex * 16.0, 1.0);
    }

    // -------------------------------------------------------------------------
    // HIGHLIGHT TOP LIFT
    // -------------------------------------------------------------------------
    // Above the scene's diffuse white W (the held anchor, at least HL_W_MIN),
    // on the smooth base: lift = exp(h u^1.5), u = position between W and
    // clip in log luminance, h = ln(cf_highlight) x span / HL_SPAN_REF with
    // span = 2.4 ln(1 / W). h is proportional to the span, so the steepest
    // added log-log slope (1.5 h / span at clip) is the same in every scene
    // and the lift shrinks to nothing as W nears clip: a soft gradient or a
    // flat clipped area never gets the whole factor over a few code values.
    // cf_strength rides along (0 = no-op).
    float hl_lift;
    {
        float hl_xw = 2.4 * log(max(sh_spec_anchor, HL_W_MIN));
        float hl_u = clamp((2.4 * log(max(Yb_gamma, 1e-4)) - hl_xw) / max(-hl_xw, 1e-4), 0.0, 1.0);
        float hl_h = log(cf_highlight) * (-hl_xw) * (1.0 / HL_SPAN_REF);
        // compact highlights only (see HL_SIZE_*)
        hl_h *= 1.0 - smoothstep(HL_SIZE_LO, HL_SIZE_HI, hl_cov);
        hl_lift = exp(max(hl_h, 0.0) * hl_u * sqrt(hl_u) * cf_strength);
        expansion *= hl_lift;
    }

    #if DEBUG_SHOW_TOPLIFT
    {
        float lift_dbg = (hl_lift - 1.0) * 4.0;
        return vec4(gamma709_to_pq2020(vec3(0.0, lift_dbg, lift_dbg)), 1.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // LIGHT PUMP — augment sudden sustained brightening (after the top lift)
    // -------------------------------------------------------------------------
    // Exposure-like gain on the finished expansion, weighted by pixel
    // brightness so shadows hold; capped by PUMP_GAIN_CEIL.
    float pump_gain = 0.0;
    // Hoisted so the pump debug view shows the exact mask used.
    float pump_mask = 1.0;
    {
        float pump_w = smoothstep(PUMP_Y_LOW, 1.0, Y_decision_gamma);
        // Bilinear sample of the 16x9 presentation mask.
        vec2  pg  = vec2(HOOKED_pos.x * 16.0 - 0.5, HOOKED_pos.y * 9.0 - 0.5);
        vec2  pgf = fract(pg);
        pgf = pgf * pgf * (3.0 - 2.0 * pgf);    // smoothstep-eased fraction (C1): no bilinear kink seams
        ivec2 pib = ivec2(floor(pg));
        ivec2 pi0 = clamp(pib,     ivec2(0), ivec2(15, 8));
        ivec2 pi1 = clamp(pib + 1, ivec2(0), ivec2(15, 8));
        float m00 = sh_pump_mask_cell[pi0.y * 16 + pi0.x];
        float m10 = sh_pump_mask_cell[pi0.y * 16 + pi1.x];
        float m01 = sh_pump_mask_cell[pi1.y * 16 + pi0.x];
        float m11 = sh_pump_mask_cell[pi1.y * 16 + pi1.x];
        pump_mask = mix(mix(m00, m10, pgf.x), mix(m01, m11, pgf.x), pgf.y);
        #if SPATIAL_PUMP_ADDITIVE
        // ADDITIVE: the mask is the local amplitude; the cover gate is the
        // scene fade guard. pump_env is not used: no scene-global amplitude
        // may change a local light's strength or rhythm.
        float pump_local = pump_mask * sh_pump_cover_gate;
        #else
        // SUBTRACTIVE: the mask only SUPPRESSES the scalar pump_env; no global
        // event means no pump, whatever the local rise.
        float pump_local = sh_pump_env * pump_mask;
        #endif
        // Down-gated where growth mode already restores expansion, so pump +
        // growth + top lift do not stack a peak past the display's ceiling (the
        // pump is monotone, so gradation is never at risk). cf_strength rides
        // along (strength 0 = no-op); PUMP_GAIN_CEIL is a roof, not scaled.
        float pump_str = PUMP_STRENGTH * cf_pump * cf_strength
                       * (1.0 - PUMP_GROWTH_DAMP * sh_growth_mode);
        pump_gain = min(pump_local * pump_str * pump_w, PUMP_GAIN_CEIL);
        expansion *= 1.0 + pump_gain;
    }

    #if DEBUG_SHOW_PUMP
    {
        // Red = scalar pump (a reference only under ADDITIVE), green = applied
        // gain, blue = the cell mask used.
        return vec4(gamma709_to_pq2020(vec3(pump_env, pump_gain, pump_mask)), 1.0);
    }
    #endif

    // -------------------------------------------------------------------------
    // DEBUG: Warm-shift / pale-skin visualization
    // -------------------------------------------------------------------------
    // Self-contained, so production can defer Oklab past the early exit.
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
            // Mirrors the production 5-tap median, shown unconditionally.
            #define S2(x,y) { float t = min(x,y); y = max(x,y); x = t; }
            float m0 = color.a;
            float m1 = HOOKED_texOff(vec2( 2.0,  0.0)).a;
            float m2 = HOOKED_texOff(vec2(-2.0,  0.0)).a;
            float m3 = HOOKED_texOff(vec2( 0.0,  2.0)).a;
            float m4 = HOOKED_texOff(vec2( 0.0, -2.0)).a;
            S2(m0,m1); S2(m3,m4); S2(m0,m3); S2(m1,m4); S2(m1,m2); S2(m2,m3); S2(m1,m2);
            float Yd = eotf_gamma(m2 * 2.0);
            #undef S2
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
    // Warm shift and pale skin apply to expanded pixels only.
    if (expansion < 1.001) {
        return vec4(gamma709_to_pq2020(color.rgb), 1.0);
    }
    // The onset blend must see the expansion BEFORE the pale-skin lift (the
    // lifted value stepped a warm AA pixel +9 % for a 0.7 % source change).
    float expansion_pre = expansion;

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
    // Near-neutral pixels: Oklab reduces to rgb_linear * expansion.
    vec3 rgb_expanded;
    #if ENABLE_OKLAB_BYPASS
    if (sat_gamma < SAT_BYPASS_THRESH) {
        rgb_expanded = rgb_linear * expansion;
    } else
    #endif
    {
        vec3 oklab_orig = rgb_to_oklab(rgb_linear);
        float chroma_orig = sqrt(oklab_orig.y * oklab_orig.y + oklab_orig.z * oklab_orig.z);

        // Shared 1/chroma for the hue tests (WS_CHROMA_FLOOR checked below).
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
            // Reuses Y_decision_gamma (== color.a * 2.0 here).
            float Y_decision = eotf_gamma(Y_decision_gamma);
            // Brightness gate from a 5-tap cross MEDIAN of the decision luma:
            // PS_BRIGHT_FLOOR's steep smoothstep (0.50..0.65 linear, where lit
            // skin sits) turned surviving grain into PS_LIFT speckle on faces.
            // Only the gate input is filtered; applied values stay per pixel.
            // Needs the stabilizer (alpha holds the decision luma only then).
            // The median commutes with the monotone EOTF, so rank in gamma and
            // linearize once. 7-comparator network; m2 = median. One deviant or
            // edge tap cannot move it (a mean dragged the gate across line art).
            #define S2(x,y) { float t = min(x,y); y = max(x,y); x = t; }
            float m0 = Y_decision_gamma * 0.5;
            float m1 = HOOKED_texOff(vec2( 2.0,  0.0)).a;
            float m2 = HOOKED_texOff(vec2(-2.0,  0.0)).a;
            float m3 = HOOKED_texOff(vec2( 0.0,  2.0)).a;
            float m4 = HOOKED_texOff(vec2( 0.0, -2.0)).a;
            S2(m0,m1); S2(m3,m4); S2(m0,m3); S2(m1,m4); S2(m1,m2); S2(m2,m3); S2(m1,m2);
            float ps_bright_in = eotf_gamma(m2 * 2.0);
            #undef S2
            #else
            float Y_decision = get_luma(rgb_linear);
            float ps_bright_in = Y_decision;
            #endif
            #if HL_RAW_LUMA
            // The gate reads the smooth base instead (grain-free like the
            // median, without pass 1's pull toward a brighter neighbour).
            ps_bright_in = eotf_gamma(Yb_gamma);
            #endif
            #if DEC_CHROMA_PS
            #if PS_DEC_WIDE && DEC_OWN_LOOK
            vec2 ab_ps = (want_skin || sat_gamma >= SAT_BYPASS_THRESH) ? ab_dec_w : ab_dec;
            #else
            vec2 ab_ps = ab_dec;
            #endif
            float ps_c = length(ab_ps);
            float ps_cos_dh = (ps_c > 1e-6) ? (ab_ps.x * PS_HUE_COS + ab_ps.y * PS_HUE_SIN) / ps_c : 0.0;
            #else
            float ps_c = chroma_orig;
            float ps_cos_dh = (oklab_orig.y * PS_HUE_COS + oklab_orig.z * PS_HUE_SIN) * inv_chroma;
            #endif
            float ps_hue_w = pow(max(ps_cos_dh, 0.0), PS_HUE_POWER);
            float ps_gate = smoothstep(PS_BRIGHT_FRAC_LOW, PS_BRIGHT_FRAC_HIGH, sh_bright_frac);
            float ps_chroma_w = smoothstep(0.015, 0.06, ps_c)
                              * (1.0 - smoothstep(0.04, PS_CHROMA_CEIL + 0.04, ps_c));
            float ps_bright_w = smoothstep(PS_BRIGHT_FLOOR, PS_BRIGHT_FLOOR + 0.15, ps_bright_in);
            float ps_w = ps_hue_w * ps_chroma_w * ps_bright_w * ps_gate;
            float ps_sat = PS_SAT_BOOST * ps_w;
            // Skin lift (see PS_LIFT): multiplicative on expansion; its gates
            // vary in chroma or neighbourhood luma only, so the composite curve
            // stays monotone per pixel.
            float psl_chroma_w = smoothstep(0.015, 0.05, ps_c)
                               * (1.0 - smoothstep(PS_LIFT_CHROMA_HI, PS_LIFT_CHROMA_CEIL, ps_c));
            float psl_w = ps_hue_w * psl_chroma_w * ps_bright_w * ps_gate
                        * smoothstep(0.5, 1.0, apl_t);
            expansion *= 1.0 + PS_LIFT * psl_w;
        #endif

        // ---- APPLY WARM SHIFT ----
        // Small-angle rotation toward red (chroma grows ~theta^2/2, ~0.2 %).
        #if ENABLE_WARM_SHIFT
        if (ws_angle > 0.0) {
            float a_shifted = oklab_orig.y + oklab_orig.z * ws_angle;
            float b_shifted = oklab_orig.z - oklab_orig.y * ws_angle;
            oklab_orig.y = a_shifted;
            oklab_orig.z = b_shifted;
        }
        #endif

        // ---- APPLY EXPANSION (Oklab) ----
        // L and chroma both scale by cbrt(expansion): constant chromaticity.
        vec3 oklab_exp = oklab_orig;
        float cbrt_exp = fast_cbrt(expansion);
        oklab_exp.x *= cbrt_exp;
        oklab_exp.yz *= cbrt_exp;
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
        float onset_blend = smoothstep(1.001, 1.05, expansion_pre);
        if (onset_blend < 1.0) {
            vec3 rgb_pq_pass = gamma709_to_pq2020(rgb_gamma);
            rgb_pq = mix(rgb_pq_pass, rgb_pq, onset_blend);
        }
    }
    #endif

    // -------------------------------------------------------------------------
    // PQ-AWARE DITHER — break 8-bit source banding after expansion
    // -------------------------------------------------------------------------
    // Triangular dither (two PCG uniforms, frame-varying), one 10-bit PQ step:
    // 8-bit source steps become visible after 2-3x expansion.
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

    // Alpha out = 1.0 (color.a held pass 1's decision luma).
    return vec4(rgb_pq, 1.0);
}

void hook() {
    // Snapshot first, before any data-dependent return (see above).
    if (gl_LocalInvocationIndex == 0u) {
        sh_bright_frac   = smoothed_bright_frac;
        sh_growth_mode   = smoothed_growth_mode;
        // APL axis: the held scene key when cf_key_hold is on (see pass 8);
        // off selects the raw key itself, bit-identical to before the hold.
        sh_log_avg       = (cf_key_hold > 0.0) ? held_log_avg : smoothed_log_avg;
        sh_contrast      = smoothed_contrast;
        sh_spec_anchor   = spec_anchor_held;
        sh_pump_env        = pump_env;
        sh_pump_cover_gate = pump_cover_gate;
    }
    // One cell per lane: keep >= 144 lanes per group (16x16 measured faster
    // than 32x32 for this register-heavy TU).
    if (gl_LocalInvocationIndex < 144u)
        sh_pump_mask_cell[gl_LocalInvocationIndex] =
            pump_mask_cell[gl_LocalInvocationIndex];
    barrier();

    vec4 c = cf_shade();
    // Padding lanes past the frame edge still reach the barrier but never
    // store.
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (all(lessThan(gid, ivec2(HOOKED_size))))
        imageStore(out_image, gid, c);
}
