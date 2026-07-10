// Debandit v2.0 — Detect-and-Reconstruct Deband, plane-fit estimator
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// Second-generation Debandit. The detect-and-reconstruct philosophy, the
// two-regime apply (snap vs LF-only), the four soft gates, the chroma
// opponent path and the trust-mask machinery carry over from v1.11
// UNCHANGED — what changes is the RAMP ESTIMATOR and the contamination
// story around it:
//
// 1. WEIGHTED PLANE FIT replaces the premultiplied Gaussian chains. Per
//    1/4-res texel, the pre-quantization ramp is estimated as the weighted
//    least-squares plane over a box window (radius db_rw texels), weights =
//    trust mask x chroma fit-mask. A plane fit has no curvature bias to
//    first order, so the 2S - S2 deconvolution (and its four kernel-table
//    copies) retires; genuinely smooth shading is a structural no-op by
//    construction, and the fit tracks CURVED shading (faces, glows) that
//    the sigma-48 refit misread by ~1.7 codes — the v1.x luma-on-shaded-
//    surfaces class. One-sided windows at frame borders fit a plane
//    through in-frame texels only: the v1.x border DC tilt (~0.19 codes)
//    measures ~0.01 codes here.
// 2. FIT RESIDUAL as pointwise contamination signal. The fit hands us,
//    almost free (three quadratic moments), the RMS of M minus the fitted
//    plane over the window — per axis (luma / opponent). A banding
//    staircase leaves ~0.3-0.6 codes of sawtooth residual; a bulk-contrast
//    edge inside the window leaves tens. The apply gates each scale's
//    validity on its residual with NO dilation (the residual already
//    integrates exactly the window the fit used), which shrinks the
//    bulk-edge collar to roughly the window radius. The +-8-texel
//    contamination envelope stays as remote-bias insurance (seeds and
//    knees unchanged from v1.11).
// 3. VALIDITY-NORMALIZED SCALE BLEND. Wide (db_rw) and short (db_rs)
//    fits blend by relative validity: c_eff = (wL*cL + wS*cS)/(wL+wS),
//    wS = (1-vL)*vS*db_ms. Unlike the v1.10 linear mix, a fully-condemned
//    scale contributes NOTHING — a plane fit near a bulk edge can carry
//    |c| ~ 10 codes where the Gaussian bias was ~2, so any leftover
//    fraction of an invalid field is visible (measured: 1.1-code paint
//    inside colored squares with the linear mix, 0.06 with this blend).
//    The gate is the validity of what was actually blended:
//    w_env = (wL*vL + wS*vS)/(wL+wS).
// 4. FRAME NOISE FLOOR (SSBO) + ANISOTROPY DISCRIMINATOR. A COMPUTE
//    reduce estimates the frame's true dither/grain floor as the p10 of
//    raw block MADs (histogram walk, 2-3 frame EMA). It de-circularizes
//    the structure-tensor texture flag that v1.11 built and reverted:
//    significant ISOTROPIC gradient energy (2-D mottle, measured against
//    the FLOOR, not against local MAD which texture inflates) closes the
//    snap regime — the LF correction stays, grain-region semantics.
//    Banding contours are locally 1-D (coherence ~1) and never flag;
//    dither is isotropic but sits at ~0.3x floor, under the significance
//    knee. Measured on a 10-bit BD stress set: fine-texture
//    retention inside former snap zones +10 to +34 points, dithered-8-bit
//    suite bit-flat (the failure that killed the v1.11 experiment).
//
// Chain order: list BEFORE CelFlare in glsl-shaders (same-hook passes run
// in load order; CelFlare's expansion multiplies any step not flattened
// first). The previous generation is parked as Debanditv1.glsl for the
// ongoing field A/B — same db_* PARAM names on purpose so shampv
// profiles transfer; do not load both at once.
//
// Known limits, documented honestly:
// - Hair-class dense structure (strand clusters every ~15px) still
//   passes through: both windows are mostly masked there, conf fades the
//   fit out. Passthrough beats fabrication; unchanged from v1.x.
// - The collar around bulk-contrast edges shrinks to ~window radius
//   (short scale: ~15-20px) but does not vanish; a fit whose window
//   straddles a 50-code edge is unusable no matter the estimator. The
//   honest full fix remains similarity-aware weights (a v2.1 candidate:
//   tonal kernel, non-separable).
// - Mid-scale (8-40px) mottle BELOW the noise-significance knee is still
//   absorbed in snap zones; the floor turns this from "30-50% of texture
//   energy" into a bounded tail.
// - The frame floor is a per-frame scalar: content mixing a pristine
//   quiet region with a heavily dithered one reads a low floor and the
//   anisotropy flag gets conservative (more texture protected, slightly
//   less dither absorbed) — the safe direction.
// - EMA state: db2_floor lives in an SSBO and survives shader reload
//   (2-3 frame convergence makes this harmless).
// - mod-4 dimension tap-phase caveat unchanged from v1.x.

// ---- shampv shader API (plain comments to libplacebo) ----
//@shampv input any

// =============================================================================
//  USER TUNING
// =============================================================================
// Sliders are DYNAMIC: glsl-shader-opts changes apply on the next frame, no
// recompile. db_debug / db_rw / db_rs are DEFINEs and recompile on change.
// Defaults = the shipped tune. NOTE: no comment lines may sit between a PARAM
// block and the next directive — the parser folds them into the value and
// fails to load.

//!PARAM db_strength
//!DESC Deband strength — fraction of the reconstructed gradient applied. 1 = full reconstruction (the shipped tune), 0 = off.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_thr
//!DESC Banding step-size ceiling in 8-bit code values; drives the amplitude gate (full up to half this, zero at 1x), the snap reach (0.375x), the contamination-envelope tolerance and the fit-residual knees. 2 covers 1-2 code banding (typical web encodes); 4 = brutal sources.
//!TYPE DYNAMIC float
//!MINIMUM 0.5
//!MAXIMUM 4.0
2.0

//!PARAM db_flat
//!DESC Flatness ceiling in 8-bit code values — luma range over a +-32px window above this marks texture/detail and fades correction out (zero at 2x). Raise to reach banding on steeper gradients, lower to protect more detail.
//!TYPE DYNAMIC float
//!MINIMUM 2.0
//!MAXIMUM 24.0
6.0

//!PARAM db_chroma
//!DESC Chroma-path strength — fraction of the opponent-axis (chroma) correction applied. Chroma banding is gated by its OWN flatness/envelope/residual/noise fields, so it is fixed even where steep luma shading closes the luma gates. 1 = full (shipped tune), 0 = luma-only. The luma path is unaffected either way.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_flat_c
//!DESC Chroma flatness ceiling in RGB-projected code values — opponent-axis (B-Y / R-Y) range of the median field over a +-32px window above this marks colored structure and fades the chroma correction out (zero at 2x). Banded surfaces measure ~0.5-1, colored art/iris structure 8+.
//!TYPE DYNAMIC float
//!MINIMUM 1.0
//!MAXIMUM 24.0
4.0

//!PARAM db_ms
//!DESC Multi-scale rescue strength. Near bulk-contrast edges the wide fit is residual-condemned and banding would survive in a collar; this blends in the short fit where only IT is clean, shrinking the collar to ~15-20px. 1 = full (shipped tune), 0 = wide fit only.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_snr
//!DESC SNR-adaptive texture protection (v1.11 semantics, unchanged). At 1 the structure gate's knee scales with the measured local noise floor so faint coherent texture on quiet high-bit-depth sources is protected from snap absorption; dithered 8-bit content clamps to the static knee (behavior unchanged). 0 = static knee.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_aniso
//!DESC Isotropic-texture snap protection (new in v2). Scales the structure-tensor flag: significant 2-D texture energy above the measured FRAME noise floor closes the snap regime (the low-frequency correction stays), so mid-scale mottle — stone, cloud, DoF texture — stops being absorbed as if it were dither. Banding contours are 1-D and never flag; dither sits under the significance knee. 1 = full (shipped tune), 0 = off (v1.11-class snap behavior).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_dither
//!DESC Dither absorption / contour snap, scaling BOTH the luma and chroma snap regimes. 1 = full: cleanest possible gradients; lower to preserve sub-code faint texture at the cost of contour removal on clean flats; 0 = low-frequency correction only.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_rw
//!DESC Wide plane-fit window radius in 1/4-res texels (recompiles on change). 16 = +-64px full res: spans the plateaus of coarse banding. Raise for ultra-wide plateaus at the cost of a wider bulk-edge collar; lower to tighten the collar at the cost of coarse-banding reach.
//!TYPE DEFINE
//!MINIMUM 8
//!MAXIMUM 24
16

//!PARAM db_rs
//!DESC Short plane-fit window radius in 1/4-res texels (recompiles on change). 3 = +-12px full res: the bulk-edge rescue scale. Values above 5 exhaust the +-8-texel envelope margin that stands in for the short scale's residual gate, and a wider short window also widens its own rescue collar — 3-5 is the validated range.
//!TYPE DEFINE
//!MINIMUM 2
//!MAXIMUM 6
3

//!PARAM db_debug
//!DESC Debug views: 0 = off, 1 = bypass, 2 = applied correction x32 on mid-gray, 3 = luma gate map (R = flatness kill, G = applied weight, B = snap regime), 4 = chroma gate map, 5 = v2 fields (R = wide luma fit residual /4 codes, G = wide chroma residual /8, B = anisotropy flag).
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 5
0

//!BUFFER DB2_STATE
//!VAR float db2_floor
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!SAVE DB2_DS
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC Debandit: Downsample 1/4

// =============================================================================
// PASS 1: DOWNSAMPLE 1/4 — plateau field P (exact 4x4 box mean)
// =============================================================================
// Identical to v1.11. Each 1/4-res texel is the uniform mean of its 4x4
// full-res block via 4 bilinear taps at half-integer positions (exact for
// mod-4 dimensions). Alpha carries BT.709 luma of the mean.

vec4 hook() {
    vec2 pt = HOOKED_pt;
    vec3 rgb  = HOOKED_tex(HOOKED_pos + vec2(-pt.x, -pt.y)).rgb;
    rgb      += HOOKED_tex(HOOKED_pos + vec2( pt.x, -pt.y)).rgb;
    rgb      += HOOKED_tex(HOOKED_pos + vec2(-pt.x,  pt.y)).rgb;
    rgb      += HOOKED_tex(HOOKED_pos + vec2( pt.x,  pt.y)).rgb;
    rgb *= 0.25;
    float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    return vec4(rgb, luma);
}

//!HOOK MAIN
//!BIND DB2_DS
//!SAVE DB2_MED_H
//!WIDTH DB2_DS.w
//!HEIGHT DB2_DS.h
//!DESC Debandit: Median H

// =============================================================================
// PASS 2/3: SEPARABLE MED5 — line-reject field M (identical to v1.11)
// =============================================================================
// med5 identity (verified exhaustively): med5(a,b,c,d,e) =
// med3(c, max(min(a,b),min(d,e)), min(max(a,b),max(d,e))); componentwise,
// so all four channels ride in SIMD.

vec4 hook() {
    vec2 pt = DB2_DS_pt;
    vec2 pos = DB2_DS_pos;
    vec4 a = DB2_DS_tex(pos - vec2(2.0 * pt.x, 0.0));
    vec4 b = DB2_DS_tex(pos - vec2(pt.x, 0.0));
    vec4 c = DB2_DS_tex(pos);
    vec4 d = DB2_DS_tex(pos + vec2(pt.x, 0.0));
    vec4 e = DB2_DS_tex(pos + vec2(2.0 * pt.x, 0.0));
    vec4 f = max(min(a, b), min(d, e));
    vec4 g = min(max(a, b), max(d, e));
    return max(min(c, f), min(max(c, f), g));
}

//!HOOK MAIN
//!BIND DB2_MED_H
//!BIND DB2_DS
//!SAVE DB2_MED
//!WIDTH DB2_MED_H.w
//!HEIGHT DB2_MED_H.h
//!DESC Debandit: Median V

// Alpha of the output is the LUMA TRUST MASK (identical to v1.11): 1 where
// this texel's box mean agrees with the median field, fading to 0 where
// they disagree (structure: line art, text strokes, dense texture). The
// RANGE passes consume it as-is — luma-only by design (v1.9 audit F2: an
// equal-luma pure-chroma structure closes the chroma range gate around
// itself, which is the correct conservative behavior). The PLANE FIT uses
// the tighter FITMASK from the next pass instead.
#define DB2_CODE_M    (1.0 / 255.0)
#define DB2_MASK_LO   0.9
#define DB2_MASK_HI   1.8

vec4 hook() {
    vec2 pt = DB2_MED_H_pt;
    vec2 pos = DB2_MED_H_pos;
    vec4 a = DB2_MED_H_tex(pos - vec2(0.0, 2.0 * pt.y));
    vec4 b = DB2_MED_H_tex(pos - vec2(0.0, pt.y));
    vec4 c = DB2_MED_H_tex(pos);
    vec4 d = DB2_MED_H_tex(pos + vec2(0.0, pt.y));
    vec4 e = DB2_MED_H_tex(pos + vec2(0.0, 2.0 * pt.y));
    vec4 f = max(min(a, b), min(d, e));
    vec4 g = min(max(a, b), max(d, e));
    vec4 med = max(min(c, f), min(max(c, f), g));

    float lum_med = dot(med.rgb, vec3(0.2126, 0.7152, 0.0722));
    float dev = abs(DB2_DS_tex(DB2_DS_pos).a - lum_med) / DB2_CODE_M;
    float mask = 1.0 - smoothstep(DB2_MASK_LO, DB2_MASK_HI, dev);
    return vec4(med.rgb, mask);
}

//!HOOK MAIN
//!BIND DB2_DS
//!BIND DB2_MED
//!SAVE DB2_FITMASK
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!COMPONENTS 1
//!DESC Debandit: Fit Mask

// =============================================================================
// PASS 4: FIT MASK — plane-fit weights = luma trust x chroma agreement
// =============================================================================
// The fit weight additionally rejects texels whose box-mean OPPONENT
// components disagree with the median field's — equal-luma chroma
// structure (colored line art, saturated edges' contour texels) that the
// luma-only trust mask cannot see. Same knees as the luma mask, in
// RGB-projected codes: chroma banding contours read ~0.5-1 here (partial
// rejection of the contour texel itself is harmless — the fit needs the
// plateaus), chroma grain sigma-2 ~1, saturated structure 2+. The RANGE
// gates deliberately do NOT use this mask (see the Median V note).
// CONSTANT SYNC: knees duplicated from the Median V pass — keep in sync.
#define DB2_CODE_M    (1.0 / 255.0)
#define DB2_MASK_LO   0.9
#define DB2_MASK_HI   1.8

vec4 hook() {
    vec4 ds = DB2_DS_tex(DB2_DS_pos);
    vec4 m  = DB2_MED_tex(DB2_MED_pos);
    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    float y_d = dot(ds.rgb, W709);
    float y_m = dot(m.rgb, W709);
    vec2 o_d = vec2(ds.b - y_d, ds.r - y_d);
    vec2 o_m = vec2(m.b - y_m, m.r - y_m);
    vec2 dv = abs(o_d - o_m) / DB2_CODE_M;
    float mask_c = 1.0 - smoothstep(DB2_MASK_LO, DB2_MASK_HI, max(dv.x, dv.y));
    return vec4(m.a * mask_c, 0.0, 0.0, 1.0);
}

// =============================================================================
// PASS 5/6/7: WIDE MOMENT ROWS — separable box sums for the plane fit
// =============================================================================
// Per output texel, sums over the row window x' in [-db_rw, +db_rw] with
// off-frame taps at ZERO WEIGHT (v1.7 border rule). Coordinates are
// window-relative and normalized (u = x'/db_rw, |u| <= 1) and M enters
// mean-subtracted against the OUTPUT texel's own median value — both are
// fp16-ulp insurance: every stored quantity stays near zero on smooth
// content (validated in the numpy prototype at <= 0.01 code fit error;
// the raw-coordinate formulation loses ~0.25 code to ulp). The fuse pass
// re-references rows to its own center in fp32 registers.
// Channel layout across the three textures (12 scalars):
//   A: (sum w, sum w*u, sum w*u^2, sum w*dY^2)
//   B: (sum w*dM.rgb, sum w*do1^2)
//   C: (sum w*u*dM.rgb, sum w*do2^2)
// dY/do1/do2 are the luma/opponent projections of dM = M(tap) - M(center):
// the quadratic moments feed the FIT RESIDUAL per gate axis.

//!HOOK MAIN
//!BIND DB2_MED
//!BIND DB2_FITMASK
//!SAVE DB2_WMOM_A
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!DESC Debandit: Wide Moments A

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;
    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    vec3 m0 = DB2_MED_tex(pos).rgb;
    float h0 = 0.0, h1 = 0.0, h2 = 0.0, hqy = 0.0;
    for (int i = -db_rw; i <= db_rw; i++) {
        float xp = pos.x + float(i) * pt.x;
        float w = DB2_FITMASK_tex(vec2(xp, pos.y)).x
                * step(0.0, xp) * step(xp, 1.0);
        vec3 dM = DB2_MED_tex(vec2(xp, pos.y)).rgb - m0;
        float u = float(i) / float(db_rw);
        float dy = dot(dM, W709);
        h0 += w; h1 += w * u; h2 += w * u * u;
        hqy += w * dy * dy;
    }
    return vec4(h0, h1, h2, hqy);
}

//!HOOK MAIN
//!BIND DB2_MED
//!BIND DB2_FITMASK
//!SAVE DB2_WMOM_B
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!DESC Debandit: Wide Moments B

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;
    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    vec3 m0 = DB2_MED_tex(pos).rgb;
    vec3 hM = vec3(0.0);
    float hq1 = 0.0;
    for (int i = -db_rw; i <= db_rw; i++) {
        float xp = pos.x + float(i) * pt.x;
        float w = DB2_FITMASK_tex(vec2(xp, pos.y)).x
                * step(0.0, xp) * step(xp, 1.0);
        vec3 dM = DB2_MED_tex(vec2(xp, pos.y)).rgb - m0;
        float do1 = dM.b - dot(dM, W709);
        hM += w * dM;
        hq1 += w * do1 * do1;
    }
    return vec4(hM, hq1);
}

//!HOOK MAIN
//!BIND DB2_MED
//!BIND DB2_FITMASK
//!SAVE DB2_WMOM_C
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!DESC Debandit: Wide Moments C

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;
    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    vec3 m0 = DB2_MED_tex(pos).rgb;
    vec3 hxM = vec3(0.0);
    float hq2 = 0.0;
    for (int i = -db_rw; i <= db_rw; i++) {
        float xp = pos.x + float(i) * pt.x;
        float w = DB2_FITMASK_tex(vec2(xp, pos.y)).x
                * step(0.0, xp) * step(xp, 1.0);
        vec3 dM = DB2_MED_tex(vec2(xp, pos.y)).rgb - m0;
        float u = float(i) / float(db_rw);
        float do2 = dM.r - dot(dM, W709);
        hxM += w * u * dM;
        hq2 += w * do2 * do2;
    }
    return vec4(hxM, hq2);
}

//!HOOK MAIN
//!BIND DB2_MED
//!BIND DB2_FITMASK
//!BIND DB2_WMOM_A
//!BIND DB2_WMOM_B
//!BIND DB2_WMOM_C
//!SAVE DB2_CORR
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!DESC Debandit: Wide Fuse

// =============================================================================
// PASS 8: WIDE FUSE — V accumulation, 2x2 weighted-LS solve, c = fit - M
// =============================================================================
// Combines the row moments over y' in [-db_rw, db_rw] (v = y'/db_rw),
// re-referencing each row's mean-subtracted moments from the row center's
// median value to this texel's in fp32 (linear moments shift linearly;
// quadratics via (X-b)^2 = (X-a)^2 + 2(a-b)(X-a) + (a-b)^2). Solves the
// centered normal equations per RGB channel (shared 2x2 geometry matrix),
// evaluates the plane AT THIS TEXEL, and stores c = (fit - M) * conf
// small-valued as 0.5 + c (both signs survive a unorm FBO fallback).
// Alpha = max-channel |c| x fitmask^2, computed HERE at fp32 — the
// envelope seed must never be re-derived from the 0.5 + c encode (v1.9
// lesson: fp16 ulp at 0.5 is ~0.06 code, measured 0.3-code apply wobble).
//
// conf fades the fit out on low trusted mass (nothing to learn a ramp
// from); g_det on an ill-conditioned normal matrix (trusted texels nearly
// collinear — frame corners, heavy masking). CONSTANT SYNC: the knee
// values and the solve here must match the Wide Residual pass EXACTLY
// (no compile-time guard across translation units).
#define DB2_CONF_LO   0.06
#define DB2_CONF_HI   0.20
#define DB2_DET_LO    0.02
#define DB2_DET_HI    0.08

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;
    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    vec3 m0 = DB2_MED_tex(pos).rgb;
    float fm0 = DB2_FITMASK_tex(pos).x;

    float W = 0.0, Wu = 0.0, Wv = 0.0, Wuu = 0.0, Wuv = 0.0, Wvv = 0.0;
    vec3 WM = vec3(0.0), WuM = vec3(0.0), WvM = vec3(0.0);
    for (int j = -db_rw; j <= db_rw; j++) {
        float yp = pos.y + float(j) * pt.y;
        float inb = step(0.0, yp) * step(yp, 1.0);
        float v = float(j) / float(db_rw);
        vec4 A = DB2_WMOM_A_tex(vec2(pos.x, yp)) * inb;
        vec3 hM = DB2_WMOM_B_tex(vec2(pos.x, yp)).rgb * inb;
        vec3 hxM = DB2_WMOM_C_tex(vec2(pos.x, yp)).rgb * inb;
        vec3 d = (DB2_MED_tex(vec2(pos.x, yp)).rgb - m0) * inb;
        W += A.x; Wu += A.y; Wv += v * A.x;
        Wuu += A.z; Wuv += v * A.y; Wvv += v * v * A.x;
        vec3 rowM = hM + A.x * d;
        WM += rowM;
        WuM += hxM + A.y * d;
        WvM += v * rowM;
    }

    float Ws = max(W, 1e-6);
    float mu = Wu / Ws, mv = Wv / Ws;
    vec3 Mb = WM / Ws;
    float Suu = Wuu - mu * Wu;
    float Suv = Wuv - mu * Wv;
    float Svv = Wvv - mv * Wv;
    vec3 SuM = WuM - mu * WM;
    vec3 SvM = WvM - mv * WM;

    float det = Suu * Svv - Suv * Suv;
    float tr = Suu + Svv;
    float cond = det / max(tr * tr, 1e-9);
    float g_det = smoothstep(DB2_DET_LO, DB2_DET_HI, cond);
    float inv_det = 1.0 / max(det, 1e-9);
    vec3 b  = (Svv * SuM - Suv * SvM) * inv_det * g_det;
    vec3 cf = (Suu * SvM - Suv * SuM) * inv_det * g_det;

    float area = float((2 * db_rw + 1) * (2 * db_rw + 1));
    float conf = smoothstep(DB2_CONF_LO, DB2_CONF_HI, W / area);
    // Centroid-offset fade (audit F2): a plane EXTRAPOLATES — when the
    // trusted mass is lopsided (chroma/trust holes), the fit at (0,0) is
    // an extrapolation the in-window residual cannot see. Full one-sided
    // frame-border windows measure |mu| ~ 0.5 and stay unfaded; only
    // genuinely hole-punched support (|centroid| > 0.6 window radii)
    // fades out.
    conf *= 1.0 - smoothstep(0.6, 0.95, max(abs(mu), abs(mv)));
    vec3 c = (Mb + b * (0.0 - mu) + cf * (0.0 - mv)) * (conf * g_det);
    // A condemned fit near a bulk edge can carry |c| past the 0.5 encode
    // headroom (measured 0.61 units on a real scene). Clamp far above any
    // gate-relevant magnitude (0.2 units = 51 codes vs knees <= ~8): the
    // encode invariant holds by construction and every consumer already
    // saturates long before the clamp engages.
    c = clamp(c, vec3(-0.2), vec3(0.2));
    float seed = max(max(abs(c.r), abs(c.g)), abs(c.b)) * fm0 * fm0;
    return vec4(vec3(0.5) + c, seed);
}

//!HOOK MAIN
//!BIND DB2_MED
//!BIND DB2_WMOM_A
//!BIND DB2_WMOM_B
//!BIND DB2_WMOM_C
//!SAVE DB2_RES
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!COMPONENTS 2
//!DESC Debandit: Wide Residual

// =============================================================================
// PASS 9: WIDE FIT RESIDUAL — pointwise contamination, per gate axis
// =============================================================================
// Re-runs the wide V-accumulation and solve (constants and math must stay
// IN SYNC with Wide Fuse — see its header) and emits what the fuse pass
// has no channels left for: the fit's RMS residual on the luma axis (x)
// and the worse of the two opponent axes (y), in linear [0,1] units.
// resvar_X = S_XX - b_X*S_uX - c_X*S_vX per axis; axis moments are linear
// combinations of the RGB ones. The apply gates each PATH on its own
// axis (steep luma structure cannot veto a clean chroma fit — the v1.9
// face lesson carries into v2), pointwise, with no dilation.
// The SHORT scale carries no residual on purpose: its +-8-texel envelope
// dilation already covers the whole +-(db_rs+2) contamination reach
// (verified inert in the prototype).
#define DB2_CONF_LO   0.06
#define DB2_CONF_HI   0.20
#define DB2_DET_LO    0.02
#define DB2_DET_HI    0.08

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;
    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    vec3 m0 = DB2_MED_tex(pos).rgb;

    float W = 0.0, Wu = 0.0, Wv = 0.0, Wuu = 0.0, Wuv = 0.0, Wvv = 0.0;
    vec3 WM = vec3(0.0), WuM = vec3(0.0), WvM = vec3(0.0);
    vec3 WQ = vec3(0.0);   // quadratics on (Y, o1, o2)
    for (int j = -db_rw; j <= db_rw; j++) {
        float yp = pos.y + float(j) * pt.y;
        float inb = step(0.0, yp) * step(yp, 1.0);
        float v = float(j) / float(db_rw);
        vec4 A = DB2_WMOM_A_tex(vec2(pos.x, yp)) * inb;
        vec4 B = DB2_WMOM_B_tex(vec2(pos.x, yp)) * inb;
        vec4 C = DB2_WMOM_C_tex(vec2(pos.x, yp)) * inb;
        vec3 d = (DB2_MED_tex(vec2(pos.x, yp)).rgb - m0) * inb;
        W += A.x; Wu += A.y; Wv += v * A.x;
        Wuu += A.z; Wuv += v * A.y; Wvv += v * v * A.x;
        vec3 rowM = B.rgb + A.x * d;
        WM += rowM;
        WuM += C.rgb + A.y * d;
        WvM += v * rowM;
        // quadratic shift: per-axis d projections against per-axis row
        // linear moments (B.rgb are RGB moments; project to axes)
        float dy  = dot(d, W709);
        float do1 = d.b - dy;
        float do2 = d.r - dy;
        float hy  = dot(B.rgb, W709);
        float h1_ = B.rgb.b - hy;
        float h2_ = B.rgb.r - hy;
        WQ.x += A.w + 2.0 * dy  * hy  + dy  * dy  * A.x;
        WQ.y += B.a + 2.0 * do1 * h1_ + do1 * do1 * A.x;
        WQ.z += C.a + 2.0 * do2 * h2_ + do2 * do2 * A.x;
    }

    float Ws = max(W, 1e-6);
    float mu = Wu / Ws, mv = Wv / Ws;
    float Suu = Wuu - mu * Wu;
    float Suv = Wuv - mu * Wv;
    float Svv = Wvv - mv * Wv;
    vec3 SuM = WuM - mu * WM;
    vec3 SvM = WvM - mv * WM;

    float det = Suu * Svv - Suv * Suv;
    float cond = det / max((Suu + Svv) * (Suu + Svv), 1e-9);
    float g_det = smoothstep(DB2_DET_LO, DB2_DET_HI, cond);
    float inv_det = 1.0 / max(det, 1e-9);
    vec3 b  = (Svv * SuM - Suv * SvM) * inv_det * g_det;
    vec3 cf = (Suu * SvM - Suv * SuM) * inv_det * g_det;

    // Per-axis residuals, statically unrolled (no dynamic vector indexing
    // — the one construct SPIRV-Cross->HLSL has historically fumbled).
    // Axis vectors: Y, o1 = B - Y, o2 = R - Y.
    #define DB2_RES_AXIS(axis, q) \
        sqrt(max((q) - dot(WM, axis) * dot(WM, axis) / Ws \
                 - dot(b, axis) * dot(SuM, axis) \
                 - dot(cf, axis) * dot(SvM, axis), 0.0) / Ws)
    float res_y  = DB2_RES_AXIS(W709, WQ.x);
    float res_o1 = DB2_RES_AXIS(vec3(0.0, 0.0, 1.0) - W709, WQ.y);
    float res_o2 = DB2_RES_AXIS(vec3(1.0, 0.0, 0.0) - W709, WQ.z);
    return vec4(res_y, max(res_o1, res_o2), 0.0, 1.0);
}

//!HOOK MAIN
//!BIND DB2_MED
//!BIND DB2_FITMASK
//!SAVE DB2_SCORR
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!DESC Debandit: Short Fuse

// =============================================================================
// PASS 10: SHORT FUSE — direct 2D plane fit, radius db_rs
// =============================================================================
// The bulk-edge rescue scale. At (2*db_rs+1)^2 = ~49 taps a direct 2D
// accumulation is cheaper than a separable pair and keeps every moment in
// fp32 registers end to end (no storage boundary at all). Same solve,
// same knees as the wide fuse; c_s stored as 0.5 + c, alpha = fp32 seed.
#define DB2_CONF_LO   0.06
#define DB2_CONF_HI   0.20
#define DB2_DET_LO    0.02
#define DB2_DET_HI    0.08

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;
    vec3 m0 = DB2_MED_tex(pos).rgb;
    float fm0 = DB2_FITMASK_tex(pos).x;

    float W = 0.0, Wu = 0.0, Wv = 0.0, Wuu = 0.0, Wuv = 0.0, Wvv = 0.0;
    vec3 WM = vec3(0.0), WuM = vec3(0.0), WvM = vec3(0.0);
    for (int j = -db_rs; j <= db_rs; j++) {
        float yp = pos.y + float(j) * pt.y;
        float vin = step(0.0, yp) * step(yp, 1.0);
        float v = float(j) / float(db_rs);
        for (int i = -db_rs; i <= db_rs; i++) {
            float xp = pos.x + float(i) * pt.x;
            float w = DB2_FITMASK_tex(vec2(xp, yp)).x * vin
                    * step(0.0, xp) * step(xp, 1.0);
            vec3 dM = DB2_MED_tex(vec2(xp, yp)).rgb - m0;
            float u = float(i) / float(db_rs);
            W += w; Wu += w * u; Wv += w * v;
            Wuu += w * u * u; Wuv += w * u * v; Wvv += w * v * v;
            WM += w * dM; WuM += w * u * dM; WvM += w * v * dM;
        }
    }

    float Ws = max(W, 1e-6);
    float mu = Wu / Ws, mv = Wv / Ws;
    vec3 Mb = WM / Ws;
    float Suu = Wuu - mu * Wu;
    float Suv = Wuv - mu * Wv;
    float Svv = Wvv - mv * Wv;
    vec3 SuM = WuM - mu * WM;
    vec3 SvM = WvM - mv * WM;

    float det = Suu * Svv - Suv * Suv;
    float cond = det / max((Suu + Svv) * (Suu + Svv), 1e-9);
    float g_det = smoothstep(DB2_DET_LO, DB2_DET_HI, cond);
    float inv_det = 1.0 / max(det, 1e-9);
    vec3 b  = (Svv * SuM - Suv * SvM) * inv_det * g_det;
    vec3 cf = (Suu * SvM - Suv * SuM) * inv_det * g_det;

    float area = float((2 * db_rs + 1) * (2 * db_rs + 1));
    float conf = smoothstep(DB2_CONF_LO, DB2_CONF_HI, W / area);
    // Same centroid-offset fade as the wide fuse (audit F2).
    conf *= 1.0 - smoothstep(0.6, 0.95, max(abs(mu), abs(mv)));
    vec3 c = (Mb + b * (0.0 - mu) + cf * (0.0 - mv)) * (conf * g_det);
    // Same encode-headroom clamp as the wide fuse (see its note).
    c = clamp(c, vec3(-0.2), vec3(0.2));
    float seed = max(max(abs(c.r), abs(c.g)), abs(c.b)) * fm0 * fm0;
    return vec4(vec3(0.5) + c, seed);
}

//!HOOK MAIN
//!BIND DB2_CORR
//!BIND DB2_SCORR
//!SAVE DB2_ENV_H
//!WIDTH DB2_CORR.w
//!HEIGHT DB2_CORR.h
//!DESC Debandit: Contamination Env H

// =============================================================================
// PASS 11/12: CONTAMINATION ENVELOPE — separable max of seeds over +-8
// =============================================================================
// Identical machinery to v1.11 (remote-bias insurance): x/y = wide
// luma/chroma, z/w = short luma/chroma. Each luma seed is the CORR alpha
// verbatim (fp32-exact); each chroma seed is that seed times the bounded
// opponent ratio of the decoded c. With the pointwise fit residual doing
// the primary contamination work, this envelope mostly matters for the
// SHORT scale (which has no residual) and as a second opinion on |c|.

const vec3 DB2_ENVH_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB2_CORR_pt;
    vec2 pos = DB2_CORR_pos;
    vec4 m = vec4(0.0);
    for (int i = -8; i <= 8; i++) {
        vec2 tp = pos + vec2(float(i) * pt.x, 0.0);
        vec4 tL = DB2_CORR_tex(tp);
        vec4 tS = DB2_SCORR_tex(tp);
        vec3 cL = tL.rgb - vec3(0.5);
        vec3 cS = tS.rgb - vec3(0.5);
        vec3 oL = cL - vec3(dot(cL, DB2_ENVH_W709));
        vec3 oS = cS - vec3(dot(cS, DB2_ENVH_W709));
        float mcL = max(max(abs(cL.r), abs(cL.g)), abs(cL.b));
        float mcS = max(max(abs(cS.r), abs(cS.g)), abs(cS.b));
        float moL = max(max(abs(oL.r), abs(oL.g)), abs(oL.b));
        float moS = max(max(abs(oS.r), abs(oS.g)), abs(oS.b));
        m.x = max(m.x, tL.a);
        m.y = max(m.y, tL.a * moL / max(mcL, 1e-6));
        m.z = max(m.z, tS.a);
        m.w = max(m.w, tS.a * moS / max(mcS, 1e-6));
    }
    return m;
}

//!HOOK MAIN
//!BIND DB2_ENV_H
//!SAVE DB2_ENV
//!WIDTH DB2_ENV_H.w
//!HEIGHT DB2_ENV_H.h
//!DESC Debandit: Contamination Env V

vec4 hook() {
    vec2 pt = DB2_ENV_H_pt;
    vec2 pos = DB2_ENV_H_pos;
    vec4 m = DB2_ENV_H_tex(pos);
    for (int i = 1; i <= 8; i++) {
        m = max(m, DB2_ENV_H_tex(pos + vec2(0.0, float(i) * pt.y)));
        m = max(m, DB2_ENV_H_tex(pos - vec2(0.0, float(i) * pt.y)));
    }
    return m;
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND DB2_DS
//!BIND DB2_MED
//!SAVE DB2_RANGE_H
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!DESC Debandit: Flatness Range H

// =============================================================================
// PASS 13/14: FLATNESS RANGE + NOISE FIELD (identical to v1.11)
// =============================================================================
// x/y: trust-masked separable min/max of median-field luma over +-8
// texels (center always seeds). z: block MAD of the full-res 4x4 block
// vs the median field, max across channels (RAW here — the FLOOR pass
// reads this pre-dilation; the V pass dilates +-2). w: structure field
// (med5 mean-bias + signed-median coherence x2.5).

const vec3 DB2_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;

    vec4 center = DB2_MED_tex(pos);
    float lum_c = dot(center.rgb, DB2_W709);
    float lo = lum_c;
    float hi = lum_c;

    for (int i = 1; i <= 8; i++) {
        vec4 tp = DB2_MED_tex(pos + vec2(float(i) * pt.x, 0.0));
        vec4 tn = DB2_MED_tex(pos - vec2(float(i) * pt.x, 0.0));
        float op = step(0.5, tp.a);
        float on = step(0.5, tn.a);
        lo = min(lo, min(mix(1e3, dot(tp.rgb, DB2_W709), op),
                         mix(1e3, dot(tn.rgb, DB2_W709), on)));
        hi = max(hi, max(mix(-1e3, dot(tp.rgb, DB2_W709), op),
                         mix(-1e3, dot(tn.rgb, DB2_W709), on)));
    }

    vec2 fpt = HOOKED_pt;
    float mad = 0.0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 off = vec2(float(x) - 1.5, float(y) - 1.5) * fpt;
            vec3 d = abs(HOOKED_tex(HOOKED_pos + off).rgb - center.rgb);
            mad += max(d.r, max(d.g, d.b));
        }
    }
    mad *= (1.0 / 16.0);

    float s0 = DB2_DS_tex(pos - vec2(2.0 * pt.x, 0.0)).a - lum_c;
    float s1 = DB2_DS_tex(pos - vec2(pt.x, 0.0)).a - lum_c;
    float s2 = DB2_DS_tex(pos).a - lum_c;
    float s3 = DB2_DS_tex(pos + vec2(pt.x, 0.0)).a - lum_c;
    float s4 = DB2_DS_tex(pos + vec2(2.0 * pt.x, 0.0)).a - lum_c;

    float d0 = abs(s0); float d1 = abs(s1); float d2 = abs(s2);
    float d3 = abs(s3); float d4 = abs(s4);
    float f = max(min(d0, d1), min(d3, d4));
    float g = min(max(d0, d1), max(d3, d4));
    float tex = max(min(d2, f), min(max(d2, f), g));

    float fs = max(min(s0, s1), min(s3, s4));
    float gs = min(max(s0, s1), max(s3, s4));
    float coh = abs(max(min(s2, fs), min(max(s2, fs), gs)));

    return vec4(lo, hi, mad, max(tex, 2.5 * coh));
}

//!HOOK MAIN
//!BIND DB2_RANGE_H
//!BIND DB2_STATE
//!SAVE DB2_FLOORTEX
//!WIDTH 1
//!HEIGHT 1
//!COMPUTE 16 9
//!DESC Debandit: Noise Floor

// =============================================================================
// PASS 15: FRAME NOISE FLOOR — COMPUTE reduce (CelFlare Frame Stats pattern)
// =============================================================================
// One 16x9 workgroup, 144 lanes. Each lane strides the 1/4-res grid
// accumulating a shared 64-bin log histogram of the RAW block MADs
// (DB2_RANGE_H.z, pre-dilation); thread 0 walks the CDF to the 10th
// percentile — the frame's true dither/grain floor: quiet-but-noisy flat
// blocks land there, textured blocks sit far above, pristine digital
// flats below (the knee floors catch that case). A 2-3 frame EMA smooths
// knee flicker on cuts; first frame (or SSBO fresh from init) locks on
// instantly, so headless single-frame runs are deterministic. The SSBO
// survives shader reload (known caveat) — convergence in ~3 frames makes
// a stale carry harmless.
#define DB2_FLOOR_BINS   64
#define DB2_FLOOR_LO     (0.02 / 255.0)
#define DB2_FLOOR_HI     (24.0 / 255.0)
#define DB2_FLOOR_PCT    0.10
#define DB2_FLOOR_ALPHA  0.4

shared uint db2_hist[DB2_FLOOR_BINS];
shared uint db2_count;

void hook() {
    uint lid = gl_LocalInvocationIndex;   // 0..143
    if (lid < uint(DB2_FLOOR_BINS)) db2_hist[lid] = 0u;
    if (lid == 0u) db2_count = 0u;
    barrier();

    // 2x2-strided subsample: a global percentile is statistically
    // indifferent to it and it quarters the reduce cost (measured 0.28 ms
    // full-grid on M2-class hardware — the one pass that overshot the
    // design budget).
    ivec2 sz = ivec2(DB2_RANGE_H_size) / 2;
    int total = sz.x * sz.y;
    float log_lo = log2(DB2_FLOOR_LO);
    float log_hi = log2(DB2_FLOOR_HI);
    uint mine = 0u;
    for (int t = int(lid); t < total; t += 144) {
        ivec2 p = ivec2((t % sz.x) * 2, (t / sz.x) * 2);
        float mad = texelFetch(DB2_RANGE_H_raw, p, 0).z;
        float lg = log2(max(mad, DB2_FLOOR_LO));
        int bin = clamp(int((lg - log_lo) / (log_hi - log_lo)
                            * float(DB2_FLOOR_BINS)), 0, DB2_FLOOR_BINS - 1);
        atomicAdd(db2_hist[bin], 1u);
        mine++;
    }
    atomicAdd(db2_count, mine);
    barrier();

    if (lid == 0u) {
        uint target = uint(float(db2_count) * DB2_FLOOR_PCT);
        uint acc = 0u;
        int hit = DB2_FLOOR_BINS - 1;
        for (int i = 0; i < DB2_FLOOR_BINS; i++) {
            acc += db2_hist[i];
            if (acc >= target) { hit = i; break; }
        }
        // geometric bin center
        float frac = (float(hit) + 0.5) / float(DB2_FLOOR_BINS);
        float new_floor = exp2(log_lo + frac * (log_hi - log_lo));
        // Instant re-lock on first frame AND on large jumps (audit F1):
        // the floor gates the SNAP REGIME via the aniso knees, so a 2-3
        // frame EMA settle after a hard cut (grainy -> quiet or back)
        // would flicker texture protection exactly when the eye resets.
        // A >50% relative move is a content change, not grain wobble —
        // lock on immediately; the EMA only smooths within-scene motion.
        bool relock = db2_floor <= 0.0
                   || abs(new_floor - db2_floor) > 0.5 * db2_floor;
        db2_floor = relock ? new_floor
                           : mix(db2_floor, new_floor, DB2_FLOOR_ALPHA);
        imageStore(out_image, ivec2(0), vec4(db2_floor, 0.0, 0.0, 1.0));
    }
}

//!HOOK MAIN
//!BIND DB2_RANGE_H
//!SAVE DB2_RANGE
//!WIDTH DB2_RANGE_H.w
//!HEIGHT DB2_RANGE_H.h
//!DESC Debandit: Flatness Range V

// V half of the range/noise reduction (identical to v1.11): min/max over
// +-8; MAD max-dilates over +-2; structure field means over +-2.

vec4 hook() {
    vec2 pt = DB2_RANGE_H_pt;
    vec2 pos = DB2_RANGE_H_pos;

    vec4 mm = DB2_RANGE_H_tex(pos);

    for (int i = 1; i <= 8; i++) {
        vec4 sp = DB2_RANGE_H_tex(pos + vec2(0.0, float(i) * pt.y));
        vec4 sn = DB2_RANGE_H_tex(pos - vec2(0.0, float(i) * pt.y));
        mm.x = min(mm.x, min(sp.x, sn.x));
        mm.y = max(mm.y, max(sp.y, sn.y));
        if (i <= 2) {
            mm.z = max(mm.z, max(sp.z, sn.z));
            mm.w += sp.w + sn.w;
        }
    }
    mm.w *= (1.0 / 5.0);

    return mm;
}

//!HOOK MAIN
//!BIND DB2_MED
//!SAVE DB2_RANGEC_H
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!DESC Debandit: Chroma Range H

// =============================================================================
// PASS 16/17: CHROMA FLATNESS RANGE + CHROMA NOISE (identical to v1.11)
// =============================================================================
// Trust-masked separable min/max of the median field's opponent axes,
// stored o * 0.5 + 0.5 (unorm-fallback-safe); the V pass decodes to
// ranges and adds the opponent block MAD (Y-plane dither cancels exactly
// in the projection) + the chroma structure/coherence field.

const vec3 DB2_RC_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;

    vec4 center = DB2_MED_tex(pos);
    float y_c = dot(center.rgb, DB2_RC_W709);
    vec2 o_c = vec2(center.b - y_c, center.r - y_c);
    vec2 lo = o_c;
    vec2 hi = o_c;

    for (int i = 1; i <= 8; i++) {
        vec4 tp = DB2_MED_tex(pos + vec2(float(i) * pt.x, 0.0));
        vec4 tn = DB2_MED_tex(pos - vec2(float(i) * pt.x, 0.0));
        float yp = dot(tp.rgb, DB2_RC_W709);
        float yn = dot(tn.rgb, DB2_RC_W709);
        vec2 op = vec2(tp.b - yp, tp.r - yp);
        vec2 on = vec2(tn.b - yn, tn.r - yn);
        float wp = step(0.5, tp.a);
        float wn = step(0.5, tn.a);
        lo = min(lo, min(mix(vec2( 1e3), op, wp), mix(vec2( 1e3), on, wn)));
        hi = max(hi, max(mix(vec2(-1e3), op, wp), mix(vec2(-1e3), on, wn)));
    }

    return vec4(lo * 0.5 + 0.5, hi * 0.5 + 0.5).xzyw;
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND DB2_DS
//!BIND DB2_MED
//!BIND DB2_RANGEC_H
//!SAVE DB2_RANGEC
//!WIDTH DB2_RANGEC_H.w
//!HEIGHT DB2_RANGEC_H.h
//!DESC Debandit: Chroma Range V

const vec3 DB2_RCV_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB2_RANGEC_H_pt;
    vec2 pos = DB2_RANGEC_H_pos;

    vec4 mm = DB2_RANGEC_H_tex(pos);
    vec2 lo = mm.xz;
    vec2 hi = mm.yw;

    for (int i = 1; i <= 8; i++) {
        vec4 sp = DB2_RANGEC_H_tex(pos + vec2(0.0, float(i) * pt.y));
        vec4 sn = DB2_RANGEC_H_tex(pos - vec2(0.0, float(i) * pt.y));
        lo = min(lo, min(sp.xz, sn.xz));
        hi = max(hi, max(sp.yw, sn.yw));
    }
    vec2 rng = (hi - lo) * 2.0;

    vec4 center = DB2_MED_tex(DB2_MED_pos);
    vec2 fpt = HOOKED_pt;
    float mad_c = 0.0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 off = vec2(float(x) - 1.5, float(y) - 1.5) * fpt;
            vec3 d = HOOKED_tex(HOOKED_pos + off).rgb - center.rgb;
            vec3 d_op = d - vec3(dot(d, DB2_RCV_W709));
            mad_c += max(max(abs(d_op.r), abs(d_op.g)), abs(d_op.b));
        }
    }
    mad_c *= (1.0 / 16.0);

    float y_c = dot(center.rgb, DB2_RCV_W709);
    vec2 o_c = vec2(center.b - y_c, center.r - y_c);
    vec4 d0t = DB2_DS_tex(DB2_DS_pos - vec2(0.0, 2.0 * pt.y));
    vec4 d1t = DB2_DS_tex(DB2_DS_pos - vec2(0.0, pt.y));
    vec4 d2t = DB2_DS_tex(DB2_DS_pos);
    vec4 d3t = DB2_DS_tex(DB2_DS_pos + vec2(0.0, pt.y));
    vec4 d4t = DB2_DS_tex(DB2_DS_pos + vec2(0.0, 2.0 * pt.y));
    vec2 s0 = vec2(d0t.b - d0t.a, d0t.r - d0t.a) - o_c;
    vec2 s1 = vec2(d1t.b - d1t.a, d1t.r - d1t.a) - o_c;
    vec2 s2 = vec2(d2t.b - d2t.a, d2t.r - d2t.a) - o_c;
    vec2 s3 = vec2(d3t.b - d3t.a, d3t.r - d3t.a) - o_c;
    vec2 s4 = vec2(d4t.b - d4t.a, d4t.r - d4t.a) - o_c;

    vec2 a0 = abs(s0); vec2 a1 = abs(s1); vec2 a2 = abs(s2);
    vec2 a3 = abs(s3); vec2 a4 = abs(s4);
    vec2 f = max(min(a0, a1), min(a3, a4));
    vec2 g = min(max(a0, a1), max(a3, a4));
    vec2 tex = max(min(a2, f), min(max(a2, f), g));

    vec2 fs = max(min(s0, s1), min(s3, s4));
    vec2 gs = min(max(s0, s1), max(s3, s4));
    vec2 coh = abs(max(min(s2, fs), min(max(s2, fs), gs)));

    vec2 sc = max(tex, 2.5 * coh);
    return vec4(rng, mad_c, max(sc.x, sc.y));
}

//!HOOK MAIN
//!BIND DB2_MED
//!SAVE DB2_TENS_H
//!WIDTH DB2_MED.w
//!HEIGHT DB2_MED.h
//!COMPONENTS 3
//!DESC Debandit: Tensor H

// =============================================================================
// PASS 18/19: STRUCTURE-TENSOR ANISOTROPY — the re-armed v1.11 experiment
// =============================================================================
// Central-difference gradients of median-field luma, tensor components
// row-summed over +-4 texels here, column-summed and eigen-analyzed in
// the V pass. Banding contours are locally 1-D (all gradient energy on
// one axis, coherence ~1); mid-scale mottle is 2-D. What made this
// UNUSABLE in v1.11 was significance-vs-local-MAD (texture inflates MAD
// and licenses its own absorption — circular); significance is now
// measured against the FRAME NOISE FLOOR from the SSBO. Dither is
// isotropic too but measures ~0.3x floor on the minor axis: under the
// knee. Clamp-sampling throughout (order-stat-class pass, bias-free).

const vec3 DB2_TH_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB2_MED_pt;
    vec2 pos = DB2_MED_pos;
    vec3 J = vec3(0.0);   // (sum gx^2, sum gy^2, sum gx*gy)
    for (int i = -4; i <= 4; i++) {
        vec2 tp = pos + vec2(float(i) * pt.x, 0.0);
        float xr = dot(DB2_MED_tex(tp + vec2(pt.x, 0.0)).rgb, DB2_TH_W709);
        float xl = dot(DB2_MED_tex(tp - vec2(pt.x, 0.0)).rgb, DB2_TH_W709);
        float yd = dot(DB2_MED_tex(tp + vec2(0.0, pt.y)).rgb, DB2_TH_W709);
        float yu = dot(DB2_MED_tex(tp - vec2(0.0, pt.y)).rgb, DB2_TH_W709);
        float gx = (xr - xl) * 0.5;
        float gy = (yd - yu) * 0.5;
        J += vec3(gx * gx, gy * gy, gx * gy);
    }
    return vec4(J, 1.0);
}

//!HOOK MAIN
//!BIND DB2_TENS_H
//!BIND DB2_STATE
//!SAVE DB2_ANISO
//!WIDTH DB2_TENS_H.w
//!HEIGHT DB2_TENS_H.h
//!COMPONENTS 1
//!DESC Debandit: Tensor V + Aniso Flag

// Eigen split: lambda_min = tr/2 - sqrt((dJ/2)^2 + Jxy^2) is the gradient
// energy on the MINOR axis — zero for anything locally 1-D regardless of
// orientation. amp_iso = its rms in code-value units; coherence kappa in
// [0,1]. Flag = significant AND isotropic. The floor is clamped to 0.05
// codes so pristine synthetic content cannot produce a zero knee.
#define DB2_AN_AMP_LO  0.35
#define DB2_AN_AMP_HI  0.9
#define DB2_AN_KAP_LO  0.55
#define DB2_AN_KAP_HI  0.95
#define DB2_AN_FLOOR_MIN (0.05 / 255.0)

vec4 hook() {
    vec2 pt = DB2_TENS_H_pt;
    vec2 pos = DB2_TENS_H_pos;
    vec3 J = vec3(0.0);
    for (int j = -4; j <= 4; j++) {
        J += DB2_TENS_H_tex(pos + vec2(0.0, float(j) * pt.y)).xyz;
    }
    J *= (1.0 / 81.0);

    float tr = J.x + J.y;
    float dq = sqrt(0.25 * (J.x - J.y) * (J.x - J.y) + J.z * J.z);
    float amp_iso = sqrt(max(0.5 * tr - dq, 0.0));
    float kappa = 2.0 * dq / max(tr, 1e-12);

    float fl = max(db2_floor, DB2_AN_FLOOR_MIN);
    float flag = smoothstep(DB2_AN_AMP_LO * fl, DB2_AN_AMP_HI * fl, amp_iso)
               * (1.0 - smoothstep(DB2_AN_KAP_LO, DB2_AN_KAP_HI, kappa));
    return vec4(flag, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND DB2_MED
//!BIND DB2_CORR
//!BIND DB2_SCORR
//!BIND DB2_RES
//!BIND DB2_ENV
//!BIND DB2_RANGE
//!BIND DB2_RANGEC
//!BIND DB2_ANISO
//!DESC Debandit: Apply

// =============================================================================
// PASS 20: APPLY (full resolution)
// =============================================================================
// v1.11's apply with the three v2 changes marked inline:
//  (v2-a) per-scale validity = envelope gate x pointwise WIDE fit-residual
//         gate per axis (no dilation on the residual);
//  (v2-b) validity-normalized convex scale blend + blended-validity gate;
//  (v2-c) the anisotropy flag closes the snap regime (LF correction stays).
// Everything else — knees, snap regimes, dark bias, SNR knee, authority,
// opponent decomposition, clip guard — is v1.11 verbatim.

#define DB2_CODE (1.0 / 255.0)

// Snap regime knees on the DILATED MAD, in code values (v1.11 verbatim).
#define DB2_SNAP_MAD_LO  1.3
#define DB2_SNAP_MAD_HI  2.1
#define DB2_SNAP_MAD_LO_D  0.6
#define DB2_SNAP_MAD_HI_D  1.6
#define DB2_DARK_LO      0.10
#define DB2_DARK_HI      0.30
// Structure-field knee + SNR adaptation (v1.11 verbatim; the chroma knee
// is EXEMPT from SNR adaptation — v1.11 audit F2).
#define DB2_TEX_LO       0.45
#define DB2_TEX_SNR      0.4
#define DB2_TEX_LO_MIN   0.18
#define DB2_SNAP_REACH   0.375
#define DB2_CLIP_LO      0.97
#define DB2_CLIP_HI      0.995
#define DB2_CTHR_GAIN    2.11
#define DB2_SNAP_MAD_LO_C  1.2
#define DB2_SNAP_MAD_HI_C  2.0

vec4 hook() {
    vec4 orig = HOOKED_tex(HOOKED_pos);

    vec4 M = DB2_MED_tex(DB2_MED_pos);

    // 5-tap tent over the range/noise fields (v1.11 verbatim).
    vec2 rpt = DB2_RANGE_pt;
    vec4 mm  = DB2_RANGE_tex(DB2_RANGE_pos);
    vec4 mmL = DB2_RANGE_tex(DB2_RANGE_pos - vec2(2.0 * rpt.x, 0.0));
    vec4 mmR = DB2_RANGE_tex(DB2_RANGE_pos + vec2(2.0 * rpt.x, 0.0));
    vec4 mmU = DB2_RANGE_tex(DB2_RANGE_pos - vec2(0.0, 2.0 * rpt.y));
    vec4 mmD = DB2_RANGE_tex(DB2_RANGE_pos + vec2(0.0, 2.0 * rpt.y));
    float range_soft = (mm.y - mm.x) * 0.4
                     + ((mmL.y - mmL.x) + (mmR.y - mmR.x)
                     +  (mmU.y - mmU.x) + (mmD.y - mmD.x)) * 0.15;
    float mad1 = max(max(DB2_RANGE_tex(DB2_RANGE_pos - vec2(rpt.x, 0.0)).z,
                         DB2_RANGE_tex(DB2_RANGE_pos + vec2(rpt.x, 0.0)).z),
                     max(DB2_RANGE_tex(DB2_RANGE_pos - vec2(0.0, rpt.y)).z,
                         DB2_RANGE_tex(DB2_RANGE_pos + vec2(0.0, rpt.y)).z));
    float mad_codes = max(max(mm.z, mad1),
                          max(max(mmL.z, mmR.z), max(mmU.z, mmD.z))) / DB2_CODE;
    float tex_codes = (mm.w + mmL.w + mmR.w + mmU.w + mmD.w) * 0.2 / DB2_CODE;

    // Chroma range/noise fields, 13-tap MAD max (v1.11 verbatim).
    vec2 cpt = DB2_RANGEC_pt;
    vec4 cc  = DB2_RANGEC_tex(DB2_RANGEC_pos);
    vec4 ccL = DB2_RANGEC_tex(DB2_RANGEC_pos - vec2(2.0 * cpt.x, 0.0));
    vec4 ccR = DB2_RANGEC_tex(DB2_RANGEC_pos + vec2(2.0 * cpt.x, 0.0));
    vec4 ccU = DB2_RANGEC_tex(DB2_RANGEC_pos - vec2(0.0, 2.0 * cpt.y));
    vec4 ccD = DB2_RANGEC_tex(DB2_RANGEC_pos + vec2(0.0, 2.0 * cpt.y));
    vec2 rangec_soft = cc.xy * 0.4
                     + (ccL.xy + ccR.xy + ccU.xy + ccD.xy) * 0.15;
    float madc1 = max(max(DB2_RANGEC_tex(DB2_RANGEC_pos - vec2(cpt.x, 0.0)).z,
                          DB2_RANGEC_tex(DB2_RANGEC_pos + vec2(cpt.x, 0.0)).z),
                      max(DB2_RANGEC_tex(DB2_RANGEC_pos - vec2(0.0, cpt.y)).z,
                          DB2_RANGEC_tex(DB2_RANGEC_pos + vec2(0.0, cpt.y)).z));
    float madc2 = max(max(DB2_RANGEC_tex(DB2_RANGEC_pos + vec2( cpt.x,  cpt.y)).z,
                          DB2_RANGEC_tex(DB2_RANGEC_pos + vec2( cpt.x, -cpt.y)).z),
                      max(DB2_RANGEC_tex(DB2_RANGEC_pos + vec2(-cpt.x,  cpt.y)).z,
                          DB2_RANGEC_tex(DB2_RANGEC_pos + vec2(-cpt.x, -cpt.y)).z));
    float madc_codes = max(max(cc.z, max(madc1, madc2)),
                           max(max(ccL.z, ccR.z), max(ccU.z, ccD.z))) / DB2_CODE;
    float texc_codes = (cc.w + ccL.w + ccR.w + ccU.w + ccD.w) * 0.2 / DB2_CODE;

    // The two plane-fit corrections, precomputed at 1/4 res.
    vec3 c  = DB2_CORR_tex(DB2_CORR_pos).rgb - vec3(0.5);
    vec3 cs = DB2_SCORR_tex(DB2_SCORR_pos).rgb - vec3(0.5);

    // (v2-a) Per-scale validities: envelope gate x pointwise residual gate
    // (wide scale only — the short scale's envelope covers its whole
    // contamination reach). Residual knees mirror the envelope knees:
    // half a step is legitimate staircase residual, a full step is not.
    float cthr = db_thr * DB2_CTHR_GAIN;
    vec4 env_codes = DB2_ENV_tex(DB2_ENV_pos) / DB2_CODE;
    vec2 res_codes = DB2_RES_tex(DB2_RES_pos).xy / DB2_CODE;
    float v_Ly = (1.0 - smoothstep(0.5 * db_thr, db_thr, env_codes.x))
               * (1.0 - smoothstep(0.5 * db_thr, db_thr, res_codes.x));
    float v_Sy =  1.0 - smoothstep(0.5 * db_thr, db_thr, env_codes.z);
    float v_Lc = (1.0 - smoothstep(0.5 * cthr, cthr, env_codes.y))
               * (1.0 - smoothstep(0.5 * cthr, cthr, res_codes.y));
    float v_Sc =  1.0 - smoothstep(0.5 * cthr, cthr, env_codes.w);

    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    float cy_L = dot(c,  W709);
    float cy_S = dot(cs, W709);
    vec3 cop_L = c  - vec3(cy_L);
    vec3 cop_S = cs - vec3(cy_S);

    // (v2-b) Validity-normalized convex blend. A condemned plane fit can
    // carry |c| ~ 10 codes near a bulk edge, so at v_L = 0 the estimate
    // must be PURELY the other scale — a linear mix's leftover fraction
    // paints measurable garbage (1.1 codes, colored-square synthetic).
    // At zero total validity the blend resolves to 0 and the gate below
    // kills the apply. db_ms = 0 degenerates to the wide fit alone.
    float wLy = v_Ly;
    float wSy = (1.0 - v_Ly) * v_Sy * db_ms;
    float dy_ = max(wLy + wSy, 1e-6);
    float cy_eff  = (wLy * cy_L + wSy * cy_S) / dy_;
    float w_env   = (wLy * v_Ly + wSy * v_Sy) / dy_;
    float wLc = v_Lc;
    float wSc = (1.0 - v_Lc) * v_Sc * db_ms;
    float dc_ = max(wLc + wSc, 1e-6);
    vec3 cop_eff  = (wLc * cop_L + wSc * cop_S) / dc_;
    float w_env_c = (wLc * v_Lc + wSc * v_Sc) / dc_;
    vec3 c_eff    = vec3(cy_eff) + cop_eff;

    // Gate 1: correction amplitude (v1.11 verbatim).
    float c_codes = max(max(abs(c_eff.r), abs(c_eff.g)), abs(c_eff.b)) / DB2_CODE;
    float w_amp = 1.0 - smoothstep(0.5 * db_thr, db_thr, c_codes);

    // Gate 2: local flatness (v1.11 verbatim).
    float range_codes = range_soft / DB2_CODE;
    float w_flat = 1.0 - smoothstep(db_flat, 2.0 * db_flat, range_codes);

    // Gate 3: structure, SNR-adaptive knee (v1.11 verbatim).
    float tex_lo = mix(DB2_TEX_LO,
                       clamp(DB2_TEX_SNR * mad_codes, DB2_TEX_LO_MIN, DB2_TEX_LO),
                       db_snr);
    float w_tex = 1.0 - smoothstep(tex_lo, 2.0 * tex_lo, tex_codes);

    float w = w_amp * w_flat * w_tex * w_env * db_strength;

    // Chroma path (v1.9/v1.11 verbatim except the v2 envelope/blend above).
    vec3 c_op = cop_eff;
    vec3 c_lp = vec3(cy_eff) + c_op * (1.0 - db_chroma);

    float cop_codes = max(max(abs(c_op.r), abs(c_op.g)), abs(c_op.b)) / DB2_CODE;
    float w_amp_c = 1.0 - smoothstep(0.5 * cthr, cthr, cop_codes);
    float rangec_codes = max(rangec_soft.x, rangec_soft.y) / DB2_CODE;
    float w_flat_c = 1.0 - smoothstep(db_flat_c, 2.0 * db_flat_c, rangec_codes);
    float w_texc = 1.0 - smoothstep(DB2_TEX_LO, 2.0 * DB2_TEX_LO, texc_codes);

    float w_c = w_amp_c * w_flat_c * w_tex * w_texc * w_env_c
              * db_strength * db_chroma;

    // Regime blend (v1.11 verbatim) + (v2-c) the anisotropy flag: mottle
    // that is significant against the FRAME noise floor and 2-D closes
    // the snap — the texture rides through; the LF correction stays.
    float lum = dot(M.rgb, vec3(0.2126, 0.7152, 0.0722));
    float dark_w = 1.0 - smoothstep(DB2_DARK_LO, DB2_DARK_HI, lum);
    float mad_lo = mix(DB2_SNAP_MAD_LO, DB2_SNAP_MAD_LO_D, dark_w);
    float mad_hi = mix(DB2_SNAP_MAD_HI, DB2_SNAP_MAD_HI_D, dark_w);
    float aniso = DB2_ANISO_tex(DB2_ANISO_pos).x * db_aniso;
    float snap = (1.0 - smoothstep(mad_lo, mad_hi, mad_codes))
               * (1.0 - aniso) * db_dither;
    float snap_c = (1.0 - smoothstep(DB2_SNAP_MAD_LO_C, DB2_SNAP_MAD_HI_C,
                                     madc_codes)) * db_dither;

    float auth   = (DB2_SNAP_REACH * db_thr + mad_codes * snap) * DB2_CODE;
    float auth_c = (DB2_SNAP_REACH * cthr + madc_codes * snap_c) * DB2_CODE;
    // Snap target reconstructed as c_eff + M (v1.11 verbatim): the stored
    // M's quantization cancels between this M and the -M inside each c.
    vec3 t = c_eff + M.rgb - orig.rgb;
    vec3 t_y  = vec3(dot(t, vec3(0.2126, 0.7152, 0.0722)));
    vec3 t_op = t - t_y;
    vec3 t_lp = t_y + t_op * (1.0 - db_chroma);
    vec3 r_lp = clamp(t_lp, vec3(-auth),   vec3(auth));
    vec3 r_op = clamp(t_op, vec3(-auth_c), vec3(auth_c));
    vec3 corr_l = mix(c_lp, r_lp, snap) * w;
    vec3 corr_c = mix(c_op, r_op, snap_c) * w_c;

    // Clip guard: LUMA-PATH TERM ONLY (v1.9 audit F1, verbatim).
    float clip_w = smoothstep(DB2_CLIP_LO, DB2_CLIP_HI, lum);
    corr_l = mix(corr_l, max(corr_l, vec3(0.0)), clip_w);
    vec3 corr = corr_l + corr_c;

#if db_debug == 1
    return orig;
#elif db_debug == 2
    return vec4(vec3(0.5) + corr * 32.0, 1.0);
#elif db_debug == 3
    return vec4(1.0 - w_flat, w, snap, 1.0);
#elif db_debug == 4
    return vec4(1.0 - w_flat_c, w_c, snap_c, 1.0);
#elif db_debug == 5
    return vec4(res_codes.x * 0.25, res_codes.y * 0.125, aniso, 1.0);
#else
    return vec4(orig.rgb + corr, orig.a);
#endif
}
