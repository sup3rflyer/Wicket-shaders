// GradientRestore v1.8 — Detect-and-Reconstruct Deband
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// Design goal: restore the smooth gradients the master had before 8-bit
// quantization banded them — reconstruction, not concealment. Where banding
// is positively identified, the pre-quantization ramp is refit and the pixel
// corrected toward it at fp16 precision; everywhere else the image passes
// through bit-exact. No noise is injected: re-quantization dither belongs to
// the grain shaders / libplacebo downstream.
//
// Grain protection is STRUCTURAL, not a heuristic: the correction field is
// derived exclusively from quarter-resolution box-mean fields, so it
// contains only low frequencies. The full-resolution high-frequency
// component — grain, texture, line art — rides through the additive apply
// untouched by construction.
//
// Method: P is the 4x4 box-mean plateau field, median-filtered (separable
// med5 at 1/4 res) into M so that line art and thin structure (up to ~2
// texels = ~8-10px) vanish from every derived field; S is a wide Gaussian
// of M (sigma 48px full-res — the refit ramp spanning several plateaus),
// DECONVOLVED to second order as S_refit = 2S - blur(S): a plain Gaussian
// carries a curvature bias (~sigma^2/2 * Laplacian) that reads real soft
// shading — cel shading, glows, vignettes — as "banding" and flattens it
// (field-found on a dark anime scene: soft maxima darkened ~0.2-2 codes,
// gate boxes visible as rectangular seams). The 2S - S2 refit cancels that
// bias exactly to O(sigma^4) on smooth content, so the correction is a
// structural no-op on anything genuinely smooth, while a staircase's
// anti-sawtooth (zero-mean at blur scale) passes through unchanged.
// c = S_refit - M is the low-frequency anti-staircase, and because all
// fields are line-free, debanding continues right up to line art instead
// of leaving a protected banded strip around it (and the line itself
// simply shifts with its background — its contrast is preserved exactly).
//
// The apply is two-regime, discriminated by a per-block noise measure (MAD
// of full-res pixels vs the median field, max across channels so chroma
// grain counts): CLEAN gradients (digital anime skies) snap to the refit
// ramp exactly — removing the contour itself, not just the plateau tilt —
// with authority bounded by 0.375*gr_thr + the measured local noise, so
// the snap always has exactly enough reach to finish the job (flatten the
// step AND absorb the encode dither that was masking it) but can never
// erase structure beyond sub-step scale plus measured noise. GRAINED
// regions (MAD above the knee) get the low-frequency correction only:
// grain rides through bit-exact, and the grain already dithers any
// residual contour. The deliberate cost: encode dither in flat regions is
// absorbed into the reconstruction — its whole purpose was masking the
// banding being reconstructed; real grain sits above the knee. Noise
// around sigma ~1.5 codes is a genuine continuum between the two and gets
// a proportional blend.
//
// Four soft gates zero all correction where the flat-ramp model does not
// hold (all fade to ZERO, never clamp, so no residual halos):
//   1. |c| gate — a true staircase correction never exceeds ~half the step
//      size, so the knee sits at gr_thr/2 (gr_thr is the STEP ceiling);
//      anything larger is remote-structure bias or uncancelled curvature.
//   2. Flatness gate — luma range of the median field over a +-32px window
//      above gr_flat code values means texture/detail; leave it alone.
//   3. Structure gate — the |DS - M| field (med5's own mean-bias): where
//      dense micro-texture makes the median an invalid ramp estimate,
//      nothing is applied. H-median keeps isolated lines invisible to it.
//   4. Contamination envelope — trust-weighted max of |c| over +-32px: a
//      banded flat needs sub-step corrections EVERYWHERE nearby; if any
//      trusted point wants more, the zone's ramp estimate is contaminated
//      (bulk edge / frame corner in blur reach) and small corrections
//      there are the tail of that bias, not banding. Pass through.
// A one-sided guard additionally forbids darkening where the plateau field
// sits at the signal ceiling (house rule: never attenuate source clipping).
//
// Chain order: this shader must be listed BEFORE CelFlare in glsl-shaders —
// same-hook passes run in shader load order, and CelFlare's expansion
// multiplies any step this shader has not flattened first.
//
// Known v1 limits, documented honestly:
// - The ramp is a MASKED normalized convolution (v1.4): texels whose box
//   mean disagrees with the median field (text strokes, line clusters,
//   dense detail) are excluded from S and from the flatness range, so
//   thin-and-dense structure no longer projects a banded halo around
//   itself. What remains protected: MID-SIZE flat objects (~12-32px) pass
//   the trust mask (their interior IS flat), so they legitimately close
//   the range gate for +-32px around themselves — banding that close to a
//   solid object edge survives. The |c| gate kills the worst S bias there.
// - The 1/4-res tap geometry is exact when source dimensions divide by 4;
//   other sizes drift the tap phase by up to ~2px across the frame — the
//   box means become approximations (soft, spatially smooth; harmless to
//   the gates but the "exact" claims below assume mod-4 dimensions).
// - The min/max range field is bilinearly interpolated at apply time, which
//   smooths the gate transition by ~1 low-res texel (~4px).
// - Frame borders: off-frame area is ZERO-WEIGHT in the normalized
//   convolution (v1.7) — the ramp near borders is estimated from in-frame
//   texels only, no replicate-clamp bias; a one-sided kernel still biases
//   sloped fields slightly (the refit cancels most of it).
// - EMA-free and stateless: nothing to unwind at cuts, no temporal shimmer
//   sources beyond per-frame MAD wobble at the regime knee (dilated +-2
//   texels in both axes, and the knees place dither on the saturated
//   plateau; sigma ~1.5 continuum noise still sits mid-knee and can
//   breathe slightly).
// - Bulk-edge collar (audit F1, THE top v2 candidate): the flatness and
//   envelope gates protect ~32px around solid-object boundaries, and near
//   CLIPPED highlights this hands residual bright banding to CelFlare's
//   expansion — the one place this shader bows out where the suite
//   amplifies. The honest fix is similarity-aware ramp estimation
//   (region-respecting weights), a v2 architecture step; a directional
//   envelope was considered and rejected (both brighter and darker
//   neighbors corrupt a ramp estimate equally).
// - The trust mask and flatness range are luma-only: equal-luma chroma
//   seams are neither masked nor range-gated (the max-channel |c| and MAD
//   gates still bound the damage). Revisit if colored seams surface.
// - At extreme frame borders the one-sided refit is not strictly monotone
//   -bounded; the envelope guards it only above its own knee.

// ---- shampv shader API (plain comments to libplacebo) ----
//@shampv input any

// =============================================================================
//  USER TUNING
// =============================================================================
// Sliders are DYNAMIC: glsl-shader-opts changes apply on the next frame, no
// recompile. gr_debug is a DEFINE and recompiles on change. Defaults = the
// shipped tune. NOTE: no comment lines may sit between a PARAM block and the
// next directive — the parser folds them into the value and fails to load.

//!PARAM gr_strength
//!DESC Deband strength — fraction of the reconstructed gradient applied. 1 = full reconstruction (the shipped tune), 0 = off.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM gr_thr
//!DESC Banding step-size ceiling in 8-bit code values; drives the amplitude gate (full up to half this, zero at 1x), the snap reach (0.375x), and the contamination-envelope tolerance. 2 covers 1-2 code banding (typical web encodes). Patchy HALF-corrected strong bands are the signal to raise it (gate and reach compound below the needed size); 4 = brutal sources.
//!TYPE DYNAMIC float
//!MINIMUM 0.5
//!MAXIMUM 4.0
2.0

//!PARAM gr_flat
//!DESC Flatness ceiling in 8-bit code values — luma range over a +-32px window above this marks texture/detail and fades correction out (zero at 2x). Raise to reach banding on steeper gradients, lower to protect more detail.
//!TYPE DYNAMIC float
//!MINIMUM 2.0
//!MAXIMUM 24.0
6.0

//!PARAM gr_dither
//!DESC Dither absorption / contour snap. 1 = full: cleanest possible gradients, but sub-code faint texture in quiet areas (smoke, haze) can read as dither and be absorbed with it. Lower to preserve such texture at the cost of contour removal on clean flats; 0 = low-frequency correction only (all fine texture bit-exact, faint contours may survive).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM gr_debug
//!DESC Debug views: 0 = off, 1 = bypass, 2 = applied correction x32 on mid-gray, 3 = gate map (R = flatness kill, G = applied weight, B = snap regime).
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 3
0

//!HOOK MAIN
//!BIND HOOKED
//!SAVE GR_DS
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC GradientRestore: Downsample 1/4

// =============================================================================
// PASS 1: DOWNSAMPLE 1/4 — plateau field P (exact 4x4 box mean)
// =============================================================================
// Each 1/4-res texel is the uniform mean of its 4x4 full-res block: the
// block center sits on a pixel corner, so bilinear taps at +-1 px are
// half-integer positions averaging a 2x2 quad each — 4 taps cover the 4x4
// exactly (mod-4 dimensions; see header for the drift caveat otherwise).
// Grain sigma drops 4x here and the median + Gaussian downstream remove the
// rest; banding plateaus (tens of px wide) keep their step structure.
// Alpha carries BT.709 luma of the mean.

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
//!BIND GR_DS
//!SAVE GR_MED_H
//!WIDTH GR_DS.w
//!HEIGHT GR_DS.h
//!DESC GradientRestore: Median H

// =============================================================================
// PASS 2/3: SEPARABLE MED5 — line-reject field M
// =============================================================================
// Per-channel median-of-5 along each axis removes structures up to ~2
// texels wide (~8-10px full res: anime line art) from the plateau field
// while preserving monotone staircases, so neither the refit ramp S nor
// the flatness range ever sees a line. Without this, every line would
// close the flatness gate for +-32px around itself (a banding lattice
// tracing the art) AND bias S into a ~1-2 code halo along the line.
// med5 identity (verified exhaustively): med5(a,b,c,d,e) =
// med3(e, max(min(a,b),min(c,d)), min(max(a,b),max(c,d))); min/max are
// componentwise, so all four channels (rgb + luma alpha) ride in SIMD.

vec4 hook() {
    vec2 pt = GR_DS_pt;
    vec2 pos = GR_DS_pos;
    vec4 a = GR_DS_tex(pos - vec2(2.0 * pt.x, 0.0));
    vec4 b = GR_DS_tex(pos - vec2(pt.x, 0.0));
    vec4 c = GR_DS_tex(pos);
    vec4 d = GR_DS_tex(pos + vec2(pt.x, 0.0));
    vec4 e = GR_DS_tex(pos + vec2(2.0 * pt.x, 0.0));
    vec4 f = max(min(a, b), min(d, e));
    vec4 g = min(max(a, b), max(d, e));
    return max(min(c, f), min(max(c, f), g));
}

//!HOOK MAIN
//!BIND GR_MED_H
//!BIND GR_DS
//!SAVE GR_MED
//!WIDTH GR_MED_H.w
//!HEIGHT GR_MED_H.h
//!DESC GradientRestore: Median V

// Alpha of the output is the TRUST MASK, not luma: 1 where this texel's
// box mean agrees with the median field (flat / banded / dithered /
// grained — all safe to learn the ramp from), fading to 0 where they
// disagree (structure: line art, text strokes, dense texture). The blur
// chain downstream is a NORMALIZED convolution against this mask, so
// structure is EXCLUDED from the ramp estimate instead of merely diluted —
// and the range gate ignores masked texels, so a title or line cluster no
// longer projects a +-32px protected strip of surviving banding around
// itself (field-found halo). Banding steps read ~0.3 here, sigma-2 grain
// ~0.7, structure 2+ — the knee sits above grain, below strokes.
#define GR_CODE_M    (1.0 / 255.0)
#define GR_MASK_LO   0.9
#define GR_MASK_HI   1.8

vec4 hook() {
    vec2 pt = GR_MED_H_pt;
    vec2 pos = GR_MED_H_pos;
    vec4 a = GR_MED_H_tex(pos - vec2(0.0, 2.0 * pt.y));
    vec4 b = GR_MED_H_tex(pos - vec2(0.0, pt.y));
    vec4 c = GR_MED_H_tex(pos);
    vec4 d = GR_MED_H_tex(pos + vec2(0.0, pt.y));
    vec4 e = GR_MED_H_tex(pos + vec2(0.0, 2.0 * pt.y));
    vec4 f = max(min(a, b), min(d, e));
    vec4 g = min(max(a, b), max(d, e));
    vec4 med = max(min(c, f), min(max(c, f), g));

    float lum_med = dot(med.rgb, vec3(0.2126, 0.7152, 0.0722));
    float dev = abs(GR_DS_tex(GR_DS_pos).a - lum_med) / GR_CODE_M;
    float mask = 1.0 - smoothstep(GR_MASK_LO, GR_MASK_HI, dev);
    return vec4(med.rgb, mask);
}

//!HOOK MAIN
//!BIND GR_MED
//!SAVE GR_BLUR_H
//!WIDTH GR_MED.w
//!HEIGHT GR_MED.h
//!DESC GradientRestore: Ramp Blur H

// =============================================================================
// PASS 4/5: RAMP BLUR — refit ramp S (separable Gaussian on M)
// =============================================================================
// sigma=12 at 1/4 res = effective ~48px at full res: a staircase with 100px
// plateaus attenuates its fundamental to ~1% (exp(-2 pi^2 sigma^2 / T^2)),
// i.e. the blur genuinely spans several plateaus and lands on the
// pre-quantization ramp. Weights are data-independent — exact separability,
// no boundary artifacts. sigma=12, radius=30, stride-2 bilinear tap merging
// (15 pairs, 31 fetches). The go/gw tables and the INV_WSUM constant are
// duplicated FOUR times total (Ramp Blur H/V, Refit Blur H, Correction)
// — all four copies MUST stay in sync (no compile-time guard across
// translation units).

vec4 hook() {
    const float go[15] = {
        1.4973958569, 3.4939239102, 5.4904525495, 7.4869821093,
        9.4835129238, 11.4800453267, 13.4765796511, 15.4731162293,
        17.4696553930, 19.4661974725, 21.4627427973, 23.4592916956,
        25.4558444940, 27.4524015179, 29.4489630909
    };
    const float gw[15] = {
        1.9827409157, 1.9151927034, 1.7993522583, 1.6442850520,
        1.4614878798, 1.2634862294, 1.0624365004, 0.8689456523,
        0.6912567644, 0.5348639224, 0.4025356305, 0.2946608540,
        0.2097962048, 0.1452880373, 0.0978631307
    };

    // Precomputed: 1.0 / (1.0 + 2.0 * sum(gw[]))
    #define GR_BLUR_INV_WSUM 0.0336152719

    vec2 pt = GR_MED_pt;
    vec2 pos = GR_MED_pos;

    // Mask-premultiplied (normalized convolution): rgb * mask in rgb,
    // mask in alpha. The Correction pass divides the two blurred halves.
    vec4 m0 = GR_MED_tex(pos);
    vec4 sum = vec4(m0.rgb * m0.a, m0.a);

    for (int i = 0; i < 15; i++) {
        float xp = pos.x + go[i] * pt.x;
        float xn = pos.x - go[i] * pt.x;
        vec4 sp = GR_MED_tex(vec2(xp, pos.y)) * step(xp, 1.0);
        vec4 sn = GR_MED_tex(vec2(xn, pos.y)) * step(0.0, xn);
        sum += (vec4(sp.rgb * sp.a, sp.a) + vec4(sn.rgb * sn.a, sn.a)) * gw[i];
    }

    return sum * GR_BLUR_INV_WSUM;
}

//!HOOK MAIN
//!BIND GR_BLUR_H
//!SAVE GR_SMOOTH
//!WIDTH GR_BLUR_H.w
//!HEIGHT GR_BLUR_H.h
//!DESC GradientRestore: Ramp Blur V

vec4 hook() {
    const float go[15] = {
        1.4973958569, 3.4939239102, 5.4904525495, 7.4869821093,
        9.4835129238, 11.4800453267, 13.4765796511, 15.4731162293,
        17.4696553930, 19.4661974725, 21.4627427973, 23.4592916956,
        25.4558444940, 27.4524015179, 29.4489630909
    };
    const float gw[15] = {
        1.9827409157, 1.9151927034, 1.7993522583, 1.6442850520,
        1.4614878798, 1.2634862294, 1.0624365004, 0.8689456523,
        0.6912567644, 0.5348639224, 0.4025356305, 0.2946608540,
        0.2097962048, 0.1452880373, 0.0978631307
    };

    // Precomputed: 1.0 / (1.0 + 2.0 * sum(gw[]))
    #define GR_BLUR_INV_WSUM 0.0336152719

    vec2 pt = GR_BLUR_H_pt;
    vec2 pos = GR_BLUR_H_pos;

    // Input is already mask-premultiplied; blur all four channels as-is.
    vec4 sum = GR_BLUR_H_tex(pos);

    for (int i = 0; i < 15; i++) {
        float yp = pos.y + go[i] * pt.y;
        float yn = pos.y - go[i] * pt.y;
        vec4 sp = GR_BLUR_H_tex(vec2(pos.x, yp)) * step(yp, 1.0);
        vec4 sn = GR_BLUR_H_tex(vec2(pos.x, yn)) * step(0.0, yn);
        sum += (sp + sn) * gw[i];
    }

    return sum * GR_BLUR_INV_WSUM;
}

//!HOOK MAIN
//!BIND GR_SMOOTH
//!SAVE GR_BLUR2_H
//!WIDTH GR_SMOOTH.w
//!HEIGHT GR_SMOOTH.h
//!DESC GradientRestore: Refit Blur H

// =============================================================================
// PASS 6/7: REFIT BLUR — S2 = blur(S), same kernel, for the 2S - S2
// deconvolution in the apply pass. Third of FOUR go/gw table copies
// (the Correction pass holds the fourth) — all four MUST stay in sync,
// and the deconvolution identity REQUIRES the same sigma as the ramp
// blur.
// =============================================================================

vec4 hook() {
    const float go[15] = {
        1.4973958569, 3.4939239102, 5.4904525495, 7.4869821093,
        9.4835129238, 11.4800453267, 13.4765796511, 15.4731162293,
        17.4696553930, 19.4661974725, 21.4627427973, 23.4592916956,
        25.4558444940, 27.4524015179, 29.4489630909
    };
    const float gw[15] = {
        1.9827409157, 1.9151927034, 1.7993522583, 1.6442850520,
        1.4614878798, 1.2634862294, 1.0624365004, 0.8689456523,
        0.6912567644, 0.5348639224, 0.4025356305, 0.2946608540,
        0.2097962048, 0.1452880373, 0.0978631307
    };

    // Precomputed: 1.0 / (1.0 + 2.0 * sum(gw[]))
    #define GR_BLUR_INV_WSUM 0.0336152719

    vec2 pt = GR_SMOOTH_pt;
    vec2 pos = GR_SMOOTH_pos;

    // Premultiplied throughout; the division happens once, in Correction.
    vec4 sum = GR_SMOOTH_tex(pos);

    for (int i = 0; i < 15; i++) {
        float xp = pos.x + go[i] * pt.x;
        float xn = pos.x - go[i] * pt.x;
        vec4 sp = GR_SMOOTH_tex(vec2(xp, pos.y)) * step(xp, 1.0);
        vec4 sn = GR_SMOOTH_tex(vec2(xn, pos.y)) * step(0.0, xn);
        sum += (sp + sn) * gw[i];
    }

    return sum * GR_BLUR_INV_WSUM;
}

//!HOOK MAIN
//!BIND GR_BLUR_H
//!BIND GR_BLUR2_H
//!BIND GR_MED
//!SAVE GR_CORR
//!WIDTH GR_BLUR_H.w
//!HEIGHT GR_BLUR_H.h
//!DESC GradientRestore: Correction

// Fused precision pass: recomputes BOTH V-blurs in fp32 registers (the
// fp16 storage noise of the H-blurred inputs averages out across 31 taps,
// ~0.03 code) and emits the COMPLETE correction field as one small value:
//   c = S_refit - M = 2*S - S2 - M
// Storing c directly instead of S/S2 separately matters: 2S - S2 built
// from fp16-stored large fields doubles the ~0.125-code ulp near white
// into visible 0.25-code plateaus (measured); a small-valued c lives in
// fp16's near-zero sweet spot (~0.001-code ulp). M enters c exactly once,
// and the apply pass reconstructs the snap target as c + M so the stored
// M's quantization cancels between the two terms. Encoded as 0.5 + c so
// both signs survive a unorm FBO fallback; the apply decodes with - 0.5.

vec4 hook() {
    const float go[15] = {
        1.4973958569, 3.4939239102, 5.4904525495, 7.4869821093,
        9.4835129238, 11.4800453267, 13.4765796511, 15.4731162293,
        17.4696553930, 19.4661974725, 21.4627427973, 23.4592916956,
        25.4558444940, 27.4524015179, 29.4489630909
    };
    const float gw[15] = {
        1.9827409157, 1.9151927034, 1.7993522583, 1.6442850520,
        1.4614878798, 1.2634862294, 1.0624365004, 0.8689456523,
        0.6912567644, 0.5348639224, 0.4025356305, 0.2946608540,
        0.2097962048, 0.1452880373, 0.0978631307
    };

    // Precomputed: 1.0 / (1.0 + 2.0 * sum(gw[]))
    #define GR_BLUR_INV_WSUM 0.0336152719

    vec2 pt = GR_BLUR_H_pt;
    vec2 pos = GR_BLUR_H_pos;

    vec4 sum_s  = GR_BLUR_H_tex(pos);
    vec4 sum_s2 = GR_BLUR2_H_tex(pos);

    for (int i = 0; i < 15; i++) {
        float yp = pos.y + go[i] * pt.y;
        float yn = pos.y - go[i] * pt.y;
        float vp = step(yp, 1.0);
        float vn = step(0.0, yn);
        sum_s  += (GR_BLUR_H_tex(vec2(pos.x, yp)) * vp  + GR_BLUR_H_tex(vec2(pos.x, yn)) * vn)  * gw[i];
        sum_s2 += (GR_BLUR2_H_tex(vec2(pos.x, yp)) * vp + GR_BLUR2_H_tex(vec2(pos.x, yn)) * vn) * gw[i];
    }

    sum_s  *= GR_BLUR_INV_WSUM;
    sum_s2 *= GR_BLUR_INV_WSUM;

    // Normalized-convolution division: the ramp is estimated from TRUSTED
    // texels only. Confidence fades the correction out where the local
    // neighborhood is mostly masked (dense structure everywhere — nothing
    // to learn a ramp from; the apply-side gates close there anyway).
    vec3 s  = sum_s.rgb  / max(sum_s.a,  1e-4);
    vec3 s2 = sum_s2.rgb / max(sum_s2.a, 1e-4);
    float conf = smoothstep(0.06, 0.20, min(sum_s.a, sum_s2.a));
    vec4 m = GR_MED_tex(GR_MED_pos);
    vec3 c = (2.0 * s - s2 - m.rgb) * conf;
    // Alpha: |c| (max channel) WEIGHTED BY THE TRUST MASK — dilated by the
    // ENV passes into the contamination envelope the apply pass gates on.
    // Masked texels (text strokes, line clusters) have huge but never-
    // applied c; counting them would spread a false protection halo around
    // structure the masking machinery already handles.
    return vec4(vec3(0.5) + c, max(max(abs(c.r), abs(c.g)), abs(c.b)) * m.a * m.a);
}

//!HOOK MAIN
//!BIND GR_CORR
//!SAVE GR_ENV_H
//!WIDTH GR_CORR.w
//!HEIGHT GR_CORR.h
//!DESC GradientRestore: Contamination Env H

// =============================================================================
// PASS 8/9: CONTAMINATION ENVELOPE — separable max of |c| over +-8 texels
// =============================================================================
// A banded flat needs corrections of at most ~half a step EVERYWHERE in a
// neighborhood; if ANY point within +-32px wants more, the ramp estimate
// there is contaminated (bulk-contrast edge or frame corner within blur
// reach) and the small corrections nearby are the TAIL of that bias, not
// banding — pointwise |c| passes through innocent-looking values inside a
// contaminated zone (field-found: ~1-code band painted onto a flat mech
// leg in a corner pocket). The apply pass fades all correction on this
// envelope. Legit fields measure well under the knee: 1-code ramps ~0.5,
// curved glow interiors ~0.9 (the refit keeps them small).

vec4 hook() {
    vec2 pt = GR_CORR_pt;
    vec2 pos = GR_CORR_pos;
    float m = GR_CORR_tex(pos).a;
    for (int i = 1; i <= 8; i++) {
        m = max(m, GR_CORR_tex(pos + vec2(float(i) * pt.x, 0.0)).a);
        m = max(m, GR_CORR_tex(pos - vec2(float(i) * pt.x, 0.0)).a);
    }
    return vec4(m, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND GR_ENV_H
//!SAVE GR_ENV
//!WIDTH GR_ENV_H.w
//!HEIGHT GR_ENV_H.h
//!DESC GradientRestore: Contamination Env V

vec4 hook() {
    vec2 pt = GR_ENV_H_pt;
    vec2 pos = GR_ENV_H_pos;
    float m = GR_ENV_H_tex(pos).x;
    for (int i = 1; i <= 8; i++) {
        m = max(m, GR_ENV_H_tex(pos + vec2(0.0, float(i) * pt.y)).x);
        m = max(m, GR_ENV_H_tex(pos - vec2(0.0, float(i) * pt.y)).x);
    }
    return vec4(m, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND GR_DS
//!BIND GR_MED
//!SAVE GR_RANGE_H
//!WIDTH GR_MED.w
//!HEIGHT GR_MED.h
//!DESC GradientRestore: Flatness Range H

// =============================================================================
// PASS 10/11: FLATNESS RANGE + NOISE FIELD
// =============================================================================
// Separable min/max of the MEDIAN-field luma over +-8 texels (+-32px full
// res). Box min/max separates exactly (min over rows then columns = min
// over the box); measuring on M (not P) means line art cannot hold the
// flatness gate open... or closed — only structure that survives the
// median (~10px+) closes the gate. Full-density taps — a strided min/max
// could skip a thin structure.
// Z carries the noise field for the snap regime: MAD of the full-res 4x4
// block against the median field, MAX across channels (chroma grain
// counts). A banding contour crossing the block contributes ~0.5 code;
// sigma-2 grain ~1.6 codes, and the +-2 max-dilation (V in the next pass,
// H at apply) biases it further up — the snap knee in the apply pass is
// placed against the DILATED statistics of each.
// W carries the STRUCTURE field: |DS block-mean - median field|, taken as
// the MEDIAN over +-2 texels H (then mean-combined V and at apply).
// Dither leaves block means nearly constant (sigma/4, reads ~0.2 codes);
// dense micro-texture — hatching strokes, fine pattern below the med5
// scale — moves ALL nearby block means by 1-2 codes and reads high. This
// is exactly med5's mean-bias: where it is large, M is not a valid ramp
// estimate (median skips the strokes, blur(M) - M grows +-1-code patch-
// boundary transients — field-found on a dark scene), so the apply pass
// fades ALL correction out on it. The H-median keeps isolated LINES
// invisible to the gate (1-2 contaminated samples of 5), preserving
// deband-up-to-the-line, and the knee doubles as a slope gate: gradients
// steeper than ~0.5 code / 8px trip it, where banding is invisible anyway.

// The min/max is MASKED: texels the trust mask rejects (text strokes, line
// clusters) do not count toward the range, so structure no longer projects
// a +-32px "still banded" strip around itself. The center's own luma seeds
// both extrema so a fully-masked window degrades to range 0 (the structure
// gate owns that case).

const vec3 GR_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = GR_MED_pt;
    vec2 pos = GR_MED_pos;

    vec4 center = GR_MED_tex(pos);
    float lum_c = dot(center.rgb, GR_W709);
    float lo = lum_c;
    float hi = lum_c;

    for (int i = 1; i <= 8; i++) {
        vec4 tp = GR_MED_tex(pos + vec2(float(i) * pt.x, 0.0));
        vec4 tn = GR_MED_tex(pos - vec2(float(i) * pt.x, 0.0));
        float op = step(0.5, tp.a);
        float on = step(0.5, tn.a);
        lo = min(lo, min(mix(1e3, dot(tp.rgb, GR_W709), op),
                         mix(1e3, dot(tn.rgb, GR_W709), on)));
        hi = max(hi, max(mix(-1e3, dot(tp.rgb, GR_W709), op),
                         mix(-1e3, dot(tn.rgb, GR_W709), on)));
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

    // MEDIANS of the 5 deviations, not means: an isolated line contaminates
    // only 1-2 samples and must not trip the gate (debanding continues over
    // and around lines); dense texture contaminates all 5 and does. Two
    // reads of the same taps:
    //  - UNSIGNED median = raw activity (dense texture, steep slope).
    //  - |SIGNED median| = COHERENCE: dither's block-mean residue is random
    //    and self-cancels (~0.1 code); a smoke wisp / soft surface detail
    //    of the SAME energy is spatially coherent and survives in full.
    //    x2.5 gain puts sub-half-code coherent texture past the knee while
    //    dither stays under it — this is what stops the snap regime from
    //    eating faint structured texture it cannot distinguish from dither
    //    by magnitude alone (field-found behind smoke, ~0.5-code detail).
    //    Banding contours self-cancel too (thin, 1-2 samples of 5).
    float s0 = GR_DS_tex(pos - vec2(2.0 * pt.x, 0.0)).a - lum_c;
    float s1 = GR_DS_tex(pos - vec2(pt.x, 0.0)).a - lum_c;
    float s2 = GR_DS_tex(pos).a - lum_c;
    float s3 = GR_DS_tex(pos + vec2(pt.x, 0.0)).a - lum_c;
    float s4 = GR_DS_tex(pos + vec2(2.0 * pt.x, 0.0)).a - lum_c;

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
//!BIND GR_RANGE_H
//!SAVE GR_RANGE
//!WIDTH GR_RANGE_H.w
//!HEIGHT GR_RANGE_H.h
//!DESC GradientRestore: Flatness Range V

// The MAD field max-dilates over +-2 here (V axis); the apply pass does the
// matching +-2 H dilation, so the snap regime is stabilized isotropically
// against per-frame wobble at grain-region borders.

vec4 hook() {
    vec2 pt = GR_RANGE_H_pt;
    vec2 pos = GR_RANGE_H_pos;

    vec4 mm = GR_RANGE_H_tex(pos);

    for (int i = 1; i <= 8; i++) {
        vec4 sp = GR_RANGE_H_tex(pos + vec2(0.0, float(i) * pt.y));
        vec4 sn = GR_RANGE_H_tex(pos - vec2(0.0, float(i) * pt.y));
        mm.x = min(mm.x, min(sp.x, sn.x));
        mm.y = max(mm.y, max(sp.y, sn.y));
        if (i <= 2) {
            mm.z = max(mm.z, max(sp.z, sn.z));
            mm.w += sp.w + sn.w;
        }
    }
    // w: MEAN over the +-2 V window (structure is extended; a max would
    // inflate on the field's own sampling noise and misfire on dither).
    mm.w *= (1.0 / 5.0);

    return mm;
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND GR_MED
//!BIND GR_CORR
//!BIND GR_ENV
//!BIND GR_RANGE
//!DESC GradientRestore: Apply

// =============================================================================
// PASS 12: APPLY (full resolution)
// =============================================================================
// out = orig + mix(c, snap, regime) * w, with the ramp taken as the
// deconvolved S_refit = 2S - S2 (curvature-unbiased: a structural no-op on
// genuinely smooth shading). The LF correction c leaves the full-res
// high-frequency component (grain, texture) bit-exact; the snap branch
// removes the contour itself on clean gradients. Gates fade to zero —
// never clamp — so structure the blur disagrees with is left alone rather
// than haloed; the range/noise fields are read through a 5-tap tent so
// gate transitions spread over ~12px instead of tracing the +-8-texel
// box's rectangular geometry. At w = 0 this pass returns the source value
// exactly.

// One 8-bit code value in full-range [0,1] terms. Limited-range sources land
// ~1.17x this after expansion to full — the thresholds are soft, so the
// approximation just shifts the knee slightly; tune gr_thr, not this.
#define GR_CODE (1.0 / 255.0)

// Snap regime knees on the DILATED MAD, in code values: sigma-1 encode
// dither dilates to ~1.1 — the LO knee sits ABOVE it so dither lands on
// the saturated plateau (snap = 1, temporally stable against per-frame
// MAD wobble; audit-found breathing risk at the old knee foot). Sigma-2
// grain dilates to ~2.2, fully outside — grain untouched. The continuum
// around sigma ~1.5 blends proportionally.
#define GR_SNAP_MAD_LO  1.3
#define GR_SNAP_MAD_HI  2.1
// In DARKS the knees tighten: faint sub-code signal in dark regions is
// far more likely deliberate texture (smoke, haze, shadow detail behind
// transparency) than encode dither, and CelFlare never expands darks so
// residual contours there are not amplified downstream. Clean dark flats
// (MAD well under the dark knee) keep FULL snap — dark banding is the
// most visible kind, and this bias must not soften it. The dark HI knee
// leaves DITHERED clean dark gradients roughly half-snapped (audit: the
// old 1.2 floored them at ~0 with no recovery path) — gr_dither scales
// from there in both directions of taste; coherent texture is protected
// independently by the structure and envelope gates.
#define GR_SNAP_MAD_LO_D  0.6
#define GR_SNAP_MAD_HI_D  1.6
// Dark ramp in M-field luma (gamma domain): fully dark-biased below
// ~26/255, fully bright-calibrated above ~77/255.
#define GR_DARK_LO      0.10
#define GR_DARK_HI      0.30
// Structure-field knee (codes): kills the snap on coherent micro-texture
// (hatching, fine pattern) whose full-res MAD mimics dither, and on
// gradients too steep for banding to be visible. Dither reads ~0.2 here.
#define GR_TEX_LO       0.45
#define GR_TEX_HI       0.9
// Snap authority = GR_SNAP_REACH * gr_thr + local MAD, in code values:
// enough to flatten a half-step quantization offset (0.75 codes at the
// default gr_thr = 2) PLUS the measured local noise (so an open snap can
// always finish absorbing the dither instead of stalling half-way), while
// staying physically unable to erase structure beyond sub-step scale plus
// noise. Scales with gr_thr so the brutal-source knob raises real reach.
#define GR_SNAP_REACH   0.375
// Never darken where the plateau field sits AT the signal ceiling — a
// clipped region gets expansion gradation downstream (CelFlare rule #1),
// not attenuation here. Deliberately tight (engages only above ~247/255):
// near-white banded skies below that need both correction signs.
#define GR_CLIP_LO      0.97
#define GR_CLIP_HI      0.995

vec4 hook() {
    vec4 orig = HOOKED_tex(HOOKED_pos);

    vec4 M = GR_MED_tex(GR_MED_pos);

    // 5-tap tent over the range/noise fields: softens gate transitions AND
    // completes the +-2 texel MAD dilation isotropically (V half in PASS 9).
    vec2 rpt = GR_RANGE_pt;
    vec4 mm  = GR_RANGE_tex(GR_RANGE_pos);
    vec4 mmL = GR_RANGE_tex(GR_RANGE_pos - vec2(2.0 * rpt.x, 0.0));
    vec4 mmR = GR_RANGE_tex(GR_RANGE_pos + vec2(2.0 * rpt.x, 0.0));
    vec4 mmU = GR_RANGE_tex(GR_RANGE_pos - vec2(0.0, 2.0 * rpt.y));
    vec4 mmD = GR_RANGE_tex(GR_RANGE_pos + vec2(0.0, 2.0 * rpt.y));
    float range_soft = (mm.y - mm.x) * 0.4
                     + ((mmL.y - mmL.x) + (mmR.y - mmR.x)
                     +  (mmU.y - mmU.x) + (mmD.y - mmD.x)) * 0.15;
    // +-1 taps close the H-dilation gap (audit: {-2,0,+2} alone skips a MAD
    // spike one texel away; the V pass covers +-2 gaplessly on its axis).
    float mad1 = max(max(GR_RANGE_tex(GR_RANGE_pos - vec2(rpt.x, 0.0)).z,
                         GR_RANGE_tex(GR_RANGE_pos + vec2(rpt.x, 0.0)).z),
                     max(GR_RANGE_tex(GR_RANGE_pos - vec2(0.0, rpt.y)).z,
                         GR_RANGE_tex(GR_RANGE_pos + vec2(0.0, rpt.y)).z));
    float mad_codes = max(max(mm.z, mad1),
                          max(max(mmL.z, mmR.z), max(mmU.z, mmD.z))) / GR_CODE;
    float tex_codes = (mm.w + mmL.w + mmR.w + mmU.w + mmD.w) * 0.2 / GR_CODE;

    // The complete LF correction, precomputed at 1/4 res: c = S_refit - M.
    vec3 c = GR_CORR_tex(GR_CORR_pos).rgb - vec3(0.5);

    // Gate 1: correction amplitude, max across channels so a big miss in any
    // channel kills the whole correction (no per-channel hue twist). The
    // knee is HALF of gr_thr: gr_thr is a STEP-size ceiling, and the true
    // correction for a staircase never exceeds ~half the step — anything
    // larger is remote-structure bias or mid-scale curvature the refit
    // could not cancel (field-found: the 2*gr_thr knee passed 1.5-2.5-code
    // pseudo-corrections on murky large-amplitude soft structure).
    float c_codes = max(max(abs(c.r), abs(c.g)), abs(c.b)) / GR_CODE;
    float w_amp = 1.0 - smoothstep(0.5 * gr_thr, gr_thr, c_codes);

    // Gate 2: local flatness — median-field luma range over +-32px.
    float range_codes = range_soft / GR_CODE;
    float w_flat = 1.0 - smoothstep(gr_flat, 2.0 * gr_flat, range_codes);

    // Gate 3: structure — med5's mean-bias field; where the median field
    // is not a valid ramp estimate (dense micro-texture), apply nothing.
    float w_tex = 1.0 - smoothstep(GR_TEX_LO, GR_TEX_HI, tex_codes);

    // Gate 4: contamination envelope — if any point within +-32px needs a
    // correction beyond banding scale, this whole zone's ramp estimate is
    // untrustworthy; pass the source through. Scales with gr_thr like the
    // amplitude gate.
    float env_codes = GR_ENV_tex(GR_ENV_pos).x / GR_CODE;
    float w_env = 1.0 - smoothstep(0.5 * gr_thr, 1.0 * gr_thr, env_codes);

    float w = w_amp * w_flat * w_tex * w_env * gr_strength;

    // Regime blend: clean gradients snap to the ramp itself (removes the
    // contour, absorbs the dither that masked it); grained regions take the
    // LF-only correction (grain rides through bit-exact).
    float lum = dot(M.rgb, vec3(0.2126, 0.7152, 0.0722));
    float dark_w = 1.0 - smoothstep(GR_DARK_LO, GR_DARK_HI, lum);
    float mad_lo = mix(GR_SNAP_MAD_LO, GR_SNAP_MAD_LO_D, dark_w);
    float mad_hi = mix(GR_SNAP_MAD_HI, GR_SNAP_MAD_HI_D, dark_w);
    float snap = (1.0 - smoothstep(mad_lo, mad_hi, mad_codes)) * gr_dither;
    // The +MAD authority exists to finish absorbing dither; grant it only in
    // proportion to dither-confidence (snap), or blend-zone texture (fine
    // hatching, weak grain) gets pulled ~1 code toward the refit.
    float auth = (GR_SNAP_REACH * gr_thr + mad_codes * snap) * GR_CODE;
    // Snap target reconstructed as c + M (= S_refit): the stored M's
    // quantization cancels between this M and the -M inside c.
    vec3 r = clamp(c + M.rgb - orig.rgb, vec3(-auth), vec3(auth));
    vec3 corr = mix(c, r, snap);

    // Clip guard: one-sided near the ceiling — never pull a clipped/near-
    // clipped plateau down.
    float clip_w = smoothstep(GR_CLIP_LO, GR_CLIP_HI, lum);
    corr = mix(corr, max(corr, vec3(0.0)), clip_w);

#if gr_debug == 1
    return orig;
#elif gr_debug == 2
    return vec4(vec3(0.5) + corr * w * 32.0, 1.0);
#elif gr_debug == 3
    return vec4(1.0 - w_flat, w, snap, 1.0);
#else
    return vec4(orig.rgb + corr * w, orig.a);
#endif
}
