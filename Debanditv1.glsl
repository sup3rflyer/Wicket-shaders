// Debanditv1 (v1.11) — Detect-and-Reconstruct Deband
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// PARKED previous generation, kept loadable for A/B against the v2.0
// default (Debandit.glsl — plane-fit estimator). Same db_* PARAM names
// in both; load one at a time.
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
// with authority bounded by 0.375*db_thr + the measured local noise, so
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
//      size, so the knee sits at db_thr/2 (db_thr is the STEP ceiling);
//      anything larger is remote-structure bias or uncancelled curvature.
//   2. Flatness gate — luma range of the median field over a +-32px window
//      above db_flat code values means texture/detail; leave it alone.
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
// - Bulk-edge collar (audit F1) — REDUCED by the v1.10 MULTI-SCALE
//   RESCUE, not eliminated: a second refit at sigma 12 full-res (built
//   with the composition identity S2 = G(sigma*sqrt2) conv M, three
//   passes) carries its own contamination envelopes in the spare ENV
//   channels, and the apply blends per axis — the wide ramp where its
//   envelope trusts it, the short ramp where only IT is clean. The short
//   refit's bias reach is ~2 sigma = 6 quarter-res texels, so with the
//   +-8-texel envelope window the collar shrinks from ~130px to ~55px,
//   and plateaus up to ~30-40px flatten inside the rescue zone (wider
//   plateaus near edges still wait for v2). Field driver: LUMA banding on
//   a dark uniform collar beside white trim (w_amp median 0.00 — the
//   sigma-48 bias exceeded db_thr region-wide) and in hair shading
//   between strand clusters (w_env median 0.25). db_ms = 0 restores
//   exact v1.9 behavior (and composes with db_chroma = 0 to exact v1.8).
//   The honest full fix is still similarity-aware ramp estimation
//   (region-respecting weights), a v2 architecture step; a directional
//   envelope was considered and rejected (both brighter and darker
//   neighbors corrupt a ramp estimate equally).
// - v1.9 CHROMA PATH: the correction is opponent-decomposed at apply
//   (since v1.10 on the multi-scale-blended field c_eff) — the luma
//   projection rides the original (unchanged) luma gates, while the
//   opponent part is gated by CHROMA-native fields: opponent-axis
//   flatness ranges, a chroma contamination envelope, and a chroma MAD
//   whose opponent projection cancels Y-plane dither EXACTLY (the Y
//   coefficient is identical for R/G/B, so common-mode dither vanishes).
//   Field-found motivation (2026-07-10, night face close-up): chroma
//   banding riding a steep luma shading field — the luma flatness gate
//   plus ~1.7 codes of 50-150px curvature error in the shared max-channel
//   |c| zeroed EVERYTHING, vetoing a chroma correction whose own model
//   error was 0.11 codes on a 0.55-code-flat chroma field. Chroma
//   quantities live in RGB-projected units: one YCbCr chroma code spans
//   ~2.1 RGB codes on the B axis, so the chroma knees/reach scale db_thr
//   by DB_CTHR_GAIN. db_chroma = 0 restores exact v1.8 behavior.
//   The chroma path carries its own structure/coherence term (DB_RANGEC
//   w channel) so faint coherent colored texture is not absorbed by the
//   chroma snap; the trust mask stays luma-only (correct — med5 validity
//   there is a luma-structure question; an equal-luma pure-chroma line is
//   rare and simply closes the chroma range gate around itself).
// - At extreme frame borders the one-sided refit is not strictly monotone
//   -bounded; the envelope guards it only above its own knee.

// ---- shampv shader API (plain comments to libplacebo) ----
//@shampv input any

// =============================================================================
//  USER TUNING
// =============================================================================
// Sliders are DYNAMIC: glsl-shader-opts changes apply on the next frame, no
// recompile. db_debug is a DEFINE and recompiles on change. Defaults = the
// shipped tune. NOTE: no comment lines may sit between a PARAM block and the
// next directive — the parser folds them into the value and fails to load.

//!PARAM db_strength
//!DESC Deband strength — fraction of the reconstructed gradient applied. 1 = full reconstruction (the shipped tune), 0 = off.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_thr
//!DESC Banding step-size ceiling in 8-bit code values; drives the amplitude gate (full up to half this, zero at 1x), the snap reach (0.375x), and the contamination-envelope tolerance. 2 covers 1-2 code banding (typical web encodes). Patchy HALF-corrected strong bands are the signal to raise it (gate and reach compound below the needed size); 4 = brutal sources.
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
//!DESC Chroma-path strength — fraction of the opponent-axis (chroma) correction applied. Chroma banding is gated by its OWN flatness/envelope/noise fields, so it is fixed even where steep luma shading closes the luma gates (the field-found miss: colored banding on a shaded face). 1 = full (shipped tune), 0 = v1.8 luma-only behavior. The luma path is unaffected either way.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_flat_c
//!DESC Chroma flatness ceiling in RGB-projected code values — opponent-axis (B-Y / R-Y) range of the median field over a +-32px window above this marks colored structure and fades the chroma correction out (zero at 2x). Chroma fields are far flatter than luma: banded surfaces measure ~0.5-1, colored art/iris structure 8+.
//!TYPE DYNAMIC float
//!MINIMUM 1.0
//!MAXIMUM 24.0
4.0

//!PARAM db_ms
//!DESC Multi-scale rescue strength. Near bulk-contrast edges (a bright trim beside a dark uniform, hair strand clusters) the wide sigma-48 ramp estimate is contamination-gated and banding survives in a ~130px collar; this blends in a short sigma-12 refit where only IT is clean, shrinking the collar to ~55px and flattening plateaus up to ~30-40px there. 1 = full (shipped tune), 0 = v1.9 behavior (wide ramp only).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_snr
//!DESC SNR-adaptive texture protection. The structure gate's knee is calibrated so 8-bit encode dither (~0.2 codes) cannot close it — but on quiet high-bit-depth sources real texture lives BELOW that static knee (10-bit stone/cloud detail at 0.1-0.35 codes) and the snap absorbs it as if it were dither. At 1 the knee scales with the MEASURED local noise floor (the MAD field): dithered 8-bit content clamps to the static knee (behavior unchanged), quiet sources tighten so faint coherent texture is protected while clean banded flats (whose structure field reads ~0.05) stay correctable. 0 = static knee (v1.10 behavior).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_dither
//!DESC Dither absorption / contour snap, scaling BOTH the luma and chroma snap regimes. 1 = full: cleanest possible gradients, but sub-code faint texture in quiet areas (smoke, haze) can read as dither and be absorbed with it. Lower to preserve such texture at the cost of contour removal on clean flats; 0 = low-frequency correction only (all fine texture bit-exact, faint contours may survive).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM db_debug
//!DESC Debug views: 0 = off, 1 = bypass, 2 = applied correction x32 on mid-gray, 3 = gate map (R = flatness kill, G = applied weight, B = snap regime), 4 = chroma gate map (R = chroma flatness kill, G = applied chroma weight, B = chroma snap regime).
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 4
0

//!HOOK MAIN
//!BIND HOOKED
//!SAVE DB_DS
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC Debanditv1: Downsample 1/4

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
//!BIND DB_DS
//!SAVE DB_MED_H
//!WIDTH DB_DS.w
//!HEIGHT DB_DS.h
//!DESC Debanditv1: Median H

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
    vec2 pt = DB_DS_pt;
    vec2 pos = DB_DS_pos;
    vec4 a = DB_DS_tex(pos - vec2(2.0 * pt.x, 0.0));
    vec4 b = DB_DS_tex(pos - vec2(pt.x, 0.0));
    vec4 c = DB_DS_tex(pos);
    vec4 d = DB_DS_tex(pos + vec2(pt.x, 0.0));
    vec4 e = DB_DS_tex(pos + vec2(2.0 * pt.x, 0.0));
    vec4 f = max(min(a, b), min(d, e));
    vec4 g = min(max(a, b), max(d, e));
    return max(min(c, f), min(max(c, f), g));
}

//!HOOK MAIN
//!BIND DB_MED_H
//!BIND DB_DS
//!SAVE DB_MED
//!WIDTH DB_MED_H.w
//!HEIGHT DB_MED_H.h
//!DESC Debanditv1: Median V

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
#define DB_CODE_M    (1.0 / 255.0)
#define DB_MASK_LO   0.9
#define DB_MASK_HI   1.8

vec4 hook() {
    vec2 pt = DB_MED_H_pt;
    vec2 pos = DB_MED_H_pos;
    vec4 a = DB_MED_H_tex(pos - vec2(0.0, 2.0 * pt.y));
    vec4 b = DB_MED_H_tex(pos - vec2(0.0, pt.y));
    vec4 c = DB_MED_H_tex(pos);
    vec4 d = DB_MED_H_tex(pos + vec2(0.0, pt.y));
    vec4 e = DB_MED_H_tex(pos + vec2(0.0, 2.0 * pt.y));
    vec4 f = max(min(a, b), min(d, e));
    vec4 g = min(max(a, b), max(d, e));
    vec4 med = max(min(c, f), min(max(c, f), g));

    float lum_med = dot(med.rgb, vec3(0.2126, 0.7152, 0.0722));
    float dev = abs(DB_DS_tex(DB_DS_pos).a - lum_med) / DB_CODE_M;
    float mask = 1.0 - smoothstep(DB_MASK_LO, DB_MASK_HI, dev);
    return vec4(med.rgb, mask);
}

//!HOOK MAIN
//!BIND DB_MED
//!SAVE DB_BLUR_H
//!WIDTH DB_MED.w
//!HEIGHT DB_MED.h
//!DESC Debanditv1: Ramp Blur H

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
    #define DB_BLUR_INV_WSUM 0.0336152719

    vec2 pt = DB_MED_pt;
    vec2 pos = DB_MED_pos;

    // Mask-premultiplied (normalized convolution): rgb * mask in rgb,
    // mask in alpha. The Correction pass divides the two blurred halves.
    vec4 m0 = DB_MED_tex(pos);
    vec4 sum = vec4(m0.rgb * m0.a, m0.a);

    for (int i = 0; i < 15; i++) {
        float xp = pos.x + go[i] * pt.x;
        float xn = pos.x - go[i] * pt.x;
        vec4 sp = DB_MED_tex(vec2(xp, pos.y)) * step(xp, 1.0);
        vec4 sn = DB_MED_tex(vec2(xn, pos.y)) * step(0.0, xn);
        sum += (vec4(sp.rgb * sp.a, sp.a) + vec4(sn.rgb * sn.a, sn.a)) * gw[i];
    }

    return sum * DB_BLUR_INV_WSUM;
}

//!HOOK MAIN
//!BIND DB_BLUR_H
//!SAVE DB_SMOOTH
//!WIDTH DB_BLUR_H.w
//!HEIGHT DB_BLUR_H.h
//!DESC Debanditv1: Ramp Blur V

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
    #define DB_BLUR_INV_WSUM 0.0336152719

    vec2 pt = DB_BLUR_H_pt;
    vec2 pos = DB_BLUR_H_pos;

    // Input is already mask-premultiplied; blur all four channels as-is.
    vec4 sum = DB_BLUR_H_tex(pos);

    for (int i = 0; i < 15; i++) {
        float yp = pos.y + go[i] * pt.y;
        float yn = pos.y - go[i] * pt.y;
        vec4 sp = DB_BLUR_H_tex(vec2(pos.x, yp)) * step(yp, 1.0);
        vec4 sn = DB_BLUR_H_tex(vec2(pos.x, yn)) * step(0.0, yn);
        sum += (sp + sn) * gw[i];
    }

    return sum * DB_BLUR_INV_WSUM;
}

//!HOOK MAIN
//!BIND DB_SMOOTH
//!SAVE DB_BLUR2_H
//!WIDTH DB_SMOOTH.w
//!HEIGHT DB_SMOOTH.h
//!DESC Debanditv1: Refit Blur H

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
    #define DB_BLUR_INV_WSUM 0.0336152719

    vec2 pt = DB_SMOOTH_pt;
    vec2 pos = DB_SMOOTH_pos;

    // Premultiplied throughout; the division happens once, in Correction.
    vec4 sum = DB_SMOOTH_tex(pos);

    for (int i = 0; i < 15; i++) {
        float xp = pos.x + go[i] * pt.x;
        float xn = pos.x - go[i] * pt.x;
        vec4 sp = DB_SMOOTH_tex(vec2(xp, pos.y)) * step(xp, 1.0);
        vec4 sn = DB_SMOOTH_tex(vec2(xn, pos.y)) * step(0.0, xn);
        sum += (sp + sn) * gw[i];
    }

    return sum * DB_BLUR_INV_WSUM;
}

//!HOOK MAIN
//!BIND DB_BLUR_H
//!BIND DB_BLUR2_H
//!BIND DB_MED
//!SAVE DB_CORR
//!WIDTH DB_BLUR_H.w
//!HEIGHT DB_BLUR_H.h
//!DESC Debanditv1: Correction

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
    #define DB_BLUR_INV_WSUM 0.0336152719

    vec2 pt = DB_BLUR_H_pt;
    vec2 pos = DB_BLUR_H_pos;

    vec4 sum_s  = DB_BLUR_H_tex(pos);
    vec4 sum_s2 = DB_BLUR2_H_tex(pos);

    for (int i = 0; i < 15; i++) {
        float yp = pos.y + go[i] * pt.y;
        float yn = pos.y - go[i] * pt.y;
        float vp = step(yp, 1.0);
        float vn = step(0.0, yn);
        sum_s  += (DB_BLUR_H_tex(vec2(pos.x, yp)) * vp  + DB_BLUR_H_tex(vec2(pos.x, yn)) * vn)  * gw[i];
        sum_s2 += (DB_BLUR2_H_tex(vec2(pos.x, yp)) * vp + DB_BLUR2_H_tex(vec2(pos.x, yn)) * vn) * gw[i];
    }

    sum_s  *= DB_BLUR_INV_WSUM;
    sum_s2 *= DB_BLUR_INV_WSUM;

    // Normalized-convolution division: the ramp is estimated from TRUSTED
    // texels only. Confidence fades the correction out where the local
    // neighborhood is mostly masked (dense structure everywhere — nothing
    // to learn a ramp from; the apply-side gates close there anyway).
    vec3 s  = sum_s.rgb  / max(sum_s.a,  1e-4);
    vec3 s2 = sum_s2.rgb / max(sum_s2.a, 1e-4);
    float conf = smoothstep(0.06, 0.20, min(sum_s.a, sum_s2.a));
    vec4 m = DB_MED_tex(DB_MED_pos);
    vec3 c = (2.0 * s - s2 - m.rgb) * conf;
    // Alpha: |c| (max channel) WEIGHTED BY THE TRUST MASK — dilated by the
    // ENV passes into the contamination envelope the apply pass gates on.
    // Masked texels (text strokes, line clusters) have huge but never-
    // applied c; counting them would spread a false protection halo around
    // structure the masking machinery already handles. Computed HERE at
    // fp32 and stored near zero (sweet-spot ulp) — the luma envelope must
    // NOT be re-derived from the 0.5 + c encode (fp16 ulp at 0.5 is ~0.06
    // code, measured 0.3-code apply wobble on env-knee content); the
    // chroma ENV seed is derived as a bounded RATIO of this exact seed.
    return vec4(vec3(0.5) + c, max(max(abs(c.r), abs(c.g)), abs(c.b)) * m.a * m.a);
}

//!HOOK MAIN
//!BIND DB_MED
//!SAVE DB_SBLUR_H
//!WIDTH DB_MED.w
//!HEIGHT DB_MED.h
//!DESC Debanditv1: Short Ramp Blur H

// =============================================================================
// PASS 8/9/10 (v1.10): SHORT REFIT — sigma 3 at 1/4 res (12px full res)
// =============================================================================
// The multi-scale rescue ramp: same masked normalized convolution and
// 2S - S2 deconvolution as the wide chain, at a quarter of its reach.
// S2 is built with the Gaussian composition identity blur_s(blur_s(M)) =
// G(s*sqrt2) conv M — one direct sigma-4.2426 blur of M instead of
// re-blurring S — so the short chain costs three passes, not four (the
// discrete-kernel difference from true double-blurring is ~1e-3 of the
// weight mass, far below code scale). A sigma-12-full-res refit spans
// plateaus up to ~30-40px and its bias reach near a bulk edge is ~2
// sigma = 6 quarter-res texels, vs ~25 for the wide ramp — that reach
// difference is the whole point (see the apply-pass blend). Tables are
// stride-2 bilinear-merged like the wide chain; the sigma-3 pair appears
// TWICE (here + Short Correction) and the sigma-4.2426 pair twice
// (Short Refit Blur H + Short Correction) — copies MUST stay in sync.

vec4 hook() {
    const float go[5] = {
        1.4584295168, 3.4039848067, 5.3518057801, 7.3029407160,
        9.2581597095
    };
    const float gw[5] = {
        1.7466968718, 1.0176429502, 0.3846874920, 0.0942940294,
        0.0149749167
    };

    // Precomputed: 1.0 / (1.0 + 2.0 * sum(gw[]))
    #define DB_SBLUR_INV_WSUM 0.1330390063

    vec2 pt = DB_MED_pt;
    vec2 pos = DB_MED_pos;

    // Mask-premultiplied normalized convolution, off-frame taps zero-weight.
    vec4 m0 = DB_MED_tex(pos);
    vec4 sum = vec4(m0.rgb * m0.a, m0.a);

    for (int i = 0; i < 5; i++) {
        float xp = pos.x + go[i] * pt.x;
        float xn = pos.x - go[i] * pt.x;
        vec4 sp = DB_MED_tex(vec2(xp, pos.y)) * step(xp, 1.0);
        vec4 sn = DB_MED_tex(vec2(xn, pos.y)) * step(0.0, xn);
        sum += (vec4(sp.rgb * sp.a, sp.a) + vec4(sn.rgb * sn.a, sn.a)) * gw[i];
    }

    return sum * DB_SBLUR_INV_WSUM;
}

//!HOOK MAIN
//!BIND DB_MED
//!SAVE DB_SBLUR2_H
//!WIDTH DB_MED.w
//!HEIGHT DB_MED.h
//!DESC Debanditv1: Short Refit Blur H

// H half of the sigma-4.2426 (= 3*sqrt2) blur of M for the short S2.

vec4 hook() {
    const float go[7] = {
        1.4791787146, 3.4515414720, 5.4241999464, 7.3973146620,
        9.3710353351, 11.3454977507, 13.3208213008
    };
    const float gw[7] = {
        1.8674437939, 1.4199811715, 0.8672312298, 0.4253890721,
        0.1675757486, 0.0530123245, 0.0134661865
    };

    // Precomputed: 1.0 / (1.0 + 2.0 * sum(gw[]))
    #define DB_SBLUR2_INV_WSUM 0.0940893179

    vec2 pt = DB_MED_pt;
    vec2 pos = DB_MED_pos;

    vec4 m0 = DB_MED_tex(pos);
    vec4 sum = vec4(m0.rgb * m0.a, m0.a);

    for (int i = 0; i < 7; i++) {
        float xp = pos.x + go[i] * pt.x;
        float xn = pos.x - go[i] * pt.x;
        vec4 sp = DB_MED_tex(vec2(xp, pos.y)) * step(xp, 1.0);
        vec4 sn = DB_MED_tex(vec2(xn, pos.y)) * step(0.0, xn);
        sum += (vec4(sp.rgb * sp.a, sp.a) + vec4(sn.rgb * sn.a, sn.a)) * gw[i];
    }

    return sum * DB_SBLUR2_INV_WSUM;
}

//!HOOK MAIN
//!BIND DB_SBLUR_H
//!BIND DB_SBLUR2_H
//!BIND DB_MED
//!SAVE DB_SCORR
//!WIDTH DB_SBLUR_H.w
//!HEIGHT DB_SBLUR_H.h
//!DESC Debanditv1: Short Correction

// Fused precision pass, mirror of the wide Correction: both V blurs in
// fp32 registers, normalized-conv division once, and the complete short
// correction c_s = 2*S_s - S2_s - M stored small-valued as 0.5 + c_s.
// Alpha = max-channel |c_s| * trust-mask^2, computed HERE at fp32 (the
// v1.9 envelope-precision lesson applies identically to this seed).

vec4 hook() {
    const float goA[5] = {
        1.4584295168, 3.4039848067, 5.3518057801, 7.3029407160,
        9.2581597095
    };
    const float gwA[5] = {
        1.7466968718, 1.0176429502, 0.3846874920, 0.0942940294,
        0.0149749167
    };
    const float goB[7] = {
        1.4791787146, 3.4515414720, 5.4241999464, 7.3973146620,
        9.3710353351, 11.3454977507, 13.3208213008
    };
    const float gwB[7] = {
        1.8674437939, 1.4199811715, 0.8672312298, 0.4253890721,
        0.1675757486, 0.0530123245, 0.0134661865
    };

    #define DB_SBLURA_INV_WSUM 0.1330390063
    #define DB_SBLURB_INV_WSUM 0.0940893179

    vec2 pt = DB_SBLUR_H_pt;
    vec2 pos = DB_SBLUR_H_pos;

    vec4 sum_s  = DB_SBLUR_H_tex(pos);
    vec4 sum_s2 = DB_SBLUR2_H_tex(pos);

    for (int i = 0; i < 5; i++) {
        float yp = pos.y + goA[i] * pt.y;
        float yn = pos.y - goA[i] * pt.y;
        sum_s += (DB_SBLUR_H_tex(vec2(pos.x, yp)) * step(yp, 1.0)
                + DB_SBLUR_H_tex(vec2(pos.x, yn)) * step(0.0, yn)) * gwA[i];
    }
    for (int i = 0; i < 7; i++) {
        float yp = pos.y + goB[i] * pt.y;
        float yn = pos.y - goB[i] * pt.y;
        sum_s2 += (DB_SBLUR2_H_tex(vec2(pos.x, yp)) * step(yp, 1.0)
                 + DB_SBLUR2_H_tex(vec2(pos.x, yn)) * step(0.0, yn)) * gwB[i];
    }

    sum_s  *= DB_SBLURA_INV_WSUM;
    sum_s2 *= DB_SBLURB_INV_WSUM;

    vec3 s  = sum_s.rgb  / max(sum_s.a,  1e-4);
    vec3 s2 = sum_s2.rgb / max(sum_s2.a, 1e-4);
    float conf = smoothstep(0.06, 0.20, min(sum_s.a, sum_s2.a));
    vec4 m = DB_MED_tex(DB_MED_pos);
    vec3 c = (2.0 * s - s2 - m.rgb) * conf;
    return vec4(vec3(0.5) + c, max(max(abs(c.r), abs(c.g)), abs(c.b)) * m.a * m.a);
}

//!HOOK MAIN
//!BIND DB_CORR
//!BIND DB_SCORR
//!SAVE DB_ENV_H
//!WIDTH DB_CORR.w
//!HEIGHT DB_CORR.h
//!DESC Debanditv1: Contamination Env H

// =============================================================================
// PASS 11/12: CONTAMINATION ENVELOPE — separable max of |c| over +-8 texels
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
// v1.9: separate luma/chroma envelopes — so luma curvature contamination
// (a shaded face) cannot veto a chroma correction whose own neighborhood
// is clean, and vice versa. v1.10: FOUR envelopes ride the pair, the
// same two per refit scale: x/y = wide (sigma 48) luma/chroma, z/w =
// short (sigma 12) luma/chroma — the apply blends toward the short ramp
// where only ITS envelope is clean (bulk-edge rescue). Each luma seed is
// the corresponding CORR alpha verbatim (fp32-exact); each chroma seed
// is that exact seed times the opponent RATIO of the decoded c: the
// ratio is bounded in [0, 2] by construction (|c_op| <= 2 max|c|), so
// the 0.5 + c decode's fp16 ulp only perturbs the ratio multiplicatively
// (~10% at half-code |c|) on the wide 0.5-1.0x cthr chroma knee — never
// additively.

const vec3 DB_ENVH_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB_CORR_pt;
    vec2 pos = DB_CORR_pos;
    vec4 m = vec4(0.0);
    for (int i = -8; i <= 8; i++) {
        vec2 tp = pos + vec2(float(i) * pt.x, 0.0);
        vec4 tL = DB_CORR_tex(tp);
        vec4 tS = DB_SCORR_tex(tp);
        vec3 cL = tL.rgb - vec3(0.5);
        vec3 cS = tS.rgb - vec3(0.5);
        vec3 oL = cL - vec3(dot(cL, DB_ENVH_W709));
        vec3 oS = cS - vec3(dot(cS, DB_ENVH_W709));
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
//!BIND DB_ENV_H
//!SAVE DB_ENV
//!WIDTH DB_ENV_H.w
//!HEIGHT DB_ENV_H.h
//!DESC Debanditv1: Contamination Env V

vec4 hook() {
    vec2 pt = DB_ENV_H_pt;
    vec2 pos = DB_ENV_H_pos;
    vec4 m = DB_ENV_H_tex(pos);
    for (int i = 1; i <= 8; i++) {
        m = max(m, DB_ENV_H_tex(pos + vec2(0.0, float(i) * pt.y)));
        m = max(m, DB_ENV_H_tex(pos - vec2(0.0, float(i) * pt.y)));
    }
    return m;
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND DB_DS
//!BIND DB_MED
//!SAVE DB_RANGE_H
//!WIDTH DB_MED.w
//!HEIGHT DB_MED.h
//!DESC Debanditv1: Flatness Range H

// =============================================================================
// PASS 13/14: FLATNESS RANGE + NOISE FIELD
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

const vec3 DB_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB_MED_pt;
    vec2 pos = DB_MED_pos;

    vec4 center = DB_MED_tex(pos);
    float lum_c = dot(center.rgb, DB_W709);
    float lo = lum_c;
    float hi = lum_c;

    for (int i = 1; i <= 8; i++) {
        vec4 tp = DB_MED_tex(pos + vec2(float(i) * pt.x, 0.0));
        vec4 tn = DB_MED_tex(pos - vec2(float(i) * pt.x, 0.0));
        float op = step(0.5, tp.a);
        float on = step(0.5, tn.a);
        lo = min(lo, min(mix(1e3, dot(tp.rgb, DB_W709), op),
                         mix(1e3, dot(tn.rgb, DB_W709), on)));
        hi = max(hi, max(mix(-1e3, dot(tp.rgb, DB_W709), op),
                         mix(-1e3, dot(tn.rgb, DB_W709), on)));
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
    float s0 = DB_DS_tex(pos - vec2(2.0 * pt.x, 0.0)).a - lum_c;
    float s1 = DB_DS_tex(pos - vec2(pt.x, 0.0)).a - lum_c;
    float s2 = DB_DS_tex(pos).a - lum_c;
    float s3 = DB_DS_tex(pos + vec2(pt.x, 0.0)).a - lum_c;
    float s4 = DB_DS_tex(pos + vec2(2.0 * pt.x, 0.0)).a - lum_c;

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
//!BIND DB_RANGE_H
//!SAVE DB_RANGE
//!WIDTH DB_RANGE_H.w
//!HEIGHT DB_RANGE_H.h
//!DESC Debanditv1: Flatness Range V

// The MAD field max-dilates over +-2 here (V axis); the apply pass does the
// matching +-2 H dilation, so the snap regime is stabilized isotropically
// against per-frame wobble at grain-region borders.

vec4 hook() {
    vec2 pt = DB_RANGE_H_pt;
    vec2 pos = DB_RANGE_H_pos;

    vec4 mm = DB_RANGE_H_tex(pos);

    for (int i = 1; i <= 8; i++) {
        vec4 sp = DB_RANGE_H_tex(pos + vec2(0.0, float(i) * pt.y));
        vec4 sn = DB_RANGE_H_tex(pos - vec2(0.0, float(i) * pt.y));
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
//!BIND DB_MED
//!SAVE DB_RANGEC_H
//!WIDTH DB_MED.w
//!HEIGHT DB_MED.h
//!DESC Debanditv1: Chroma Range H

// =============================================================================
// PASS 15/16 (v1.9): CHROMA FLATNESS RANGE + CHROMA NOISE FIELD
// =============================================================================
// Separable trust-masked min/max of the median field's OPPONENT axes
// (o1 = B - Y, o2 = R - Y, both in RGB-projected units) over +-8 texels —
// the chroma analogue of the luma flatness range, feeding the chroma
// flatness gate at apply. The same M.a trust mask excludes line art and
// text; a rare equal-luma pure-chroma structure is unmasked and simply
// closes this gate around itself (conservative, correct). Stored as
// o * 0.5 + 0.5 so both signs survive a unorm FBO fallback (the DB_CORR
// pattern); the V pass decodes and collapses to plain ranges.

const vec3 DB_RC_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB_MED_pt;
    vec2 pos = DB_MED_pos;

    vec4 center = DB_MED_tex(pos);
    float y_c = dot(center.rgb, DB_RC_W709);
    vec2 o_c = vec2(center.b - y_c, center.r - y_c);
    vec2 lo = o_c;
    vec2 hi = o_c;

    for (int i = 1; i <= 8; i++) {
        vec4 tp = DB_MED_tex(pos + vec2(float(i) * pt.x, 0.0));
        vec4 tn = DB_MED_tex(pos - vec2(float(i) * pt.x, 0.0));
        float yp = dot(tp.rgb, DB_RC_W709);
        float yn = dot(tn.rgb, DB_RC_W709);
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
//!BIND DB_DS
//!BIND DB_MED
//!BIND DB_RANGEC_H
//!SAVE DB_RANGEC
//!WIDTH DB_RANGEC_H.w
//!HEIGHT DB_RANGEC_H.h
//!DESC Debanditv1: Chroma Range V

// V-axis min/max reduction, then collapse to ranges: x = o1 range,
// y = o2 range (RGB-projected codes; the raw range can theoretically
// reach ~1.86 — past unorm 1.0 — but any clamp engages ~5x beyond the
// widest possible fully-closed flatness knee, so a unorm FBO fallback
// changes nothing). Z carries the CHROMA noise field: MAD of the
// full-res 4x4 block's OPPONENT deviation against the median field. The
// opponent projection cancels Y-plane encode dither EXACTLY (Y moves
// R/G/B by the same coefficient, so common-mode vanishes) — this field
// reads chroma-plane noise and chroma grain only, which is what the
// chroma snap regime must discriminate on. No +-2 pre-dilation here
// (channel budget); the apply pass reads it through a 13-tap max
// ({0,+-1,+-2} both axes plus the +-1 diagonals) — weaker than the luma
// path's full 5x5, accepted because the chroma knees are plateau-placed
// with a wider margin (see apply) so the blend zone this stabilizes is
// narrower.
// W carries the CHROMA STRUCTURE field (v1.9 audit F2): opponent-MAD
// magnitude alone cannot tell faint COHERENT colored texture (haze, a
// soft colored rim — luma-flat, so the shared luma structure gate never
// sees it) from chroma dither; without this the chroma snap would absorb
// it. Mirror of the luma term on the V axis: signed deviations of the
// raw plateau field's opponent components against the median field's,
// med5 per axis — UNSIGNED median = raw activity, |SIGNED median| x 2.5
// = coherence (dither self-cancels, structure survives; banding contours
// contaminate only 1-2 samples of 5 and stay invisible). Max over axes.

const vec3 DB_RCV_W709 = vec3(0.2126, 0.7152, 0.0722);

vec4 hook() {
    vec2 pt = DB_RANGEC_H_pt;
    vec2 pos = DB_RANGEC_H_pos;

    vec4 mm = DB_RANGEC_H_tex(pos);
    vec2 lo = mm.xz;
    vec2 hi = mm.yw;

    for (int i = 1; i <= 8; i++) {
        vec4 sp = DB_RANGEC_H_tex(pos + vec2(0.0, float(i) * pt.y));
        vec4 sn = DB_RANGEC_H_tex(pos - vec2(0.0, float(i) * pt.y));
        lo = min(lo, min(sp.xz, sn.xz));
        hi = max(hi, max(sp.yw, sn.yw));
    }
    // Decode the 0.5-offset affine encode: range = (hi_e - lo_e) * 2.
    vec2 rng = (hi - lo) * 2.0;

    vec4 center = DB_MED_tex(DB_MED_pos);
    vec2 fpt = HOOKED_pt;
    float mad_c = 0.0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 off = vec2(float(x) - 1.5, float(y) - 1.5) * fpt;
            vec3 d = HOOKED_tex(HOOKED_pos + off).rgb - center.rgb;
            vec3 d_op = d - vec3(dot(d, DB_RCV_W709));
            mad_c += max(max(abs(d_op.r), abs(d_op.g)), abs(d_op.b));
        }
    }
    mad_c *= (1.0 / 16.0);

    // Chroma structure/coherence: opponent components of DB_DS (alpha IS
    // its luma) vs the median field's, five V taps.
    float y_c = dot(center.rgb, DB_RCV_W709);
    vec2 o_c = vec2(center.b - y_c, center.r - y_c);
    vec4 d0t = DB_DS_tex(DB_DS_pos - vec2(0.0, 2.0 * pt.y));
    vec4 d1t = DB_DS_tex(DB_DS_pos - vec2(0.0, pt.y));
    vec4 d2t = DB_DS_tex(DB_DS_pos);
    vec4 d3t = DB_DS_tex(DB_DS_pos + vec2(0.0, pt.y));
    vec4 d4t = DB_DS_tex(DB_DS_pos + vec2(0.0, 2.0 * pt.y));
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
//!BIND HOOKED
//!BIND DB_MED
//!BIND DB_CORR
//!BIND DB_SCORR
//!BIND DB_ENV
//!BIND DB_RANGE
//!BIND DB_RANGEC
//!DESC Debanditv1: Apply

// =============================================================================
// PASS 17: APPLY (full resolution)
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
// approximation just shifts the knee slightly; tune db_thr, not this.
#define DB_CODE (1.0 / 255.0)

// Snap regime knees on the DILATED MAD, in code values: sigma-1 encode
// dither dilates to ~1.1 — the LO knee sits ABOVE it so dither lands on
// the saturated plateau (snap = 1, temporally stable against per-frame
// MAD wobble; audit-found breathing risk at the old knee foot). Sigma-2
// grain dilates to ~2.2, fully outside — grain untouched. The continuum
// around sigma ~1.5 blends proportionally.
#define DB_SNAP_MAD_LO  1.3
#define DB_SNAP_MAD_HI  2.1
// In DARKS the knees tighten: faint sub-code signal in dark regions is
// far more likely deliberate texture (smoke, haze, shadow detail behind
// transparency) than encode dither, and CelFlare never expands darks so
// residual contours there are not amplified downstream. Clean dark flats
// (MAD well under the dark knee) keep FULL snap — dark banding is the
// most visible kind, and this bias must not soften it. The dark HI knee
// leaves DITHERED clean dark gradients roughly half-snapped (audit: the
// old 1.2 floored them at ~0 with no recovery path) — db_dither scales
// from there in both directions of taste; coherent texture is protected
// independently by the structure and envelope gates.
#define DB_SNAP_MAD_LO_D  0.6
#define DB_SNAP_MAD_HI_D  1.6
// Dark ramp in M-field luma (gamma domain): fully dark-biased below
// ~26/255, fully bright-calibrated above ~77/255.
#define DB_DARK_LO      0.10
#define DB_DARK_HI      0.30
// Structure-field knee (codes): kills the snap on coherent micro-texture
// (hatching, fine pattern) whose full-res MAD mimics dither, and on
// gradients too steep for banding to be visible. Dither reads ~0.2 here.
// v1.11: the static knee is the CEILING of an SNR-adaptive knee (scaled
// by db_snr): tex_lo = ratio x local MAD, clamped to [floor, static].
// Rationale: the 0.45 floor exists so 8-bit dither cannot close the
// gate — but dither reads ~0.2 ONLY on dithered sources. On quiet
// high-bit-depth content (10-bit BD anime, field 2026-07-10: 30-50% of
// texture energy eaten inside snap zones) the noise floor is ~0.1 and
// real texture lives at 0.1-0.35 codes, under the static knee. Scaling
// the knee to the measured MAD makes the gate a signal-to-NOISE test:
// texture must merely exceed the local noise floor to be protected.
// Banding contours stay invisible to the field at ANY knee (they
// contaminate only 1-2 of 5 median samples), so clean flats still
// deband. The floor keeps sampling noise (~0.05) from closing it.
#define DB_TEX_LO       0.45
#define DB_TEX_SNR      0.4
#define DB_TEX_LO_MIN   0.18
// Snap authority = DB_SNAP_REACH * db_thr + local MAD, in code values:
// enough to flatten a half-step quantization offset (0.75 codes at the
// default db_thr = 2) PLUS the measured local noise (so an open snap can
// always finish absorbing the dither instead of stalling half-way), while
// staying physically unable to erase structure beyond sub-step scale plus
// noise. Scales with db_thr so the brutal-source knob raises real reach.
#define DB_SNAP_REACH   0.375
// Never darken where the plateau field sits AT the signal ceiling — a
// clipped region gets expansion gradation downstream (CelFlare rule #1),
// not attenuation here. Deliberately tight (engages only above ~247/255):
// near-white banded skies below that need both correction signs.
#define DB_CLIP_LO      0.97
#define DB_CLIP_HI      0.995
// ---- v1.9 chroma path ----
// Chroma quantities live in RGB-PROJECTED units: one YCbCr chroma code
// spans ~2.11 RGB codes on its dominant axis (255/224 * 1.8556 for Cb->B;
// Cr->R is 1.79 — the max is used so both axes gate TIGHT, never loose),
// so every chroma knee/reach derives from db_thr * this gain — db_thr
// keeps its "step ceiling in source-plane codes" meaning on both paths.
#define DB_CTHR_GAIN    2.11
// Chroma snap knees on the plus-dilated opponent MAD, in RGB codes. The
// opponent projection already cancels Y-plane dither exactly, so what
// remains is chroma-plane noise and the chroma contours themselves: a
// full 1-chroma-code step contour crossing a block reads ~1.0 here
// (2.1 RGB codes x the crossing fraction) — the LO knee sits ABOVE that
// so banding contours land on the full-snap plateau (a knee below 1.0
// would half-close the snap exactly on the contours it must remove).
// Chroma-plane sigma-1 dither reads ~0.8 (upscale-smoothed): plateau.
// Chromatic film grain sigma 1.5+ reads past 2.0: fully outside. NO dark
// tightening (unlike luma): the dark-texture prior does not transfer —
// Y-dither is already projected out, faint dark chroma texture is rare,
// and dark chroma banding (night scenes) is the motivating case and must
// keep full snap.
#define DB_SNAP_MAD_LO_C  1.2
#define DB_SNAP_MAD_HI_C  2.0

vec4 hook() {
    vec4 orig = HOOKED_tex(HOOKED_pos);

    vec4 M = DB_MED_tex(DB_MED_pos);

    // 5-tap tent over the range/noise fields: softens gate transitions AND
    // completes the +-2 texel MAD dilation isotropically (V half in PASS 9).
    vec2 rpt = DB_RANGE_pt;
    vec4 mm  = DB_RANGE_tex(DB_RANGE_pos);
    vec4 mmL = DB_RANGE_tex(DB_RANGE_pos - vec2(2.0 * rpt.x, 0.0));
    vec4 mmR = DB_RANGE_tex(DB_RANGE_pos + vec2(2.0 * rpt.x, 0.0));
    vec4 mmU = DB_RANGE_tex(DB_RANGE_pos - vec2(0.0, 2.0 * rpt.y));
    vec4 mmD = DB_RANGE_tex(DB_RANGE_pos + vec2(0.0, 2.0 * rpt.y));
    float range_soft = (mm.y - mm.x) * 0.4
                     + ((mmL.y - mmL.x) + (mmR.y - mmR.x)
                     +  (mmU.y - mmU.x) + (mmD.y - mmD.x)) * 0.15;
    // +-1 taps close the H-dilation gap (audit: {-2,0,+2} alone skips a MAD
    // spike one texel away; the V pass covers +-2 gaplessly on its axis).
    float mad1 = max(max(DB_RANGE_tex(DB_RANGE_pos - vec2(rpt.x, 0.0)).z,
                         DB_RANGE_tex(DB_RANGE_pos + vec2(rpt.x, 0.0)).z),
                     max(DB_RANGE_tex(DB_RANGE_pos - vec2(0.0, rpt.y)).z,
                         DB_RANGE_tex(DB_RANGE_pos + vec2(0.0, rpt.y)).z));
    float mad_codes = max(max(mm.z, mad1),
                          max(max(mmL.z, mmR.z), max(mmU.z, mmD.z))) / DB_CODE;
    float tex_codes = (mm.w + mmL.w + mmR.w + mmU.w + mmD.w) * 0.2 / DB_CODE;

    // Chroma range/noise fields, same tent geometry; the MAD max adds the
    // +-1 diagonals (13 taps total) so an isolated diagonal chroma-grain
    // filament cannot thread between the plus arms and breathe the snap
    // (v1.9 audit F5) — still short of the luma path's full 5x5, see the
    // Chroma Range V note.
    vec2 cpt = DB_RANGEC_pt;
    vec4 cc  = DB_RANGEC_tex(DB_RANGEC_pos);
    vec4 ccL = DB_RANGEC_tex(DB_RANGEC_pos - vec2(2.0 * cpt.x, 0.0));
    vec4 ccR = DB_RANGEC_tex(DB_RANGEC_pos + vec2(2.0 * cpt.x, 0.0));
    vec4 ccU = DB_RANGEC_tex(DB_RANGEC_pos - vec2(0.0, 2.0 * cpt.y));
    vec4 ccD = DB_RANGEC_tex(DB_RANGEC_pos + vec2(0.0, 2.0 * cpt.y));
    vec2 rangec_soft = cc.xy * 0.4
                     + (ccL.xy + ccR.xy + ccU.xy + ccD.xy) * 0.15;
    float madc1 = max(max(DB_RANGEC_tex(DB_RANGEC_pos - vec2(cpt.x, 0.0)).z,
                          DB_RANGEC_tex(DB_RANGEC_pos + vec2(cpt.x, 0.0)).z),
                      max(DB_RANGEC_tex(DB_RANGEC_pos - vec2(0.0, cpt.y)).z,
                          DB_RANGEC_tex(DB_RANGEC_pos + vec2(0.0, cpt.y)).z));
    float madc2 = max(max(DB_RANGEC_tex(DB_RANGEC_pos + vec2( cpt.x,  cpt.y)).z,
                          DB_RANGEC_tex(DB_RANGEC_pos + vec2( cpt.x, -cpt.y)).z),
                      max(DB_RANGEC_tex(DB_RANGEC_pos + vec2(-cpt.x,  cpt.y)).z,
                          DB_RANGEC_tex(DB_RANGEC_pos + vec2(-cpt.x, -cpt.y)).z));
    float madc_codes = max(max(cc.z, max(madc1, madc2)),
                           max(max(ccL.z, ccR.z), max(ccU.z, ccD.z))) / DB_CODE;
    float texc_codes = (cc.w + ccL.w + ccR.w + ccU.w + ccD.w) * 0.2 / DB_CODE;

    // The complete LF corrections, precomputed at 1/4 res: wide (sigma 48)
    // and short (sigma 12) refits, c = S_refit - M each.
    vec3 c  = DB_CORR_tex(DB_CORR_pos).rgb - vec3(0.5);
    vec3 cs = DB_SCORR_tex(DB_SCORR_pos).rgb - vec3(0.5);

    // ---- v1.10 multi-scale blend, per axis ----
    // Envelope validities of each scale (the envelope IS the gate 4
    // quantity — see below): near a bulk edge the wide ramp's envelope
    // condemns a ~25-texel reach while the short ramp stays clean beyond
    // ~6, so blending toward the short refit where only IT is valid
    // shrinks the unfixable collar from ~130px to ~55px. Wide is
    // preferred wherever valid (it spans the plateaus the short one
    // cannot). db_ms = 0 collapses every blend to the wide field: exact
    // v1.9 behavior (and with db_chroma = 0, exact v1.8).
    float cthr = db_thr * DB_CTHR_GAIN;
    vec4 env_codes = DB_ENV_tex(DB_ENV_pos) / DB_CODE;
    float v_Ly = 1.0 - smoothstep(0.5 * db_thr, db_thr, env_codes.x);
    float v_Sy = 1.0 - smoothstep(0.5 * db_thr, db_thr, env_codes.z);
    float v_Lc = 1.0 - smoothstep(0.5 * cthr, cthr, env_codes.y);
    float v_Sc = 1.0 - smoothstep(0.5 * cthr, cthr, env_codes.w);

    const vec3 W709 = vec3(0.2126, 0.7152, 0.0722);
    float cy_L = dot(c,  W709);
    float cy_S = dot(cs, W709);
    vec3 cop_L = c  - vec3(cy_L);
    vec3 cop_S = cs - vec3(cy_S);
    // The substitution weight requires the short scale to be valid ON ITS
    // OWN TERMS: pref_S = (1 - v_L) * v_S. Falling back on (1 - v_L)
    // alone swaps in the short refit's BIAS wherever both scales are
    // contaminated (hair strand clusters: v_S ~ 0 too) — field-measured
    // as a total-weight collapse 0.49 -> 0.01 there. With both invalid
    // the blend stays on the wide field and the envelope gate (below)
    // kills it, exactly as v1.9 did.
    float pref_Sy = (1.0 - v_Ly) * v_Sy;
    float pref_Sc = (1.0 - v_Lc) * v_Sc;
    float cy_eff  = mix(cy_L,  cy_S,  pref_Sy * db_ms);
    vec3 cop_eff  = mix(cop_L, cop_S, pref_Sc * db_ms);
    vec3 c_eff    = vec3(cy_eff) + cop_eff;

    // Gate 1: correction amplitude, max across channels so a big miss in any
    // channel kills the whole correction (no per-channel hue twist). The
    // knee is HALF of db_thr: db_thr is a STEP-size ceiling, and the true
    // correction for a staircase never exceeds ~half the step — anything
    // larger is remote-structure bias or mid-scale curvature the refit
    // could not cancel (field-found: the 2*db_thr knee passed 1.5-2.5-code
    // pseudo-corrections on murky large-amplitude soft structure).
    float c_codes = max(max(abs(c_eff.r), abs(c_eff.g)), abs(c_eff.b)) / DB_CODE;
    float w_amp = 1.0 - smoothstep(0.5 * db_thr, db_thr, c_codes);

    // Gate 2: local flatness — median-field luma range over +-32px.
    float range_codes = range_soft / DB_CODE;
    float w_flat = 1.0 - smoothstep(db_flat, 2.0 * db_flat, range_codes);

    // Gate 3: structure — med5's mean-bias field; where the median field
    // is not a valid ramp estimate (dense micro-texture), apply nothing.
    // v1.11: SNR-adaptive knee (see the define block) — tightens toward
    // the measured noise floor on quiet sources so faint coherent
    // texture is protected from snap absorption; clamps to the static
    // knee on dithered content (bit-identical there at any db_snr).
    float tex_lo = mix(DB_TEX_LO,
                       clamp(DB_TEX_SNR * mad_codes, DB_TEX_LO_MIN, DB_TEX_LO),
                       db_snr);
    float w_tex = 1.0 - smoothstep(tex_lo, 2.0 * tex_lo, tex_codes);

    // Gate 4: contamination envelope — if any point within +-32px needs a
    // correction beyond banding scale, that scale's ramp estimate is
    // untrustworthy there. With the multi-scale blend the gate is the
    // validity of the field ACTUALLY BLENDED, mix(v_L, v_S, pref_S) — a
    // plain max(v_L, v_S) over-reports confidence in the graduated
    // mid-zone where c_eff is still mostly the less-trusted wide field
    // (v1.10 audits, both passes). At db_ms = 0 it degenerates to the
    // wide validity alone.
    float w_env = mix(v_Ly, mix(v_Ly, v_Sy, pref_Sy), db_ms);

    float w = w_amp * w_flat * w_tex * w_env * db_strength;

    // ---- v1.9 chroma path: opponent decomposition ----
    // The luma-path component rides every gate above exactly as v1.8
    // applied the full c (the amplitude gate stays max-channel: when c is
    // luma-dominated that is the v1.8 conservatism, and when it is
    // chroma-dominated the luma part is ~0 anyway). The opponent part
    // gets chroma-native gates in chroma units (db_thr x DB_CTHR_GAIN).
    // db_chroma MIGRATES the opponent component between the paths: at 0
    // it rides the luma path exactly as v1.8 routed it (bit-exact
    // fallback), at 1 it is gated purely by its own fields.
    vec3 c_op = cop_eff;
    vec3 c_lp = vec3(cy_eff) + c_op * (1.0 - db_chroma);

    // Chroma gate 1: amplitude — a staircase correction never exceeds
    // ~half the step, in the chroma step's own units.
    float cop_codes = max(max(abs(c_op.r), abs(c_op.g)), abs(c_op.b)) / DB_CODE;
    float w_amp_c = 1.0 - smoothstep(0.5 * cthr, cthr, cop_codes);
    // Chroma gate 2: opponent-axis flatness over +-32px.
    float rangec_codes = max(rangec_soft.x, rangec_soft.y) / DB_CODE;
    float w_flat_c = 1.0 - smoothstep(db_flat_c, 2.0 * db_flat_c, rangec_codes);
    // Chroma gate 3: structure, two layers. The shared luma term (w_tex)
    // covers med5-invalidity — where the median is not a valid ramp
    // estimate, no field derived from M is, chroma included. The chroma
    // term (w_texc, audit F2) covers coherent COLORED texture the luma
    // field cannot see (luma-flat haze, soft colored rims): same knees —
    // chroma dither's self-cancelled residue reads ~0.2 here too.
    // The CHROMA knee is EXEMPT from SNR adaptation (audit v1.11 F2):
    // madc is Y-projected, so it reads low even on 8-bit luma-dithered
    // content and the knee would tighten exactly where the v1.9 chroma
    // banding fix operates — measured: the face-scene chroma deband
    // halves at the luma floor, while the 10-bit texture-retention win is
    // entirely luma-driven (retention identical with or without chroma
    // adaptation). Weak upside, concrete downside: static knee stays.
    float w_texc = 1.0 - smoothstep(DB_TEX_LO, 2.0 * DB_TEX_LO, texc_codes);
    // Chroma gate 4: chroma contamination envelope, blend-weighted like
    // gate 4 above.
    float w_env_c = mix(v_Lc, mix(v_Lc, v_Sc, pref_Sc), db_ms);

    float w_c = w_amp_c * w_flat_c * w_tex * w_texc * w_env_c
              * db_strength * db_chroma;

    // Regime blend: clean gradients snap to the ramp itself (removes the
    // contour, absorbs the dither that masked it); grained regions take the
    // LF-only correction (grain rides through bit-exact).
    float lum = dot(M.rgb, vec3(0.2126, 0.7152, 0.0722));
    float dark_w = 1.0 - smoothstep(DB_DARK_LO, DB_DARK_HI, lum);
    float mad_lo = mix(DB_SNAP_MAD_LO, DB_SNAP_MAD_LO_D, dark_w);
    float mad_hi = mix(DB_SNAP_MAD_HI, DB_SNAP_MAD_HI_D, dark_w);
    float snap = (1.0 - smoothstep(mad_lo, mad_hi, mad_codes)) * db_dither;
    // Chroma snap: keyed to the OPPONENT MAD (Y-plane dither projected
    // out), flat knees — see the define block. db_dither scales both
    // regimes: it is the one taste knob for contour-vs-texture.
    float snap_c = (1.0 - smoothstep(DB_SNAP_MAD_LO_C, DB_SNAP_MAD_HI_C,
                                     madc_codes)) * db_dither;
    // The +MAD authority exists to finish absorbing dither; grant it only in
    // proportion to dither-confidence (snap), or blend-zone texture (fine
    // hatching, weak grain) gets pulled ~1 code toward the refit.
    // (An env-scaled snap reach was tried and measured INERT here: the
    // texture itself inflates |c| and thus the envelope, licensing its
    // own absorption — amplitude fields cannot make this distinction.
    // DC-coherence and structure-tensor discriminators were also built
    // and reverted this session: see CHANGELOG v1.11 before retrying.)
    float auth   = (DB_SNAP_REACH * db_thr + mad_codes * snap) * DB_CODE;
    float auth_c = (DB_SNAP_REACH * cthr + madc_codes * snap_c) * DB_CODE;
    // Snap target reconstructed as c_eff + M (= the blended S_refit): the
    // stored M's quantization cancels between this M and the -M inside
    // each c. The total pull t is decomposed BEFORE clamping so each axis
    // spends only its own authority (a luma-clamp cannot eat chroma reach
    // or vice versa).
    vec3 t = c_eff + M.rgb - orig.rgb;
    vec3 t_y  = vec3(dot(t, vec3(0.2126, 0.7152, 0.0722)));
    vec3 t_op = t - t_y;
    vec3 t_lp = t_y + t_op * (1.0 - db_chroma);
    vec3 r_lp = clamp(t_lp, vec3(-auth),   vec3(auth));
    vec3 r_op = clamp(t_op, vec3(-auth_c), vec3(auth_c));
    vec3 corr_l = mix(c_lp, r_lp, snap) * w;
    vec3 corr_c = mix(c_op, r_op, snap_c) * w_c;

    // Clip guard: one-sided near the ceiling — never pull a clipped/near-
    // clipped plateau down. LUMA-PATH TERM ONLY (v1.9 audit F1): the
    // chroma term is luma-neutral by construction, so it cannot darken a
    // plateau — but a per-channel max() on a signed opponent vector clamps
    // its negative channels, injecting luma and twisting hue exactly on
    // near-clip colored banding, which CelFlare then expands. At
    // db_chroma = 0 the opponent component rides corr_l and receives the
    // v1.8 per-channel guard verbatim.
    float clip_w = smoothstep(DB_CLIP_LO, DB_CLIP_HI, lum);
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
#else
    return vec4(orig.rgb + corr, orig.a);
#endif
}
