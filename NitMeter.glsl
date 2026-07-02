// NitMeter v1.1 — HDR peak-nit analysis overlay for mpv (gpu-next)
// Copyright (C) 2026 Agust Ari · GPL-3.0
//
// Companion dev tool for CelFlare: decodes the PQ frame and displays
// absolute-nit statistics as an on-screen panel, plus an optional
// full-frame false-color heatmap. Panel/readout design borrows from
// Lilium's HDR analysis (ReShade) and DesktopLUT's analysis overlay.
//
// WHAT IT MEASURES
//   Per-pixel light level = pixel CLL: PQ-EOTF(max(R,G,B)) * 10000 — the
//   same convention as MaxCLL/MaxFALL in HDR10 static metadata.
//   Panel rows (7-seg letter labels):
//     P (white)  — frame peak CLL in nits (sample-and-hold, ~0.5 s refresh)
//     H (yellow) — peak hold: held ~3 s after the last new peak, then decays
//     A (cyan)   — FALL: frame-average CLL (letterbox bars included, same as
//                  the MaxFALL averaging; ~0.5 s refresh; dilution on scope)
//     C (orange) — session MaxCLL: running max frame peak since toggle-on
//     F (green)  — session MaxFALL: running max FALL since toggle-on
//   Values under 100 nits gain a tenths digit (e.g. 94.3). The small square
//   on the P row is the display-ceiling indicator vs nitmeter_target:
//   green < 95% of target, yellow approaching, red = content peak exceeds
//   the display (mpv/display will compress or clip).
//   Histogram — screen-area distribution of 16x16-block mean CLL on a
//   log2 axis from 1 to 10000 nits. Grey ticks at 100 / 203 / 1000 / 4000
//   nits; bright line = current frame peak. Bars take the heatmap ramp hue
//   of their bin (grey below 100).
//
// REQUIREMENTS / USAGE
//   The frame at MAIN must already be PQ-encoded when this shader runs:
//   load it AFTER CelFlare (which emits PQ in-shader), or on native PQ HDR
//   content without CelFlare (HDR10/HDR10+/DV — DV is reshaped to plain PQ
//   before MAIN). HLG is NOT supported (deliberately — PQ only). On plain
//   SDR content the numbers are meaningless (gamma decoded as PQ); the
//   NITMETER_SDR_GUARD tripwire below blanks the panel when it detects that.
//
//   Convenient driving is a small lua script cycling panel -> heatmap ->
//   off on one key: it appends this file to glsl-shaders and sets the
//   nitmeter_mode shader param via glsl-shader-opts. Manual equivalent:
//     change-list glsl-shader-opts append nitmeter_mode=2
//     change-list glsl-shaders append "~~/shaders/NitMeter.glsl"
//
// PARAMS (runtime, via glsl-shader-opts; NB: never write the magic
// comment-marker sequence itself inside a comment — the libplacebo section
// splitter is a raw substring search, not line-anchored, and it WILL split
// the file mid-comment and reject the whole shader)
//   nitmeter_mode:
//     1 = panel only (default)
//     2 = false-color heatmap by pixel CLL + panel. Lilium-style ramp:
//       <100 grey (self-luminous)      100-203 cyan->green
//       203-400 green->yellow          400-1000 yellow->red
//       1000-4000 red->pink            4000-10000 pink->blue
//       at signal ceiling (~10000)     bright red flag
//     3 = heatmap encoded as plain gamma-2.2 SDR instead of PQ, for
//       screenshots/encodes to share on SDR screens. Looks dim on the live
//       PQ-passthrough display — capture mode, not viewing mode.
//     "Off" is not a mode: unload the shader (the script does this).
//   nitmeter_target (float, default 1000) — display peak in nits for the
//     P-row ceiling indicator; match your target-peak. Live uniform, no
//     recompile: change-list glsl-shader-opts append nitmeter_target=800
//
// KNOBS (PASS 3, "MAIN TUNING")
//   NITMETER_SDR_GUARD (default 1) — SDR-input tripwire. A user shader
//     cannot query the frame's tagged transfer, so detection is heuristic:
//     any clipped SDR white misread as PQ decodes to 10000 nits, while real
//     masters stay <= ~4000 and CelFlare output sits far below that. If the
//     smoothed MAX 16x16-BLOCK-AVERAGE exceeds NM_SDR_SUSPECT_NITS (8000,
//     PASS 2) the panel blanks its readouts to "---" and draws a red border
//     instead of reporting nonsense. Block-mean, not pixel max, so a lone
//     YUV-overshoot pixel on legit HDR can't false-trip it; an SDR source
//     trips as soon as any 16x16 clipped-white area is on screen ~0.5 s.
//     Session MaxCLL/MaxFALL reset while tripped (no garbage poisoning).
//     Disable if you ever feed legit 8000+ nit material.
//   NITMETER_CORNER — 0 TL / 1 TR (default) / 2 BL / 3 BR
//   NITMETER_SCALE  — panel size multiplier (1.0 = ~136x182 px at 1080p)
//
// The two measurement passes run BEFORE the overlay draw, so the panel and
// heatmap never contaminate their own statistics.

//!PARAM nitmeter_mode
//!DESC 1 = panel, 2 = heatmap + panel, 3 = heatmap (gamma-2.2 SDR export) + panel
//!TYPE DEFINE
//!MINIMUM 1
//!MAXIMUM 3
1

//!PARAM nitmeter_target
//!DESC display peak nits for the P-row ceiling indicator (match target-peak)
//!TYPE float
//!MINIMUM 100.0
//!MAXIMUM 10000.0
1000.0

//!BUFFER NITMETER_STATE
//!VAR float nm_peak
//!VAR float nm_fall
//!VAR float nm_hold
//!VAR float nm_hold_age
//!VAR float nm_sdr
//!VAR float nm_maxcll
//!VAR float nm_maxfall
//!VAR float nm_show_peak
//!VAR float nm_show_fall
//!VAR float nm_show_ctr
//!VAR float nm_hist[24]
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!SAVE NITMETER_BLK
//!WIDTH HOOKED.w 15 + 16 /
//!HEIGHT HOOKED.h 15 + 16 /
//!COMPUTE 16 16
//!DESC NitMeter: block reduce (16x16 max/mean CLL)

// ST.2084 PQ, exact forms only (this is a measurement tool — no fast
// approx). Duplicated in every NitMeter pass: separate translation units,
// keep in sync.
float pq_eotf_norm(float e) {   // PQ code -> linear, 1.0 = 10000 nits
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    // clamp: codes past 1.0 (YUV->RGB overshoot) cross the EOTF denominator's
    // zero at e~1.009 and explode to inf; overrange = signal ceiling (10k nits)
    float p = pow(clamp(e, 0.0, 1.0), 1.0 / m2);
    return pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}
float pq_oetf_norm(float L) {   // linear (1.0 = 10000 nits) -> PQ code
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float Lm = pow(max(L, 0.0), m1);
    return pow((c1 + c2 * Lm) / (1.0 + c3 * Lm), m2);
}

void hook() {
    ivec2 bpos = ivec2(gl_GlobalInvocationID.xy);
    vec2 base = vec2(bpos) * 16.0;
    float emax = 0.0;   // block max PQ code of per-pixel max(R,G,B)
    float lsum = 0.0;   // block sum of linear CLL (1.0 = 10000 nits)
    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 16; i++) {
            vec2 uv = (base + vec2(float(i), float(j)) + 0.5) * HOOKED_pt;
            // clamp-to-edge duplicates at the frame edge: no effect on max,
            // negligible edge-weighting on the mean
            vec3 rgb = HOOKED_tex(uv).rgb;
            float e = max(rgb.r, max(rgb.g, rgb.b));
            emax = max(emax, e);
            lsum += pq_eotf_norm(e);
        }
    }
    // Store PQ codes (perceptually uniform in [0,1]) so the intermediate is
    // robust to whatever SAVE format the fbo setting picks.
    imageStore(out_image, bpos, vec4(emax, pq_oetf_norm(lsum / 256.0), 0.0, 1.0));
    // Threads past the padded right/bottom edge imageStore out of bounds —
    // a defined no-op, same pattern as CelFlare's prefilter dispatch.
}

//!HOOK MAIN
//!BIND NITMETER_BLK
//!BIND NITMETER_STATE
//!SAVE NITMETER_STATS
//!WIDTH 1
//!HEIGHT 1
//!COMPUTE 16 9
//!DESC NitMeter: frame reduce + peak hold

// PQ duplicate — keep in sync with the other passes.
float pq_eotf_norm(float e) {   // PQ code -> linear, 1.0 = 10000 nits
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    // clamp: codes past 1.0 (YUV->RGB overshoot) cross the EOTF denominator's
    // zero at e~1.009 and explode to inf; overrange = signal ceiling (10k nits)
    float p = pow(clamp(e, 0.0, 1.0), 1.0 / m2);
    return pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}

#define NM_HIST_BINS   24       // MUST match PASS 3 (hist drawing)
#define NM_HIST_LOG_HI 13.2877  // log2(10000) — MUST match PASS 3
#define NM_HOLD_FRAMES 72.0     // ~3 s @24p before the hold starts decaying
#define NM_HOLD_DECAY  0.985    // per-frame decay once expired (halves in ~1.4 s)
#define NM_SDR_SUSPECT_NITS 8000.0  // sustained block-mean above this = SDR misread as PQ
#define NM_SDR_ALPHA        0.06    // suspicion EMA (~0.5 s to engage/release @24p)
#define NM_SHOW_FRAMES      12.0    // ~0.5 s @24p numeric sample-and-hold (readability)

shared float s_max[144];
shared float s_sum[144];
shared float s_mmean[144];   // max block-MEAN — SDR-tripwire driver
shared uint  s_hist[NM_HIST_BINS];

void hook() {
    uint lid = gl_LocalInvocationIndex;    // 0..143
    if (lid < uint(NM_HIST_BINS)) s_hist[lid] = 0u;
    barrier();

    uint W = uint(NITMETER_BLK_size.x);
    uint H = uint(NITMETER_BLK_size.y);
    uint total = W * H;
    float lmax = 0.0;
    float lsum = 0.0;
    float lmm  = 0.0;
    for (uint i = lid; i < total; i += 144u) {
        // texelFetch, not center-sampling: a sub-ULP off-center uv through
        // the LINEAR sampler could bleed a neighbor block into a MAX
        vec2 s = (NITMETER_BLK_mul *
                  texelFetch(NITMETER_BLK_raw, ivec2(int(i % W), int(i / W)), 0)).rg;
        float bmax  = pq_eotf_norm(s.r) * 10000.0;   // block peak CLL, nits
        float bmean = pq_eotf_norm(s.g) * 10000.0;   // block mean CLL, nits
        lmax = max(lmax, bmax);
        lsum += bmean;
        lmm  = max(lmm, bmean);
        // area histogram over block means; everything <= 1 nit lands in bin 0
        float hx = clamp(log2(max(bmean, 1.0)) / NM_HIST_LOG_HI, 0.0, 0.99999);
        // shared-atomic histogram (InterlockedAdd on groupshared via
        // SPIRV-Cross); if a backend ever misbehaves here, fall back to
        // per-lane bins + a thread-0 serial tally like CelFlare PASS 5
        atomicAdd(s_hist[uint(hx * float(NM_HIST_BINS))], 1u);
    }
    s_max[lid]   = lmax;
    s_sum[lid]   = lsum;
    s_mmean[lid] = lmm;
    barrier();

    if (lid == 0u) {
        float pk  = 0.0;
        float sm  = 0.0;
        float pkm = 0.0;
        for (uint i = 0u; i < 144u; i++) {
            pk  = max(pk, s_max[i]);
            sm  += s_sum[i];
            pkm = max(pkm, s_mmean[i]);
        }
        float inv = 1.0 / max(float(total), 1.0);
        nm_peak = pk;
        nm_fall = sm * inv;
        // classic peak hold: latch upward instantly, sit for NM_HOLD_FRAMES,
        // then decay multiplicatively but never below the live peak
        if (pk >= nm_hold) {
            nm_hold = pk;
            nm_hold_age = 0.0;
        } else {
            nm_hold_age += 1.0;
            if (nm_hold_age > NM_HOLD_FRAMES)
                nm_hold = max(pk, nm_hold * NM_HOLD_DECAY);
        }
        // SDR-input tripwire (heuristic): gamma misread as PQ decodes any
        // clipped white to 10000 nits, and real PQ masters stay <= ~4000.
        // Driven by the max 16x16-block MEAN, not the pixel max: an SDR
        // clipped-white AREA still reads 10000, but a lone YUV-overshoot
        // pixel on legit HDR can't lift a 256-px average past the gate.
        // EMA'd so a single hot frame can't trip it either.
        nm_sdr += NM_SDR_ALPHA * (((pkm > NM_SDR_SUSPECT_NITS) ? 1.0 : 0.0) - nm_sdr);
        // session maxima (MaxCLL/MaxFALL semantics; reset per shader load,
        // which the cycle script triggers on every toggle step). Reset while
        // the tripwire is engaged so SDR garbage can't poison them.
        if (nm_sdr > 0.5) {
            nm_maxcll  = 0.0;
            nm_maxfall = 0.0;
        } else {
            nm_maxcll  = max(nm_maxcll, pk);
            nm_maxfall = max(nm_maxfall, nm_fall);
        }
        // numeric sample-and-hold (Lilium-style ~0.5 s readout refresh):
        // measurement stays per-frame, only the SHOWN digits are held.
        // Countdown from 0 so the very first frame populates immediately.
        if (nm_show_ctr <= 0.0) {
            nm_show_peak = pk;
            nm_show_fall = nm_fall;
            nm_show_ctr  = NM_SHOW_FRAMES;
        }
        nm_show_ctr -= 1.0;
        for (int b = 0; b < NM_HIST_BINS; b++)
            nm_hist[b] = float(s_hist[b]) * inv;
        imageStore(out_image, ivec2(0), vec4(0.0));
    }
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND NITMETER_STATE
//!BIND NITMETER_STATS
//!DESC NitMeter v1.1: overlay draw

// NITMETER_STATS is bound only as the explicit data dependency on PASS 2 —
// the stats themselves arrive through the NITMETER_STATE SSBO (same pattern
// as CelFlare PASS 6 / CELFLARE_STATS: don't remove the bind without
// verifying pass ordering/visibility on every backend).

// panel/heatmap selection is the runtime nitmeter_mode PARAM (file header)
// ---------------------------- MAIN TUNING ----------------------------
#define NITMETER_SDR_GUARD   1     // blank the panel when input looks like SDR misread as PQ
#define NITMETER_CORNER      1     // 0 TL / 1 TR / 2 BL / 3 BR
#define NITMETER_SCALE       1.0   // panel size multiplier
#define NM_FC_PAINT_NITS     180.0 // display brightness of heatmap bands (PQ mode)
// ------------------------------------------------------------------

#define NM_HIST_BINS   24          // MUST match PASS 2
#define NM_HIST_LOG_HI 13.2877     // MUST match PASS 2
#define NM_PANEL_W     136.0
#define NM_PANEL_H     182.0

// PQ duplicate — keep in sync with the other passes.
float pq_eotf_norm(float e) {   // PQ code -> linear, 1.0 = 10000 nits
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    // clamp: codes past 1.0 (YUV->RGB overshoot) cross the EOTF denominator's
    // zero at e~1.009 and explode to inf; overrange = signal ceiling (10k nits)
    float p = pow(clamp(e, 0.0, 1.0), 1.0 / m2);
    return pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}
vec3 pq_oetf3(vec3 L) {         // linear (1.0 = 10000 nits) -> PQ code
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 Lm = pow(max(L, vec3(0.0)), vec3(m1));
    return pow((c1 + c2 * Lm) / (1.0 + c3 * Lm), vec3(m2));
}
vec3 pq_nits(vec3 nits) { return pq_oetf3(nits / 10000.0); }

// Lilium-style luminance heatmap ramp (10000-nit cutoff), linear RGB.
// Band edges 100 (SDR ref white) / 203 (BT.2408 HDR ref white) / 400 /
// 1000 / 4000, continuous crossfade inside each band.
vec3 heat_lin(float n) {
    if (n <  100.0) return vec3(n / 100.0 * 0.25);                              // grey
    if (n <  203.0) return vec3(0.0, 1.0, 1.0 - (n - 100.0) / 103.0);           // cyan->green
    if (n <  400.0) return vec3((n - 203.0) / 197.0, 1.0, 0.0);                 // green->yellow
    if (n < 1000.0) return vec3(1.0, 1.0 - (n - 400.0) / 600.0, 0.0);           // yellow->red
    if (n < 4000.0) return vec3(1.0, 0.0, (n - 1000.0) / 3000.0);               // red->pink
    return vec3(1.0 - clamp((n - 4000.0) / 6000.0, 0.0, 1.0), 0.0, 1.0);        // pink->blue
}

// 7-segment glyphs. Bits: A=1 B=2 C=4 D=8 E=16 F=32 G=64.
const int NM_SEG[10] = int[10](0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F);
// row label glyphs: P, H, A, C, F
const int NM_LBL[5] = int[5](0x73, 0x76, 0x77, 0x39, 0x71);

float seg_rect(vec2 p, vec2 c, vec2 h) {
    vec2 d = abs(p - c) - h;
    return step(max(d.x, d.y), 0.0);
}
// glyph occupies [0,0.66]x[0,1] of q, y down; m = segment bitmask
float glyph_mask(int m, vec2 q) {
    float r = 0.0;
    if ((m &  1) != 0) r += seg_rect(q, vec2(0.33, 0.05),  vec2(0.21,  0.05));  // A top
    if ((m &  2) != 0) r += seg_rect(q, vec2(0.60, 0.275), vec2(0.055, 0.185)); // B top-right
    if ((m &  4) != 0) r += seg_rect(q, vec2(0.60, 0.725), vec2(0.055, 0.185)); // C bottom-right
    if ((m &  8) != 0) r += seg_rect(q, vec2(0.33, 0.95),  vec2(0.21,  0.05));  // D bottom
    if ((m & 16) != 0) r += seg_rect(q, vec2(0.06, 0.725), vec2(0.055, 0.185)); // E bottom-left
    if ((m & 32) != 0) r += seg_rect(q, vec2(0.06, 0.275), vec2(0.055, 0.185)); // F top-left
    if ((m & 64) != 0) r += seg_rect(q, vec2(0.33, 0.50),  vec2(0.21,  0.05));  // G middle
    return min(r, 1.0);
}
// right-aligned integer, leading zeros blanked down to mind digits.
// p = (leftward distance from the field's right edge, downward from row top)
float number_mask(int v, int mind, vec2 p, float h) {
    if (p.y < 0.0 || p.y >= h || p.x < 0.0) return 0.0;
    float slotw = 0.85 * h;
    int nd = (v >= 10000) ? 5 : (v >= 1000) ? 4 : (v >= 100) ? 3 : (v >= 10) ? 2 : 1;
    nd = max(nd, mind);
    int slot = int(p.x / slotw);               // 0 = rightmost digit
    if (slot >= nd) return 0.0;
    const int P10[5] = int[5](1, 10, 100, 1000, 10000);
    int d = (v / P10[slot]) % 10;
    float xl = (float(slot) + 1.0) * slotw - p.x;   // rightward from the slot's left edge
    return glyph_mask(NM_SEG[d], vec2(xl / h, p.y / h));
}
// nit value readout: tenths digit + decimal point below 100 nits
// (dark-scene FALL at integer precision is useless), integer above.
float value_mask(float nits, vec2 p, float h) {
    if (p.y < 0.0 || p.y >= h || p.x < 0.0) return 0.0;
    if (nits < 99.95) {
        float m = number_mask(int(nits * 10.0 + 0.5), 2, p, h);   // "0.4" not ".4"
        // decimal point: small square in the inter-digit gap, near baseline
        vec2 dd = abs(p - vec2(0.945 * h, 0.90 * h)) - vec2(0.055 * h);
        return max(m, step(max(dd.x, dd.y), 0.0));
    }
    return number_mask(int(min(nits, 99999.0) + 0.5), 1, p, h);
}
// "---" (three G segments), right-aligned like number_mask — the blanked
// readout when the SDR tripwire is engaged
float dashes_mask(vec2 p, float h) {
    if (p.y < 0.0 || p.y >= h || p.x < 0.0) return 0.0;
    float slotw = 0.85 * h;
    int slot = int(p.x / slotw);
    if (slot >= 3) return 0.0;
    float xl = (float(slot) + 1.0) * slotw - p.x;
    return seg_rect(vec2(xl / h, p.y / h), vec2(0.33, 0.50), vec2(0.21, 0.05));
}

// panel geometry: 5 label+digit rows, then the histogram strip
const float NM_ROW_Y[5] = float[5](8.0, 33.0, 58.0, 83.0, 108.0);
// row colors as display nits: P white, H yellow, A cyan, C orange, F green
const vec3 NM_ROW_NITS[5] = vec3[5](
    vec3(230.0, 230.0, 230.0),
    vec3(220.0, 180.0, 15.0),
    vec3(15.0,  190.0, 230.0),
    vec3(230.0, 110.0, 15.0),
    vec3(60.0,  220.0, 60.0));

vec4 hook() {
    vec4 color = HOOKED_texOff(0);
    vec3 col = color.rgb;

#if nitmeter_mode >= 2
    {
        float e = max(col.r, max(col.g, col.b));
        float n = pq_eotf_norm(e) * 10000.0;
        bool ceil_flag = n >= 9999.5;   // at signal ceiling — scream
#if nitmeter_mode == 3
        // capture/export encode: plain gamma 2.2 so screenshots read as a
        // normal SDR image. Grey ramp tops out at 0.8 so the band colors
        // (encoded at full swing) stay separable above it.
        col = ceil_flag   ? vec3(1.0, 0.0, 0.0)
            : (n < 100.0) ? vec3(0.8 * pow(clamp(n / 100.0, 0.0, 1.0), 1.0 / 2.2))
                          : pow(heat_lin(n), vec3(1.0 / 2.2));
#else
        // SDR range reproduces as self-luminous grey; bands paint flat
        col = ceil_flag   ? pq_nits(vec3(400.0, 8.0, 8.0))
            : (n < 100.0) ? pq_nits(vec3(n))
                          : pq_oetf3(heat_lin(n) * (NM_FC_PAINT_NITS / 10000.0));
#endif
    }
#endif

    float S = NITMETER_SCALE * HOOKED_size.y / 1080.0;
#if NITMETER_CORNER == 0
    vec2 org = vec2(12.0, 12.0);
#elif NITMETER_CORNER == 1
    vec2 org = vec2(HOOKED_size.x / S - 12.0 - NM_PANEL_W, 12.0);
#elif NITMETER_CORNER == 2
    vec2 org = vec2(12.0, HOOKED_size.y / S - 12.0 - NM_PANEL_H);
#else
    vec2 org = vec2(HOOKED_size.x / S - 12.0 - NM_PANEL_W,
                    HOOKED_size.y / S - 12.0 - NM_PANEL_H);
#endif
    vec2 lp = HOOKED_pos * HOOKED_size / S - org;   // panel-local units

    if (all(greaterThanEqual(lp, vec2(0.0))) && all(lessThan(lp, vec2(NM_PANEL_W, NM_PANEL_H)))) {
        col = mix(col, pq_nits(vec3(2.0)), 0.92);   // ~2-nit translucent background

        float rx = NM_PANEL_W - 8.0;
        float m;
        // row labels (always drawn, even while tripped — they say what's blank)
        for (int r = 0; r < 5; r++) {
            m = glyph_mask(NM_LBL[r], vec2((lp.x - 8.0) / 20.0, (lp.y - NM_ROW_Y[r]) / 20.0));
            col = mix(col, pq_nits(NM_ROW_NITS[r]), m);
        }

        bool sdr_suspect = false;
#if NITMETER_SDR_GUARD
        sdr_suspect = nm_sdr > 0.5;
#endif
        if (sdr_suspect) {
            // input looks like SDR gamma misread as PQ: red border, and all
            // readouts blank to "---" instead of reporting nonsense
            vec2 db = min(lp, vec2(NM_PANEL_W, NM_PANEL_H) - lp);
            col = mix(col, pq_nits(vec3(180.0, 8.0, 8.0)),
                      0.9 * step(min(db.x, db.y), 3.0));
            for (int r = 0; r < 5; r++) {
                m = dashes_mask(vec2(rx - lp.x, lp.y - NM_ROW_Y[r]), 20.0);
                col = mix(col, pq_nits(NM_ROW_NITS[r]), m);
            }
            return vec4(col, color.a);
        }

        // digit rows: P peak (shown, ~0.5 s hold), H peak-hold, A FALL
        // (shown), C session MaxCLL, F session MaxFALL
        float vals[5] = float[5](nm_show_peak, nm_hold, nm_show_fall, nm_maxcll, nm_maxfall);
        for (int r = 0; r < 5; r++) {
            m = value_mask(vals[r], vec2(rx - lp.x, lp.y - NM_ROW_Y[r]), 20.0);
            col = mix(col, pq_nits(NM_ROW_NITS[r]), m);
        }

        // display-ceiling indicator (P row): shown peak vs nitmeter_target
        vec2 td = abs(lp - vec2(30.0, 18.0)) - vec2(4.0);
        float tm = step(max(td.x, td.y), 0.0);
        vec3 tc = (nm_show_peak > nitmeter_target)        ? vec3(230.0, 15.0, 15.0)
                : (nm_show_peak > 0.95 * nitmeter_target) ? vec3(230.0, 200.0, 20.0)
                                                          : vec3(25.0, 210.0, 25.0);
        col = mix(col, pq_nits(tc), tm);

        // histogram strip: log2 axis, 1 -> 10000 nits
        if (lp.x >= 8.0 && lp.x < 128.0 && lp.y >= 134.0 && lp.y < 174.0) {
            float hx = (lp.x - 8.0) / 120.0;
            // reference ticks at 100 / 203 / 1000 / 4000 nits
            float tick = step(abs(hx - 0.5000), 0.004);
            tick = max(tick, step(abs(hx - 0.5768), 0.004));
            tick = max(tick, step(abs(hx - 0.7500), 0.004));
            tick = max(tick, step(abs(hx - 0.9005), 0.004));
            col = mix(col, pq_nits(vec3(25.0)), tick);

            // guard bounds already force bin < 24; clamp anyway so a future
            // geometry edit can't silently reopen an OOB SSBO read
            int bin = clamp(int(hx * float(NM_HIST_BINS)), 0, NM_HIST_BINS - 1);
            float bh = pow(clamp(nm_hist[bin], 0.0, 1.0), 0.4);   // gamma'd so small areas stay visible
            float yy = (174.0 - lp.y) / 40.0;                     // 0 bottom .. 1 top
            if (yy < bh) {
                float cn = exp2((float(bin) + 0.5) / float(NM_HIST_BINS) * NM_HIST_LOG_HI);
                vec3 bc = (cn < 100.0) ? vec3(0.5) : heat_lin(cn);
                col = mix(col, pq_oetf3(bc * (NM_FC_PAINT_NITS / 10000.0)), 0.9);
            }
            // bright marker at the current frame peak (live, not held)
            float pkx = clamp(log2(max(nm_peak, 1.0)) / NM_HIST_LOG_HI, 0.0, 1.0);
            col = mix(col, pq_nits(vec3(300.0)), step(abs(hx - pkx), 0.005));
        }
    }
    return vec4(col, color.a);
}
