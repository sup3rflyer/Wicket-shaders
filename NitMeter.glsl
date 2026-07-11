// NitMeter v1.6 — HDR peak-nit analysis overlay for mpv (gpu-next)
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
//     P (white)  — frame peak CLL in nits (sample-and-hold, ~0.5 s refresh).
//                  STRICT max pixel — on web re-encodes this is often codec
//                  ringing, not content (calibration vs a DoVi-labeled
//                  reference measured spike pixels ~4x the master's peak)
//     9 (grey)   — 99.99th-percentile pixel CLL: the practical/content
//                  peak, immune to lone spike pixels (~830 px ignored at
//                  4K). Same convention as madVR measurements and the
//                  ST 2094-40 maxscl distribution. P >> 9 means the "peak"
//                  is encode noise; P close to 9 means it is real content
//     H (yellow) — hold of the 9 row: latches its highest recent value,
//                  sits ~3 s, then decays — spike-free recent content peak
//     A (cyan)   — FALL: frame-average CLL (letterbox bars included, same as
//                  the MaxFALL averaging; ~0.5 s refresh; dilution on scope)
//     C (orange) — session MaxCLL: running max frame peak since toggle-on
//     F (green)  — session MaxFALL: running max FALL since toggle-on
//   Values under 100 nits gain a tenths digit (e.g. 94.3). The small square
//   on the P row is the display-ceiling indicator vs nitmeter_target,
//   driven by the 9 row (CONTENT peak — a strict-max driver went red on
//   lone encode-ringing spikes no display visibly clips): green < 95% of
//   target, yellow approaching, red = content exceeds the display
//   (mpv/display will compress or clip). Strict-max-only exceedance shows
//   in the P digits / P-9 gap / dim histogram tick, not in the square.
//   Histogram — screen-area distribution of per-pixel CLL on a log2 axis
//   from 1 to 10000 nits, rebinned from the same 256-bin pixel histogram
//   that feeds the 9 row (until v1.5 the bars were 16x16-block MEANS,
//   which diluted small speculars into mid bins — the tail is the point).
//   Grey ticks at 100 / 203 / 1000 / 4000 nits; bright line = current
//   frame 9 (content peak), thin dim line = strict-max P. Bars take the
//   heatmap ramp hue of their bin (grey below 100).
//   nitmeter_graph=1 appends a time-series strip under the histogram:
//   the last 120 rendered frames (~5 s @24p) of P (white) / 9 (grey) /
//   A (cyan) on the same log2 axis, newest at the right, dim line at
//   nitmeter_target — a live scope for CelFlare's light-pump envelope
//   (onset / hold / release). 0-samples draw as gaps: tripped frames by
//   design, true-black frames incidentally.
//
// REQUIREMENTS / USAGE
//   The frame at MAIN must already be PQ-encoded when this shader runs:
//   load it AFTER CelFlare (which emits PQ in-shader), or on native PQ HDR
//   content without CelFlare (HDR10/HDR10+/DV — DV is reshaped to plain PQ
//   before MAIN). HLG is NOT supported (deliberately — PQ only). On plain
//   SDR content the numbers are meaningless (gamma decoded as PQ); the
//   nitmeter_sdr_guard tripwire below blanks the panel when it detects that.
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
//   nitmeter_target (DYNAMIC float, default 1000) — display peak in nits
//     for the P-row ceiling indicator and the graph target line; match
//     your target-peak. True live uniform (v1.6; the plain-float PARAM it
//     replaced took the constant/rebuild path).
//   nitmeter_corner (1) — 0 TL / 1 TR / 2 BL / 3 BR
//   nitmeter_scale (DYNAMIC, 1.0) — panel size multiplier (1.0 = ~136 px
//     wide at 1080p)
//   nitmeter_graph (0) — 1 appends the P/9/A time-series strip
//   nitmeter_heat_nits (DYNAMIC, 180) — display brightness of the heatmap
//     bands and histogram bars (PQ mode)
//   nitmeter_pct (DYNAMIC, 0.0001) — fraction of frame pixels allowed
//     ABOVE the 9-row readout (0.0001 = 99.99th percentile)
//   nitmeter_reset (DYNAMIC, 0) — session reset edge: whenever the value
//     CHANGES, session state (C / F / H / graph ring) zeroes. Needed
//     because the STORAGE buffer survives the cycle script's
//     remove+re-append (field-observed: C/F persist across N-key cycles),
//     so "resets on shader reload" was never true. The cycle script bumps
//     this on every toggle-on; manual:
//       change-list glsl-shader-opts append nitmeter_reset=7
//   nitmeter_sdr_guard (1) — SDR-input tripwire. A user shader cannot
//     query the frame's tagged transfer, so detection is heuristic, on
//     two OR'd signals (PASS 2): (a) smoothed max 16x16-block-average
//     above NM_SDR_SUSPECT_NITS (8000) — a solid clipped-white AREA; (b)
//     fraction of frame pixels at the signal ceiling (code >= 0.9995,
//     ~9950 nits) above NM_SDR_CEIL_FRAC (0.05% ~ one small line of white
//     text). Clipped SDR whites decode to exactly 10000 in droves, while
//     graded HDR essentially never reaches ceiling codes (masters cap
//     ~1000-4000 nits, code <= ~0.92, and YUV-overshoot ringing from there
//     cannot climb to 0.9995) — so (b) catches ordinary SDR frames whose
//     whites are scattered pixels, not solid blocks. Fast attack (~0.3 s),
//     slow release (~2 s) so scene-to-scene evidence doesn't strobe the
//     panel. While tripped: readouts blank to "---", red border, and ALL
//     temporal state is quarantined. Limitation: SDR frames with no
//     clipped whites at all (dark scenes) are undetectable — numbers
//     return until the next white. Set 0 if you ever feed legit 8000+ nit
//     material — that compiles out the PASS 2 state quarantine too, not
//     just the panel blanking (pre-v1.6 it only disarmed the display and
//     the quarantine kept zeroing C/F/H).
//
// The two measurement passes run BEFORE the overlay draw, so the panel and
// heatmap never contaminate their own statistics.

//!PARAM nitmeter_mode
//!DESC Display mode. 1 = panel · 2 = heatmap + panel · 3 = heatmap (gamma-2.2 SDR export) + panel.
//!TYPE DEFINE
//!MINIMUM 1
//!MAXIMUM 3
1

//!PARAM nitmeter_target
//!DESC Display peak (nits) for the P-row ceiling indicator + graph target line. Match your target-peak.
//!TYPE DYNAMIC float
//!MINIMUM 100.0
//!MAXIMUM 10000.0
1000.0

//!PARAM nitmeter_corner
//!DESC Panel corner. 0 = top-left · 1 = top-right · 2 = bottom-left · 3 = bottom-right.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 3
1

//!PARAM nitmeter_scale
//!DESC Panel size multiplier. ↑ = bigger · ↓ = smaller. 1 = ~136 px wide at 1080p.
//!TYPE DYNAMIC float
//!MINIMUM 0.5
//!MAXIMUM 3.0
1.0

//!PARAM nitmeter_graph
//!DESC Time-series strip (toggle). 1 = append the P/9/A graph (last ~5 s, log2 nit axis) under the histogram · 0 = off.
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM nitmeter_heat_nits
//!DESC Display brightness (nits) of the heatmap bands + histogram bars in PQ mode. ↑ = brighter overlay.
//!TYPE DYNAMIC float
//!MINIMUM 40.0
//!MAXIMUM 1000.0
180.0

//!PARAM nitmeter_pct
//!DESC 9-row percentile = frame fraction allowed above the 9 value. ↑ = more tolerant, 9 reads lower · 0 = strict max. 0.0001 = 99.99th pct.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 0.01
0.0001

//!PARAM nitmeter_reset
//!DESC Session reset edge — session maxima + hold + graph ring zero whenever this value CHANGES. Magnitude is meaningless (cycle script bumps it per toggle-on).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 16777216.0
0.0

//!PARAM nitmeter_sdr_guard
//!DESC SDR-input tripwire (toggle). 1 = blank panel + quarantine state when input looks like SDR misread as PQ · 0 = off (only if feeding real 8000+ nit material).
//!TYPE DEFINE
//!MINIMUM 0
//!MAXIMUM 1
1

//!BUFFER NITMETER_STATE
//!VAR float nm_peak
//!VAR float nm_fall
//!VAR float nm_hold
//!VAR float nm_hold_age
//!VAR float nm_sdr
//!VAR float nm_maxcll
//!VAR float nm_maxfall
//!VAR float nm_p9999
//!VAR float nm_show_peak
//!VAR float nm_show_p9999
//!VAR float nm_show_fall
//!VAR float nm_show_ctr
//!VAR float nm_hist[24]
//!VAR uint nm_phist[256]
//!VAR float nm_reset_seen
//!VAR float nm_ring_idx
//!VAR float nm_ring_p[120]
//!VAR float nm_ring_9[120]
//!VAR float nm_ring_a[120]
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!BIND NITMETER_STATE
//!SAVE NITMETER_BLK
//!WIDTH HOOKED.w 7 + 16 /
//!HEIGHT HOOKED.h 7 + 16 /
//!COMPUTE 16 16
//!DESC NitMeter: block reduce (16x16 max/mean CLL) + pixel histogram

// SIZE RPN: libplacebo evaluates the expression in FLOAT and roundf()s the
// result (half away from zero) — it is NOT C integer division. The naive
// ceil-divide idiom (dim+15)/16 therefore lands one block HIGH whenever
// dim%16 is 0 or 9..15 (i.e. nearly every standard resolution), leaving a
// phantom column/row that PASS 2 reads but nothing writes. (dim+7)/16 under
// roundf equals floor((dim+15)/16) = ceil(dim/16) exactly, for every dim —
// that is what the WIDTH/HEIGHT above encode. Audit-caught in v1.4; in
// v1.1-v1.3 the oversized texture was masked by unguarded edge-clamped
// writes filling the phantom texels with duplicated edge content.
//
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

// Per-workgroup staging for the frame-global 256-bin pixel-code histogram
// (percentile peak + drawn histogram, PASS 2). Each invocation reduces one
// 16x16 block, so a workgroup covers 256 blocks; staging in shared memory
// keeps the SSBO atomic traffic to at most 256 adds per workgroup instead
// of per pixel. Shared-memory atomicAdd = InterlockedAdd on groupshared
// via SPIRV-Cross; if a backend ever misbehaves here, fall back to
// per-lane bins + a thread-0 serial tally like CelFlare PASS 5.
shared uint s_ph[256];

void hook() {
    uint lid = gl_LocalInvocationIndex;
    s_ph[lid] = 0u;
    barrier();

    ivec2 bpos = ivec2(gl_GlobalInvocationID.xy);
    // live = this invocation's block exists in the grid. Padded-dispatch
    // phantom blocks used to be handled by the imageStore OOB no-op alone;
    // the histogram is an SSBO (no OOB safety net), so gate explicitly.
    bool live = all(lessThan(bpos, (ivec2(HOOKED_size) + 15) / 16));
    if (live) {
        vec2 base = vec2(bpos) * 16.0;
        float emax = 0.0;   // block max PQ code of per-pixel max(R,G,B)
        float lsum = 0.0;   // block sum of linear CLL (1.0 = 10000 nits)
        float ceil_cnt = 0.0;   // pixels at the signal ceiling (SDR-tripwire evidence)
        for (int j = 0; j < 16; j++) {
            for (int i = 0; i < 16; i++) {
                vec2 uv = (base + vec2(float(i), float(j)) + 0.5) * HOOKED_pt;
                // clamp-to-edge duplicates at the frame edge: no effect on
                // max, negligible edge-weighting on the mean and histogram
                vec3 rgb = HOOKED_tex(uv).rgb;
                float e = max(rgb.r, max(rgb.g, rgb.b));
                emax = max(emax, e);
                lsum += pq_eotf_norm(e);
                ceil_cnt += step(0.9995, e);
                atomicAdd(s_ph[uint(clamp(e, 0.0, 0.999999) * 256.0)], 1u);
            }
        }
        // Store PQ codes (perceptually uniform in [0,1]) so the intermediate
        // is robust to whatever SAVE format the fbo setting picks.
        imageStore(out_image, bpos, vec4(emax, pq_oetf_norm(lsum / 256.0), ceil_cnt / 256.0, 1.0));
    }
    barrier();
    // flush this workgroup's staging bins; PASS 2 reads then zeroes them
    uint c = s_ph[lid];
    if (c > 0u) atomicAdd(nm_phist[lid], c);
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

// hist axis top: log2(10000 nits). Derived (constant-folded), not a magic
// number — the PASS 3 copy is the identical expression, so the only way to
// desync is editing the 10000.0 in one pass.
const float NM_HIST_LOG_HI = log2(10000.0);

#define NM_HIST_BINS   24       // MUST match the nm_hist[24] SSBO array + PASS 3
#define NM_RING_LEN    120      // MUST match the nm_ring_*[120] SSBO arrays + PASS 3
#define NM_HOLD_FRAMES 72.0     // ~3 s @24p before the hold starts decaying
#define NM_HOLD_DECAY  0.985    // per-frame decay once expired (halves in ~1.9 s @24p)
#define NM_SDR_SUSPECT_NITS 8000.0  // block-mean gate: solid clipped-white AREA
#define NM_SDR_CEIL_FRAC    0.0005  // ceiling-pixel area gate (0.05% of frame)
#define NM_SDR_ALPHA_UP     0.10    // suspicion attack (~0.3 s @24p)
#define NM_SDR_ALPHA_DN     0.015   // suspicion release (~2 s @24p)
#define NM_SHOW_FRAMES      12.0    // ~0.5 s @24p numeric sample-and-hold (readability)
#define NM_PHIST_BINS       256     // pixel-code histogram bins — MUST match PASS 1 s_ph

shared float s_max[144];
shared float s_sum[144];
shared float s_mmean[144];   // max block-MEAN — SDR-tripwire driver (a)
shared float s_ceil[144];    // ceiling-pixel fraction sum — SDR-tripwire driver (b)

void hook() {
    uint lid = gl_LocalInvocationIndex;    // 0..143

    uint W = uint(NITMETER_BLK_size.x);
    uint H = uint(NITMETER_BLK_size.y);
    uint total = W * H;
    float lmax = 0.0;
    float lsum = 0.0;
    float lmm  = 0.0;
    float lcf  = 0.0;
    for (uint i = lid; i < total; i += 144u) {
        // texelFetch, not center-sampling: a sub-ULP off-center uv through
        // the LINEAR sampler could bleed a neighbor block into a MAX
        vec3 s = (NITMETER_BLK_mul *
                  texelFetch(NITMETER_BLK_raw, ivec2(int(i % W), int(i / W)), 0)).rgb;
        float bmax  = pq_eotf_norm(s.r) * 10000.0;   // block peak CLL, nits
        float bmean = pq_eotf_norm(s.g) * 10000.0;   // block mean CLL, nits
        lmax = max(lmax, bmax);
        lsum += bmean;
        lmm  = max(lmm, bmean);
        lcf  += s.b;
    }
    s_max[lid]   = lmax;
    s_sum[lid]   = lsum;
    s_mmean[lid] = lmm;
    s_ceil[lid]  = lcf;
    barrier();

    if (lid == 0u) {
        float pk  = 0.0;
        float sm  = 0.0;
        float pkm = 0.0;
        float cfs = 0.0;
        for (uint i = 0u; i < 144u; i++) {
            pk  = max(pk, s_max[i]);
            sm  += s_sum[i];
            pkm = max(pkm, s_mmean[i]);
            cfs += s_ceil[i];
        }
        float inv = 1.0 / max(float(total), 1.0);
        nm_peak = pk;
        nm_fall = sm * inv;
        // explicit session reset (edge on the nitmeter_reset DYNAMIC
        // param). Reload does NOT reset: the STORAGE buffer survives the
        // cycle script's remove+re-append (field-observed — C/F persisted
        // across N-key cycles), so session semantics need a real edge.
        // Also self-heals first-load garbage if a backend ever hands us a
        // non-zeroed buffer (reset fires because nm_reset_seen mismatches).
        if (nitmeter_reset != nm_reset_seen) {
            nm_reset_seen = nitmeter_reset;
            nm_maxcll   = 0.0;
            nm_maxfall  = 0.0;
            nm_hold     = 0.0;
            nm_hold_age = 0.0;
            // expire the P/9/A sample-and-hold too, or the first frames of
            // a new session keep showing the OLD session's held digits for
            // up to 12 frames (audit-caught) — this forces the show
            // snapshot to re-arm with current values further down
            nm_show_ctr = 0.0;
            nm_ring_idx = 0.0;
            for (int i = 0; i < NM_RING_LEN; i++) {
                nm_ring_p[i] = 0.0;
                nm_ring_9[i] = 0.0;
                nm_ring_a[i] = 0.0;
            }
        }
        // pixel-level percentile peak from the 256-bin code histogram:
        // walk from the top bin until nitmeter_pct of the frame's
        // pixels lie above, then interpolate inside the stopping bin
        // (uniform-within-bin). Immune to the lone codec-ringing spike
        // pixels that dominate a strict max on web re-encodes — the same
        // practical-peak convention as madVR measurements / ST 2094-40.
        // Total from the histogram itself (== frame pixel count: PASS 1
        // gates phantom padded blocks off the histogram).
        uint tot = 0u;
        for (int b = 0; b < NM_PHIST_BINS; b++) tot += nm_phist[b];
        nm_p9999 = 0.0;
        if (tot > 0u) {
            // floor at 1 px so small frames still ignore the single hottest
            // pixel instead of degenerating into a second P row; compare in
            // uint (acc can pass 2^24 at 8K, where float comparisons drift)
            uint target = max(uint(nitmeter_pct * float(tot)), 1u);
            uint acc = 0u;
            int bi = NM_PHIST_BINS - 1;
            for (; bi >= 0; bi--) {
                acc += nm_phist[bi];
                if (acc >= target) break;
            }
            bi = max(bi, 0);
            float bcnt = max(float(nm_phist[bi]), 1.0);
            // target sits 'into' of the way down from the bin's top edge
            float into = clamp((float(target) - (float(acc) - bcnt)) / bcnt, 0.0, 1.0);
            float code = (float(bi) + 1.0 - into) / float(NM_PHIST_BINS);
            // a percentile can never exceed the max; the min also snaps the
            // uniform-frame case (everything in one bin, interpolation lands
            // on the bin's upper edge) back to the exact frame value
            nm_p9999 = min(pq_eotf_norm(code) * 10000.0, pk);
        }
        // drawn histogram: PIXEL-level since v1.6 — rebin the 256 PQ-code
        // bins onto the 24-bin log2-nit axis (bin center through the exact
        // EOTF). Replaces the 16x16-block-MEAN histogram, which diluted
        // small speculars into mid bins; the tail is the point of a nit
        // histogram. Denominator = tot (pixel count), same [0,1]
        // area-fraction semantics PASS 3 already draws.
        float hinv = 1.0 / max(float(tot), 1.0);
        for (int b = 0; b < NM_HIST_BINS; b++) nm_hist[b] = 0.0;
        for (int b = 0; b < NM_PHIST_BINS; b++) {
            float cn = pq_eotf_norm((float(b) + 0.5) / float(NM_PHIST_BINS)) * 10000.0;
            float hx = clamp(log2(max(cn, 1.0)) / NM_HIST_LOG_HI, 0.0, 0.99999);
            nm_hist[int(hx * float(NM_HIST_BINS))] += float(nm_phist[b]) * hinv;
        }
        // zero the bins for the next frame's PASS 1 accumulation
        for (int b = 0; b < NM_PHIST_BINS; b++) nm_phist[b] = 0u;
        // classic peak hold: latch upward instantly, sit for NM_HOLD_FRAMES,
        // then decay multiplicatively but never below the live value.
        // Driven by the PERCENTILE peak, not the strict max — H is "recent
        // content peak", spike-free; a hold latched to strict max would
        // pin the worst codec-ringing pixel of the last 3 s instead.
        // (Must stay upstream of the tripwire quarantine below, which
        // zeroes it on tripped frames.)
        if (nm_p9999 >= nm_hold) {
            nm_hold = nm_p9999;
            nm_hold_age = 0.0;
        } else {
            nm_hold_age += 1.0;
            if (nm_hold_age > NM_HOLD_FRAMES)
                nm_hold = max(nm_p9999, nm_hold * NM_HOLD_DECAY);
        }
        // SDR-input tripwire (heuristic), two OR'd signals — see header:
        // (a) a solid clipped-white AREA (block mean past the gate);
        // (b) enough scattered pixels at the exact signal ceiling — SDR
        //     whites clip to code 1.0 en masse, graded HDR never gets
        //     there (masters cap ~0.92) and lone YUV-overshoot pixels
        //     stay far under the area threshold.
        // Asymmetric EMA: fast attack, slow release (no scene strobing).
        float cf  = cfs * inv;   // frame fraction of ceiling pixels
        float sus = ((pkm > NM_SDR_SUSPECT_NITS) || (cf > NM_SDR_CEIL_FRAC)) ? 1.0 : 0.0;
        nm_sdr += ((sus > nm_sdr) ? NM_SDR_ALPHA_UP : NM_SDR_ALPHA_DN) * (sus - nm_sdr);
        // With the guard compiled out, the tripwire must be fully inert
        // here too, not just unblanked in PASS 3 — pre-v1.6 this
        // quarantine ran unconditionally, so guard=0 still zeroed C/F/H
        // on legit 8000+ nit material (the documented escape hatch was
        // broken). The suspicion EMA above stays warm either way (cheap).
        bool tripped = false;
#if nitmeter_sdr_guard
        tripped = nm_sdr > 0.5;
#endif
        // session maxima (MaxCLL/MaxFALL semantics; reset on the
        // nitmeter_reset edge above). Reset while the tripwire is engaged
        // so SDR garbage can't poison them.
        if (tripped) {
            nm_maxcll  = 0.0;
            nm_maxfall = 0.0;
            // quarantine ALL temporal readout state while tripped, not just
            // the session maxima: the hold and P/A snapshots used to survive
            // a trip carrying garbage, so after recovery H decayed from
            // ~10000 and sat ABOVE C for seconds (field-observed). Hold
            // re-latches instantly from clean content on release.
            nm_hold     = 0.0;
            nm_hold_age = 0.0;
        } else {
            nm_maxcll  = max(nm_maxcll, pk);
            nm_maxfall = max(nm_maxfall, nm_fall);
        }
        // numeric sample-and-hold (Lilium-style ~0.5 s readout refresh):
        // measurement stays per-frame, only the SHOWN digits are held.
        // Countdown from 0 so the very first frame populates immediately.
        // while tripped, refresh every frame (blanked anyway) so the
        // FIRST unblanked frame shows current values, not a stale snapshot.
        // jolt = hard discontinuity (seek, scene cut): resample NOW instead
        // of showing digits from up to 12 rendered frames ago — matters
        // most when paused-and-seeking, where renders are scarce and the
        // hold used to span several seek points (field-observed). Floors
        // keep near-black scenes from churning the digits every frame.
        // 9 gets its own jolt term: it's the row designed to move
        // independently of P (spike-immune) and A (percentile, not mean —
        // and undiluted by letterbox bars), so a cut can shift it hard
        // while neither of the other two crosses its 50% threshold.
        bool jolt = abs(pk - nm_show_peak)          > 0.5 * max(nm_show_peak, 100.0)
                 || abs(nm_p9999 - nm_show_p9999)   > 0.5 * max(nm_show_p9999, 100.0)
                 || abs(nm_fall - nm_show_fall)     > 0.5 * max(nm_show_fall, 20.0);
        if (nm_show_ctr <= 0.0 || jolt || tripped) {
            nm_show_peak  = pk;
            nm_show_p9999 = nm_p9999;
            nm_show_fall  = nm_fall;
            nm_show_ctr   = NM_SHOW_FRAMES;
        }
        nm_show_ctr -= 1.0;
        // per-frame P/9/A ring for the time-series strip. Written even with
        // nitmeter_graph off (3 floats/frame) so toggling it on has history;
        // tripped frames write 0 = a gap in the traces. Index clamp guards
        // a garbage first-load value from becoming an OOB SSBO write.
        int ri = clamp(int(nm_ring_idx), 0, NM_RING_LEN - 1);
        nm_ring_p[ri] = tripped ? 0.0 : pk;
        nm_ring_9[ri] = tripped ? 0.0 : nm_p9999;
        nm_ring_a[ri] = tripped ? 0.0 : nm_fall;
        nm_ring_idx = float((ri + 1) % NM_RING_LEN);
        imageStore(out_image, ivec2(0), vec4(0.0));
    }
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND NITMETER_STATE
//!BIND NITMETER_STATS
//!DESC NitMeter v1.6: overlay draw

// NITMETER_STATS is bound only as the explicit data dependency on PASS 2 —
// the stats themselves arrive through the NITMETER_STATE SSBO (same pattern
// as CelFlare PASS 6 / CELFLARE_STATS: don't remove the bind without
// verifying pass ordering/visibility on every backend).

// All tuning lives in the top-of-file PARAM block (v1.6; the old MAIN
// TUNING defines here are gone — corner/scale/guard/paint-nits are
// runtime params now, single-sourced across passes).

// derived, identical expression in PASS 2 — see the comment there
const float NM_HIST_LOG_HI = log2(10000.0);

#define NM_HIST_BINS   24       // MUST match the nm_hist[24] SSBO array + PASS 2
#define NM_RING_LEN    120      // MUST match the nm_ring_*[120] SSBO arrays + PASS 2
#define NM_PANEL_W     136.0
#if nitmeter_graph
#define NM_PANEL_H     255.0   // + time-series strip at y [207,247)
#else
#define NM_PANEL_H     207.0
#endif

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
// row label glyphs: P, 9 (99.99th-pct peak), H, A, C, F
const int NM_LBL[6] = int[6](0x73, 0x6F, 0x76, 0x77, 0x39, 0x71);

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
    // negative v (only reachable from NaN/garbage SSBO state on a
    // first load) would index NM_SEG out of bounds via GLSL's
    // sign-following % — clamp rather than UB
    v = max(v, 0);
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

// panel geometry: 6 label+digit rows, then the histogram strip
const float NM_ROW_Y[6] = float[6](8.0, 33.0, 58.0, 83.0, 108.0, 133.0);
// row colors as display nits: P white, 9 dim grey-white, H yellow, A cyan,
// C orange, F green
const vec3 NM_ROW_NITS[6] = vec3[6](
    vec3(230.0, 230.0, 230.0),
    vec3(150.0, 150.0, 150.0),
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
                          : pq_oetf3(heat_lin(n) * (nitmeter_heat_nits / 10000.0));
#endif
    }
#endif

    float S = nitmeter_scale * HOOKED_size.y / 1080.0;
#if nitmeter_corner == 0
    vec2 org = vec2(12.0, 12.0);
#elif nitmeter_corner == 1
    vec2 org = vec2(HOOKED_size.x / S - 12.0 - NM_PANEL_W, 12.0);
#elif nitmeter_corner == 2
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
        for (int r = 0; r < 6; r++) {
            m = glyph_mask(NM_LBL[r], vec2((lp.x - 8.0) / 20.0, (lp.y - NM_ROW_Y[r]) / 20.0));
            col = mix(col, pq_nits(NM_ROW_NITS[r]), m);
        }

        bool sdr_suspect = false;
#if nitmeter_sdr_guard
        sdr_suspect = nm_sdr > 0.5;
#endif
        if (sdr_suspect) {
            // input looks like SDR gamma misread as PQ: red border, and all
            // readouts blank to "---" instead of reporting nonsense
            vec2 db = min(lp, vec2(NM_PANEL_W, NM_PANEL_H) - lp);
            col = mix(col, pq_nits(vec3(180.0, 8.0, 8.0)),
                      0.9 * step(min(db.x, db.y), 3.0));
            for (int r = 0; r < 6; r++) {
                m = dashes_mask(vec2(rx - lp.x, lp.y - NM_ROW_Y[r]), 20.0);
                col = mix(col, pq_nits(NM_ROW_NITS[r]), m);
            }
            return vec4(col, color.a);
        }

        // digit rows: P peak (shown, ~0.5 s hold), 9 percentile peak
        // (shown), H peak-hold, A FALL (shown), C session MaxCLL,
        // F session MaxFALL
        float vals[6] = float[6](nm_show_peak, nm_show_p9999, nm_hold,
                                 nm_show_fall, nm_maxcll, nm_maxfall);
        for (int r = 0; r < 6; r++) {
            m = value_mask(vals[r], vec2(rx - lp.x, lp.y - NM_ROW_Y[r]), 20.0);
            col = mix(col, pq_nits(NM_ROW_NITS[r]), m);
        }

        // display-ceiling indicator (P row): CONTENT peak (9 row) vs
        // nitmeter_target — v1.5 repointed H at the percentile for spike
        // immunity; v1.6 finishes that for the verdict square (strict-max
        // drive went red on lone ringing spikes no display visibly clips).
        // Deliberately BLIND to the strict max: a "spike-only" state was
        // tried and audit-killed (P > target is the COMMON case on ringing
        // web re-encodes, so it made green/yellow unreachable exactly
        // there) — the P digits, the P-9 gap, and the dim histogram tick
        // already carry the spike story.
        vec2 td = abs(lp - vec2(30.0, 18.0)) - vec2(4.0);
        float tm = step(max(td.x, td.y), 0.0);
        vec3 tc = (nm_show_p9999 > nitmeter_target)        ? vec3(230.0, 15.0, 15.0)
                : (nm_show_p9999 > 0.95 * nitmeter_target) ? vec3(230.0, 200.0, 20.0)
                                                           : vec3(25.0, 210.0, 25.0);
        col = mix(col, pq_nits(tc), tm);

        // histogram strip: log2 axis, 1 -> 10000 nits
        if (lp.x >= 8.0 && lp.x < 128.0 && lp.y >= 159.0 && lp.y < 199.0) {
            float hx = (lp.x - 8.0) / 120.0;
            // reference ticks at 100 / 203 / 1000 / 4000 nits (positions
            // derived, constant-folded — v1.5 hardcoded the four fractions,
            // which silently depended on the axis-top constant)
            float tick = step(abs(hx - log2(100.0)  / NM_HIST_LOG_HI), 0.004);
            tick = max(tick, step(abs(hx - log2(203.0)  / NM_HIST_LOG_HI), 0.004));
            tick = max(tick, step(abs(hx - log2(1000.0) / NM_HIST_LOG_HI), 0.004));
            tick = max(tick, step(abs(hx - log2(4000.0) / NM_HIST_LOG_HI), 0.004));
            col = mix(col, pq_nits(vec3(25.0)), tick);

            // guard bounds already force bin < 24; clamp anyway so a future
            // geometry edit can't silently reopen an OOB SSBO read
            int bin = clamp(int(hx * float(NM_HIST_BINS)), 0, NM_HIST_BINS - 1);
            float bh = pow(clamp(nm_hist[bin], 0.0, 1.0), 0.4);   // gamma'd so small areas stay visible
            float yy = (199.0 - lp.y) / 40.0;                     // 0 bottom .. 1 top
            if (yy < bh) {
                float cn = exp2((float(bin) + 0.5) / float(NM_HIST_BINS) * NM_HIST_LOG_HI);
                vec3 bc = (cn < 100.0) ? vec3(0.5) : heat_lin(cn);
                col = mix(col, pq_oetf3(bc * (nitmeter_heat_nits / 10000.0)), 0.9);
            }
            // markers (live, not held): thin dim = strict-max P, bright =
            // 9 (content peak). The bright one used to be P, which floats
            // right of every bar on ringing spikes — same repoint as the
            // ceiling square.
            float pmx = clamp(log2(max(nm_peak, 1.0)) / NM_HIST_LOG_HI, 0.0, 1.0);
            col = mix(col, pq_nits(vec3(60.0)), step(abs(hx - pmx), 0.004));
            float pkx = clamp(log2(max(nm_p9999, 1.0)) / NM_HIST_LOG_HI, 0.0, 1.0);
            col = mix(col, pq_nits(vec3(300.0)), step(abs(hx - pkx), 0.005));
        }

#if nitmeter_graph
        // time-series strip: last NM_RING_LEN rendered frames of P (white)
        // / 9 (grey) / A (cyan) on the same log2 1..10000 axis, newest at
        // the right — a live scope for the light-pump envelope. nm_ring_idx
        // is the next write slot = the OLDEST sample, so column c maps to
        // (idx + c) % len. Tripped-frame samples are 0 and draw as gaps.
        if (lp.x >= 8.0 && lp.x < 128.0 && lp.y >= 207.0 && lp.y < 247.0) {
            int c = int(lp.x - 8.0);
            int ri = (clamp(int(nm_ring_idx), 0, NM_RING_LEN - 1) + c) % NM_RING_LEN;
            float yy = (247.0 - lp.y) / 40.0;                     // 0 bottom .. 1 top
            // dim line at the display target
            float ty = clamp(log2(max(nitmeter_target, 1.0)) / NM_HIST_LOG_HI, 0.0, 1.0);
            col = mix(col, pq_nits(vec3(25.0)), step(abs(yy - ty), 0.02));
            // traces, ±1.5 px band; P drawn last so it wins overlaps. The
            // v>0 gate makes 0-samples gaps: tripped frames by design, and
            // true-black frames land there too (nothing to plot anyway)
            vec3 tv = vec3(nm_ring_a[ri], nm_ring_9[ri], nm_ring_p[ri]);
            vec3 th = clamp(log2(max(tv, vec3(1.0))) / NM_HIST_LOG_HI, 0.0, 1.0);
            if (tv.x > 0.0 && abs(yy - th.x) < 0.0375) col = mix(col, pq_nits(NM_ROW_NITS[3]), 1.0);
            if (tv.y > 0.0 && abs(yy - th.y) < 0.0375) col = mix(col, pq_nits(NM_ROW_NITS[1]), 1.0);
            if (tv.z > 0.0 && abs(yy - th.z) < 0.0375) col = mix(col, pq_nits(NM_ROW_NITS[0]), 1.0);
        }
#endif
    }
    return vec4(col, color.a);
}
