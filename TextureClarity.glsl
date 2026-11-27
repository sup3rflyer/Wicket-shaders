// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE

// User tuning: the tc_* params below (DYNAMIC — glsl-shader-opts
// changes apply on the next frame, no recompile). Defaults = the shipped
// tune. Everything below the param block is internal.
// shampv shader API (plain comments to libplacebo):
//@shampv input any

//!PARAM tc_strength
//!DESC Sharpening strength. ↑ = sharper · 12 = shipped tune · 0 = off.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 24.0
12.0

//!PARAM tc_coring
//!DESC Coring threshold — high-pass deltas below this are ignored (stops noise being sharpened). ↑ for noisier sources · ↓ sharpens finer detail.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 0.003
0.0005

//!PARAM tc_texture_thresh
//!DESC Texture floor — minimum local variance treated as real texture, not noise. ↑ = sharpen only stronger detail · ↓ = also lift faint texture.
//!TYPE DYNAMIC float
//!MINIMUM 0.005
//!MAXIMUM 0.08
0.02

//!PARAM tc_max_delta
//!DESC Per-pixel luma change cap — hard ceiling on how far sharpening may move any pixel. ↑ = allow bigger swings · ↓ = safer.
//!TYPE DYNAMIC float
//!MINIMUM 0.001
//!MAXIMUM 0.05
0.012

//!PARAM tc_line_guard
//!DESC Line guard. The sharpening fades where the 5x5 neighbourhood holds an edge or a line (oriented gradient), so the foot of an outline is not steepened and the faint ripple beside it is not lifted. Grain and texture, which point every way, pass. 1 = on (shipped) · 0 = off, only the centre pixel's own edge test applies.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM tc_grain_bypass
//!DESC Grain bypass. The grain level is measured over the whole picture, per brightness band, from its flattest parts; the sharpening acts only where the local detail rises above it, so grain on flat areas is left as it is while texture, cloth and painted detail are still lifted (with the grain riding on them). 1 = on (shipped) · 0 = off, grain is sharpened like texture.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM tc_range
//!DESC Luma range of the video (choice). 1 = limited, 16-235 (nearly all video: discs, broadcast, streaming, the AnimeJaNai upscaler output): the near-black / near-white window and the grain-floor bands sit at signal levels · 0 = full range, 0-255 (screen captures, yuvj): also the picture of the file before this switch.
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM tc_pq
//!DESC PQ source (native HDR10, Dolby Vision base layer). 1 = every level test, the texture and grain tests and the size of the change work on the luma an SDR picture of the same light would have (BT.1886 2.4, white = tc_ref_white), so HDR is sharpened like the same picture in SDR; the near-white fade runs from about 0.68 to 0.88 x tc_ref_white in light (137-180 nits at 203) and nothing brighter is touched · 0 = the plane as it is (all SDR; SDR retagged as PQ for CelFlare must stay 0; HLG stays 0).
//!TYPE DYNAMIC float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM tc_ref_white
//!DESC With tc_pq: the nit level of the source's diffuse white, which plays SDR white. 203 = BT.2408 (where HDR masters place diffuse white). Lower = the window closes lower and the same nits read brighter.
//!TYPE DYNAMIC float
//!MINIMUM 80.0
//!MAXIMUM 480.0
203.0

//!HOOK LUMA
//!BIND HOOKED
//!SAVE TC_GRAIN_E
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!COMPONENTS 2
//!DESC Texture Clarity: local energy 1/4

// x: mean high-pass energy of the central 2x2 pixels of this 4x4 cell (the 4x4 read covers their 3x3
// neighbourhoods), times GE_SCALE (keeps grain-level energies far from the half-float floor).
// y: mean luma of the 16 pixels.
#define GE_SCALE 10000.0

// Range and transfer of the luma plane (tc_range, tc_pq). Levels are compared as SIGNAL: on limited range
// code 16/255 is black and 235/255 is white (the hook sees the plane as decoded, not normalised). With
// tc_pq each PQ plane value is carried to the code an SDR picture of the same light would have (BT.1886
// gamma 2.4, white = tc_ref_white nits), in the same plane range, so every threshold, the coring and the
// cap keep their SDR meaning in light; above white it continues past 1 (no clip). Differences are not
// rescaled by the range. The texture test, coring, cap, edge gate and detail tests were tuned by eye in
// the player, on limited-range planes, so they already mean plane units. The line-guard thresholds
// (LINE_GUARD_LO / HI) were set on a bench that fed full-range signal, so in the player they act at a
// signal gradient about 16 % (255 / 219) larger than scored; that was so before tc_range and still is.
// Each hook is its own unit, so this block is repeated where it is needed.
const float TC_LIM_BLACK = 16.0 / 255.0;
const float TC_LIM_SPAN = 219.0 / 255.0;
float tc_black() { return tc_range > 0.5 ? TC_LIM_BLACK : 0.0; }
float tc_span() { return tc_range > 0.5 ? TC_LIM_SPAN : 1.0; }
float tc_level(float v) { return tc_range > 0.5 ? (v - TC_LIM_BLACK) * (1.0 / TC_LIM_SPAN) : v; }
// ST 2084 constants: m1 0.1593017578125, m2 78.84375, c1 0.8359375, c2 18.8515625, c3 18.6875.
float tc_to_sdr(float v) {
    float e = clamp((v - tc_black()) / tc_span(), 0.0, 1.0);
    float q = pow(e, 1.0 / 78.84375);
    float r = max(q - 0.8359375, 0.0) / (18.8515625 - 18.6875 * q);
    float s = pow(10000.0 / max(tc_ref_white, 1.0), 1.0 / 2.4) * pow(r, 1.0 / (2.4 * 0.1593017578125));
    return tc_black() + s * tc_span();
}
float tc_from_sdr(float v) {
    float s = max((v - tc_black()) / tc_span(), 0.0);
    float y = pow(max(tc_ref_white, 1.0) / 10000.0, 0.1593017578125) * pow(s, 2.4 * 0.1593017578125);
    float e = pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), 78.84375);
    return tc_black() + e * tc_span();
}

vec4 hook() {
    if (tc_grain_bypass <= 0.0)
        return vec4(0.0);
    // Cell origin on the pixel grid: with a size that is not a multiple of 4
    // the cell pitch is not 4 px, and taps around this texel's own centre
    // drifted off pixel centres (bilinear-blurred in bands across the frame).
    vec2 cell = (floor(HOOKED_pos * HOOKED_size * 0.25) * 4.0 + 2.0) * HOOKED_pt;
    float g[16];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            g[j * 4 + i] = HOOKED_tex(cell + vec2(float(i) - 1.5, float(j) - 1.5) * HOOKED_pt).x;
    // tc_pq: the energy and the mean luma of the cell on the SDR-equivalent luma. Its own copy and its
    // own return, so the plain path below is the same code as before the switch.
    if (tc_pq > 0.5) {
        float h[16];
        h[0] = tc_to_sdr(g[0]);
        h[1] = tc_to_sdr(g[1]);
        h[2] = tc_to_sdr(g[2]);
        h[3] = tc_to_sdr(g[3]);
        h[4] = tc_to_sdr(g[4]);
        h[5] = tc_to_sdr(g[5]);
        h[6] = tc_to_sdr(g[6]);
        h[7] = tc_to_sdr(g[7]);
        h[8] = tc_to_sdr(g[8]);
        h[9] = tc_to_sdr(g[9]);
        h[10] = tc_to_sdr(g[10]);
        h[11] = tc_to_sdr(g[11]);
        h[12] = tc_to_sdr(g[12]);
        h[13] = tc_to_sdr(g[13]);
        h[14] = tc_to_sdr(g[14]);
        h[15] = tc_to_sdr(g[15]);
#define GE_TQ(I) (h[I] - (4.0 * h[I] + 2.0 * (h[I - 1] + h[I + 1] + h[I - 4] + h[I + 4]) + h[I - 5] + h[I - 3] + h[I + 3] + h[I + 5]) * (1.0 / 16.0))
        float q0 = GE_TQ(5);
        float q1 = GE_TQ(6);
        float q2 = GE_TQ(9);
        float q3 = GE_TQ(10);
        float lq = h[0] + h[1] + h[2] + h[3] + h[4] + h[5] + h[6] + h[7] + h[8] + h[9] + h[10] + h[11] + h[12] + h[13] + h[14] + h[15];
        return vec4((q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) * (0.25 * GE_SCALE), lq * (1.0 / 16.0), 0.0, 1.0);
    }
#define GE_T(I) (g[I] - (4.0 * g[I] + 2.0 * (g[I - 1] + g[I + 1] + g[I - 4] + g[I + 4]) + g[I - 5] + g[I - 3] + g[I + 3] + g[I + 5]) * (1.0 / 16.0))
    float t0 = GE_T(5);
    float t1 = GE_T(6);
    float t2 = GE_T(9);
    float t3 = GE_T(10);
    float l = 0.0;
    for (int k = 0; k < 16; k++)
        l += g[k];
    return vec4((t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3) * (0.25 * GE_SCALE), l * (1.0 / 16.0), 0.0, 1.0);
}

//!HOOK LUMA
//!BIND HOOKED
//!BIND TC_GRAIN_E
//!SAVE TC_GRAIN_FLOOR
//!WIDTH 8
//!HEIGHT 1
//!COMPONENTS 1
//!COMPUTE 32 32
//!DESC Texture Clarity: grain floor

// One workgroup. Block k of a GF_GX x GF_GY grid = one 2x2 group of TC_GRAIN_E cells, read with a single
// bilinear fetch at the group's centre (equal weights: the mean of the four). Blocks outside luma
// GF_Y_LO-GF_Y_HI are left out (crushed blacks, letterbox and clipped whites hold no grain). Per luma band
// (GF_BINS), a histogram of log2(energy); the band's floor is its GF_PCT percentile, stored as RMS. A band
// with fewer than GF_MIN_COUNT blocks gets 0 (no bypass at that brightness).
// The band limits and the bands are signal levels (tc_level, tc_range).
// Every barrier sits at top level: FXC X3663 needs barriers in uniform control flow.
#define GF_GX 96
#define GF_GY 54
#define GF_BINS 8
#define GF_HB 64
#define GF_LOG_MIN (-27.0)
#define GF_LOG_SPAN 21.0
#define GF_PCT 15.0000
#define GF_MIN_COUNT 12u
#define GF_Y_LO 0.05
#define GF_Y_HI 0.95

// Range and transfer of the luma plane (tc_range, tc_pq). Levels are compared as SIGNAL: on limited range
// code 16/255 is black and 235/255 is white (the hook sees the plane as decoded, not normalised). With
// tc_pq each PQ plane value is carried to the code an SDR picture of the same light would have (BT.1886
// gamma 2.4, white = tc_ref_white nits), in the same plane range, so every threshold, the coring and the
// cap keep their SDR meaning in light; above white it continues past 1 (no clip). Differences are not
// rescaled by the range. The texture test, coring, cap, edge gate and detail tests were tuned by eye in
// the player, on limited-range planes, so they already mean plane units. The line-guard thresholds
// (LINE_GUARD_LO / HI) were set on a bench that fed full-range signal, so in the player they act at a
// signal gradient about 16 % (255 / 219) larger than scored; that was so before tc_range and still is.
// Each hook is its own unit, so this block is repeated where it is needed.
const float TC_LIM_BLACK = 16.0 / 255.0;
const float TC_LIM_SPAN = 219.0 / 255.0;
float tc_black() { return tc_range > 0.5 ? TC_LIM_BLACK : 0.0; }
float tc_span() { return tc_range > 0.5 ? TC_LIM_SPAN : 1.0; }
float tc_level(float v) { return tc_range > 0.5 ? (v - TC_LIM_BLACK) * (1.0 / TC_LIM_SPAN) : v; }

shared uint gf_hist[GF_BINS * GF_HB];

void hook() {
    uint lid = gl_LocalInvocationIndex;
    if (lid < uint(GF_BINS * GF_HB))
        gf_hist[lid] = 0u;
    barrier();
    vec2 cells = TC_GRAIN_E_size;
    vec2 pairs = floor(cells * 0.5);
    for (uint k = lid; k < uint(GF_GX * GF_GY); k += 1024u) {
        vec2 ab = floor((vec2(float(k % uint(GF_GX)), float(k / uint(GF_GX))) + 0.5) * pairs / vec2(GF_GX, GF_GY));
        vec2 v = TC_GRAIN_E_tex((2.0 * ab + 1.0) / cells).xy;
        float e = v.x * (1.0 / 10000.0);
        float l = v.y;
        if (tc_range > 0.5)
            l = tc_level(l);
        if (l >= GF_Y_LO && l <= GF_Y_HI) {
            int bin = clamp(int(l * float(GF_BINS)), 0, GF_BINS - 1);
            int hb = clamp(int((log2(max(e, 1e-30)) - GF_LOG_MIN) * (float(GF_HB) / GF_LOG_SPAN)), 0, GF_HB - 1);
            atomicAdd(gf_hist[bin * GF_HB + hb], 1u);
        }
    }
    barrier();
    if (lid < uint(GF_BINS)) {
        uint n = 0u;
        for (int h = 0; h < GF_HB; h++)
            n += gf_hist[int(lid) * GF_HB + h];
        float rms = 0.0;
        if (n >= GF_MIN_COUNT) {
            // rank inside the cumulative histogram, interpolated inside the bin (log domain)
            float target = GF_PCT * 0.01 * float(n);
            float cum = 0.0;
            float lg = GF_LOG_MIN;
            for (int h = 0; h < GF_HB; h++) {
                float c = float(gf_hist[int(lid) * GF_HB + h]);
                if (c > 0.0 && cum + c >= target) {
                    lg = GF_LOG_MIN + (float(h) + clamp((target - cum) / c, 0.0, 1.0)) * (GF_LOG_SPAN / float(GF_HB));
                    break;
                }
                cum += c;
            }
            rms = exp2(0.5 * lg);
        }
        imageStore(out_image, ivec2(int(lid), 0), vec4(rms, 0.0, 0.0, 1.0));
    }
}

//!HOOK LUMA
//!BIND HOOKED
//!BIND TC_GRAIN_FLOOR
//!BIND TC_GRAIN_E
//!DESC Texture Clarity

// Grain suppression parameters
const float grain_threshold = 0.045;
const float grain_transition_width = 0.50;

// Texture discrimination
const float texture_transition_width = 0.018;

// Black/white area safeguards: the sharpening fades in over signal 0.05-0.12 and
// out over 0.85-0.95 (signal levels: on limited range 0 = code 16, 1 = code 235; tc_range). (Was 0.15-0.25 and 0.35-0.55: that window held 21-65 % of
// the texture in a set of 34 masters, and nothing outside it was ever touched.
// Scored band by band against the masters, what the shader adds agrees with
// the master's detail equally from near-black to near-white.) Below 0.05 sit
// crushed blacks and letterbox; above 0.95 there is nothing left to lift, and a
// highlight expansion that runs later multiplies small luma differences there.
const float black_threshold = 0.05;
const float black_falloff = 0.07;
const float white_threshold = 0.95;
const float white_falloff = 0.10;

// Enhanced control parameters
const float EDGE_MULTIPLIER = 6.0;
// Line guard (tc_line_guard): fade by the oriented gradient energy of the 5x5.
// The centre's own Sobel is small at the FOOT of an edge, where the curvature
// the high-pass reads is largest, so the edge gate alone let the shader steepen
// the shoulders of lines. The structure tensor keeps an edge's or a line's
// energy (all gradients along one axis, either polarity) and cancels grain,
// which is isotropic. Units: luma per pixel.
const float LINE_GUARD_LO = 0.010;
const float LINE_GUARD_HI = 0.018;
#define TC_TENSOR(GX, GY) { float gx = GX; float gy = GY; jxx += gx * gx; jyy += gy * gy; jxy += gx * gy; }
const float DETAIL_LOW = 0.003;
// Grain bypass (tc_grain_bypass): the sharpening fades in as the local high-pass energy (TC_GRAIN_E) rises
// from GB_LO to GB_HI times the grain floor at the pixel's brightness (TC_GRAIN_FLOOR). On a field of grain
// the ratio sits around 2 (the floor is a low percentile of 16-sample blocks).
const float GB_LO = 1.500;
const float GB_HI = 6.000;
#define GB_PASS(R) smoothstep(GB_LO, GB_HI, R)

// Range and transfer of the luma plane (tc_range, tc_pq). Levels are compared as SIGNAL: on limited range
// code 16/255 is black and 235/255 is white (the hook sees the plane as decoded, not normalised). With
// tc_pq each PQ plane value is carried to the code an SDR picture of the same light would have (BT.1886
// gamma 2.4, white = tc_ref_white nits), in the same plane range, so every threshold, the coring and the
// cap keep their SDR meaning in light; above white it continues past 1 (no clip). Differences are not
// rescaled by the range. The texture test, coring, cap, edge gate and detail tests were tuned by eye in
// the player, on limited-range planes, so they already mean plane units. The line-guard thresholds
// (LINE_GUARD_LO / HI) were set on a bench that fed full-range signal, so in the player they act at a
// signal gradient about 16 % (255 / 219) larger than scored; that was so before tc_range and still is.
// Each hook is its own unit, so this block is repeated where it is needed.
const float TC_LIM_BLACK = 16.0 / 255.0;
const float TC_LIM_SPAN = 219.0 / 255.0;
float tc_black() { return tc_range > 0.5 ? TC_LIM_BLACK : 0.0; }
float tc_span() { return tc_range > 0.5 ? TC_LIM_SPAN : 1.0; }
float tc_level(float v) { return tc_range > 0.5 ? (v - TC_LIM_BLACK) * (1.0 / TC_LIM_SPAN) : v; }
// ST 2084 constants: m1 0.1593017578125, m2 78.84375, c1 0.8359375, c2 18.8515625, c3 18.6875.
float tc_to_sdr(float v) {
    float e = clamp((v - tc_black()) / tc_span(), 0.0, 1.0);
    float q = pow(e, 1.0 / 78.84375);
    float r = max(q - 0.8359375, 0.0) / (18.8515625 - 18.6875 * q);
    float s = pow(10000.0 / max(tc_ref_white, 1.0), 1.0 / 2.4) * pow(r, 1.0 / (2.4 * 0.1593017578125));
    return tc_black() + s * tc_span();
}
float tc_from_sdr(float v) {
    float s = max((v - tc_black()) / tc_span(), 0.0);
    float y = pow(max(tc_ref_white, 1.0) / 10000.0, 0.1593017578125) * pow(s, 2.4 * 0.1593017578125);
    float e = pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), 78.84375);
    return tc_black() + e * tc_span();
}
const float GE_SCALE_INV = 1.0 / 10000.0;
const float DETAIL_HIGH = 0.013;

// Cinematic boost parameters
// Max multiplier applied to sharpen_amount in lowest-variance areas.
const float LOWDETAIL_MAX_BOOST = 1.5;  // final cap (1.0 = no boost, 1.5 = +50%)
// Curve shape is a fixed sqrt() (was pow(.., 0.5)); see the boost block below.

float get_grid(int x, int y, float grid[25]) {
    return grid[(y + 2) * 5 + (x + 2)];
}

vec4 hook() {
    // Pre-sample 5x5 neighborhood
    float grid[25];
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            grid[(y + 2) * 5 + (x + 2)] = HOOKED_texOff(vec2(x, y)).x;
        }
    }
    // tc_pq: everything below works on the SDR-equivalent luma; the change is carried back at
    // the end (constant indices: see the line guard's note on D3D11)
    float p_raw = grid[12];
    if (tc_pq > 0.5) {
        grid[0] = tc_to_sdr(grid[0]);
        grid[1] = tc_to_sdr(grid[1]);
        grid[2] = tc_to_sdr(grid[2]);
        grid[3] = tc_to_sdr(grid[3]);
        grid[4] = tc_to_sdr(grid[4]);
        grid[5] = tc_to_sdr(grid[5]);
        grid[6] = tc_to_sdr(grid[6]);
        grid[7] = tc_to_sdr(grid[7]);
        grid[8] = tc_to_sdr(grid[8]);
        grid[9] = tc_to_sdr(grid[9]);
        grid[10] = tc_to_sdr(grid[10]);
        grid[11] = tc_to_sdr(grid[11]);
        grid[12] = tc_to_sdr(grid[12]);
        grid[13] = tc_to_sdr(grid[13]);
        grid[14] = tc_to_sdr(grid[14]);
        grid[15] = tc_to_sdr(grid[15]);
        grid[16] = tc_to_sdr(grid[16]);
        grid[17] = tc_to_sdr(grid[17]);
        grid[18] = tc_to_sdr(grid[18]);
        grid[19] = tc_to_sdr(grid[19]);
        grid[20] = tc_to_sdr(grid[20]);
        grid[21] = tc_to_sdr(grid[21]);
        grid[22] = tc_to_sdr(grid[22]);
        grid[23] = tc_to_sdr(grid[23]);
        grid[24] = tc_to_sdr(grid[24]);
    }

    float p_x = grid[12];
    vec4 p = vec4(p_x, 0.0, 0.0, 1.0);

    // 3x3 window analysis
    float sum_grain = 0.0;
    float sum_texture = 0.0;
    float window_sum = 0.0;
    float window_sum_sq = 0.0;
    float grain_center = 0.0;
    float texture_center = 0.0;
    float max_dev_center = 0.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float n = get_grid(x, y, grid);

            // 4 diagonals around (x,y)
            float s0[4];
            s0[0] = get_grid(x + 1, y + 1, grid);
            s0[1] = get_grid(x + 1, y - 1, grid);
            s0[2] = get_grid(x - 1, y + 1, grid);
            s0[3] = get_grid(x - 1, y - 1, grid);

            // Compute deviation and local statistics
            float max_dev = 0.0;
            float min_val = s0[0];
            float max_val = s0[0];
            float local_mean_dev = 0.0;

            for (int j = 0; j < 4; j++) {
                float dv = abs(n - s0[j]);
                max_dev = max(max_dev, dv);
                min_val = min(min_val, s0[j]);
                max_val = max(max_val, s0[j]);
                local_mean_dev += dv;
            }
            local_mean_dev *= 0.25; // true mean deviation

            float thresh = grain_threshold + 0.5 * local_mean_dev;
            float grain = 1.0 - smoothstep(thresh, thresh + grain_transition_width, max_dev);
            float variance = max_val - min_val;
            float texture = smoothstep(tc_texture_thresh, tc_texture_thresh + texture_transition_width, variance);

            sum_grain += grain;
            sum_texture += texture;
            window_sum += n;
            window_sum_sq += n * n;

            if (x == 0 && y == 0) {
                grain_center = grain;
                texture_center = texture;
                max_dev_center = max_dev;
            }
        }
    }

    float avg_grain = sum_grain / 9.0;
    float avg_texture = sum_texture / 9.0;
    float window_mean = window_sum / 9.0;
    // E[x^2] - E[x]^2 form (numerically loose, but fine for luma in [0,1]); the tiny
    // negative FP results it can produce are absorbed by the smoothstep/max() below.
    float window_variance = window_sum_sq / 9.0 - window_mean * window_mean;

    // Local texture blending
    float blend = smoothstep(0.0, 0.003, window_variance);
    float blend_sq = blend * blend;
    float final_grain = mix(avg_grain, grain_center, blend_sq);
    float final_texture = mix(avg_texture, texture_center, blend_sq);

    // Multi-scale factor (vectorized)
    vec4 s1 = vec4(grid[24], grid[4], grid[20], grid[0]);
    vec4 dv_large = abs(p.x - s1);
    float max_dev_large = max(max(dv_large.x, dv_large.y), max(dv_large.z, dv_large.w));
    float scale_diff = max_dev_center - max_dev_large;
    final_grain *= (1.0 - smoothstep(0.0, 0.05, scale_diff));

    // Unsharp mask — isotropic 3x3 binomial high-pass (center 4, edges 2, corners 1; / 16).
    // Replaces the old 5-tap axis cross, which had no diagonal taps and therefore
    // under-sharpened diagonal texture/edges (mild axis-aligned anisotropy). All taps
    // already live in grid[] — this costs zero extra texture fetches.
    float blurred = (4.0 * p.x
                     + 2.0 * (get_grid(1,0,grid) + get_grid(-1,0,grid) +
                              get_grid(0,1,grid) + get_grid(0,-1,grid))
                     + (get_grid(1,1,grid)  + get_grid(-1,1,grid) +
                        get_grid(1,-1,grid) + get_grid(-1,-1,grid))) * (1.0 / 16.0);
    float t = p.x - blurred;
    float t_sharp = t - clamp(t, -tc_coring, tc_coring);

    // Sobel edge detection
    float sobel_h = (grid[8] - grid[6]) + 2.0 * (grid[13] - grid[11]) + (grid[18] - grid[16]);
    float sobel_v = (grid[16] - grid[6]) + 2.0 * (grid[17] - grid[7]) + (grid[18] - grid[8]);
    float edge = abs(sobel_h) + abs(sobel_v);
    float edge_factor = clamp(1.0 - edge * EDGE_MULTIPLIER, 0.0, 1.0);

    // Line guard: structure tensor of plain differences at all 25 positions
    // (central inside the window, one-sided on its rim). CONSTANT indices on
    // purpose: a loop over get_grid() made the D3D11 compiler index the array
    // dynamically and copy it per call (measured 20 ms against 0.14 ms at 4K).
    float jxx = 0.0;
    float jyy = 0.0;
    float jxy = 0.0;
    TC_TENSOR(grid[1] - grid[0], grid[5] - grid[0])
    TC_TENSOR(0.5 * (grid[2] - grid[0]), grid[6] - grid[1])
    TC_TENSOR(0.5 * (grid[3] - grid[1]), grid[7] - grid[2])
    TC_TENSOR(0.5 * (grid[4] - grid[2]), grid[8] - grid[3])
    TC_TENSOR(grid[4] - grid[3], grid[9] - grid[4])
    TC_TENSOR(grid[6] - grid[5], 0.5 * (grid[10] - grid[0]))
    TC_TENSOR(0.5 * (grid[7] - grid[5]), 0.5 * (grid[11] - grid[1]))
    TC_TENSOR(0.5 * (grid[8] - grid[6]), 0.5 * (grid[12] - grid[2]))
    TC_TENSOR(0.5 * (grid[9] - grid[7]), 0.5 * (grid[13] - grid[3]))
    TC_TENSOR(grid[9] - grid[8], 0.5 * (grid[14] - grid[4]))
    TC_TENSOR(grid[11] - grid[10], 0.5 * (grid[15] - grid[5]))
    TC_TENSOR(0.5 * (grid[12] - grid[10]), 0.5 * (grid[16] - grid[6]))
    TC_TENSOR(0.5 * (grid[13] - grid[11]), 0.5 * (grid[17] - grid[7]))
    TC_TENSOR(0.5 * (grid[14] - grid[12]), 0.5 * (grid[18] - grid[8]))
    TC_TENSOR(grid[14] - grid[13], 0.5 * (grid[19] - grid[9]))
    TC_TENSOR(grid[16] - grid[15], 0.5 * (grid[20] - grid[10]))
    TC_TENSOR(0.5 * (grid[17] - grid[15]), 0.5 * (grid[21] - grid[11]))
    TC_TENSOR(0.5 * (grid[18] - grid[16]), 0.5 * (grid[22] - grid[12]))
    TC_TENSOR(0.5 * (grid[19] - grid[17]), 0.5 * (grid[23] - grid[13]))
    TC_TENSOR(grid[19] - grid[18], 0.5 * (grid[24] - grid[14]))
    TC_TENSOR(grid[21] - grid[20], grid[20] - grid[15])
    TC_TENSOR(0.5 * (grid[22] - grid[20]), grid[21] - grid[16])
    TC_TENSOR(0.5 * (grid[23] - grid[21]), grid[22] - grid[17])
    TC_TENSOR(0.5 * (grid[24] - grid[22]), grid[23] - grid[18])
    TC_TENSOR(grid[24] - grid[23], grid[24] - grid[19])
    float j_d = (jxx - jyy) * (1.0 / 25.0);
    float j_o = jxy * (2.0 / 25.0);
    float edge_oriented = sqrt(sqrt(j_d * j_d + j_o * j_o));
    float line_guard = 1.0 - tc_line_guard * smoothstep(LINE_GUARD_LO, LINE_GUARD_HI, edge_oriented);

    // Suppress sharpening in high-detail areas
    float detail_saturation = 1.0 - smoothstep(DETAIL_LOW, DETAIL_HIGH, window_variance);

    // Combine influences
    float combined = edge_factor * line_guard * final_grain * final_texture * detail_saturation;

    // Grain bypass: local energy against the grain floor at this brightness (bands centred at (k + 0.5) / 8,
    // linear between them, flat beyond the end ones). A band without enough blocks has floor 0: no bypass.
    if (tc_grain_bypass > 0.0) {
        float fb = clamp(window_mean * 8.0 - 0.5, 0.0, 7.0);
        if (tc_range > 0.5)
            fb = clamp(tc_level(window_mean) * 8.0 - 0.5, 0.0, 7.0);
        float f0i = floor(fb);
        float f0 = TC_GRAIN_FLOOR_tex(vec2((f0i + 0.5) / 8.0, 0.5)).x;
        float f1 = TC_GRAIN_FLOOR_tex(vec2((min(f0i + 1.0, 7.0) + 0.5) / 8.0, 0.5)).x;
        float floor_e = mix(f0 * f0, f1 * f1, fb - f0i);
        float e_loc = TC_GRAIN_E_tex(HOOKED_pos).x * GE_SCALE_INV;
        float grain_pass = floor_e > 0.0 ? GB_PASS(e_loc / floor_e) : 1.0;
        combined *= 1.0 - tc_grain_bypass * (1.0 - grain_pass);
    }
    float sharpen_amount = t_sharp * tc_strength * combined;

    // -----------------------
    // Mild-detail boost (applied BEFORE black/white mask)
    // -----------------------
    // Goal: lift the sharpening a little more in MILD / low-amplitude texture, while
    // leaving already-sharp (high-variance) detail untouched. The boost rides on top of
    // sharpen_amount, which the texture/detail gates have already zeroed in flat areas,
    // so this only acts where there is some texture to begin with.
    // NOTE: keyed on window_variance alone, so it cannot distinguish mild distributed
    // texture from sparse isolated detail at the same low variance — see audit notes.
    float lowdetail_gain = 1.0 - smoothstep(0.0, DETAIL_LOW, window_variance);
    // Concave (square-root) curve emphasizes the milder end of the range.
    // sqrt(x) == pow(x, 0.5); the guard keeps the input non-negative.
    lowdetail_gain = sqrt(max(lowdetail_gain, 0.0));
    // Map to a multiplier between 1.0 and LOWDETAIL_MAX_BOOST
    float lowdetail_boost = mix(1.0, LOWDETAIL_MAX_BOOST, lowdetail_gain);

    sharpen_amount *= lowdetail_boost;
    // -----------------------
    // End cinematic boost
    // -----------------------

    // Black/white luminance masks
    float mask = smoothstep(black_threshold, black_threshold + black_falloff, p.x) *
                 (1.0 - smoothstep(white_threshold - white_falloff, white_threshold, p.x));
    if (tc_range > 0.5) {
        float lv = tc_level(p.x);
        mask = smoothstep(black_threshold, black_threshold + black_falloff, lv) *
               (1.0 - smoothstep(white_threshold - white_falloff, white_threshold, lv));
    }

    sharpen_amount *= mask;

    // Cap on the brightness change, eased: small changes pass unchanged, large
    // ones approach tc_max_delta smoothly (the hard clip left a third of the
    // touched pixels of a grainy picture sitting exactly on the cap).
    float result = p.x + tc_max_delta * tanh(clamp(sharpen_amount / tc_max_delta, -10.0, 10.0));
    // tc_pq: the same change in light, carried back to the PQ plane (exactly 0 where nothing moved)
    if (tc_pq > 0.5)
        result = p_raw + (tc_from_sdr(result) - tc_from_sdr(p.x));

    return vec4(result, 0.0, 0.0, 1.0);
}
