// FilmGrainSmooth.fx - ReShade port of mpv filmgrain-smooth
// Original GLSL by Agust Ari, GPL-3.0
// Six presets (SDR/HDR x Light/Medium/Heavy) in a single shader.
//
// HDR-safe: no saturate(), no clamp(0,1), no backbuffer format override.

#include "ReShade.fxh"

// =============================================================================
// UI
// =============================================================================

uniform int Preset <
    ui_type = "combo";
    ui_label = "Preset";
    ui_items = "SDR Light\0SDR Medium\0SDR Heavy\0HDR Light\0HDR Medium\0HDR Heavy\0";
    ui_tooltip = "Match your backbuffer encoding.\n"
                 "  SDR presets: gamma sRGB backbuffer.\n"
                 "  HDR presets: PQ BT.2020 (HDR10) backbuffer.\n"
                 "scRGB linear backbuffers: HDR presets will under-grain super-white pixels.";
> = 1;

uniform float IntensityScale <
    ui_type = "slider";
    ui_label = "Intensity";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
    ui_tooltip = "Multiplier on the preset's intensity. 1.0 = preset default.";
> = 1.0;

uniform float GrainScale <
    ui_type = "slider";
    ui_label = "Grain Scale";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
    ui_tooltip = "Grain spatial size. Multiplier on the Gaussian blur sigma.\n"
                 "0 = per-pixel (sharpest), 1 = preset default, 2 = double.\n"
                 "Scales with render resolution (2160p reference).";
> = 1.0;

uniform float SaturationScale <
    ui_type = "slider";
    ui_label = "Color Saturation";
    ui_min = 0.0; ui_max = 2.0; ui_step = 0.05;
    ui_tooltip = "Multiplier on chroma saturation. 0 = monochrome grain, 1.0 = preset default.";
> = 1.0;

uniform float GrainPeak <
    ui_type = "slider";
    ui_label = "Grain Mid";
    ui_min = 0.5; ui_max = 2.0; ui_step = 0.05;
    ui_tooltip = "Shifts where grain is most visible. Multiplier on the response midpoint.\n"
                 "1.0 = preset default. Higher = brighter tones, lower = darker tones.";
> = 1.0;

uniform float GrainRate <
    ui_type = "slider";
    ui_label = "Grain Rate (target fps)";
    ui_min = 0.0; ui_max = 60.0; ui_step = 1.0;
    ui_tooltip = "Target grain animation rate. Snaps to the nearest integer divisor "
                 "of display refresh for even pattern persistence.\n"
                 "30 = filmic (default), 24 = cinema, 12 = soft, 0 = static.";
> = 30.0;

uniform bool MatchBlur <
    ui_label = "Match Blur";
    ui_tooltip = "On real film, grain IS the imaging medium - image detail is never "
                 "finer than the grain. This softens the image to match, so the grain "
                 "becomes the apparent detail floor.";
> = false;

uniform float MatchBind <
    ui_type = "slider";
    ui_label = "Match Bind";
    ui_min = 0.0; ui_max = 1.0; ui_step = 0.05;
    ui_tooltip = "Requires Match Blur. Grain pattern gates which image detail "
                 "survives the blur — where grain is active, original detail bleeds "
                 "through; where grain is quiet, the blur wins. Binds grain and blur "
                 "into a single coherent texture.\n"
                 "0 = independent layers (blur + grain), 1 = full detail-gating.";
> = 0.5;

uniform float frame_time  < source = "frametime";  >;
uniform int   frame_count < source = "framecount"; >;

// =============================================================================
// Preset constants (separate arrays - FXC cannot dynamically index struct arrays)
//
// Order: SDR Light, SDR Medium, SDR Heavy, HDR Light, HDR Medium, HDR Heavy
//
// Per-channel sigma derived from mpv baked Gaussian weights:
//   1-tap: sigma 0.80 | 2-tap: sigma 1.00 | 3-tap: sigma 1.20
// =============================================================================

static const float P_INT[6]    = { 0.05,  0.12,  0.20,  0.05,  0.08,  0.12  };
static const float P_SAT[6]    = { 0.10,  0.25,  0.50,  0.20,  0.40,  0.60  };
static const float P_TKS[6]    = { 0.459, 0.459, 0.459, 0.383, 0.383, 0.383 };

static const float P_RIMUL[6]  = { 1.0,  1.0,  1.0,  1.0,  1.05, 1.2  };
static const float P_GIMUL[6]  = { 1.0,  1.0,  1.0,  1.0,  1.0,  1.1  };
static const float P_BIMUL[6]  = { 1.0,  1.0,  1.1,  1.0,  1.1,  1.0  };

static const float P_RVAR[6]   = { 1.0,  1.0,  1.1,  1.0,  1.0,  1.1  };
static const float P_GVAR[6]   = { 1.0,  1.0,  1.0,  1.0,  1.0,  1.0  };
static const float P_BVAR[6]   = { 1.0,  1.0,  1.1,  1.0,  1.0,  1.0  };

static const float P_RMID[6]   = { 0.50, 0.40, 0.35, 0.22, 0.22, 0.25 };
static const float P_GMID[6]   = { 0.50, 0.40, 0.35, 0.22, 0.22, 0.25 };
static const float P_BMID[6]   = { 0.50, 0.40, 0.35, 0.22, 0.22, 0.25 };

static const float P_RSTEEP[6] = { 2.0,  1.6,  1.2,  15.0, 10.0, 5.0  };
static const float P_GSTEEP[6] = { 2.0,  1.6,  1.2,  15.0, 10.0, 5.0  };
static const float P_BSTEEP[6] = { 2.0,  1.6,  1.2,  15.0, 10.0, 5.0  };

static const float P_RSAT[6]   = { 0.5,  0.6,  0.7,  0.5,  0.6,  0.7  };
static const float P_GSAT[6]   = { 0.5,  0.5,  0.6,  0.5,  0.5,  0.6  };
static const float P_BSAT[6]   = { 0.5,  0.4,  0.5,  0.5,  0.5,  0.5  };

static const float P_RSIG[6]   = { 0.80, 0.80, 0.80, 0.80, 1.00, 1.20 };
static const float P_GSIG[6]   = { 1.00, 1.00, 1.00, 0.80, 1.00, 1.00 };
static const float P_BSIG[6]   = { 1.20, 1.20, 1.20, 0.80, 0.80, 0.80 };

// =============================================================================
// Core functions
// =============================================================================

#define MAX_R 3

uint pcg_hash(uint s)
{
    uint state = s * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float grain_scale_curve(float lum, float mid, float steepness, float tukey_scale)
{
    float d2 = steepness * (lum - mid) * (lum - mid);
    float t = 1.0 - d2 * tukey_scale;
    float curve = t > 0.0 ? t * t : 0.0;
    float protection = smoothstep(0.0, 0.05, lum);
    return curve * protection;
}

// PCG chain order R -> G -> B matches mpv's rand_triangular(inout state).
float3 raw_triangular_rgb(uint2 nxy, uint frame_seed, float r_var, float g_var, float b_var)
{
    uint state = nxy.x * 1664525u + nxy.y * 22695477u + frame_seed * 314159265u;

    uint a = pcg_hash(state); state = a;
    uint b = pcg_hash(state); state = b;
    float ru = float(a) * (1.0 / 4294967296.0);
    float rv = float(b) * (1.0 / 4294967296.0);
    float gr = (ru + rv - 1.0) * 0.612 * r_var;

    uint c = pcg_hash(state); state = c;
    uint d = pcg_hash(state); state = d;
    float gu = float(c) * (1.0 / 4294967296.0);
    float gv = float(d) * (1.0 / 4294967296.0);
    float gg = (gu + gv - 1.0) * 0.612 * g_var;

    uint e = pcg_hash(state); state = e;
    uint f = pcg_hash(state); state = f;
    float bu = float(e) * (1.0 / 4294967296.0);
    float bv = float(f) * (1.0 / 4294967296.0);
    float gb = (bu + bv - 1.0) * 0.612 * b_var;

    return float3(gr, gg, gb);
}

// =============================================================================
// Pixel shader
// =============================================================================

float4 PS_FilmGrain(float4 vpos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float3 color = tex2D(ReShade::BackBuffer, uv).rgb;
    uint2 pxy = uint2(vpos.xy);

    // Frame seed: snap to nearest integer divisor of display refresh
    uint frame_seed = 0u;
    if (GrainRate > 0.0)
    {
        int divisor = max(1, (int)round(1000.0 / (max(frame_time, 1.0) * GrainRate)));
        frame_seed = uint(frame_count / divisor);
    }

    // Load preset parameters
    float intensity  = P_INT[Preset];
    float saturation = P_SAT[Preset];
    float tukey_s    = P_TKS[Preset];
    float r_imul = P_RIMUL[Preset], g_imul = P_GIMUL[Preset], b_imul = P_BIMUL[Preset];
    float r_var  = P_RVAR[Preset],  g_var  = P_GVAR[Preset],  b_var  = P_BVAR[Preset];
    float r_mid  = P_RMID[Preset],  g_mid  = P_GMID[Preset],  b_mid  = P_BMID[Preset];
    float r_stp  = P_RSTEEP[Preset],g_stp  = P_GSTEEP[Preset],b_stp  = P_BSTEEP[Preset];
    float r_sat  = P_RSAT[Preset],  g_sat  = P_GSAT[Preset],  b_sat  = P_BSAT[Preset];

    // Per-channel sigma: base * user scale * resolution scale (2160p reference)
    float res_scale = float(BUFFER_HEIGHT) / 2160.0;
    float sigma_r = max(P_RSIG[Preset] * GrainScale * res_scale, 0.001);
    float sigma_g = max(P_GSIG[Preset] * GrainScale * res_scale, 0.001);
    float sigma_b = max(P_BSIG[Preset] * GrainScale * res_scale, 0.001);

    // Pre-compute 1D Gaussian weights (symmetric half: index = |offset|)
    float wr[MAX_R + 1], wgw[MAX_R + 1], wbw[MAX_R + 1];
    [unroll] for (int i = 0; i <= MAX_R; i++)
    {
        float ii = float(i * i);
        wr[i]  = exp(-ii / (2.0 * sigma_r * sigma_r));
        wgw[i] = exp(-ii / (2.0 * sigma_g * sigma_g));
        wbw[i] = exp(-ii / (2.0 * sigma_b * sigma_b));
    }

    // Grain convolution
    float sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    float wsum_r = 0.0, wsum_g = 0.0, wsum_b = 0.0;

    [unroll] for (int dy = -MAX_R; dy <= MAX_R; dy++)
    {
        [unroll] for (int dx = -MAX_R; dx <= MAX_R; dx++)
        {
            uint2 nxy = uint2(int2(pxy) + int2(dx, dy));

            float3 graw = raw_triangular_rgb(nxy, frame_seed, r_var, g_var, b_var);
            float gly = dot(graw, float3(0.299, 0.587, 0.114));
            float gnr = lerp(gly, graw.r, r_sat * saturation * SaturationScale);
            float gng = lerp(gly, graw.g, g_sat * saturation * SaturationScale);
            float gnb = lerp(gly, graw.b, b_sat * saturation * SaturationScale);

            int adx = abs(dx), ady = abs(dy);
            float w_r = wr[adx]  * wr[ady];
            float w_g = wgw[adx] * wgw[ady];
            float w_b = wbw[adx] * wbw[ady];

            sum_r  += w_r * gnr;  wsum_r += w_r;
            sum_g  += w_g * gng;  wsum_g += w_g;
            sum_b  += w_b * gnb;  wsum_b += w_b;
        }
    }

    sum_r /= wsum_r;
    sum_g /= wsum_g;
    sum_b /= wsum_b;

    float3 src = color;
    [branch] if (MatchBlur)
    {
        float sigma_img = sigma_g;
        float wimgs[MAX_R + 1];
        [unroll] for (int k = 0; k <= MAX_R; k++)
            wimgs[k] = exp(-float(k * k) / (2.0 * sigma_img * sigma_img));

        float3 blurred = float3(0.0, 0.0, 0.0);
        float wsum_img = 0.0;
        float2 pixel_size = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

        [unroll] for (int sy = -MAX_R; sy <= MAX_R; sy++)
        {
            [unroll] for (int sx = -MAX_R; sx <= MAX_R; sx++)
            {
                float w = wimgs[abs(sx)] * wimgs[abs(sy)];
                float2 nuv = uv + float2(sx, sy) * pixel_size;
                blurred += w * tex2Dlod(ReShade::BackBuffer, float4(nuv, 0.0, 0.0)).rgb;
                wsum_img += w;
            }
        }
        src = blurred / wsum_img;

        if (MatchBind > 0.0)
        {
            float3 detail = color - src;
            float grain_energy = abs(dot(float3(sum_r, sum_g, sum_b), float3(0.299, 0.587, 0.114)));
            src += detail * grain_energy * MatchBind;
        }
    }

    // Tukey response with grain peak shift
    float scale_r = grain_scale_curve(src.r, r_mid * GrainPeak, r_stp, tukey_s);
    float scale_g = grain_scale_curve(src.g, g_mid * GrainPeak, g_stp, tukey_s);
    float scale_b = grain_scale_curve(src.b, b_mid * GrainPeak, b_stp, tukey_s);

    float3 vsum = float3(sum_r * r_imul, sum_g * g_imul, sum_b * b_imul);
    float3 result = src + intensity * IntensityScale * vsum * float3(scale_r, scale_g, scale_b);

    return float4(result, 1.0);
}

technique FilmGrainSmooth <
    ui_label = "Film Grain (Smooth)";
    ui_tooltip = "Triangular-noise film grain with per-channel asymmetric Gaussian blur, "
                 "luminance-shaped Tukey response, resolution scaling, and optional image "
                 "softening. Six presets (SDR/HDR x Light/Medium/Heavy). HDR-signal-safe.";
>
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader  = PS_FilmGrain;
    }
}
