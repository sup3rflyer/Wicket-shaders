// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE

//!HOOK OUTPUT
//!BIND HOOKED
//!DESC Film Grain (HDR-Medium)
//!COMPUTE 32 32

// --- SETTINGS ---
#define GRAIN_RATE 1.0
#define INTENSITY 0.08
#define SATURATION 0.40

#define RED_TAPS 2
#define GREEN_TAPS 2
#define BLUE_TAPS 1

#define RED_INTENSITY_MULTIPLIER 1.05
#define GREEN_INTENSITY_MULTIPLIER 1.0
#define BLUE_INTENSITY_MULTIPLIER 1.1

#define RED_VARIANCE_SCALE 1.0
#define GREEN_VARIANCE_SCALE 1.0
#define BLUE_VARIANCE_SCALE 1.0

#define RED_LUMA_THRESHOLD 0.00
#define GREEN_LUMA_THRESHOLD 0.00
#define BLUE_LUMA_THRESHOLD 0.00

#define RED_MID 0.22
#define GREEN_MID 0.22
#define BLUE_MID 0.22

#define RED_STEEPNESS 10.0
#define GREEN_STEEPNESS 10.0
#define BLUE_STEEPNESS 10.0

#define RED_SATURATION 0.6
#define GREEN_SATURATION 0.5
#define BLUE_SATURATION 0.5

#define USE_LINEAR 0 
#define MAX_TAPS 2

const uint row_size = 2 * MAX_TAPS + 1;

// 2-Tap (Sigma ~1.2) for R/G, 1-Tap (Sigma ~0.8) for B
const float weights_red[5] = { 0.05449, 0.24420, 0.40262, 0.24420, 0.05449 };
const float weights_green[5] = { 0.05449, 0.24420, 0.40262, 0.24420, 0.05449 };
const float weights_blue[3] = { 0.23899, 0.52202, 0.23899 };

const uvec2 isize = uvec2(gl_WorkGroupSize) + uvec2(2 * MAX_TAPS);
shared float grain_r[isize.y][isize.x];
shared float grain_g[isize.y][isize.x];
shared float grain_b[isize.y][isize.x];

// --- PRNG: PCG Hash ---
uint pcg_hash(uint s) {
    uint state = s * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_gaussian(inout uint state, float variance_scale) {
    uint u_int = pcg_hash(state); state = u_int;
    uint v_int = pcg_hash(state); state = v_int;
    float u = float(u_int) * (1.0 / 4294967296.0);
    float v = float(v_int) * (1.0 / 4294967296.0);
    if (u < 1e-6) u = 1e-6;
    float r = sqrt(-2.0 * log(u));
    float theta = 6.28318530718 * v;
    return r * cos(theta) * 0.25 * variance_scale;
}

// Grain scaling function with "Soft Toe" to protect black levels
float grain_scale(float lum, float mid, float steepness, float threshold) {
    // 1. Calculate the Gaussian curve
    float gaussian = exp(-steepness * (lum - mid) * (lum - mid));
    
    // 2. Protect the blacks: Smoothly fade grain out between 0.0 and 0.05 luminance.
    // This ensures pure black (0.0) is ALWAYS free of noise.
    float protection_mask = smoothstep(0.00, 0.05, lum); 

    return gaussian * protection_mask;
}

void hook() {
    uint num_threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    uint frame_seed = uint(floor(float(frame) * GRAIN_RATE));

    for (uint i = gl_LocalInvocationIndex; i < isize.y * isize.x; i += num_threads) {
        uvec2 local_pos = uvec2(i % isize.x, i / isize.x);
        ivec2 global_coord_i = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy) + ivec2(local_pos) - ivec2(MAX_TAPS);
        uvec2 global_pos = uvec2(global_coord_i);
        
        uint seed_init = (global_pos.x * 1664525u) + (global_pos.y * 22695477u) + (frame_seed * 314159265u);
        uint seed = pcg_hash(seed_init);
        
        float g_r = rand_gaussian(seed, RED_VARIANCE_SCALE);
        float g_g = rand_gaussian(seed, GREEN_VARIANCE_SCALE);
        float g_b = rand_gaussian(seed, BLUE_VARIANCE_SCALE);
        
        float grain_lum = dot(vec3(g_r, g_g, g_b), vec3(0.299, 0.587, 0.114));
        grain_r[local_pos.y][local_pos.x] = mix(grain_lum, g_r, RED_SATURATION * SATURATION);
        grain_g[local_pos.y][local_pos.x] = mix(grain_lum, g_g, GREEN_SATURATION * SATURATION);
        grain_b[local_pos.y][local_pos.x] = mix(grain_lum, g_b, BLUE_SATURATION * SATURATION);
    }

    barrier();

    for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
        float hsum_r = 0.0, hsum_g = 0.0, hsum_b = 0.0;
        for (int x = 0; x < 2 * MAX_TAPS + 1; x++) {
            if (x < 2 * RED_TAPS + 1)   hsum_r += weights_red[x]   * grain_r[y][gl_LocalInvocationID.x + x + (MAX_TAPS - RED_TAPS)];
            if (x < 2 * GREEN_TAPS + 1) hsum_g += weights_green[x] * grain_g[y][gl_LocalInvocationID.x + x + (MAX_TAPS - GREEN_TAPS)];
            if (x < 2 * BLUE_TAPS + 1)  hsum_b += weights_blue[x]  * grain_b[y][gl_LocalInvocationID.x + x + (MAX_TAPS - BLUE_TAPS)];
        }
        grain_r[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_r;
        grain_g[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_g;
        grain_b[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_b;
    }

    barrier();

    float vsum_r = 0.0, vsum_g = 0.0, vsum_b = 0.0;
    for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
        if (y < 2 * RED_TAPS + 1)   vsum_r += weights_red[y]   * grain_r[gl_LocalInvocationID.y + y + (MAX_TAPS - RED_TAPS)][gl_LocalInvocationID.x + MAX_TAPS];
        if (y < 2 * GREEN_TAPS + 1) vsum_g += weights_green[y] * grain_g[gl_LocalInvocationID.y + y + (MAX_TAPS - GREEN_TAPS)][gl_LocalInvocationID.x + MAX_TAPS];
        if (y < 2 * BLUE_TAPS + 1)  vsum_b += weights_blue[y]  * grain_b[gl_LocalInvocationID.y + y + (MAX_TAPS - BLUE_TAPS)][gl_LocalInvocationID.x + MAX_TAPS];
    }

    vec4 color = HOOKED_tex(HOOKED_pos);
    if (USE_LINEAR == 1) color = linearize(color);

    float scale_r = grain_scale(color.r, RED_MID, RED_STEEPNESS, RED_LUMA_THRESHOLD);
    float scale_g = grain_scale(color.g, GREEN_MID, GREEN_STEEPNESS, GREEN_LUMA_THRESHOLD);
    float scale_b = grain_scale(color.b, BLUE_MID, BLUE_STEEPNESS, BLUE_LUMA_THRESHOLD);

    vec3 vsum = vec3(vsum_r * RED_INTENSITY_MULTIPLIER, vsum_g * GREEN_INTENSITY_MULTIPLIER, vsum_b * BLUE_INTENSITY_MULTIPLIER);
    vec3 scale_vec = vec3(scale_r, scale_g, scale_b);
    color.rgb += INTENSITY * vsum * scale_vec;

    if (USE_LINEAR == 1) color = delinearize(color);
    imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
}