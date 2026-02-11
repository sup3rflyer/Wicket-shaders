// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE

//!HOOK OUTPUT
//!BIND HOOKED
//!DESC Film Grain (Light)
//!COMPUTE 32 32

#define GRAIN_RATE 1.0
#define USE_LINEAR 0
#define INTENSITY 0.05
#define SATURATION 0.10
#define MAX_TAPS 3

#define RED_TAPS 1
#define GREEN_TAPS 2
#define BLUE_TAPS 3

#define RED_INTENSITY_MULTIPLIER 1.0
#define GREEN_INTENSITY_MULTIPLIER 1.0
#define BLUE_INTENSITY_MULTIPLIER 1.0

#define RED_VARIANCE_SCALE 1.0
#define GREEN_VARIANCE_SCALE 1.0
#define BLUE_VARIANCE_SCALE 1.0

#define RED_LUMA_THRESHOLD 0.02
#define GREEN_LUMA_THRESHOLD 0.02
#define BLUE_LUMA_THRESHOLD 0.02

#define RED_MID 0.5
#define GREEN_MID 0.5
#define BLUE_MID 0.5

#define RED_STEEPNESS 2.0
#define GREEN_STEEPNESS 2.0
#define BLUE_STEEPNESS 2.0

#define RED_SATURATION 0.5
#define GREEN_SATURATION 0.5
#define BLUE_SATURATION 0.5

const uint row_size = 2 * MAX_TAPS + 1;
const uvec2 isize = uvec2(gl_WorkGroupSize) + uvec2(2 * MAX_TAPS);

shared float grain_r[isize.y][isize.x];
shared float grain_g[isize.y][isize.x];
shared float grain_b[isize.y][isize.x];

const float weights_red[3] = { 0.23899, 0.52202, 0.23899 };
const float weights_green[5] = { 0.05449, 0.24420, 0.40262, 0.24420, 0.05449 };
const float weights_blue[7] = { 0.01465, 0.08312, 0.23556, 0.33335, 0.23556, 0.08312, 0.01465 };

// --- PRNG: PCG Hash (Stateless, No Patterns) ---
uint pcg_hash(uint s) {
    uint state = s * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_gaussian(inout uint state, float variance_scale) {
    uint s = pcg_hash(state);
    state = s; // Advance state
    float u = float(s) * (1.0 / 4294967296.0);
    
    s = pcg_hash(state); 
    state = s;
    float v = float(s) * (1.0 / 4294967296.0);

    if (u < 1e-6) u = 1e-6;
    float r = sqrt(-2.0 * log(u));
    float theta = 6.28318530718 * v;
    return r * cos(theta) * 0.25 * variance_scale;
}

float grain_scale(float lum, float mid, float steepness, float threshold) {
    if (lum < threshold) return 0.0;
    return exp(-steepness * (lum - mid) * (lum - mid));
}

void hook() {
    uint num_threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    uint frame_seed = uint(floor(float(frame) * GRAIN_RATE));

    // 1. Generate Grain
    for (uint i = gl_LocalInvocationIndex; i < isize.y * isize.x; i += num_threads) {
        uvec2 local_pos = uvec2(i % isize.x, i / isize.x);
        ivec2 global_coord_i = ivec2(gl_WorkGroupID.xy * gl_WorkGroupSize.xy) + ivec2(local_pos) - ivec2(MAX_TAPS);
        uvec2 global_pos = uvec2(global_coord_i); 

        // Asymmetric Seed: x*Prime1 + y*Prime2 to prevent mirroring
        uint seed_init = (global_pos.x * 1664525u) + (global_pos.y * 22695477u) + (frame_seed * 314159265u);
        
        float g_r = rand_gaussian(seed_init, RED_VARIANCE_SCALE);
        float g_g = rand_gaussian(seed_init, GREEN_VARIANCE_SCALE);
        float g_b = rand_gaussian(seed_init, BLUE_VARIANCE_SCALE);

        float grain_lum = dot(vec3(g_r, g_g, g_b), vec3(0.299, 0.587, 0.114));
        grain_r[local_pos.y][local_pos.x] = mix(grain_lum, g_r, RED_SATURATION * SATURATION);
        grain_g[local_pos.y][local_pos.x] = mix(grain_lum, g_g, GREEN_SATURATION * SATURATION);
        grain_b[local_pos.y][local_pos.x] = mix(grain_lum, g_b, BLUE_SATURATION * SATURATION);
    }

    barrier();

    // 2. Convolve Horizontal
    for (uint y = gl_LocalInvocationID.y; y < isize.y; y += gl_WorkGroupSize.y) {
        float hsum_r = 0.0;
        float hsum_g = 0.0;
        float hsum_b = 0.0;
        
        for (int x = 0; x < 2 * MAX_TAPS + 1; x++) {
            // Safety checks to handle variable tap sizes in one shader
            if (x < 2 * RED_TAPS + 1)   hsum_r += weights_red[x]   * grain_r[y][gl_LocalInvocationID.x + x + (MAX_TAPS - RED_TAPS)];
            if (x < 2 * GREEN_TAPS + 1) hsum_g += weights_green[x] * grain_g[y][gl_LocalInvocationID.x + x + (MAX_TAPS - GREEN_TAPS)];
            if (x < 2 * BLUE_TAPS + 1)  hsum_b += weights_blue[x]  * grain_b[y][gl_LocalInvocationID.x + x + (MAX_TAPS - BLUE_TAPS)];
        }
        grain_r[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_r;
        grain_g[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_g;
        grain_b[y][gl_LocalInvocationID.x + MAX_TAPS] = hsum_b;
    }

    barrier();

    // 3. Convolve Vertical + Output
    float vsum_r = 0.0;
    float vsum_g = 0.0;
    float vsum_b = 0.0;

    for (int y = 0; y < 2 * MAX_TAPS + 1; y++) {
        if (y < 2 * RED_TAPS + 1)   vsum_r += weights_red[y]   * grain_r[gl_LocalInvocationID.y + y + (MAX_TAPS - RED_TAPS)][gl_LocalInvocationID.x + MAX_TAPS];
        if (y < 2 * GREEN_TAPS + 1) vsum_g += weights_green[y] * grain_g[gl_LocalInvocationID.y + y + (MAX_TAPS - GREEN_TAPS)][gl_LocalInvocationID.x + MAX_TAPS];
        if (y < 2 * BLUE_TAPS + 1)  vsum_b += weights_blue[y]  * grain_b[gl_LocalInvocationID.y + y + (MAX_TAPS - BLUE_TAPS)][gl_LocalInvocationID.x + MAX_TAPS];
    }

    vec4 color = HOOKED_tex(HOOKED_pos);
    // Linearization moved to macros to support HDR switch
    if (USE_LINEAR == 1) color.rgb = pow(max(color.rgb, 0.0), vec3(2.2)); // Approx linearize

    float scale_r = grain_scale(color.r, RED_MID, RED_STEEPNESS, RED_LUMA_THRESHOLD);
    float scale_g = grain_scale(color.g, GREEN_MID, GREEN_STEEPNESS, GREEN_LUMA_THRESHOLD);
    float scale_b = grain_scale(color.b, BLUE_MID, BLUE_STEEPNESS, BLUE_LUMA_THRESHOLD);

    vec3 vsum = vec3(vsum_r * RED_INTENSITY_MULTIPLIER, vsum_g * GREEN_INTENSITY_MULTIPLIER, vsum_b * BLUE_INTENSITY_MULTIPLIER);
    vec3 scale_vec = vec3(scale_r, scale_g, scale_b);
    color.rgb += INTENSITY * vsum * scale_vec;

    if (USE_LINEAR == 1) color.rgb = pow(max(color.rgb, 0.0), vec3(1.0/2.2)); // Approx delinearize
    
    imageStore(out_image, ivec2(gl_GlobalInvocationID), color);
}