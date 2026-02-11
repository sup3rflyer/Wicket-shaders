// Copyright (C) 2026 Ágúst Ari
// Licensed under GPL-3.0 — see LICENSE

//!DESC Texture Clarity
//!HOOK LUMA
//!BIND HOOKED

#define PARAM 12.0
#define THRESHOLD 0.0005
#define MIN_NOISE_THRESHOLD 0.020
#define BRIGHTNESS_CLAMP 0.012

// Grain suppression parameters
const float grain_threshold = 0.045;
const float grain_transition_width = 0.50;

// Texture discrimination
const float texture_transition_width = 0.018;

// Black/white area safeguards
const float black_threshold = 0.15;
const float black_falloff = 0.10;
const float white_threshold = 0.55;
const float white_falloff = 0.20;

// Enhanced control parameters
const float EDGE_MULTIPLIER = 6.0;
const float DETAIL_LOW = 0.003;
const float DETAIL_HIGH = 0.013;

// Cinematic boost parameters
// Max multiplier applied to sharpen_amount in lowest-variance areas.
const float LOWDETAIL_MAX_BOOST = 1.5;  // final cap (1.0 = no boost, 1.5 = +50%)
// Power exponent for "cinematic" curve.
const float LOWDETAIL_EXP = 0.5;

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
            float texture = smoothstep(MIN_NOISE_THRESHOLD, MIN_NOISE_THRESHOLD + texture_transition_width, variance);

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

    // Unsharp mask
    float blurred = (p.x + get_grid(1,0,grid) + get_grid(0,1,grid) +
                     get_grid(-1,0,grid) + get_grid(0,-1,grid)) * 0.2;
    float t = p.x - blurred;
    float t_sharp = sign(t) * max(abs(t) - THRESHOLD, 0.0);

    // Sobel edge detection
    float sobel_h = (grid[8] - grid[6]) + 2.0 * (grid[13] - grid[11]) + (grid[18] - grid[16]);
    float sobel_v = (grid[16] - grid[6]) + 2.0 * (grid[17] - grid[7]) + (grid[18] - grid[8]);
    float edge = abs(sobel_h) + abs(sobel_v);
    float edge_factor = clamp(1.0 - edge * EDGE_MULTIPLIER, 0.0, 1.0);

    // Suppress sharpening in high-detail areas
    float detail_saturation = 1.0 - smoothstep(DETAIL_LOW, DETAIL_HIGH, window_variance);

    // Combine influences
    float combined = edge_factor * final_grain * final_texture * detail_saturation;
    float sharpen_amount = t_sharp * PARAM * combined;

    // -----------------------
    // Cinematic low-detail boost (applied BEFORE black/white mask)
    // -----------------------
    // We want a curve that strongly favors the very-low variance areas while
    // leaving mid/high detail alone. Use an exponent (power curve) for a
    // cinematic "lift" feel and cap the multiplier at LOWDETAIL_MAX_BOOST.
    float lowdetail_gain = 1.0 - smoothstep(0.0, DETAIL_LOW, window_variance);
    // Now bias with a power curve (exponent < 1 emphasizes lowest values)
    lowdetail_gain = pow(max(lowdetail_gain, 0.0), LOWDETAIL_EXP);
    // Map to a multiplier between 1.0 and LOWDETAIL_MAX_BOOST
    float lowdetail_boost = mix(1.0, LOWDETAIL_MAX_BOOST, lowdetail_gain);

    sharpen_amount *= lowdetail_boost;
    // -----------------------
    // End cinematic boost
    // -----------------------

    // Black/white luminance masks
    float mask = smoothstep(black_threshold, black_threshold + black_falloff, p.x) *
                 (1.0 - smoothstep(white_threshold - white_falloff, white_threshold, p.x));

    sharpen_amount *= mask;

    // Clamp final brightness changes
    float result = clamp(p.x + sharpen_amount, p.x - BRIGHTNESS_CLAMP, p.x + BRIGHTNESS_CLAMP);

    return vec4(result, 0.0, 0.0, 1.0);
}
