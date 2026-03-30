# Wicket Shaders

Custom GLSL shaders for [mpv](https://mpv.io/) video player. 

Requires **mpv** with `vo=gpu-next` (libplacebo backend).

**My personal setup:** TextureClarity + Film Grain Light (auto SDR/HDR) on everything. CelFlare for anime.

| Key | Action |
|-----|--------|
| F5  | Clear all shaders |
| F6  | Cycle Film Grain SDR (None → Light → Medium → Heavy) |
| F7  | Cycle Film Grain HDR (None → Light → Medium → Heavy) |
| F8  | Toggle CelFlare + sdr-to-hdr profile |

The keybindings above use a Lua script for mpv. Ask an AI assistant to write you a `shader-toggle.lua` tailored to your keybindings and shader combinations — describe what you want (mutual exclusivity, profile switching, OSD feedback) and it'll have a working script in under a minute.

## Shaders

### CelFlare

**Scene-adaptive SDR-to-HDR highlight expansion** with PQ BT.2020 output.

Uses a spatially-modulated expansion curve driven by an illumination field (bright-biased Gaussian blur of regional luminance). All pixels in a region share the same curve parameters — local contrast preserved by construction through multiplicative application. Frame-level scene metrics (APL, contrast, bright fraction) provide gentle adaptation. Highlights expand to ~300 nits with specular pop on top, while midtones stay close to the SDR grade. Works with anime and live-action content.

Use INTENSITY and KNEE for tuning. Other tweaks are at your own peril.

Features:
- Illumination-field driven expansion (per-pixel curve, regionally adapted)
- Scene adaptation via continuous metrics (APL, contrast, bright fraction)
- Scene cut detection with fast adaptation lockout
- Grain stabilization via bilateral log-luma filter
- Specular bonus with scene-detected highlight pop (tier separation gating)
- Bezold-Brücke hue compensation (regional warm-to-red rotation in Oklab — fixes warm-to-green shift on fire, sunsets, skin)
- Pale skin saturation protection
- Expansion in Oklab with chroma attenuation for saturated pixels
- PQ-aware temporal dither to mask 8-bit banding in expanded highlights
- Direct PQ BT.2020 output bypasses libplacebo's SDR peak clipping
- Multiple debug visualizations for tuning

**Requirements:** mpv v0.41.0+ with `vo=gpu-next`

**Usage:**

Add an mpv profile that re-tags source metadata for PQ BT.2020 output:

```ini
# mpv.conf
[sdr-to-hdr]
profile-restore=copy
target-trc=pq
target-prim=bt.2020
target-peak=1000
hdr-reference-white=110          # Match your Windows SDR brightness (nits)
sub-hdr-peak=110
image-subs-hdr-peak=110
vf-append=format:gamma=pq:primaries=bt.2020 		#Has to be set
glsl-shaders-append=~~/shaders/CelFlare.glsl
```

Set `REFERENCE_WHITE` inside the shader to match your `hdr-reference-white` value.

#### Finding your SDR white level

Both values must match your Windows **SDR content brightness** slider (Settings → Display → HDR). Windows maps this slider linearly to nits:

| Slider | Nits | Slider | Nits |
|--------|------|--------|------|
| 0%     | 80   | 30%    | 200  |
| 5%     | 100  | 40%    | 240  |
| 8%     | 112  | 50%    | 280  |
| 10%    | 120  | 60%    | 320  |
| 15%    | 140  | 75%    | 380  |
| 20%    | 160  | 100%   | 480  |

Source: [DISPLAYCONFIG_SDR_WHITE_LEVEL (Microsoft)](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-displayconfig_sdr_white_level) — the slider maps `SDRWhiteLevel` from 1000 (80 nits) to 6000 (480 nits).

---

### CelFlare Lite

**Static SDR-to-HDR highlight expansion** — lightweight variant of CelFlare using the same processing pipeline (PQ BT.2020 output, bilateral grain stabilization, chroma-adaptive expansion, PQ-aware dither) but with fixed expansion parameters instead of scene-adaptive analysis.

Uses the same mpv profile and `REFERENCE_WHITE` setup as CelFlare. Tune with `INTENSITY`, `CURVE_STEEPNESS`, `HIGHLIGHT_PEAK`, and `KNEE_END`.

---

### TextureClarity

**Subtle texture sharpening** that enhances fine detail without edge sharpening or grain amplification.

Operates on the luma channel only. Uses a 5x5 neighborhood with variance-based texture/grain discrimination and Sobel edge detection to selectively sharpen real texture while leaving edges, grain, and flat areas untouched. Works best when it's barely noticeable. This is meant to restore subtle texture detail affected by encoding.

**Usage:**

```ini
# mpv.conf
glsl-shaders-append=~~/shaders/TextureClarity.glsl
```

---

### Film Grain

**Professional film grain simulation** using GPU compute shaders. Adds photographic-like grain with per-channel control over size, intensity, and luminance response.

Technical approach:
- PCG hash PRNG (stateless, pattern-free)
- Triangular noise (sum of two uniforms — bounded, no transcendentals)
- Separable multi-tap Gaussian convolution for grain size control (per-channel tap counts create natural chromatic grain structure)
- Luminance-adaptive scaling via Tukey window (grain concentrated in midtones, finite support)

#### SDR Variants

For standard dynamic range content. Hooks at `OUTPUT` stage to always present the grain at native resolution.

| Variant | Intensity | Character |
|---------|-----------|-----------|
| **SDR Light** | 0.05 | Barely visible. Safe for any content. |
| **SDR Medium** | 0.12 | Noticeable film-like grain. Made to match grainy footage. |
| **SDR Heavy** | 0.20 | Strong, visible grain. Emulates high-ISO film stock. |

#### HDR Variants

For HDR content or use after SDR-to-HDR expansion. Include a soft-toe black level protection that keeps pure blacks grain-free, and steep Gaussian falloff that keeps highlights crystal clear. Grain is concentrated in the midtones (peak at ~22% luminance).

| Variant | Intensity | Character |
|---------|-----------|-----------|
| **HDR Light** | 0.05 | Minimal grain. Fine texture without compromising HDR clarity. |
| **HDR Medium** | 0.08 | Moderate grain with differential channel blur (R/G coarser, B sharper). |
| **HDR Heavy** | 0.12 | Strong grain with multi-scale channel structure (R coarsest, B finest). |

**Usage:**

Pick one variant and add it to your config:

```ini
# mpv.conf
glsl-shaders-append=~~/shaders/filmgrain-smooth-SDR-light.glsl
```

Available files:
- `filmgrain-smooth-SDR-light.glsl`
- `filmgrain-smooth-SDR-medium.glsl`
- `filmgrain-smooth-SDR-heavy.glsl`
- `filmgrain-smooth-HDR-light.glsl`
- `filmgrain-smooth-HDR-medium.glsl`
- `filmgrain-smooth-HDR-heavy.glsl`

## Installation

1. Copy the desired `.glsl` files to your mpv shader directory (typically `~~/shaders/`)
2. Add `glsl-shaders-append=~~/shaders/<filename>.glsl` to your `mpv.conf`

On Windows, `~~` refers to the mpv config directory (e.g. `%APPDATA%/mpv/` or `portable_config/` for portable installs).

## Shader Load Order

If using multiple shaders together, load them in this order:

```ini
glsl-shaders-append=~~/shaders/TextureClarity.glsl
glsl-shaders-append=~~/shaders/CelFlare.glsl
glsl-shaders-append=~~/shaders/filmgrain-smooth-HDR-light.glsl
```

TextureClarity runs on LUMA before expansion. CelFlare hooks at MAIN (multi-pass: blur → stats → expand). Film grain hooks at OUTPUT (final stage).

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.
