# Wicket Shaders

Custom GLSL shaders for [mpv](https://mpv.io/) video player. Designed for reference-grade image quality on both SDR and HDR displays.

Requires **mpv** with `vo=gpu-next` (libplacebo backend).

## Shaders

### CelFlare

**Scene-adaptive SDR-to-HDR highlight expansion** with PQ BT.2020 output.

Analyzes each frame's brightness, contrast, and highlight distribution to classify the scene (dark/moody, bright/specular, high contrast, etc.) and applies a tailored expansion curve. Highlights are expanded to ~195-290 nits peak while preserving the original artistic intent of the scene.

Features:
- 7 scene types with smooth blending (no hard jumps)
- Grain stabilization via bilateral log-luma filter (prevents grain from flickering in expanded regions)
- Perceptual saturation boost in Oklab color space (counters the silvery look from luminance-only expansion)
- Hue correction to prevent yellow-to-green shift on fires and sunsets
- PQ-aware temporal dither to mask 8-bit banding in expanded highlights
- Direct PQ BT.2020 output bypasses libplacebo's SDR peak clipping
- Scene cut detection with fast adaptation lockout
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
target-peak=10000
hdr-reference-white=116          # Match your Windows SDR brightness (nits)
tone-mapping=clip
gamut-mapping-mode=clip
vf-append=format:gamma=pq:primaries=bt.2020
glsl-shaders-append=~~/shaders/CelFlare.glsl
```

Set `REFERENCE_WHITE` inside the shader to match your `hdr-reference-white` value.

---

### TextureClarity

**Subtle texture sharpening** that enhances fine detail without edge artifacts or grain amplification.

Operates on the luma channel only. Uses a 5x5 neighborhood with variance-based texture/grain discrimination and Sobel edge detection to selectively sharpen real texture while leaving edges, grain, and flat areas untouched. Includes a cinematic low-detail boost that subtly enhances large smooth surfaces.

**Usage:**

```ini
# mpv.conf
glsl-shaders-append=~~/shaders/TextureClarity.glsl
```

---

### Film Grain

**Professional film grain simulation** using GPU compute shaders. Adds photographic grain with per-channel control over size, intensity, and luminance response.

Technical approach:
- PCG hash PRNG (stateless, pattern-free)
- Gaussian noise via Box-Muller transform
- Separable multi-tap Gaussian convolution for grain size control (per-channel tap counts create natural chromatic grain structure)
- Luminance-adaptive scaling via Gaussian curve (grain concentrated in midtones)

#### SDR Variants

For standard dynamic range content. Hook at `OUTPUT` stage.

| Variant | Intensity | Character |
|---------|-----------|-----------|
| **SDR Light** | 0.05 | Barely visible. Adds subtle texture to digital content without drawing attention. |
| **SDR Medium** | 0.12 | Noticeable film-like grain. Good for a cinematic look on clean digital sources. |
| **SDR Heavy** | 0.20 | Strong, visible grain. Emulates high-ISO film stock. |

#### HDR Variants

For HDR content or use after SDR-to-HDR expansion. Include a soft-toe black level protection that keeps pure blacks grain-free, and steep Gaussian falloff that keeps highlights crystal clear. Grain is concentrated in the midtones (peak at ~22% luminance).

| Variant | Intensity | Character |
|---------|-----------|-----------|
| **HDR Light** | 0.04 | Minimal grain. Fine texture without compromising HDR clarity. |
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

TextureClarity runs on LUMA before expansion. CelFlare hooks at MAIN. Film grain hooks at OUTPUT (final stage).

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.
