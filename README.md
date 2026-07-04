# Wicket Shaders

Custom shaders for [mpv](https://mpv.io/) and [ReShade](https://reshade.me/).

mpv shaders require `vo=gpu-next` (libplacebo backend).

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

Uses a spatially-modulated expansion curve driven by an illumination field (bright-biased Gaussian blur of regional luminance). All pixels in a region share the same curve parameters — local contrast preserved by construction through multiplicative application. Frame-level scene metrics (APL, contrast, bright fraction) drive continuous adaptation rather than discrete scene-type classification. Highlights expand to ~200–310 nits with specular pop on top, while midtones stay close to the SDR grade. Works with anime and live-action content.

All supported tuning lives in the **USER TUNING block at the top of the shader** — six sliders (`cf_ref_white`, `cf_strength`, `cf_curve`, `cf_shoulder`, `cf_spec`, `cf_pump`) and per-feature toggles, each documented in place. Edit the values there, or set them from `mpv.conf` without touching the file:

```ini
glsl-shader-opts=cf_ref_white=110,cf_strength=1.2,cf_spec=0.8
```

The sliders respond live during playback (bind `glsl-shader-opts` changes to keys for real-time A/B); toggles trigger a quick recompile. `cf_strength` scales the whole effect (0 = plain SDR), `cf_curve` sets how harshly expansion ramps into the highlights (peak brightness unchanged), `cf_shoulder` softens the arrival at peak for sources whose highlights are already harsh or clipped hard, `cf_spec` scales specular pop, `cf_pump` scales the light pump. Deeper internals are tunable in each pass — at your own peril.

Features:
- Spatially-modulated expansion curve (per-pixel, regionally adapted by an illumination field)
- Continuous scene adaptation (APL, contrast, bright fraction) — no discrete scene-type classifier
- Velocity-adaptive temporal smoothing: still scenes get stable averages, gradual lighting changes adapt faster, scene cuts lock on near-instantly
- Growing-object detection — explosions, fire, backlit reveals, and similar expanding-bright events keep their HDR pop instead of being progressively dampened as they grow
- Bright-scene specular recovery — chrome, sun glints, headlights, and snowfield highlights still pop in daylight where normal detection would shut off
- Specular bonus with per-pixel ramp, scene-aware peak/gamma, and saturation gating (genuine highlights pop, bright colored surfaces don't)
- Bezold-Brücke hue compensation (regional warm-to-red rotation in Oklab — fixes warm-to-green shift on fire, sunsets, skin)
- Pale skin saturation protection
- Grain stabilization via compute-shader bilateral log-luma filter with shared-memory tile load
- Expansion in Oklab with chroma attenuation for saturated pixels; fast-path bypass for near-neutrals
- PQ-aware temporal dither to mask 8-bit banding in expanded highlights
- Direct PQ BT.2020 output bypasses libplacebo's SDR peak clipping
- Multiple debug visualizations for tuning (including in-frame stats overlay)

**Requirements:** mpv v0.41.0+ with `vo=gpu-next`. Uses compute shaders (GLSL 4.30+) — works on the default Vulkan and D3D11 backends; OpenGL backend requires 4.3+ for native compute.

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
glsl-shader-opts=cf_ref_white=110      # Same value as hdr-reference-white above
```

`cf_ref_white` must match `hdr-reference-white` (set it via `glsl-shader-opts` as above, or edit the default at the top of the shader).

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

Uses the same mpv profile as CelFlare, but keeps the classic in-file setup: set `REFERENCE_WHITE` inside the shader to match `hdr-reference-white`. Tune with `INTENSITY`, `CURVE_STEEPNESS`, `HIGHLIGHT_PEAK`, and `KNEE_END`.

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

#### ReShade Port

A single-file ReShade port is available at [`ReShade/FilmGrainSmooth.fx`](ReShade/FilmGrainSmooth.fx). All six mpv variants are consolidated as selectable presets with additional tuning controls:

| Control | Description |
|---------|-------------|
| **Preset** | SDR/HDR x Light/Medium/Heavy |
| **Intensity** | Multiplier on preset intensity |
| **Grain Scale** | Grain spatial size (0 = per-pixel sharpest, 1 = preset default). Resolution-scaled to 2160p reference. |
| **Color Saturation** | Chroma saturation multiplier (0 = monochrome) |
| **Grain Mid** | Shifts where grain is most visible (response midpoint) |
| **Grain Rate** | Animation rate in target fps. Auto-snaps to integer divisor of display refresh. |
| **Match Blur** | Softens image at grain scale, emulating the film resolution limit |
| **Match Bind** | Grain pattern gates which image detail survives the blur — binds grain and blur into one coherent texture (requires Match Blur) |

HDR-signal-safe: never clamps the backbuffer. Works on SDR (sRGB), HDR10 (PQ BT.2020), and scRGB (RGBA16F) backbuffers.

**Usage:** Copy `FilmGrainSmooth.fx` to your ReShade `Shaders` folder.

---

### Match Grain (SDR)

**Adaptive grain restoration.** Instead of applying a fixed tier, it *measures* the source's own surviving film grain and auto-tunes the grain model to restore it. Compression and intermediates smooth camera-original grain unevenly; this reads what survived and rebuilds a fuller, source-matched grain rather than laying a generic overlay on top.

Two-stage compute pipeline: grain character is measured on the `LUMA` plane (pre-scale, where the signal survives), and grain is rendered on `OUTPUT` (native display resolution), with a persistent buffer carrying the measured state between stages. Per source it adapts grain **amplitude** (how much survived, extrapolated up toward the original), **tone** (which luminance range the grain occupies), and **sharpness/size** (fine "sandpaper" vs soft grain) — with motion-, cut-, and pan-aware gating so static texture, smoke, and busy detail aren't mistaken for grain.

Runtime controls (live-toggleable via `glsl-shader-opts`):

| Param | Effect |
|-------|--------|
| `match_grain` | 0 = bit-identical to the fixed Light tier, 1 = matched. `mix()` between, so A/B is non-destructive. |
| `grain_sharpness` | Global crispness dial (1 = crisp 4K-scan default, 0 = a softer look). |
| `restore_gain` | How far to extrapolate past the surviving grain toward the camera original. |
| `density_combine` | 0 = additive, 1 = multiplicative density (grain rides the tone/bloom gradients). |
| `debug_match` | Machine-readable state overlay for tuning. |

**Requirements:** mpv with `vo=gpu-next`; compute shaders (GLSL 4.30+) — Vulkan/D3D11, or OpenGL 4.3+. SDR content; for SDR→HDR chains (e.g. CelFlare, PQ BT.2020 out) set `grain_hdr=1` + `grain_ref_white=<your hdr-reference-white>` in `glsl-shader-opts`.

**Usage:**

```ini
# mpv.conf
glsl-shaders-append=~~/shaders/filmgrain-match.glsl
```

Use it *instead of* a fixed Film Grain tier (above), not on top of one.

---

## Installation

### mpv

1. Copy the desired `.glsl` files to your mpv shader directory (typically `~~/shaders/`)
2. Add `glsl-shaders-append=~~/shaders/<filename>.glsl` to your `mpv.conf`

On Windows, `~~` refers to the mpv config directory (e.g. `%APPDATA%/mpv/` or `portable_config/` for portable installs).

### ReShade

1. Copy `.fx` files from the `ReShade/` folder to your ReShade installation's `Shaders` directory
2. Enable the shader in the ReShade overlay

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
