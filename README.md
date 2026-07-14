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

Runtime controls (live via `glsl-shader-opts`, no recompile):

| Param | Effect |
|-------|--------|
| `tc_strength` | Sharpening strength (12 = shipped tune, 0 = off). |
| `tc_coring` | High-pass deltas below this are ignored (keeps noise from being sharpened). |
| `tc_texture_thresh` | Minimum local variance that counts as real texture rather than noise. |
| `tc_max_delta` | Hard cap on how much sharpening may move any pixel. |

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

#### Single file — `filmgrain-smooth.glsl`

The whole SDR/HDR × light/medium/heavy matrix is also available as **one shader** with the variant selected at runtime — the mpv counterpart of the ReShade port below. Pick a look with `grain_preset` (light/medium/heavy), or set it to `custom` and dial the individual knobs; size, cadence, and the HDR toggle stay live in every preset.

Grain is generated once per **source frame** on a fixed 3840×2160 grid and composited display-scaled every present. So its cadence is locked to the content (not the display refresh) and its size holds a constant visual angle on any panel — and on a high-refresh display it costs far less than regenerating grain every present.

Runtime controls (live via `glsl-shader-opts`, no recompile):

| Param | Effect |
|-------|--------|
| `grain_preset` | Baked look: `0` = custom, `1` = light, `2` = medium, `3` = heavy. A preset drives intensity, saturation, tone, and per-channel chroma balance — the custom-only knobs below are ignored while it's active. |
| `grain_intensity` | Grain amplitude (custom only; light ≈ 0.05, medium ≈ 0.12, heavy ≈ 0.20). |
| `grain_saturation` | Per-channel grain chroma (custom only; 0 = monochrome, 1 = full color). |
| `grain_size` | Cell size at a 4K reference (constant visual angle). 1 = calibrated, >1 coarser, <1 finer. Live in every preset. |
| `grain_mid` | Tone where grain peaks (custom only; 0 = shadows, 0.5 = midtones, 1 = highlights). |
| `grain_steepness` | Tone-bell tightness (custom only) — higher confines grain to the midtones and keeps highlights cleaner. |
| `grain_rate` | Reseed cadence in *source* frames (1 = every frame, 0.5 = on twos). Display-refresh independent. Live. |
| `grain_hdr` | `1` = PQ BT.2020 output chain (e.g. after CelFlare): grain is keyed and applied in the SDR domain via a per-pixel PQ bridge, fading out above reference white. `0` = plain SDR. Live. |
| `grain_ref_white` | SDR reference white in nits for the HDR bridge — match `hdr-reference-white` (and CelFlare's `cf_ref_white`). |

```ini
# mpv.conf
glsl-shaders-append=~~/shaders/filmgrain-smooth.glsl
glsl-shader-opts=grain_preset=2          # medium; or grain_preset=0 to use the custom knobs
```

#### Fixed-file variants

The same six looks are also shipped as individual fixed shaders — no parameters, just append one.

**SDR** — for standard dynamic range content. Hooks at `OUTPUT` stage to always present the grain at native resolution.

| Variant | Intensity | Character |
|---------|-----------|-----------|
| **SDR Light** | 0.05 | Barely visible. Safe for any content. |
| **SDR Medium** | 0.12 | Noticeable film-like grain. Made to match grainy footage. |
| **SDR Heavy** | 0.20 | Strong, visible grain. Emulates high-ISO film stock. |

**HDR** — for HDR content or use after SDR-to-HDR expansion. Include a soft-toe black level protection that keeps pure blacks grain-free, and steep Gaussian falloff that keeps highlights crystal clear. Grain is concentrated in the midtones (peak at ~22% luminance).

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

### Match Grain

**Adaptive grain restoration.** Instead of applying a fixed tier, it *measures* the source's own surviving film grain and auto-tunes the grain model to restore it. Compression and intermediates smooth camera-original grain unevenly; this reads what survived and rebuilds a fuller, source-matched grain rather than laying a generic overlay on top.

Two-stage compute pipeline: grain character is measured on the `LUMA` plane (pre-scale, where the signal survives), and grain is rendered on `OUTPUT` (native display resolution, so grain is crisp and display-native regardless of source resolution), with a persistent buffer carrying the measured state between stages. Grain re-seeds on a **source-frame counter**, so its temporal cadence is locked to the content and independent of the display refresh rate (`grain_rate` means the same thing on any panel). Per source it adapts grain **amplitude** (how much survived, extrapolated up toward the original), **tone** (which luminance range the grain occupies), and **sharpness/size** (fine "sandpaper" vs soft grain, held to a constant visual angle so it reads like a 4K film scan on any display) — with motion-, cut-, and pan-aware gating so static texture, smoke, and busy detail aren't mistaken for grain.

Runtime controls (live-toggleable via `glsl-shader-opts`):

| Param | Effect |
|-------|--------|
| `match_grain` | 0 = bit-identical to the fixed Light tier, 1 = matched. `mix()` between, so A/B is non-destructive. |
| `grain_sharpness` | Global crispness dial (1 = crisp 4K-scan default, 0 = a softer look). |
| `grain_rate` | Grain animation cadence, as a fraction of *source* frames (default 1 = fresh grain every source frame; 0.5 = every 2nd source frame, "on twos"). Display-refresh independent. |
| `grain_base_sat` | Per-channel independence of the base grain — the subtle baked-in hue speckle (default 0.25 = calibrated look; 0 = mono grain). |
| `restore_gain` | How far to extrapolate past the surviving grain toward the camera original. |
| `density_combine` | 0 = additive, 1 = multiplicative density (grain rides the tone/bloom gradients). |
| `grain_hdr` | 1 = PQ BT.2020 output chain (e.g. CelFlare): grain is keyed and applied in the measured SDR domain via a per-pixel PQ bridge, fading out shortly above reference white. 0 = plain SDR (exact prior behavior). |
| `grain_ref_white` | SDR reference white in nits for the HDR bridge — match `hdr-reference-white`. |
| `debug_match` | Machine-readable state overlay for tuning. |

**Requirements:** mpv with `vo=gpu-next`; compute shaders (GLSL 4.30+) — Vulkan/D3D11, or OpenGL 4.3+. SDR content; for SDR→HDR chains (e.g. CelFlare, PQ BT.2020 out) set `grain_hdr=1` + `grain_ref_white=<your hdr-reference-white>` in `glsl-shader-opts`.

**Usage:**

```ini
# mpv.conf
glsl-shaders-append=~~/shaders/filmgrain-match.glsl
```

Use it *instead of* a fixed Film Grain tier (above), not on top of one.

---

### NitMeter

**HDR luminance and gamut analysis overlay** — a measurement/tuning companion for CelFlare (and native HDR content), not an image effect. Decodes the PQ frame for an absolute-nit luminance scope or a Rec.709/P3/Rec.2020 color scope.

In luminance modes, per-pixel light level is computed as pixel CLL — `PQ-EOTF(max(R,G,B)) × 10000` — the same convention as MaxCLL/MaxFALL in HDR10 metadata. The luminance panel shows:

| Row | Meaning |
|-----|---------|
| **P** | Frame peak CLL (strict max pixel). On web re-encodes this is often codec ringing rather than content. |
| **9** | 99.99th-percentile CLL — the practical content peak, immune to lone spike pixels. `P` ≫ `9` means the "peak" is encode noise. |
| **H** | Hold of the `9` row: latches its recent high, then decays — a spike-free recent content peak. |
| **A** | FALL — frame-average CLL (same averaging as MaxFALL). |
| **C** | Session MaxCLL — running max frame peak since toggle-on. |
| **F** | Session MaxFALL — running max FALL since toggle-on. |

A log2 histogram (1 → 10000 nits, ticks at 100/203/1000/4000) shows the screen-area CLL distribution, and a small indicator on the `P` row goes green → yellow → red as the content peak approaches and exceeds `nitmeter_target` (your display ceiling).

Mode 4 replaces that panel with a color-only scope:

| Row | Meaning |
|-----|---------|
| **R / G / B** | Independent frame-maximum linear BT.2020 channel levels in nits. Each maximum may come from a different pixel. |
| **P3** | Percentage of the full raster in the P3-D65 shell: outside Rec.709 but inside P3-D65. |
| **20** | Percentage of the full raster in the Rec.2020 shell: outside P3-D65 but inside legal Rec.2020. |

The two gamut percentages are exclusive and include letterbox bars. A 0.01-nit + 0.25% boundary guard suppresses the nearest numerical/quantization fuzz while retaining subtle native-HDR excursions; larger encoded excursions are reported as present in the signal. Mode 4 has no luminance rows, histogram, ceiling indicator, or time graph.

Mode 4 can also warn when the signal exceeds a particular display gamut. Enable `nitmeter_display_clip` and enter the display's six CIE 1931 chromaticity values: `Rx`, `Ry`, `Gx`, `Gy`, `Bx`, and `By`. The white point is fixed at D65. Pixels outside that primary triangle receive bright near-neutral diagonal zebra alternating with the existing mode-4 view. The defaults describe P3-D65, but the warning is off by default. Invalid or degenerate coordinates safely disable the zebra and turn the panel border red. This is a **gamut-exceedance** warning only: downstream color management may compress or remap those colors, so it does not predict physical gamut clipping, peak-luminance clipping, or tone-mapping behavior.

The scope measures the **final decoded signal**, not mastering provenance. Rec.709 material composited or transcoded inside a PQ/BT.2020 video can therefore show small WCG residues from conversion, chroma resampling, scaling, graphics, or compression. These are real pixels in the delivered stream even when the original insert was SDR. The classifier deliberately has no luminance gate: genuine HDR gamut excursions can live in dark saturated regions too.

After CelFlare, the base curve, specular gain, and light pump are chromaticity-preserving scalar expansion. Its enabled warm-hue and pale-skin perceptual corrections—and the fast PQ encoder’s finite error—can produce small boundary excursions; mode 4 reports those as part of CelFlare’s actual encoded output rather than treating CelFlare as a gamut-expansion effect.

Runtime controls (live via `glsl-shader-opts`):

| Param | Effect |
|-------|--------|
| `nitmeter_mode` | `1` = luminance panel, `2` = false-color heatmap + luminance panel, `3` = gamma-2.2 SDR-export heatmap + luminance panel, `4` = Rec.709 gamut mask + color-only scope. Mode 4 grayscales Rec.709 pixels at their original luminance while out-of-Rec.709 pixels retain their real PQ/BT.2020 color and brightness. (Recompiles.) |
| `nitmeter_target` | Display peak in nits for the luminance panel's `P`-row ceiling indicator — match your `target-peak`. Ignored in mode 4. Live, no recompile. |
| `nitmeter_display_clip` | Enables the custom display-gamut zebra in mode 4. Live toggle; off by default. |
| `nitmeter_display_rx` / `ry`, `gx` / `gy`, `bx` / `by` | Display red, green, and blue CIE 1931 xy primaries. D65 white is fixed; defaults are P3-D65. Live via shampv. |

**Requirements / order:** mpv with `vo=gpu-next`. The frame at `MAIN` must already be **PQ-encoded**, so load NitMeter **after CelFlare** (which emits PQ in-shader), or on native PQ HDR content (HDR10/HDR10+/DV) without it. HLG is not supported, and on plain SDR content the numbers are meaningless — a built-in guard blanks the panel when it detects SDR input.

```ini
# mpv.conf — append after CelFlare, or on native PQ HDR
glsl-shaders-append=~~/shaders/NitMeter.glsl
glsl-shader-opts=nitmeter_mode=2,nitmeter_target=1000
```

Most convenient driven from a small Lua script that cycles panel → heatmap → Rec.709 gamut mask → off on one key (appending/unloading the shader and setting `nitmeter_mode`).

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

TextureClarity runs on LUMA before expansion. CelFlare hooks at MAIN (multi-pass: blur → stats → expand). Film grain hooks at OUTPUT (final stage). If you use NitMeter, append it after CelFlare so it reads the PQ frame.

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.
