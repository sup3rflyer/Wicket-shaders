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

The goal is a professional HDR grade of the source, not an "HDR filter" look: midtones stay close to the SDR grade, highlights expand with natural gradation, and speculars get a believable amount of extra pop. The shader does no tone mapping of its own; your display does the final mapping.

Every pixel gets a smooth, monotone expansion curve of its own brightness. The curve's shape is set by an illumination field (a wide Gaussian blur of the regional brightness), so all pixels in one region share one curve: tonal order is kept and no gradient can invert, while local contrast is scaled by the curve's slope. Frame-level statistics (bright fraction, contrast, brightness key) adapt the result continuously; there is no scene-type classifier. At `cf_strength` 1 the curve peaks at about 2.4x (bright regions) to 2.7x (dark regions) of reference white; the default 0.7 scales that down, before scene adaptation, specular pop and the light pump. Works with anime and live action.

Features:
- Spatially modulated per-pixel expansion curve, shaped by a regional illumination field
- Picture-size-aware geometry: the illumination field scales with the picture (80 px sigma at 1080p, 160 px on a 4K or 2x-upscaled picture), so an upscaler chain looks like 1080p playback of the same title; letterboxed and 4:3 encodes keep the 1080p scale
- Continuous scene adaptation (brightness key, contrast, bright fraction), with letterbox/pillarbox bars excluded from the statistics
- Velocity-adaptive temporal smoothing with cut detection: still scenes are stable, lighting changes adapt faster, cuts lock on quickly; a brightening explosion or tunnel exit is not mistaken for a cut, and strobing content does not keep the state in "cut" mode
- Growing-object detection: explosions, fire and backlit reveals keep their HDR pop instead of being dampened as they grow
- Specular bonus on near-clip highlights, with bright-scene recovery (chrome, sun glints, headlights in daylight) and saturation gating (bright colored surfaces do not get specular pop)
- Specular stabilization (`cf_spec_stab`): grain and mottle near the specular onset are no longer amplified by the steep specular ramp, lone outliers and pepper pits are locked or filled, and tiny isolated glints can be de-emphasized (`cf_spec_floor`). Edges, glints and smooth falloffs are left alone
- Light pump: a temporary, exposure-like surge on sustained brightening (explosions, tunnel exits, spells). It works per region of the picture, uses a motion search so camera pans and moving lamps do not trigger it, needs several frames of proof before it opens, and releases slowly with the source instead of snapping off
- Warm-hue correction (Bezold-Brücke): fire, sunsets and skin do not drift green as they brighten
- Pale-skin protection (saturation restore plus a small lift in bright, cooled scenes)
- Grain stabilization: a compute-shader bilateral filter on luma makes the expansion decision grain-stable, so film grain stays filmic after expansion
- Expansion in Oklab at constant chromaticity, with a fast path for near-neutral pixels
- Exact PQ encoding where the fast approximation is inaccurate (near black, and above 1800 nits per channel), plus a PQ-aware dither against 8-bit banding in expanded highlights
- Debug views with on-screen legends (`cf_debug`)

**Controls.** All controls live in the **USER TUNING block at the top of the shader**. Edit the values there, or set them from `mpv.conf` without touching the file:

```ini
glsl-shader-opts=cf_ref_white=110,cf_strength=0.8,cf_spec=1.2
```

Sliders respond live during playback (no recompile; bind `glsl-shader-opts` changes to keys for real-time A/B). Toggles and `cf_debug` trigger a quick recompile.

| Control | Range | Default | Effect |
|---------|-------|---------|--------|
| `cf_ref_white` | 80–480 | 116 | SDR white level in nits. **Must match `hdr-reference-white`.** |
| `cf_strength` | 0–2 | 0.7 | Overall strength. 0 = plain SDR, 1 = the full internal tune. Scales base expansion, specular pop and the light pump together. |
| `cf_curve` | 0.6–2 | 1.2 | Ramp shape. Below 1 = gentle, broad lift; above 1 = lift concentrated on the brightest pixels, midtones closer to the SDR grade. Peak unchanged. |
| `cf_shoulder` | 0–1 | 1.0 | How softly expansion arrives at the peak. 1 = smoothest (no steepening near clip), 0 = steepest near-clip pop. |
| `cf_spec` | 0–2 | 1.1 | Specular pop on glints, light sources and clipped highlights. 0 = off. |
| `cf_spec_stab` | 0–2 | 1.0 | Specular stabilization. 0 = raw specular ramp; 1 = default; above 1 = overdrive (texture evened harder, stronger corrections). |
| `cf_spec_floor` | 0–1 | 0.45 | Share of specular pop kept by tiny isolated points (2x2 stars, lone sparkles). Coherent highlights always get full pop. 1 = uniform, 0 = isolated points get base expansion only. Needs `cf_spec_stab` > 0. |
| `cf_pump` | 0–2 | 1.0 | Light pump strength. 0 = off. |
| `cf_grain_stab` | 0/1 | 1 | Grain stabilization toggle. |
| `cf_additive_pump` | 0/1 | 1 | 1 = each region pumps at its own strength (motion-checked opening). 0 = the older subtractive mode, where the regional mask can only suppress a frame-wide pump. |
| `cf_warm_shift` | 0/1 | 1 | Warm-hue correction toggle. |
| `cf_pale_skin` | 0/1 | 1 | Pale-skin protection toggle. |
| `cf_debug` | 0–12 | 0 | Debug view: 1 bypass, 2 illumination field, 3 expansion, 4 base expansion, 5 specular, 6 light pump, 7 warm shift / pale skin, 8 scene stats, 9 motion offset, 10 motion evidence (11 = same as 10), 12 light-pump opening proof. |

`cf_spec_stab` in detail: from 0 to 1 it fades in both parts of the stabilization (the texture-evened drive and the range lock / pit fill). From 1 to 2 those stay at full, and the texture evening gets stronger: less of the grain is passed to the specular ramp (never zero) and the largest allowed correction doubles.

**Removed in v6.0** (these keys no longer exist; drop them from `glsl-shader-opts` and saved profiles): `cf_spec_bonus` (use `cf_spec=0`), `cf_light_pump` (use `cf_pump=0`), `cf_spatial_pump` (the regional pump is always on; the frame-wide-only mode pumped on camera pans), `cf_spec_scene_reject` (always on), `cf_spec_radius` (fixed at 6 px), `cf_spec_texture` (merged into `cf_spec_stab`: `cf_spec_texture=2` with `cf_spec_stab=1` is now `cf_spec_stab=2`).

Load **one** CelFlare shader at a time. Never combine `CelFlare.glsl` with `CelFlare-transport.glsl`: the picture would be processed twice.

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

Three-stage compute pipeline: grain character is measured on the `LUMA` plane, a persistent 960×540 toroidal vocabulary is generated on source ticks, and `OUTPUT` assembles that vocabulary into a continuous active-picture field. Grain size is defined as a fraction of picture height, not as grain “at 4K”: scanning or displaying the same source at higher resolution reveals the same grain with greater sampling fidelity. The current 2160-sample synthesis lattice is a finite implementation bandwidth, not the grain's identity.

Shape and amount are separate controls. `grain_size`, `grain_contrast` and `grain_soften` change what the grain looks like at a fixed *visible* strength — its RMS after a small visibility blur (σ 1.5 lattice samples, roughly where contrast sensitivity falls off on a 4K display), held per channel against the default look — and `grain_gain` alone sets how much. Before this, those knobs held raw RMS, so soft or coarse grain read up to ~1.8× stronger at the same gain and fine grain read weaker; configs that changed them render a different amount now, so re-tune `grain_gain` by eye. At the default look nothing changes.

The density compositor retains the generator's unequal, correlated RGB grain character, but normalizes its luma energy downward when a saturated carrier would otherwise make the same measured grain look stronger than it does on an equal-luma neutral. This is covariance-aware rather than a generic saturation reduction: quiet chromatic directions are left alone, neutral pixels are unchanged, and additive mode is untouched.

Visible arrangement cadence is source-locked and display-refresh independent. `grain_rate` controls the boil cadence; `grain_gen_rate` may retain the standing vocabulary for several visible ticks while block windows and jitter still rehash every tick. Paused redraws freeze the observer, vocabulary, and arrangement when the companion shampv script supplies the machine-owned pause signal; paused seeks/frame-steps participate in normal source cadence for their new frame, then refreeze (so “on twos” still intentionally shares an arrangement). Editing a generator-baked look control while paused rebuilds the standing vocabulary once under the same seed, so tuning remains live without starting the temporal observer. A shader first enabled while paused initializes once from the held source frame. shampv also changes `state_epoch` once per file so title state cannot bleed across a playlist; standalone integrations must do the same.

Runtime controls (live-toggleable via `glsl-shader-opts`):

| Param | Effect |
|-------|--------|
| `match_grain` | 0 = no synthetic output while observation remains live; 1 = full Match Grain+. Intermediate values are valid evaluation amounts. |
| `grain_size` | Grain clump size as a fraction of picture height. 1 = calibrated neutral scale; default 1.34 = the restoration scan character. Holds the visible amount. The former `grain_sharpness` size trim is folded in and removed: an old `grain_size` s with `grain_sharpness` p is now s × (1 + 0.387·p) (the old default 1.2 at 0.3 is the new 1.34). Size rendering is evidence-frozen — delivery texture never steers it. |
| `grain_soften` | Sub-pixel capture softness in grain-lattice samples (1/2160 of active picture height — a title property like `grain_size`, not rescaled by the output raster): a common scan-aperture/optical MTF folded into every channel's kernel. 0 = point-sampled lattice (no capture MTF); 0.29 (default) = the minimal one-sample scan aperture; 1–2.5 = softer scan/optics, up to smooth Gaussian-like grain (with `value_warp` 0 — the warp flattens soft grains into mesas). Holds the visible amount, so softening no longer reads stronger. Softens the finest (red) channel most, which `grain_size` cannot do without coarsening all three. |
| `grain_rate` | Visible arrangement cadence as a fraction of *source* frames (default 1 = on ones; 0.5 = on twos). Display-refresh independent. |
| `grain_gen_rate` | Template-vocabulary regeneration rate relative to visible ticks (default 1 = every tick; 0.25 = every fourth). Arrangement still refreshes each visible tick; the saving is modest because OUTPUT dominates cost, and finite-vocabulary reuse can change higher-order temporal correlation. |
| `grain_base_sat` | Colour noise of the grain: per-channel independence (0 = mono grain; 0.75 = calibrated look; ~1.36 = channels fully independent, like separate dye layers; up to 2 = exaggerated colour noise for outlier sources). The luma amount stays the same across the range; only the colour noise grows. |
| `restore_gain` | How far to extrapolate past the surviving grain toward the camera original. Default 1 follows the inferred target; >1 is a manual override for known-damaged material. Values near 5 are aggressive and shot-specific, not a preset; 6 is the expert ceiling. Above 2, restored power rises roughly with the square and can heavily overgrain intact material. |
| `grain_fade` | Work-domain luma where grain reaches zero. Default 1.10 preserves the stock top end; 0.2-0.3 confines grain to low luminance with a widened shadow toe so the grain rises gradually out of black. Below-reference-white values only move the luma envelope; the per-channel safety ceiling stays at reference white. |
| `density_combine` | Combine mix: 0 = additive; 1 = multiplicative density (grain rides tone/bloom gradients, with neutral-referenced chromatic luma-energy protection); values between blend the two deltas linearly from the same field, so matched power holds while carrier weighting, bright-biased skew and shadow behaviour interpolate. |
| `grain_hdr` | 1 = PQ BT.2020 output chain (e.g. CelFlare): grain is keyed and applied in the measured SDR domain via a per-pixel PQ bridge, fading out shortly above reference white. 0 = plain SDR (exact prior behavior). |
| `grain_ref_white` | SDR reference white in nits for the HDR bridge — match `hdr-reference-white`. With `grain_source_trc=1` it is also the anchor native PQ is measured against. |
| `grain_source_trc` | Source transfer the observer reads: 0 = gamma (all SDR, including SDR retagged for an SDR→HDR shader); 1 = PQ native HDR (HDR10/HDR10+, Dolby Vision P7/P8 base layer), measured through a bridge to SDR-equivalent codes at `grain_ref_white` so native HDR is read in the domain the model is calibrated in. HLG and Dolby Vision P5 stay 0. mpv cannot tell the shader the source transfer — set it per profile for HDR sources. |
| `debug_match` | Machine-readable state overlay for tuning. |

**Requirements:** mpv with `vo=gpu-next`; compute shaders (GLSL 4.30+) — Vulkan/D3D11, or OpenGL 4.3+. SDR content, or native PQ HDR with `grain_source_trc=1`; for SDR→HDR chains (e.g. CelFlare, PQ BT.2020 out) set `grain_hdr=1` + `grain_ref_white=<your hdr-reference-white>` in `glsl-shader-opts`. Grain is measured in picture-relative terms, so 1080p, 4K and upscaled (e.g. an AI upscaler filter ahead of the shader) sources of the same picture read the same grain. The observer assumes limited-range video (disc, broadcast, streaming); full-range sources mis-key the darkest tones slightly.

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
