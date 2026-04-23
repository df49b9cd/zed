// Dual-Kawase blur inner passes (Marius Bjorge, ARM, SIGGRAPH 2015).
//
// Only the downsample half of the original down→up sequence runs here:
// the renderer preserves every chain[i] into pyramid mip `i` after the
// downsample loop, and `fs_blur_rect` / `fs_lens_rect` sample the pyramid
// at a fractional LOD to produce variable-radius blur. The upsample
// passes that the original Kawase paper describes are therefore dead
// code for this renderer and are intentionally omitted.

struct BlurParams {
    viewport_size: vec2<f32>,
    offset_multiplier: f32,
    pad: f32,
}

@group(0) @binding(0) var t_blur_input: texture_2d<f32>;
@group(0) @binding(1) var s_blur_input: sampler;
@group(0) @binding(2) var<uniform> blur_params: BlurParams;

struct BlurFullscreenVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Gamma conversion uses the fast γ=2.2 power-curve approximation rather
// than the full piecewise sRGB transfer. The toe (c < 0.04045) is slightly
// darker than the exact transfer, but downsample runs N passes per frame
// and the piecewise branch costs more than the perceptual error is worth
// for a backdrop blur. Keep in sync with any new gamma-sensitive sample
// sites added here.
fn linearize(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

fn encode_srgb(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(1.0 / 2.2));
}

fn sample_linear(uv: vec2<f32>) -> vec4<f32> {
    let s = textureSample(t_blur_input, s_blur_input, uv);
    return vec4<f32>(linearize(s.rgb), s.a);
}

// Fullscreen-triangle vertex shader (3 vertices cover the NDC [-1,1] square).
@vertex
fn vs_blur_fullscreen(@builtin(vertex_index) vertex_id: u32) -> BlurFullscreenVarying {
    var out = BlurFullscreenVarying();
    let uv = vec2<f32>(f32((vertex_id << 1u) & 2u), f32(vertex_id & 2u));
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    out.position = vec4<f32>(uv * 2.0 - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    return out;
}

@fragment
fn fs_blur_downsample(input: BlurFullscreenVarying) -> @location(0) vec4<f32> {
    let o = 0.5 / blur_params.viewport_size * blur_params.offset_multiplier;
    var color = sample_linear(input.uv) * 4.0;
    color += sample_linear(input.uv + vec2<f32>(-o.x, -o.y));
    color += sample_linear(input.uv + vec2<f32>( o.x, -o.y));
    color += sample_linear(input.uv + vec2<f32>(-o.x,  o.y));
    color += sample_linear(input.uv + vec2<f32>( o.x,  o.y));
    let averaged = color / 8.0;
    return vec4<f32>(encode_srgb(averaged.rgb), averaged.a);
}

