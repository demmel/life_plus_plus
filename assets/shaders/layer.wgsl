@group(0) @binding(0)
var input_texture: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(1)
var rules: texture_storage_2d<rgba32float, read>;

@group(0) @binding(2)
var output_texture: texture_storage_2d<rgba8unorm, read_write>;

fn kernel_size() -> i32 {
    let texture_size = textureDimensions(rules);
    return texture_size.x;
}

fn get_size() -> vec2<i32> {
    return textureDimensions(input_texture);
}

fn get_rule(x: i32, y: i32, channel: i32) -> vec3<f32> {
    return textureLoad(rules, vec2<i32>(x, channel * 3 + y)).rgb;
}

fn get_state(location: vec2<i32>, x: i32, y: i32) -> vec3<f32> {
    let index = ((location + vec2<i32>(x, y)) + get_size() ) % get_size();
    return textureLoad(input_texture, index).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let k = kernel_size();

    var color3 = vec3<f32>(0.0, 0.0, 0.0);
    for (var c: i32 = 0; c <= 2; c++) {
        var sum = 0.0;
        for (var x: i32 = 0; x < k; x++) {
            for (var y: i32 = 0; y < k; y++) {
                let rule = get_rule(x, y, c);
                let state = get_state(location, x - k / 2, y - k / 2);
                sum += rule.r * state.r + rule.g * state.g + rule.b * state.b;
            }
        }
        color3[c] = (tanh(sum) + 1.0) / 2.0;
    }
    let color4 = vec4<f32>(color3, 1.0);
    textureStore(output_texture, location, color4);
}