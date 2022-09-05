@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(1)
var rules: texture_storage_2d<rgba32float, read>;

fn kernel_size() -> i32 {
    return 31;
}

fn get_size() -> vec2<i32> {
    return vec2(1528, 856);
}

fn get_rule(x: i32, y: i32, channel: i32) -> vec3<f32> {
    return textureLoad(rules, vec2<i32>(x, channel * 3 + y)).rgb;
}

fn get_state(location: vec2<i32>, x: i32, y: i32) -> vec3<f32> {
    let index = ((location + vec2<i32>(x, y)) + get_size() ) % get_size();
    return textureLoad(texture, index).rgb;
}

fn new_state(location: vec2<i32>) -> vec3<f32> {
    let k = kernel_size();
    var color = vec3<f32>(0.0, 0.0, 0.0);
    for (var c: i32 = 0; c <= 2; c++) {
        var sum = 0.0;
        for (var x: i32 = 0; x < k; x++) {
            for (var y: i32 = 0; y < k; y++) {
                let rule = get_rule(x, y, c);
                let state = get_state(location, x - k / 2, y - k / 2);
                sum += rule.r * state.r + rule.g * state.g + rule.b * state.b;
            }
        }
        color[c] = clamp((tanh(sum) + 1.0), 0.0, 1.0);
    }
    return color;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let color = vec4<f32>(new_state(location), 1.0);

    storageBarrier();

    textureStore(texture, location, color);
}