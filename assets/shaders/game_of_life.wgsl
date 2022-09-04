@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, read_write>;

// @group(0) @binding(1)
// var rules: texture_storage_2d<rgba8unorm, read>;

fn get_rule(location: vec2<i32>, offset_x: i32, offset_y: i32, channel: i32) -> vec3<f32> {
    return vec3(1.0, 1.0, 1.0);
}

fn get_state(location: vec2<i32>, offset_x: i32, offset_y: i32) -> vec3<f32> {
    return 2.0 * textureLoad(texture, location + vec2<i32>(offset_x, offset_y)).rgb - 1.0;
}

fn new_state(location: vec2<i32>) -> vec3<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0);
    for (var c: i32 = 0; c <= 2; c++) {
        var weight = 0.0;
        var sum = 0.0;
        for (var x: i32 = -1; x <= 1; x++) {
            for (var y: i32 = -1; y <= 1; y++) {
                let rule = get_rule(location, x, y, c);
                weight += abs(rule.r) + abs(rule.g) + abs(rule.b);
                let state = get_state(location, x, y);
                sum += rule.r * state.r + rule.g * state.g + rule.b * state.b;
            }
        }
        color[c] = sum / weight;
    }
    return (color + 1.0) / 2.0;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let color = vec4<f32>(new_state(location), 1.0);

    storageBarrier();

    textureStore(texture, location, color);
}