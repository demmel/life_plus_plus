mod nca;

use bevy::{
    prelude::*,
    window::{WindowDescriptor, WindowMode},
};
use nca::{NeuralCellularAutomataComputePlugin, NeuralCellularAutomataSize};

const SIZE: (u32, u32) = (760, 760);

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(WindowDescriptor {
            width: SIZE.0 as f32,
            height: SIZE.1 as f32,
            mode: WindowMode::BorderlessFullscreen,
            ..default()
        })
        .insert_resource(NeuralCellularAutomataSize(SIZE.0, SIZE.1))
        .add_plugins(DefaultPlugins)
        .add_plugin(NeuralCellularAutomataComputePlugin)
        .run();
}
