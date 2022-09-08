mod nca;

use bevy::{
    prelude::*,
    window::{WindowDescriptor, WindowMode},
};
use nca::{NeuralCellularAutomataConfig, NeuralCellularAutomataPlugin};

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
        .insert_resource(NeuralCellularAutomataConfig {
            size: (SIZE.0, SIZE.1),
            population_size: 8,
            num_variants: 2,
            kernel_size: 13,
            num_layers: 5,
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(NeuralCellularAutomataPlugin)
        .run();
}
