mod render;

use bevy::{prelude::*, render::extract_resource::ExtractResource};
use rand::prelude::*;

use self::render::NeuralCellularAutomataRenderPlugin;

pub struct NeuralCellularAutomataPlugin;

#[derive(Clone, ExtractResource)]
pub struct NeuralCellularAutomataConfig {
    pub size: (u32, u32),
    pub num_variants: usize,
    pub kernel_size: usize,
}

impl Plugin for NeuralCellularAutomataPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(NeuralCellularAutomataRenderPlugin)
            .add_startup_system(setup)
            .add_system(regenerate_world);
    }
}

fn setup(mut commands: Commands, config: Res<NeuralCellularAutomataConfig>) {
    let rules = NCARules(
        (0..config.num_variants)
            .map(|_| NCARule::rand(config.kernel_size))
            .collect(),
    );

    commands.insert_resource(rules);
}

fn regenerate_world(
    keys: Res<Input<KeyCode>>,
    config: Res<NeuralCellularAutomataConfig>,
    mut rules: ResMut<NCARules>,
) {
    if !keys.just_pressed(KeyCode::Space) {
        return;
    }

    for rule in &mut rules.0 {
        *rule = NCARule::rand(config.kernel_size);
    }
}

struct NCARules(Vec<NCARule>);

struct NCARule(Vec<f32>);

impl NCARule {
    fn rand(kernel_size: usize) -> Self {
        let mut rng = thread_rng();

        let mut rules = vec![1.0; 3 * kernel_size * kernel_size * 4];

        for pixel in rules.chunks_mut(4) {
            for x in pixel.iter_mut().take(3) {
                *x = rng.gen_range(-1.0..=1.0);
            }
        }

        Self(rules)
    }
}
