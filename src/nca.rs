mod render;

use std::cmp::Ordering;

use bevy::{prelude::*, render::extract_resource::ExtractResource};
use rand::prelude::*;

use self::render::NeuralCellularAutomataRenderPlugin;

pub struct NeuralCellularAutomataPlugin;

#[derive(Clone, ExtractResource)]
pub struct NeuralCellularAutomataConfig {
    pub size: (u32, u32),
    pub population_size: usize,
    pub num_variants: usize,
    pub kernel_size: usize,
    pub num_layers: usize,
}

impl Plugin for NeuralCellularAutomataPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(NeuralCellularAutomataRenderPlugin)
            .add_startup_system(setup)
            .add_system(rank_selected);
    }
}

fn setup(mut commands: Commands, config: Res<NeuralCellularAutomataConfig>) {
    let rules = NCARules::new(
        (0..config.population_size)
            .map(|_| NCARule::rand(config.kernel_size, config.num_layers))
            .collect(),
    );

    commands.insert_resource(SelectedRules(vec![rules.current(), rules.pivot()]));
    commands.insert_resource(rules);
}

fn rank_selected(
    keys: Res<Input<KeyCode>>,
    config: Res<NeuralCellularAutomataConfig>,
    mut rules: ResMut<NCARules>,
    mut selected_rules: ResMut<SelectedRules>,
) {
    let ordering = if keys.just_pressed(KeyCode::Left) {
        Ordering::Greater
    } else if keys.just_pressed(KeyCode::Right) {
        Ordering::Less
    } else {
        return;
    };

    if !rules.rank(ordering) {
        let rules_inner = rules.rules();

        let mut rng = thread_rng();
        let mut new_rules = Vec::with_capacity(rules_inner.len());

        new_rules.push(rules_inner[0].clone());
        for _ in 0..((rules_inner.len() - 2) / 2) {
            let mut rules = rules_inner[0..(rules_inner.len() / 2)].choose_multiple(&mut rng, 2);
            let a = rules.next().unwrap();
            let b = rules.next().unwrap();
            let (a, b) = a.crossover(b);
            new_rules.push(a);
            new_rules.push(b)
        }
        new_rules.push(NCARule::rand(config.kernel_size, config.num_layers));

        *rules = NCARules::new(new_rules);
    }

    *selected_rules = SelectedRules(vec![rules.current(), rules.pivot()]);
}

struct SelectedRules(Vec<usize>);

struct NCARules {
    i: usize,
    j: usize,
    remaining: Vec<(usize, usize)>,
    rules: Vec<NCARule>,
}

impl NCARules {
    fn new(rules: Vec<NCARule>) -> Self {
        Self {
            i: 0,
            j: 0,
            remaining: vec![(0, rules.len() - 1)],
            rules,
        }
    }

    fn current(&self) -> usize {
        self.j
    }

    fn pivot(&self) -> usize {
        self.remaining.last().unwrap().1
    }

    fn rules(&self) -> &[NCARule] {
        &self.rules
    }

    fn rank(&mut self, ordering: Ordering) -> bool {
        if matches!(ordering, Ordering::Greater) {
            self.rules.swap(self.i, self.j);
            self.i += 1;
        }
        self.j += 1;
        if self.j == self.pivot() {
            let (start, end) = self.remaining.pop().unwrap();
            self.rules.swap(self.i, end);
            if start + 1 < self.i {
                self.remaining.push((start, self.i - 1));
            }
            if self.i + 1 < end {
                self.remaining.push((self.i + 1, end));
            }
            if let Some((start, end)) = self.remaining.last() {
                self.i = *start;
                self.j = *start;
                let mid = ((end + 1) - start) / 2;
                self.rules.swap(mid, *end);
                true
            } else {
                false
            }
        } else {
            true
        }
    }
}

#[derive(Clone)]
struct NCARule(Vec<Vec<f32>>);

impl NCARule {
    fn rand(kernel_size: usize, num_layers: usize) -> Self {
        let mut rng = thread_rng();

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut rules = vec![1.0; 3 * kernel_size * kernel_size * 4];

            for pixel in rules.chunks_mut(4) {
                for x in pixel.iter_mut().take(3) {
                    *x = rng.gen_range(-1.0..=1.0);
                }
            }

            layers.push(rules)
        }

        Self(layers)
    }

    fn crossover(&self, other: &NCARule) -> (NCARule, NCARule) {
        let mut rng = thread_rng();
        let (a, b): (Vec<_>, Vec<_>) = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(a, b)| {
                        if rng.gen_bool(0.5) {
                            (*a, *b)
                        } else {
                            (*b, *a)
                        }
                    })
                    .unzip()
            })
            .unzip();

        (NCARule(a), NCARule(b))
    }
}
