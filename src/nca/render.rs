use std::borrow::Cow;

use bevy::{
    ecs::schedule::ShouldRun,
    math::vec3,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        RenderApp, RenderStage,
    },
};
use rand::prelude::*;

use super::{NCARules, NeuralCellularAutomataConfig, SelectedRules};

const WORKGROUP_SIZE: u32 = 8;

pub struct NeuralCellularAutomataRenderPlugin;

impl Plugin for NeuralCellularAutomataRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractResourcePlugin::<NeuralCellularAutomataConfig>::default())
            .add_plugin(ExtractResourcePlugin::<NeuralCellularAutomataRenderResources>::default())
            .add_startup_system(setup)
            .add_system(regenerate_world);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_system_to_stage(
                RenderStage::Prepare,
                setup_pipelines.with_run_criteria(should_setup_pipelines),
            )
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("nca", NeuralCellularAutomataNode::default());
        render_graph
            .add_node_edge("nca", bevy::render::main_graph::node::CAMERA_DRIVER)
            .unwrap();
    }
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    config: Res<NeuralCellularAutomataConfig>,
    windows: Res<Windows>,
) {
    let width = if let Some(window) = windows.get_primary() {
        window.width()
    } else {
        (2 * config.size.0) as f32
    };

    let render_resources = (0..config.num_variants)
        .map(|i| {
            let view = Image::new_fill(
                Extent3d {
                    width: config.size.0,
                    height: config.size.1,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                &[0, 0, 0, 255],
                TextureFormat::Rgba8Unorm,
            );
            let view = images.add(view);
            commands.spawn_bundle(SpriteBundle {
                sprite: Sprite {
                    custom_size: Some(Vec2::new(config.size.0 as f32, config.size.1 as f32)),
                    ..default()
                },
                texture: view.clone(),
                transform: Transform::from_translation(vec3(
                    -width / 4.0 + i as f32 * width / 2.0,
                    0.0,
                    0.0,
                )),
                ..default()
            });
            let rules_image = generate_rules_texture(
                &vec![0.0; 3 * config.kernel_size * config.kernel_size * 4],
                config.kernel_size,
            );
            let rules_image = images.add(rules_image);
            NeuralCellularAutomataState { view, rules_image }
        })
        .collect();

    commands.insert_resource(NeuralCellularAutomataRenderResources(render_resources));

    commands.spawn_bundle(Camera2dBundle::default());
}

fn regenerate_world(
    config: Res<NeuralCellularAutomataConfig>,
    rules: Res<NCARules>,
    selected_rules: Res<SelectedRules>,
    mut images: ResMut<Assets<Image>>,
    mut render_resources: ResMut<NeuralCellularAutomataRenderResources>,
) {
    if !rules.is_changed() && !selected_rules.is_changed() {
        return;
    }

    for (rule, state) in selected_rules.0.iter().zip(render_resources.0.iter_mut()) {
        let rule = &rules.rules()[*rule];

        let view = generate_view_texture(config.size);
        state.view = images.set(&state.view, view);

        let rules_image = generate_rules_texture(&rule.0, config.kernel_size);
        state.rules_image = images.set(&state.rules_image, rules_image);
    }
}

fn generate_view_texture(size: (u32, u32)) -> Image {
    let mut rng = thread_rng();

    let mut view = Image::new_fill(
        Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8Unorm,
    );

    for pixel in view.data.chunks_mut(4) {
        for x in pixel.iter_mut().take(3) {
            *x = rng.gen_range(0..=255);
        }
    }

    view.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    view
}

fn generate_rules_texture(rules: &[f32], kernel_size: usize) -> Image {
    fn vf_to_u8(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    let mut rules = Image::new(
        Extent3d {
            width: kernel_size as u32,
            height: kernel_size as u32 * 3,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vf_to_u8(rules).to_vec(),
        TextureFormat::Rgba32Float,
    );

    rules.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    rules
}

pub struct NeuralCellularAutomataPipeline {
    texture_bind_group_layout: BindGroupLayout,
    pipelines: Vec<CachedComputePipelineId>,
}

fn should_setup_pipelines(world: &World) -> ShouldRun {
    if world.contains_resource::<NeuralCellularAutomataPipeline>() {
        ShouldRun::No
    } else {
        ShouldRun::Yes
    }
}

fn setup_pipelines(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    config: Res<NeuralCellularAutomataConfig>,
    asset_server: Res<AssetServer>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    let texture_bind_group_layout =
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    let shader = asset_server.load("shaders/nca.wgsl");
    let pipelines = (0..config.num_variants)
        .map(|_| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: Some(vec![texture_bind_group_layout.clone()]),
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Cow::from("update"),
            })
        })
        .collect();

    commands.insert_resource(NeuralCellularAutomataPipeline {
        texture_bind_group_layout,
        pipelines,
    });
}

#[derive(Clone, ExtractResource)]
struct NeuralCellularAutomataRenderResources(Vec<NeuralCellularAutomataState>);

#[derive(Clone)]
struct NeuralCellularAutomataState {
    view: Handle<Image>,
    rules_image: Handle<Image>,
}

struct NeuralCellularAutomataImageBindGroup(Vec<BindGroup>);

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<NeuralCellularAutomataPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    game_of_life_render_resources: Res<NeuralCellularAutomataRenderResources>,
    render_device: Res<RenderDevice>,
) {
    commands.insert_resource(NeuralCellularAutomataImageBindGroup(
        game_of_life_render_resources
            .0
            .iter()
            .map(|res| {
                let view = &gpu_images[&res.view];
                let rules = &gpu_images[&res.rules_image];
                render_device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.texture_bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(&view.texture_view),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(&rules.texture_view),
                        },
                    ],
                })
            })
            .collect(),
    ));
}

enum NeuralCellularAutomataNodeState {
    Loading,
    Update,
}

struct NeuralCellularAutomataNode {
    state: NeuralCellularAutomataNodeState,
}

impl Default for NeuralCellularAutomataNode {
    fn default() -> Self {
        Self {
            state: NeuralCellularAutomataNodeState::Loading,
        }
    }
}

impl render_graph::Node for NeuralCellularAutomataNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<NeuralCellularAutomataPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            NeuralCellularAutomataNodeState::Loading => {
                let ready = pipeline.pipelines.iter().all(|p| {
                    matches!(
                        pipeline_cache.get_compute_pipeline_state(*p),
                        CachedPipelineState::Ok(_),
                    )
                });

                if ready {
                    self.state = NeuralCellularAutomataNodeState::Update;
                }
            }
            NeuralCellularAutomataNodeState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        if !matches!(self.state, NeuralCellularAutomataNodeState::Update) {
            return Ok(());
        }

        let texture_bind_groups = world.resource::<NeuralCellularAutomataImageBindGroup>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<NeuralCellularAutomataPipeline>();
        let config = world.resource::<NeuralCellularAutomataConfig>();

        for (i, p) in pipeline.pipelines.iter().enumerate() {
            let mut pass = render_context
                .command_encoder
                .begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_bind_group(0, &texture_bind_groups.0[i], &[]);

            let update_pipeline = pipeline_cache.get_compute_pipeline(*p).unwrap();
            pass.set_pipeline(update_pipeline);
            pass.dispatch_workgroups(
                config.size.0 / WORKGROUP_SIZE,
                config.size.1 / WORKGROUP_SIZE,
                1,
            );
        }

        Ok(())
    }
}
