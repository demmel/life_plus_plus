use std::borrow::Cow;

use bevy::{
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

const WORKGROUP_SIZE: u32 = 8;
const KERNEL_SIZE: usize = 31;

pub struct NeuralCellularAutomataComputePlugin;

#[derive(Clone, ExtractResource)]
pub struct NeuralCellularAutomataSize(pub u32, pub u32);

impl Plugin for NeuralCellularAutomataComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugin(ExtractResourcePlugin::<NeuralCellularAutomataRenderResources>::default())
            .add_plugin(ExtractResourcePlugin::<NeuralCellularAutomataSize>::default())
            .add_startup_system(setup);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<NeuralCellularAutomataPipeline>()
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
    size: Res<NeuralCellularAutomataSize>,
) {
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
    let view = images.add(view);

    commands.spawn_bundle(SpriteBundle {
        sprite: Sprite {
            custom_size: Some(Vec2::new(size.0 as f32, size.1 as f32)),
            ..default()
        },
        texture: view.clone(),
        ..default()
    });

    let mut rules = vec![1.0; 3 * KERNEL_SIZE * KERNEL_SIZE * 4];

    for pixel in rules.chunks_mut(4) {
        for x in pixel.iter_mut().take(3) {
            *x = rng.gen_range(-1.0..=1.0);
        }
    }

    fn vf_to_u8(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    let mut rules = Image::new(
        Extent3d {
            width: KERNEL_SIZE as u32,
            height: KERNEL_SIZE as u32 * 3,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vf_to_u8(&rules).to_vec(),
        TextureFormat::Rgba32Float,
    );

    rules.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    let rules = images.add(rules);

    commands.insert_resource(NeuralCellularAutomataRenderResources { view, rules });

    commands.spawn_bundle(Camera2dBundle::default());
}

#[derive(Clone, ExtractResource)]
struct NeuralCellularAutomataRenderResources {
    view: Handle<Image>,
    rules: Handle<Image>,
}

struct NeuralCellularAutomataImageBindGroup(BindGroup);

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<NeuralCellularAutomataPipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    game_of_life_render_resources: Res<NeuralCellularAutomataRenderResources>,
    render_device: Res<RenderDevice>,
) {
    let view = &gpu_images[&game_of_life_render_resources.view];
    let rules = &gpu_images[&game_of_life_render_resources.rules];
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
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
    });
    commands.insert_resource(NeuralCellularAutomataImageBindGroup(bind_group));
}

pub struct NeuralCellularAutomataPipeline {
    texture_bind_group_layout: BindGroupLayout,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for NeuralCellularAutomataPipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let shader = world.resource::<AssetServer>().load("shaders/nca.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![texture_bind_group_layout.clone()]),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        NeuralCellularAutomataPipeline {
            texture_bind_group_layout,
            update_pipeline,
        }
    }
}

enum NeuralCellularAutomataState {
    Loading,
    Update,
}

struct NeuralCellularAutomataNode {
    state: NeuralCellularAutomataState,
}

impl Default for NeuralCellularAutomataNode {
    fn default() -> Self {
        Self {
            state: NeuralCellularAutomataState::Loading,
        }
    }
}

impl render_graph::Node for NeuralCellularAutomataNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<NeuralCellularAutomataPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            NeuralCellularAutomataState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = NeuralCellularAutomataState::Update;
                }
            }
            NeuralCellularAutomataState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let texture_bind_group = &world.resource::<NeuralCellularAutomataImageBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<NeuralCellularAutomataPipeline>();
        let size = world.resource::<NeuralCellularAutomataSize>();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            NeuralCellularAutomataState::Loading => {}
            NeuralCellularAutomataState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(size.0 / WORKGROUP_SIZE, size.1 / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}
