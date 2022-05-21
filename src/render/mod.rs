#![allow(unused_imports)] // TEMP

use bevy::{
    asset::{AssetEvent, Assets, Handle, HandleId, HandleUntyped},
    core::{cast_slice, FloatOrd, Pod, Time, Zeroable},
    ecs::{
        prelude::*,
        system::{lifetimeless::*, ParamSet, SystemParam, SystemState},
    },
    log::trace,
    math::{const_vec3, Mat4, Rect, Vec2, Vec3, Vec4, Vec4Swizzles},
    reflect::TypeUuid,
    render::{
        color::Color,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{Draw, DrawFunctions, RenderPhase, TrackedRenderPass},
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{BevyDefault, Image},
        view::{ComputedVisibility, ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
        RenderWorld,
    },
    transform::components::GlobalTransform,
    utils::{HashMap, HashSet},
};
use bitflags::bitflags;
use bytemuck::cast_slice_mut;
use rand::random;
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::{borrow::Cow, cmp::Ordering, num::NonZeroU64, ops::Range};

#[cfg(feature = "2d")]
use bevy::core_pipeline::Transparent2d;
#[cfg(feature = "3d")]
use bevy::core_pipeline::Transparent3d;

mod aligned_buffer_vec;
mod compute_cache;
mod effect_batcher;
mod effect_cache;
mod pipeline_template;
mod storage_vec;
mod uniform_vec;

use crate::{
    asset::EffectAsset,
    modifiers::{ForceFieldParam, FFNUM},
    spawn::{new_rng, Random},
    Gradient, ParticleEffect, ToWgslString,
};
use aligned_buffer_vec::AlignedBufferVec;
use effect_batcher::EffectBatcher;
use effect_cache::BufferKind;
use storage_vec::StorageVec;
use uniform_vec::NamedUniformVec;

pub use compute_cache::{ComputeCache, SpecializedComputePipeline};
pub use effect_cache::{EffectBuffer, EffectCache, EffectCacheId, EffectSlice};
pub use pipeline_template::PipelineRegistry;

const VFX_COMMON_SHADER_IMPORT: &'static str = include_str!("vfx_common.wgsl");

const VFX_PREPARE_SHADER_TEMPLATE: &'static str = include_str!("vfx_prepare.wgsl");
const VFX_INIT_SHADER_TEMPLATE: &'static str = include_str!("vfx_init.wgsl");
const VFX_UPDATE_SHADER_TEMPLATE: &'static str = include_str!("vfx_update.wgsl");
const VFX_RENDER_SHADER_TEMPLATE: &'static str = include_str!("vfx_render.wgsl");

const DEFAULT_POSITION_CODE: &str = r##"
    ret.pos = vec3<f32>(0., 0., 0.);
    var dir = rand3() * 2. - 1.;
    dir = normalize(dir);
    var speed = 2.;
    ret.vel = dir * speed;
"##;

const DEFAULT_FORCE_FIELD_CODE: &str = r##"
    vVel = vVel + (spawner.accel * sim_params.dt);
    vPos = vPos + vVel * sim_params.dt;
"##;

const FORCE_FIELD_CODE: &str = include_str!("force_field_code.wgsl");

/// Labels for the Hanabi systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum EffectSystems {
    /// Extract the effects to render this frame.
    ExtractEffects,
    /// Extract the effect events to process this frame.
    ExtractEffectEvents,
    /// Prepare GPU data for the extracted effects.
    PrepareEffects,
    /// Queue the GPU commands for the extracted effects.
    QueueEffects,
}

/// Trait to convert any data structure to its equivalent shader code.
trait ShaderCode {
    /// Generate the shader code for the current state of the object.
    fn to_shader_code(&self) -> String;
}

impl ShaderCode for Gradient<Vec2> {
    fn to_shader_code(&self) -> String {
        if self.keys().is_empty() {
            return String::new();
        }
        let mut s: String = self
            .keys()
            .iter()
            .enumerate()
            .map(|(index, key)| {
                format!(
                    "let t{0} = {1};\nlet v{0} = {2};",
                    index,
                    key.ratio().to_wgsl_string(),
                    key.value.to_wgsl_string()
                )
            })
            .fold("// Gradient\n".into(), |s, key| s + &key + "\n");
        if self.keys().len() == 1 {
            s + "size = v0;\n"
        } else {
            // FIXME - particle.age and particle.lifetime are unrelated to Gradient<Vec4>
            s += "let life = particle.age / particle.lifetime;\nif (life <= t0) { size = v0; }\n";
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if (life <= t{1}) {{ size = mix(v{0}, v{1}, (life - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            s += &format!("else {{ size = v{}; }}\n", self.keys().len() - 1);
            s
        }
    }
}

impl ShaderCode for Gradient<Vec4> {
    fn to_shader_code(&self) -> String {
        if self.keys().is_empty() {
            return String::new();
        }
        let mut s: String = self
            .keys()
            .iter()
            .enumerate()
            .map(|(index, key)| {
                format!(
                    "let t{0} = {1};\nlet c{0} = {2};",
                    index,
                    key.ratio().to_wgsl_string(),
                    key.value.to_wgsl_string()
                )
            })
            .fold("// Gradient\n".into(), |s, key| s + &key + "\n");
        if self.keys().len() == 1 {
            s + "out.color = c0;\n"
        } else {
            // FIXME - particle.age and particle.lifetime are unrelated to Gradient<Vec4>
            s += "let life = particle.age / particle.lifetime;\nif (life <= t0) { out.color = c0; }\n";
            let mut s = self
                .keys()
                .iter()
                .skip(1)
                .enumerate()
                .map(|(index, _key)| {
                    format!(
                        "else if (life <= t{1}) {{ out.color = mix(c{0}, c{1}, (life - t{0}) / (t{1} - t{0})); }}\n",
                        index,
                        index + 1
                    )
                })
                .fold(s, |s, key| s + &key);
            s += &format!("else {{ out.color = c{}; }}\n", self.keys().len() - 1);
            s
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd430)]
pub(crate) struct GpuSlice {
    base_index: u32,
    count: u32,
    dead_count: u32,
    max_spawn_count: u32,
}

// Single indirect draw call (via draw_indirect / multi_draw_indirect / multi_draw_indirect_count).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd430)]
pub(crate) struct GpuDrawIndirect {
    // The number of vertices to draw.
    vertex_count: u32,
    // The number of instances to draw.
    instance_count: u32,
    // The Index of the first vertex to draw.
    base_vertex: u32,
    // The instance ID of the first instance to draw.
    base_instance: u32,
}

/// Global effect-independent simulation parameters.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SimParams {
    /// Current simulation time.
    time: f64,
    /// Frame timestep.
    dt: f32,
    // TODO - Add time from beginning of VFX system activation ("simulation time") and rename
    // `time` to `app_time` or so, which is the time since the app started, and is usually
    // greater.
}

/// GPU representation of [`SimParams`].
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd140)]
struct SimParamsUniform {
    dt: f32,
    time: f32,
    __pad0: f32,
    __pad1: f32,
}

impl Default for SimParamsUniform {
    fn default() -> SimParamsUniform {
        SimParamsUniform {
            dt: 0.04,
            time: 0.0,
            __pad0: 0.0,
            __pad1: 0.0,
        }
    }
}

impl From<SimParams> for SimParamsUniform {
    fn from(src: SimParams) -> Self {
        SimParamsUniform {
            dt: src.dt,
            time: src.time as f32,
            ..Default::default()
        }
    }
}

/// Per-effect parameters.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct EffectParams {
    /// Global acceleration (gravity-like) to apply to all particles this frame.
    accel: Vec3,
    /// Offset to add to the particle index to access it in its GPU particle buffer.
    /// Same as [`SpawnerParams::particle_base`], but for the update pass.
    particle_base: u32,
    /// Force field definition.
    force_field: [ForceFieldParam; FFNUM],
}

/// GPU representation of [`SimParams`].
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd140)]
struct EffectParamsUniform {
    accel_x: f32,
    accel_y: f32,
    accel_z: f32,
    particle_base: u32,
    // FIXME - min_uniform_buffer_offset_alignment == 64B
    /// Force field components. One PullingForceFieldParam takes up 32 bytes.
    force_field: [ForceFieldStd140; FFNUM],
}

impl Default for EffectParamsUniform {
    fn default() -> EffectParamsUniform {
        EffectParamsUniform {
            accel_x: 0.0,
            accel_y: 0.0,
            accel_z: 0.0,
            particle_base: 0,
            force_field: [ForceFieldStd140::default(); FFNUM],
        }
    }
}

impl From<EffectParams> for EffectParamsUniform {
    fn from(src: EffectParams) -> Self {
        EffectParamsUniform {
            accel_x: src.accel[0],
            accel_y: src.accel[1],
            accel_z: src.accel[2],
            particle_base: src.particle_base,
            force_field: src.force_field.map(|ffp| ffp.into()),
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, AsStd140)]
pub(crate) struct ForceFieldStd140 {
    position_or_direction: Vec3, // FIXME - merge with f32 for alignemnt in std140, now max_radius will start @ 16 bytes!
    max_radius: f32,
    min_radius: f32,
    mass: f32,
    force_exponent: f32,
    conform_to_sphere: f32,
}

impl From<ForceFieldParam> for ForceFieldStd140 {
    fn from(param: ForceFieldParam) -> Self {
        ForceFieldStd140 {
            position_or_direction: param.position,
            max_radius: param.max_radius,
            min_radius: param.min_radius,
            mass: param.mass,
            force_exponent: param.force_exponent,
            conform_to_sphere: if param.conform_to_sphere { 1.0 } else { 0.0 },
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable, AsStd430)]
struct SpawnerParams {
    /// Origin of the effect. This is either added to emitted particles at spawn time, if the effect simulated
    /// in world space, or to all simulated particles if the effect is simulated in local space.
    origin: Vec3,
    /// Number of particles to spawn this frame.
    spawn_count: u32,
    /// Spawn seed, for randomized modifiers.
    seed: u32,
    /// Offset to add to the particle index to access it in its GPU particle buffer.
    /// Same as [`EffectParams::particle_base`], but for the init pass.
    particle_base: u32,
    /// Index of the slice into the [`SliceList`] array.
    slice_index: u32,
}

/// GPU representation of [`SpawnerParams`].
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd140)]
struct SpawnerParamsUniform {
    origin: Vec4,
    spawn_count: u32,
    seed: u32,
    particle_base: u32,
    slice_index: u32,

    // Pad to 64 bytes for std140 min size
    __pad2: Vec4,
    __pad3: Vec4,
}

impl Default for SpawnerParamsUniform {
    fn default() -> SpawnerParamsUniform {
        SpawnerParamsUniform {
            origin: Vec4::ZERO,
            spawn_count: 0,
            seed: 0,
            particle_base: 0,
            slice_index: 0,
            __pad2: Vec4::ZERO,
            __pad3: Vec4::ZERO,
        }
    }
}

impl From<SpawnerParams> for SpawnerParamsUniform {
    fn from(src: SpawnerParams) -> Self {
        SpawnerParamsUniform {
            origin: src.origin.extend(0.0),
            spawn_count: src.spawn_count,
            seed: src.seed,
            particle_base: src.particle_base,
            slice_index: src.slice_index,
            ..Default::default()
        }
    }
}

/// GPU representation of a dispatch buffer for indirect compute dispatch.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Pod, Zeroable, AsStd430)]
struct DispatchBuffer {
    x: u32,
    y: u32,
    z: u32,
    __pad: u32,
}

pub struct ParticlesPreparePipeline {
    dispatch_buffer_layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl FromWorld for ParticlesPreparePipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        // group(3)
        trace!(
            "DispatchBuffer: std430_size_static = {}",
            DispatchBuffer::std430_size_static()
        );
        let dispatch_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(
                            DispatchBuffer::std430_size_static() as u64
                        ),
                    },
                    count: None,
                }],
                label: Some("vfx_prepare_dispatch_buffer_layout"),
            });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("vfx_prepare_pipeline_layout"),
            bind_group_layouts: &[&dispatch_buffer_layout],
            push_constant_ranges: &[],
        });

        let mut source = VFX_PREPARE_SHADER_TEMPLATE.to_string();
        source.insert_str(0, VFX_COMMON_SHADER_IMPORT); // FIXME - #import not working on compute shaders

        //trace!("Specialized compute pipeline:\n{}", source);

        let shader_module = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("vfx_prepare.wgsl"),
            source: ShaderSource::Wgsl(Cow::Owned(source)),
        });

        let pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("vfx_prepare_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        ParticlesPreparePipeline {
            dispatch_buffer_layout,
            pipeline,
        }
    }
}

pub struct ParticlesInitPipeline {
    // common
    particles_buffer_layout: BindGroupLayout,
    dead_list_layout: BindGroupLayout,
    slice_list_layout: BindGroupLayout,
    // init
    spawner_buffer_layout: BindGroupLayout,
    dispatch_buffer_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
}

impl FromWorld for ParticlesInitPipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let limits = render_device.limits();
        bevy::log::info!(
            "GPU limits:\n- max_compute_invocations_per_workgroup={}\n- max_compute_workgroup_size_x={}\n- max_compute_workgroup_size_y={}\n- max_compute_workgroup_size_z={}\n- max_compute_workgroups_per_dimension={}\n- min_storage_buffer_offset_alignment={}",
            limits.max_compute_invocations_per_workgroup, limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z,
            limits.max_compute_workgroups_per_dimension, limits.min_storage_buffer_offset_alignment
        );
        bevy::log::info!("GPU limits = {:?}", limits);

        //
        // Common
        //

        // group(0)
        trace!(
            "Particle: std430_size_static = {}",
            Particle::std430_size_static()
        );
        let particles_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(Particle::std430_size_static() as u64),
                    },
                    count: None,
                }],
                label: Some("vfx_particles_buffer_layout"),
            });

        // group(1)
        let dead_list_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                },
                count: None,
            }],
            label: Some("vfx_dead_list_layout"),
        });

        // group(2)
        let slice_list_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(GpuSlice::std430_size_static() as u64),
                    },
                    count: None,
                }],
                label: Some("vfx_slice_list_layout"),
            });

        //
        // Init pipeline
        //

        // group(2)
        trace!(
            "SpawnerParamsUniform: std140_size_static = {}",
            SpawnerParamsUniform::std140_size_static()
        );
        let spawner_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(
                            SpawnerParamsUniform::std140_size_static() as u64,
                        ),
                    },
                    count: None,
                }],
                label: Some("vfx_spawner_buffer_layout"),
            });

        // group(3)
        trace!(
            "DispatchBuffer: std430_size_static = {}",
            DispatchBuffer::std430_size_static()
        );
        let dispatch_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(
                            DispatchBuffer::std430_size_static() as u64
                        ),
                    },
                    count: None,
                }],
                label: Some("vfx_dispatch_buffer_layout"),
            });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("vfx_init_pipeline_layout"),
            bind_group_layouts: &[
                &particles_buffer_layout,
                &dead_list_layout,
                &slice_list_layout,
                &spawner_buffer_layout,
                &dispatch_buffer_layout,
            ],
            push_constant_ranges: &[],
        });

        ParticlesInitPipeline {
            // common
            particles_buffer_layout,
            dead_list_layout,
            slice_list_layout,
            // init
            spawner_buffer_layout,
            dispatch_buffer_layout,
            pipeline_layout,
        }
    }
}

pub struct ParticlesUpdatePipeline {
    // common
    particles_buffer_layout: BindGroupLayout,
    dead_list_layout: BindGroupLayout,
    slice_list_layout: BindGroupLayout,
    // update
    draw_indirect_layout: BindGroupLayout,
    indirect_buffer_layout: BindGroupLayout,
    sim_params_layout: BindGroupLayout,
    effect_params_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
}

impl FromWorld for ParticlesUpdatePipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        // FIXME - Use `max_compute_workgroups_per_dimension` to cap the dispatch() call (and therefore the number of particles updated per call)
        let limits = render_device.limits();
        bevy::log::info!(
            "GPU limits:\n- max_compute_invocations_per_workgroup={}\n- max_compute_workgroup_size_x={}\n- max_compute_workgroup_size_y={}\n- max_compute_workgroup_size_z={}\n- max_compute_workgroups_per_dimension={}",
            limits.max_compute_invocations_per_workgroup, limits.max_compute_workgroup_size_x, limits.max_compute_workgroup_size_y, limits.max_compute_workgroup_size_z, limits.max_compute_workgroups_per_dimension
        );

        //
        // Common
        //

        // group(0)
        trace!(
            "Particle: std430_size_static = {}",
            Particle::std430_size_static()
        );
        let particles_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(Particle::std430_size_static() as u64),
                    },
                    count: None,
                }],
                label: Some("vfx_particles_buffer_layout"),
            });

        // group(1)
        let dead_list_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(std::mem::size_of::<u32>() as u64),
                },
                count: None,
            }],
            label: Some("vfx_dead_list_layout"),
        });

        // group(2)
        let slice_list_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(GpuSlice::std430_size_static() as u64),
                    },
                    count: None,
                }],
                label: Some("vfx_slice_list_layout"),
            });

        //
        // Update pipeline
        //

        // group(2)
        let draw_indirect_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(4 * 4),
                    },
                    count: None,
                }],
                label: Some("vfx_draw_indirect_layout"),
            });

        // group(3)
        let indirect_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(4),
                    },
                    count: None,
                }],
                label: Some("vfx_indirect_buffer_layout"),
            });

        // group(4)
        trace!(
            "SimParamsUniform: std140_size_static = {}",
            SimParamsUniform::std140_size_static()
        );
        let sim_params_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            SimParamsUniform::std140_size_static() as u64
                        ),
                    },
                    count: None,
                }],
                label: Some("vfx_sim_params_layout"),
            });

        // group(5)
        trace!(
            "EffectParamsUniform: std140_size_static = {}",
            EffectParamsUniform::std140_size_static()
        );
        let effect_params_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(
                            EffectParamsUniform::std140_size_static() as u64,
                        ),
                    },
                    count: None,
                }],
                label: Some("vfx_effect_params_layout"),
            });

        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("vfx_update_pipeline_layout"),
            bind_group_layouts: &[
                &particles_buffer_layout,
                &dead_list_layout,
                &slice_list_layout,
                &draw_indirect_layout,
                &indirect_buffer_layout,
                &sim_params_layout,
                &effect_params_layout,
            ],
            push_constant_ranges: &[],
        });

        ParticlesUpdatePipeline {
            // common
            particles_buffer_layout,
            dead_list_layout,
            slice_list_layout,
            // update
            draw_indirect_layout,
            indirect_buffer_layout,
            sim_params_layout,
            effect_params_layout,
            pipeline_layout,
        }
    }
}

pub struct ParticlesRenderPipeline {
    particles_buffer_layout: BindGroupLayout,
    view_layout: BindGroupLayout,
    material_layout: BindGroupLayout,
}

impl FromWorld for ParticlesRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let particles_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(Particle::std430_size_static() as u64),
                    },
                    count: None,
                }],
                label: Some("vfx_render_particle_buffer_layout"),
            });

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(ViewUniform::std140_size_static() as u64),
                },
                count: None,
            }],
            label: Some("vfx_render_view_layout"),
        });

        let material_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("vfx_render_material_layout"),
        });

        ParticlesRenderPipeline {
            particles_buffer_layout,
            view_layout,
            material_layout,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ParticleInitPipelineKey {
    /// Code for the position initialization of newly emitted particles.
    position_code: String,
}

impl Default for ParticleInitPipelineKey {
    fn default() -> Self {
        ParticleInitPipelineKey {
            position_code: Default::default(),
        }
    }
}

impl SpecializedComputePipeline for ParticlesInitPipeline {
    type Key = ParticleInitPipelineKey;

    fn specialize(&self, key: Self::Key, render_device: &RenderDevice) -> ComputePipeline {
        trace!(
            "Specializing init pipeline: position_code={:?}",
            key.position_code,
        );

        let mut source = VFX_INIT_SHADER_TEMPLATE.replace("{{INIT_POS_VEL}}", &key.position_code);
        source.insert_str(0, VFX_COMMON_SHADER_IMPORT); // FIXME - #import not working on compute shaders

        //trace!("Specialized compute pipeline:\n{}", source);

        let shader_module = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("vfx_init.wgsl"),
            source: ShaderSource::Wgsl(Cow::Owned(source)),
        });

        let pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("vfx_init_pipeline"),
            layout: Some(&self.pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        pipeline
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ParticleUpdatePipelineKey {
    force_field_code: String,
}

impl Default for ParticleUpdatePipelineKey {
    fn default() -> Self {
        ParticleUpdatePipelineKey {
            force_field_code: Default::default(),
        }
    }
}

impl SpecializedComputePipeline for ParticlesUpdatePipeline {
    type Key = ParticleUpdatePipelineKey;

    fn specialize(&self, key: Self::Key, render_device: &RenderDevice) -> ComputePipeline {
        trace!("Specializing update compute pipeline...");

        let mut source = VFX_UPDATE_SHADER_TEMPLATE.to_string();
        source = source.replace("{{FORCE_FIELD_CODE}}", &key.force_field_code);
        source.insert_str(0, VFX_COMMON_SHADER_IMPORT); // FIXME - #import not working on compute shaders

        trace!("Specialized update compute pipeline:\n{}", source);

        let shader_module = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("particles_update.wgsl"),
            source: ShaderSource::Wgsl(Cow::Owned(source)),
        });

        render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("vfx_update_pipeline"),
            layout: Some(&self.pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        })
    }
}

#[cfg(all(feature = "2d", feature = "3d"))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PipelineMode {
    Camera2d,
    Camera3d,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ParticleRenderPipelineKey {
    /// Render shader, with template applied, but not preprocessed yet.
    shader: Handle<Shader>,
    /// Key: PARTICLE_TEXTURE
    /// Define a texture sampled to modulate the particle color.
    /// This key requires the presence of UV coordinates on the particle vertices.
    particle_texture: Option<Handle<Image>>,
    /// For dual-mode configurations only, the actual mode of the current render
    /// pipeline. Otherwise the mode is implicitly determined by the active feature.
    #[cfg(all(feature = "2d", feature = "3d"))]
    pipeline_mode: PipelineMode,
}

impl Default for ParticleRenderPipelineKey {
    fn default() -> Self {
        ParticleRenderPipelineKey {
            shader: Handle::<Shader>::default(),
            particle_texture: None,
            #[cfg(all(feature = "2d", feature = "3d"))]
            pipeline_mode: PipelineMode::Camera3d,
        }
    }
}

impl SpecializedRenderPipeline for ParticlesRenderPipeline {
    type Key = ParticleRenderPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Base mandatory part of vertex buffer layout
        let vertex_buffer_layout = VertexBufferLayout {
            array_stride: 20,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                // [[location(0)]] vertex_position: vec3<f32>
                VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // [[location(1)]] vertex_uv: vec2<f32>
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 12,
                    shader_location: 1,
                },
                // [[location(1)]] vertex_color: u32
                // VertexAttribute {
                //     format: VertexFormat::Uint32,
                //     offset: 12,
                //     shader_location: 1,
                // },
                // [[location(2)]] vertex_velocity: vec3<f32>
                // VertexAttribute {
                //     format: VertexFormat::Float32x3,
                //     offset: 12,
                //     shader_location: 1,
                // },
                // [[location(3)]] vertex_uv: vec2<f32>
                // VertexAttribute {
                //     format: VertexFormat::Float32x2,
                //     offset: 28,
                //     shader_location: 3,
                // },
            ],
        };

        let mut layout = vec![
            self.particles_buffer_layout.clone(),
            self.view_layout.clone(),
        ];
        let mut shader_defs = vec![];

        // Key: PARTICLE_TEXTURE
        if key.particle_texture.is_some() {
            layout.push(self.material_layout.clone());
            shader_defs.push("PARTICLE_TEXTURE".to_string());
            // // [[location(1)]] vertex_uv: vec2<f32>
            // vertex_buffer_layout.attributes.push(VertexAttribute {
            //     format: VertexFormat::Float32x2,
            //     offset: 12,
            //     shader_location: 1,
            // });
            // vertex_buffer_layout.array_stride += 8;
        }

        #[cfg(all(feature = "2d", feature = "3d"))]
        let depth_stencil = match key.pipeline_mode {
            // Bevy's Transparent2d render phase doesn't support a depth-stencil buffer.
            PipelineMode::Camera2d => None,
            PipelineMode::Camera3d => Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                // Bevy uses reverse-Z, so Greater really means closer
                depth_compare: CompareFunction::Greater,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
        };

        #[cfg(all(feature = "2d", not(feature = "3d")))]
        let depth_stencil: Option<DepthStencilState> = None;

        #[cfg(all(feature = "3d", not(feature = "2d")))]
        let depth_stencil = Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: false,
            // Bevy uses reverse-Z, so Greater really means closer
            depth_compare: CompareFunction::Greater,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        });

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: key.shader.clone(),
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: key.shader,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            layout: Some(layout),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil,
            multisample: MultisampleState {
                count: 4, // TODO: Res<Msaa>.samples
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("particle_render_pipeline".into()),
        }
    }
}

/// A single effect instance extracted from a [`ParticleEffect`] as a [`RenderWorld`] item.
#[derive(Component)]
pub struct ExtractedEffect {
    /// Handle to the effect asset this instance is based on.
    /// The handle is weak to prevent refcount cycles and gracefully handle assets unloaded
    /// or destroyed after a draw call has been submitted.
    pub handle: Handle<EffectAsset>,
    /// Number of particles to spawn this frame for the effect.
    /// Obtained from calling [`Spawner::tick()`] on the source effect instance.
    pub spawn_count: u32,
    /// Global transform of the effect origin.
    pub transform: Mat4,
    /// Constant acceleration applied to all particles.
    pub accel: Vec3,
    /// Force field applied to all particles in the "update" phase.
    force_field: [ForceFieldParam; FFNUM],
    /// Particles tint to modulate with the texture image.
    pub color: Color,
    pub rect: Rect<f32>,
    // Texture to use for the sprites of the particles of this effect.
    //pub image: Handle<Image>,
    pub has_image: bool, // TODO -> use flags
    /// Texture to modulate the particle color.
    pub image_handle_id: HandleId,
    /// Render shader.
    pub shader: Handle<Shader>,
    /// Update position code.
    pub position_code: String,
    /// Update force field code.
    pub force_field_code: String,
}

impl ExtractedEffect {
    fn layout_flags(&self) -> LayoutFlags {
        if self.has_image {
            LayoutFlags::PARTICLE_TEXTURE
        } else {
            LayoutFlags::NONE
        }
    }
}

/// Extracted data for newly-added [`ParticleEffect`] component requiring a new GPU allocation.
pub struct AddedEffect {
    /// Entity with a newly-added [`ParticleEffect`] component.
    pub entity: Entity,
    /// Capacity of the effect (and therefore, the particle buffer), in number of particles.
    pub capacity: u32,
    /// Size in bytes of each particle.
    pub item_size: u32,
    /// Handle of the effect asset.
    pub handle: Handle<EffectAsset>,
}

/// Collection of all extracted effects for this frame, inserted into the
/// [`RenderWorld`] as a render resource.
#[derive(Default)]
pub struct ExtractedEffects {
    /// Map of extracted effects from the entity the source [`ParticleEffect`] is on.
    pub effects: HashMap<Entity, ExtractedEffect>,
    /// Entites which had their [`ParticleEffect`] component removed.
    pub removed_effect_entities: Vec<Entity>,
    /// Newly added effects without a GPU allocation yet.
    pub added_effects: Vec<AddedEffect>,
}

#[derive(Default)]
pub struct EffectAssetEvents {
    pub images: Vec<AssetEvent<Image>>,
}

pub fn extract_effect_events(
    mut render_world: ResMut<RenderWorld>,
    mut image_events: EventReader<AssetEvent<Image>>,
) {
    trace!("extract_effect_events");
    let mut events = render_world
        .get_resource_mut::<EffectAssetEvents>()
        .unwrap();
    let EffectAssetEvents { ref mut images } = *events;
    images.clear();

    for image in image_events.iter() {
        // AssetEvent: !Clone
        images.push(match image {
            AssetEvent::Created { handle } => AssetEvent::Created {
                handle: handle.clone_weak(),
            },
            AssetEvent::Modified { handle } => AssetEvent::Modified {
                handle: handle.clone_weak(),
            },
            AssetEvent::Removed { handle } => AssetEvent::Removed {
                handle: handle.clone_weak(),
            },
        });
    }
}

/// System extracting data for rendering of all active [`ParticleEffect`] components.
///
/// Extract rendering data for all [`ParticleEffect`] components in the world which are
/// visible ([`ComputedVisibility::is_visible`] is `true`), and wrap the data into a new
/// [`ExtractedEffect`] instance added to the [`ExtractedEffects`] resource.
pub(crate) fn extract_effects(
    mut render_world: ResMut<RenderWorld>,
    time: Res<Time>,
    effects: Res<Assets<EffectAsset>>,
    _images: Res<Assets<Image>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut pipeline_registry: ResMut<PipelineRegistry>,
    mut rng: ResMut<Random>,
    mut query: ParamSet<(
        // All existing ParticleEffect components
        Query<(
            Entity,
            &ComputedVisibility,
            &mut ParticleEffect, //TODO - Split EffectAsset::Spawner (desc) and ParticleEffect::SpawnerData (runtime data), and init the latter on component add without a need for the former
            &GlobalTransform,
        )>,
        // Newly added ParticleEffect components
        Query<
            (Entity, &mut ParticleEffect),
            (
                Added<ParticleEffect>,
                With<ComputedVisibility>,
                With<GlobalTransform>,
            ),
        >,
    )>,
    removed_effects: RemovedComponents<ParticleEffect>,
) {
    trace!("extract_effects");

    // Save simulation params into render world
    let mut sim_params = render_world.get_resource_mut::<SimParams>().unwrap();
    let dt = time.delta_seconds();
    sim_params.time = time.seconds_since_startup();
    sim_params.dt = dt;
    trace!("sim_params: time={} dt={}", sim_params.time, sim_params.dt);

    let mut extracted_effects = render_world.get_resource_mut::<ExtractedEffects>().unwrap();

    // Collect removed effects for later GPU data purge
    extracted_effects.removed_effect_entities = removed_effects.iter().collect();

    // Collect added effects for later GPU data allocation
    extracted_effects.added_effects = query
        .p1()
        .iter()
        .map(|(entity, effect)| {
            let handle = effect.handle.clone_weak();
            let asset = effects.get(&effect.handle).unwrap();
            AddedEffect {
                entity,
                capacity: asset.capacity,
                item_size: Particle::std430_size_static() as u32, // effect.item_size(),
                handle,
            }
        })
        .collect();

    // Loop over all existing effects to update them
    for (entity, computed_visibility, mut effect, transform) in query.p0().iter_mut() {
        // Check if visible
        if !computed_visibility.is_visible {
            continue;
        }

        // Check if asset is available, otherwise silently ignore
        let asset = match effects.get(&effect.handle) {
            Some(asset) => asset,
            None => continue,
        };
        //let size = image.texture_descriptor.size;

        // Tick the effect's spawner to determine the spawn count for this frame
        let spawner = effect.spawner(&asset.spawner);
        let spawn_count = spawner.tick(dt, &mut rng.0);

        // Extract the global effect acceleration to apply to all particles
        let accel = asset.update_layout.accel;
        let force_field = asset.update_layout.force_field;

        // Generate the shader code for the position initializing of newly emitted particles
        // TODO - Move that to a pre-pass, not each frame!
        let position_code = &asset.init_layout.position_code;
        let position_code = if position_code.is_empty() {
            DEFAULT_POSITION_CODE.to_owned()
        } else {
            position_code.clone()
        };

        // Generate the shader code for the force field of newly emitted particles
        // TODO - Move that to a pre-pass, not each frame!
        // let force_field_code = &asset.init_layout.force_field_code;
        // let force_field_code = if force_field_code.is_empty() {
        let force_field_code = if 0.0 == asset.update_layout.force_field[0].force_exponent {
            DEFAULT_FORCE_FIELD_CODE.to_owned()
        } else {
            FORCE_FIELD_CODE.to_owned()
        };

        // Generate the shader code for the color over lifetime gradient.
        // TODO - Move that to a pre-pass, not each frame!
        let mut vertex_modifiers = if let Some(grad) = &asset.render_layout.lifetime_color_gradient
        {
            grad.to_shader_code()
        } else {
            String::new()
        };
        if let Some(grad) = &asset.render_layout.size_color_gradient {
            vertex_modifiers += &grad.to_shader_code();
        }
        //trace!("vertex_modifiers={}", vertex_modifiers);

        // Configure the shader template, and make sure a corresponding shader asset exists
        let shader_source =
            VFX_RENDER_SHADER_TEMPLATE.replace("{{VERTEX_MODIFIERS}}", &vertex_modifiers);
        let shader = pipeline_registry.configure(&shader_source, &mut shaders);

        trace!(
            "extracted: handle={:?} shader={:?} has_image={}",
            effect.handle,
            shader,
            if asset.render_layout.particle_texture.is_some() {
                "Y"
            } else {
                "N"
            }
        );

        extracted_effects.effects.insert(
            entity,
            ExtractedEffect {
                handle: effect.handle.clone_weak(),
                spawn_count,
                color: Color::RED, //effect.color,
                transform: transform.compute_matrix(),
                accel,
                rect: Rect {
                    left: -0.1,
                    right: 0.1,
                    top: -0.1,
                    bottom: 0.2, // effect.custom_size.unwrap_or_else(|| Vec2::new(size.width as f32, size.height as f32)),
                },
                has_image: asset.render_layout.particle_texture.is_some(),
                image_handle_id: asset
                    .render_layout
                    .particle_texture
                    .clone()
                    .map_or(HandleId::default::<Image>(), |handle| handle.id),
                shader,
                position_code,
                force_field_code,
            },
        );
    }
}

/// A single particle as stored in a GPU buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, AsStd430)]
struct Particle {
    /// Particle position in effect space (local or world).
    pub position: [f32; 3],
    /// Current particle age in \[0:`lifetime`\].
    pub age: f32,
    /// Particle velocity in effect space (local or world).
    pub velocity: [f32; 3],
    /// Total particle lifetime.
    pub lifetime: f32,
}

/// A single vertex of a particle mesh as stored in a GPU buffer.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ParticleVertex {
    /// Vertex position.
    pub position: [f32; 3],
    /// UV coordinates of vertex.
    pub uv: [f32; 2],
}

/// Global resource containing the GPU data to draw all the particle effects in all views.
///
/// The resource is populated by [`prepare_effects()`] with all the effects to render
/// for the current frame, for all views in the frame, and consumed by [`queue_effects()`]
/// to actually enqueue the drawning commands to draw those effects.
pub(crate) struct EffectsMeta {
    /// Map from an entity with a [`ParticleEffect`] component attached to it, to the associated
    /// effect slice allocated in an [`EffectCache`].
    entity_map: HashMap<Entity, EffectSlice>,
    /// Global effect cache for all effects in use.
    effect_cache: EffectCache,
    /// Bind group for the camera view, containing the camera projection and other uniform
    /// values related to the camera.
    view_bind_group: Option<BindGroup>,
    /// Bind group for the simulation parameters, like the current time and frame delta time.
    sim_params_bind_group: Option<BindGroup>,
    /// Bind group for the particles buffer itself.
    particles_bind_group: Option<BindGroup>,
    /// Bind group for the spawning parameters (number of particles to spawn this frame, ...).
    spawner_bind_group: Option<BindGroup>,
    /// Bind group for the effect parameters.
    effect_params_bind_group: Option<BindGroup>,
    /// Bind group for all the buffer containing all DispatchBuffer instances for all effects,
    /// which are used to dispatch update compute passes based on the current number of alive
    /// particles as updated by the init compute passes.
    dispach_buffer_bind_group: Option<BindGroup>,
    /// Global effect-independent simulation parameters.
    sim_params_uniforms: NamedUniformVec<SimParamsUniform>,
    /// Per-effect simulation parameters for all effects.
    effect_params_uniforms: NamedUniformVec<EffectParamsUniform>,
    /// Spawner parameteres for all effects.
    spawner_buffer: AlignedBufferVec<SpawnerParamsUniform>,
    /// Dispatch buffers for all effects.
    dispatch_buffers: StorageVec<DispatchBuffer>,
    /// Unscaled vertices of the mesh of a single particle, generally a quad.
    /// The mesh is later scaled during rendering by the "particle size".
    // FIXME - This is a per-effect thing, unless we merge all meshes into a single buffer (makes
    // sense) but in that case we need a vertex slice too to know which mesh to draw per effect.
    vertices: BufferVec<ParticleVertex>,
}

impl EffectsMeta {
    pub fn new(device: RenderDevice) -> Self {
        let mut vertices = BufferVec::new(BufferUsages::VERTEX);
        for v in QUAD_VERTEX_POSITIONS {
            let uv = v.truncate() + 0.5;
            let v = *v * Vec3::new(1.0, 1.0, 1.0);
            vertices.push(ParticleVertex {
                position: v.into(),
                uv: uv.into(),
            });
        }

        let item_align = device.limits().min_storage_buffer_offset_alignment as usize;

        Self {
            entity_map: HashMap::default(),
            effect_cache: EffectCache::new(device),
            view_bind_group: None,
            sim_params_bind_group: None,
            particles_bind_group: None,
            spawner_bind_group: None,
            effect_params_bind_group: None,
            dispach_buffer_bind_group: None,
            sim_params_uniforms: NamedUniformVec::new(Cow::from("vfx_uniforms_sim_params")),
            effect_params_uniforms: NamedUniformVec::new(Cow::from("vfx_uniforms_effect_params")),
            spawner_buffer_uniforms: AlignedBufferVec::new(
                BufferUsages::STORAGE,
                item_align,
                Some("vfx_uniforms_spawner".to_string()),
            ),
            dispatch_buffers: StorageVec::new(Cow::from("vfx_storage_dispatch_buffers")),
            vertices,
        }
    }
}

const QUAD_VERTEX_POSITIONS: &[Vec3] = &[
    const_vec3!([-0.5, -0.5, 0.0]),
    const_vec3!([0.5, 0.5, 0.0]),
    const_vec3!([-0.5, 0.5, 0.0]),
    const_vec3!([-0.5, -0.5, 0.0]),
    const_vec3!([0.5, -0.5, 0.0]),
    const_vec3!([0.5, 0.5, 0.0]),
];

bitflags! {
    struct LayoutFlags: u32 {
        const NONE = 0;
        const PARTICLE_TEXTURE = 0b00000001;
    }
}

impl Default for LayoutFlags {
    fn default() -> Self {
        LayoutFlags::NONE
    }
}

/// A single slice inside a batch.
pub struct EffectBatchSlice {
    /// Number of particles spawned this frame, for init pass dispatch.
    spawn_count: u32,
    /// Origin of the spawner, for init pass.
    origin: Vec3,
    /// Slice in the GPU effect buffer of the particles to update for the entire batch.
    slice: Range<u32>,
    /// Global acceleration applied to all particles, for update pass.
    accel: Vec3,
    /// Index of the slice into the SliceList/GpuSlice buffer.
    slice_index: u32,
}

/// A batch of multiple instances of the same effect, rendered all together to reduce GPU shader
/// permutations and draw call overhead.
#[derive(Component)]
pub struct EffectBatch {
    /// Index of the GPU effect buffer that effects in this batch are stored in.
    /// This is also the index of the DispatchBuffer.
    buffer_index: u32,
    /// Index of the first Spawner of the effects in the batch.
    spawner_base: u32,
    /// Slices of various instances batched together.
    slices: Vec<EffectBatchSlice>,
    /// Size of a single particle.
    item_size: u32,
    /// Handle of the underlying effect asset describing the effect.
    handle: Handle<EffectAsset>,
    /// Flags describing the render layout.
    layout_flags: LayoutFlags,
    /// Texture to modulate the particle color.
    image_handle_id: HandleId,
    /// Render shader.
    shader: Handle<Shader>,
    /// Update position code.
    position_code: String,
    /// Update force field code.
    force_field_code: String,
    /// Prepare pipeline.
    prepare_pipeline: Option<ComputePipeline>,
    /// Init compute pipeline specialized for this batch.
    init_pipeline: Option<ComputePipeline>,
    /// Update compute pipeline specialized for this batch.
    update_pipeline: Option<ComputePipeline>,
}

pub(crate) fn prepare_effects(
    mut commands: Commands,
    sim_params: Res<SimParams>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    //update_pipeline: Res<ParticlesUpdatePipeline>, // TODO move update_pipeline.pipeline to EffectsMeta
    mut effects_meta: ResMut<EffectsMeta>,
    mut extracted_effects: ResMut<ExtractedEffects>,
) {
    trace!("prepare_effects");

    // Allocate simulation uniform if needed
    if effects_meta.sim_params_uniforms.is_empty() {
        effects_meta
            .sim_params_uniforms
            .push(SimParamsUniform::default());
    }

    // Update simulation parameters
    {
        let sim_params_uni = effects_meta.sim_params_uniforms.get_mut(0);
        let sim_params = *sim_params;
        *sim_params_uni = sim_params.into();
    }
    trace!(
        "Simulation parameters: time={} dt={}",
        sim_params.time,
        sim_params.dt,
    );
    effects_meta
        .sim_params_uniforms
        .write_buffer(&render_device, &render_queue);

    // Allocate spawner buffer if needed
    //if effects_meta.spawner_buffer.is_empty() {
    //    effects_meta.spawner_buffer.push(SpawnerParams::default());
    //}

    // Write vertices (TODO - lazily once only)
    effects_meta
        .vertices
        .write_buffer(&render_device, &render_queue);

    // Allocate GPU data for newly created effect instances. Do this first to ensure a group is not left
    // unused and dropped due to the last effect being removed but a new compatible one added not being
    // inserted yet. By inserting first, we ensure the group is not dropped in this case.
    let old_dispatch_index = effects_meta.effect_cache.buffers().len() as u32;
    for added_effect in extracted_effects.added_effects.drain(..) {
        let entity = added_effect.entity;
        let (_, slice) = effects_meta.effect_cache.insert(
            added_effect.handle,
            added_effect.capacity,
            added_effect.item_size,
            //update_pipeline.pipeline.clone(),
            &render_queue,
        );
        effects_meta.entity_map.insert(entity, slice);

        effects_meta.dispatch_buffers.push(DispatchBuffer {
            dead_count: added_effect.capacity,
            ..Default::default()
        });
    }
    let new_dispatch_index = effects_meta.effect_cache.buffers().len() as u32;

    // Write the new dispatch buffers if any
    let dispatch_slice = old_dispatch_index..new_dispatch_index;
    if !dispatch_slice.is_empty() {
        effects_meta
            .dispatch_buffers
            .write_slice(dispatch_slice, &render_device, &render_queue);
    }

    // Deallocate GPU data for destroyed effect instances. This will automatically drop any group where
    // there is no more effect slice.
    for _entity in extracted_effects.removed_effect_entities.iter() {
        unimplemented!("Remove particle effect.");
        //effects_meta.remove(&*entity);
    }

    // // sort first by z and then by handle. this ensures that, when possible, batches span multiple z layers
    // // batches won't span z-layers if there is another batch between them
    // extracted_effects.effects.sort_by(|a, b| {
    //     match FloatOrd(a.transform.w_axis[2]).cmp(&FloatOrd(b.transform.w_axis[2])) {
    //         Ordering::Equal => a.handle.cmp(&b.handle),
    //         other => other,
    //     }
    // });

    // Get the effect-entity mapping
    let mut effect_entity_list = extracted_effects
        .effects
        .iter()
        .filter_map(|(entity, extracted_effect)| {
            if let Some(slice) = effects_meta.entity_map.get(&entity) {
                Some((slice.clone(), extracted_effect))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    trace!("Collected {} extracted effects", effect_entity_list.len());

    // Sort first by effect buffer, then by slice range (see EffectSlice)
    effect_entity_list.sort_by(|a, b| a.0.cmp(&b.0));

    // Batch extracted effects
    effects_meta.spawner_uniforms.clear();
    effects_meta.effect_params_uniforms.clear();
    let mut batcher = EffectBatcher::new(0);
    for (slice, extracted_effect) in effect_entity_list {
        batcher.insert(extracted_effect, &slice);
    }

    // Write batches
    trace!("Write batches");
    for batch in batcher.into_batches().into_iter() {
        trace!(
            "+ batch: buffer_index={} spawner_base={} item_size={}B",
            batch.buffer_index,
            batch.spawner_base,
            batch.item_size
        );

        for slice in &batch.slices {
            trace!(
                "  + slice: spawn_count={} origin={:?} slice={:?} accel={:?}",
                slice.spawn_count,
                slice.origin,
                slice.slice,
                slice.accel
            );
            let spawner_params = SpawnerParams {
                origin: slice.origin,
                spawn_count: slice.spawn_count as i32,
                //accel: slice.accel,
                seed: random::<u32>(),
                particle_base: slice.slice.start,
                ..Default::default()
            };
            trace!("    spawner_params = {:?}", spawner_params);
            effects_meta.spawner_uniforms.push(spawner_params.into());

            let effect_params = EffectParams {
                accel: slice.accel,
                particle_base: slice.slice.start,
            };
            trace!("    effect_params = {:?}", effect_params);
            effects_meta
                .effect_params_uniforms
                .push(effect_params.into());
        }

        // Spawn the batch into the render world
        commands.spawn().insert(batch);
    }

    // Write the entire spawner buffer for this frame, for all effects combined
    effects_meta
        .spawner_uniforms
        .write_buffer(&render_device, &render_queue);

    // Write the entire effect params buffer for this frame, for all effects combined
    effects_meta
        .effect_params_uniforms
        .write_buffer(&render_device, &render_queue);
}

#[derive(Default)]
pub struct ImageBindGroups {
    values: HashMap<Handle<Image>, BindGroup>,
}

pub struct EffectBindGroup {
    common_particle_buffer: BindGroup,
    common_dead_list: BindGroup,
    common_slice_list: BindGroup,
    update_draw_indirect: BindGroup,
    update_indirect_buffer: BindGroup,
    render_particle_buffer: BindGroup,
}

#[derive(Default)]
pub struct EffectBindGroups {
    /// Bind groups for each group index for compute shader.
    group_from_index: HashMap<u32, EffectBindGroup>,
}

#[derive(SystemParam)]
struct PipelineParams<'w, 's> {
    prepare_pipeline: Res<'w, ParticlesPreparePipeline>,
    init_pipeline: Res<'w, ParticlesInitPipeline>,
    update_pipeline: Res<'w, ParticlesUpdatePipeline>,
    render_pipeline: Res<'w, ParticlesRenderPipeline>,
    #[system_param(ignore)]
    marker: std::marker::PhantomData<&'s usize>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_effects(
    #[cfg(feature = "2d")] draw_functions_2d: Res<DrawFunctions<Transparent2d>>,
    #[cfg(feature = "3d")] draw_functions_3d: Res<DrawFunctions<Transparent3d>>,
    render_device: Res<RenderDevice>,
    mut effects_meta: ResMut<EffectsMeta>,
    view_uniforms: Res<ViewUniforms>,
    pipelines: PipelineParams,
    mut init_compute_cache: ResMut<ComputeCache<ParticlesInitPipeline>>,
    mut update_compute_cache: ResMut<ComputeCache<ParticlesUpdatePipeline>>,
    mut specialized_render_pipelines: ResMut<SpecializedRenderPipelines<ParticlesRenderPipeline>>,
    mut render_pipeline_cache: ResMut<PipelineCache>,
    mut image_bind_groups: ResMut<ImageBindGroups>,
    mut effect_bind_groups: ResMut<EffectBindGroups>,
    gpu_images: Res<RenderAssets<Image>>,
    mut effect_batches: Query<(Entity, &mut EffectBatch)>,
    #[cfg(feature = "2d")] mut views_2d: Query<&mut RenderPhase<Transparent2d>>,
    #[cfg(feature = "3d")] mut views_3d: Query<&mut RenderPhase<Transparent3d>>,
    //events: Res<EffectAssetEvents>,
) {
    trace!("queue_effects");

    let prepare_pipeline = pipelines.prepare_pipeline;
    let init_pipeline = pipelines.init_pipeline;
    let update_pipeline = pipelines.update_pipeline;
    let render_pipeline = pipelines.render_pipeline;

    // If an image has changed, the GpuImage has (probably) changed
    // for event in &events.images {
    //     match event {
    //         AssetEvent::Created { .. } => None,
    //         AssetEvent::Modified { handle } => image_bind_groups.values.remove(handle),
    //         AssetEvent::Removed { handle } => image_bind_groups.values.remove(handle),
    //     };
    // }

    // Get the binding for the ViewUniform, the uniform data structure containing the Camera data
    // for the current view.
    let view_binding = match view_uniforms.uniforms.binding() {
        Some(view_binding) => view_binding,
        None => {
            return;
        }
    };

    // Create the bind group for the camera/view parameters
    effects_meta.view_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[BindGroupEntry {
            binding: 0,
            resource: view_binding,
        }],
        label: Some("vfx_view_bind_group"),
        layout: &render_pipeline.view_layout,
    }));

    // Create the bind group for the global simulation parameters
    effects_meta.sim_params_bind_group =
        Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: effects_meta.sim_params_uniforms.binding().unwrap(),
            }],
            label: Some("vfx_sim_params_bind_group"),
            layout: &update_pipeline.sim_params_layout,
        }));

    // Create the bind group for the spawner parameters
    trace!(
        "SpawnerParams::std430_size_static() = {}",
        SpawnerParams::std430_size_static()
    );
    effects_meta.spawner_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
        entries: &[BindGroupEntry {
            binding: 0,
            resource: effects_meta.spawner_uniforms.binding().unwrap(),
        }],
        label: Some("vfx_spawner_bind_group"),
        layout: &init_pipeline.spawner_buffer_layout,
    }));

    // Create the bind group for the DispatchBuffer buffer
    effects_meta.dispach_buffer_bind_group =
        Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: effects_meta.dispatch_buffers.binding().unwrap(),
            }],
            label: Some("vfx_dispach_buffer_bind_group"),
            layout: &init_pipeline.dispatch_buffer_layout,
        }));

    // Create the bind group for the effect parameters
    effects_meta.effect_params_bind_group =
        Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: effects_meta.effect_params_uniforms.binding().unwrap(),
            }],
            label: Some("vfx_effect_params_bind_group"),
            layout: &update_pipeline.effect_params_layout,
        }));

    // Queue the update compute
    trace!("queue effects from cache...");
    for (buffer_index, buffer) in effects_meta.effect_cache.buffers().iter().enumerate() {
        // Ensure all effect groups have a bind group for the entire buffer of the group,
        // since the update phase runs on an entire group/buffer at once, with all the
        // effect instances in it batched together.
        trace!("effect buffer_index=#{}", buffer_index);
        effect_bind_groups
            .group_from_index
            .entry(buffer_index as u32)
            .or_insert_with(|| {
                trace!("Create new bind group for buffer_index={}", buffer_index);

                let common_particle_buffer =
                    render_device.create_bind_group(&BindGroupDescriptor {
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: buffer.max_binding(BufferKind::Particles),
                        }],
                        label: Some(&format!("vfx_particles_bind_group{}", buffer_index)),
                        layout: &update_pipeline.particles_buffer_layout,
                    });

                let common_dead_list = render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.max_binding(BufferKind::DeadList),
                    }],
                    label: Some(&format!("vfx_dead_list_bind_group{}", buffer_index)),
                    layout: &update_pipeline.dead_list_layout,
                });

                let common_slice_list = render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.max_binding(BufferKind::SliceList),
                    }],
                    label: Some(&format!("vfx_slice_list_bind_group{}", buffer_index)),
                    layout: &update_pipeline.slice_list_layout,
                });

                let update_draw_indirect = render_device.create_bind_group(&BindGroupDescriptor {
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buffer.max_binding(BufferKind::DrawIndirect),
                    }],
                    label: Some(&format!("vfx_draw_indirect_bind_group{}", buffer_index)),
                    layout: &update_pipeline.draw_indirect_layout,
                });

                let update_indirect_buffer =
                    render_device.create_bind_group(&BindGroupDescriptor {
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: buffer.max_binding(BufferKind::IndirectBuffer),
                        }],
                        label: Some(&format!("vfx_indirect_buffer_bind_group{}", buffer_index)),
                        layout: &update_pipeline.indirect_buffer_layout,
                    });

                let render_particle_buffer =
                    render_device.create_bind_group(&BindGroupDescriptor {
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: buffer.max_binding(BufferKind::Particles),
                        }],
                        label: Some(&format!("vfx_render_particles_bind_group{}", buffer_index)),
                        layout: &render_pipeline.particles_buffer_layout,
                    });

                EffectBindGroup {
                    common_particle_buffer,
                    common_dead_list,
                    common_slice_list,
                    update_draw_indirect,
                    update_indirect_buffer,
                    render_particle_buffer,
                }
            });
    }

    // Get the prepare pipeline
    let prepare_pipeline = prepare_pipeline.pipeline;

    // Queue the update
    // TODO - Move to prepare(), there's no view-dependent thing here!
    //let buffers = effects_meta.effect_cache.buffers();
    for (_, mut batch) in effect_batches.iter_mut() {
        //let buffer = &buffers[batch.buffer_index as usize];

        // // Create bind groups
        // let common_particle_buffer = render_device.create_bind_group(&BindGroupDescriptor {
        //     entries: &[BindGroupEntry {
        //         binding: 0,
        //         resource: buffer.binding(BufferKind::Particles),
        //     }],
        //     label: Some(&format!("vfx_particles_bind_group{}", batch.buffer_index)),
        //     layout: &update_pipeline.particles_buffer_layout,
        // });

        batch.prepare_pipeline = prepare_pipeline;

        // Specialize the init pipeline based on the effect batch
        let init_pipeline = init_compute_cache.specialize(
            &init_pipeline,
            ParticleInitPipelineKey {
                position_code: batch.position_code.clone(),
            },
            &render_device,
        );
        batch.init_pipeline = Some(init_pipeline.clone());

        // Specialize the update pipeline based on the effect batch
        let update_pipeline = update_compute_cache.specialize(
            &update_pipeline,
            ParticleUpdatePipelineKey {
                force_field_code: batch.force_field_code.clone(),
            },
            &render_device,
        );
        batch.update_pipeline = Some(update_pipeline.clone());
    }

    // Loop over all 2D cameras/views that need to render effects
    #[cfg(feature = "2d")]
    {
        let draw_effects_function_2d = draw_functions_2d.read().get_id::<DrawEffects>().unwrap();
        for mut transparent_phase_2d in views_2d.iter_mut() {
            trace!("Process new Transparent2d view");
            // For each view, loop over all the effect batches to determine if the effect needs to be rendered
            // for that view, and enqueue a view-dependent batch if so.
            for (entity, batch) in effect_batches.iter() {
                trace!(
                    "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?}",
                    entity,
                    batch.buffer_index,
                    batch.spawner_base,
                    batch.slice
                );
                // Ensure the particle texture is available as a GPU resource and create a bind group for it
                let particle_texture = if batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE)
                {
                    let image_handle = Handle::weak(batch.image_handle_id);
                    if effect_bind_groups.images.get(&image_handle).is_none() {
                        trace!(
                            "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                            batch.buffer_index,
                            batch.slice
                        );
                        // If texture doesn't have a bind group yet from another instance of the same effect,
                        // then try to create one now
                        if let Some(gpu_image) = gpu_images.get(&image_handle) {
                            let bind_group =
                                render_device.create_bind_group(&BindGroupDescriptor {
                                    entries: &[
                                        BindGroupEntry {
                                            binding: 0,
                                            resource: BindingResource::TextureView(
                                                &gpu_image.texture_view,
                                            ),
                                        },
                                        BindGroupEntry {
                                            binding: 1,
                                            resource: BindingResource::Sampler(&gpu_image.sampler),
                                        },
                                    ],
                                    label: Some("particles_material_bind_group"),
                                    layout: &render_pipeline.material_layout,
                                });
                            effect_bind_groups
                                .images
                                .insert(image_handle.clone(), bind_group);
                            Some(image_handle)
                        } else {
                            // Texture is not ready; skip for now...
                            trace!("GPU image not yet available; skipping batch for now.");
                            None
                        }
                    } else {
                        // Bind group already exists, meaning texture is ready
                        Some(image_handle)
                    }
                } else {
                    // Batch doesn't use particle texture
                    None
                };

                // Specialize the render pipeline based on the effect batch
                trace!(
                    "Specializing render pipeline: shader={:?} particle_texture={:?}",
                    batch.shader,
                    particle_texture
                );
                let render_pipeline_id = specialized_render_pipelines.specialize(
                    &mut render_pipeline_cache,
                    &render_pipeline,
                    ParticleRenderPipelineKey {
                        particle_texture,
                        shader: batch.shader.clone(),
                        #[cfg(feature = "3d")]
                        pipeline_mode: PipelineMode::Camera2d,
                    },
                );
                trace!("Render pipeline specialized: id={:?}", render_pipeline_id);

                for slice in &batch.slices {
                    // Add a draw pass for the effect batch
                    trace!("Add Transparent for batch on entity {:?}: buffer_index={} spawner_base={} slice={:?} handle={:?}", entity, batch.buffer_index, batch.spawner_base, batch.slice, batch.handle);
                    transparent_phase_2d.add(Transparent2d {
                        draw_function: draw_effects_function_2d,
                        pipeline: render_pipeline_id,
                        entity,
                        sort_key: FloatOrd(0.0),
                        batch_range: None,
                    });
                }
            }
        }
    }

    // Loop over all 3D cameras/views that need to render effects
    #[cfg(feature = "3d")]
    {
        let draw_effects_function_3d = draw_functions_3d.read().get_id::<DrawEffects>().unwrap();
        for mut transparent_phase_3d in views_3d.iter_mut() {
            trace!("Process new Transparent3d view");
            // For each view, loop over all the effect batches to determine if the effect needs to be rendered
            // for that view, and enqueue a view-dependent batch if so.
            for (entity, batch) in effect_batches.iter() {
                trace!(
                    "Process batch entity={:?} buffer_index={} spawner_base={} slice={:?}",
                    entity,
                    batch.buffer_index,
                    batch.spawner_base,
                    batch.slice
                );
                // Ensure the particle texture is available as a GPU resource and create a bind group for it
                let particle_texture = if batch.layout_flags.contains(LayoutFlags::PARTICLE_TEXTURE)
                {
                    let image_handle = Handle::weak(batch.image_handle_id);
                    if effect_bind_groups.images.get(&image_handle).is_none() {
                        trace!(
                            "Batch buffer #{} slice={:?} has missing GPU image bind group, creating...",
                            batch.buffer_index,
                            batch.slice
                        );
                        // If texture doesn't have a bind group yet from another instance of the same effect,
                        // then try to create one now
                        if let Some(gpu_image) = gpu_images.get(&image_handle) {
                            let bind_group =
                                render_device.create_bind_group(&BindGroupDescriptor {
                                    entries: &[
                                        BindGroupEntry {
                                            binding: 0,
                                            resource: BindingResource::TextureView(
                                                &gpu_image.texture_view,
                                            ),
                                        },
                                        BindGroupEntry {
                                            binding: 1,
                                            resource: BindingResource::Sampler(&gpu_image.sampler),
                                        },
                                    ],
                                    label: Some("particles_material_bind_group"),
                                    layout: &render_pipeline.material_layout,
                                });
                            effect_bind_groups
                                .images
                                .insert(image_handle.clone(), bind_group);
                            Some(image_handle)
                        } else {
                            // Texture is not ready; skip for now...
                            trace!("GPU image not yet available; skipping batch for now.");
                            None
                        }
                    } else {
                        // Bind group already exists, meaning texture is ready
                        Some(image_handle)
                    }
                } else {
                    // Batch doesn't use particle texture
                    None
                };

                // Specialize the render pipeline based on the effect batch
                trace!(
                    "Specializing render pipeline: shader={:?} particle_texture={:?}",
                    batch.shader,
                    particle_texture
                );
                let render_pipeline_id = specialized_render_pipelines.specialize(
                    &mut render_pipeline_cache,
                    &render_pipeline,
                    ParticleRenderPipelineKey {
                        particle_texture,
                        shader: batch.shader.clone(),
                        #[cfg(feature = "2d")]
                        pipeline_mode: PipelineMode::Camera3d,
                    },
                );
                trace!("Render pipeline specialized: id={:?}", render_pipeline_id);

                // Add a draw pass for the effect batch
                trace!("Add Transparent for batch on entity {:?}: buffer_index={} spawner_base={} slice={:?} handle={:?}", entity, batch.buffer_index, batch.spawner_base, batch.slice, batch.handle);
                transparent_phase_3d.add(Transparent3d {
                    draw_function: draw_effects_function_3d,
                    pipeline: render_pipeline_id,
                    entity,
                    distance: 0.0, // TODO ??????
                });
            }
        }
    }
}

/// Component to hold all the entities with a [`ExtractedEffect`] component on them
/// that need to be updated this frame with a compute pass. This is view-independent
/// because the update phase itself is also view-independent (effects like camera
/// facing are applied in the render phase, which runs once per view).
#[derive(Component)]
pub struct ExtractedEffectEntities {
    pub entities: Vec<Entity>,
}

/// Draw function for rendering all active effects for the current frame.
///
/// Effects are rendered in the [`Transparent2d`] phase of the main 2D pass,
/// and the [`Transparent3d`] phase of the main 3D pass.
pub struct DrawEffects {
    params: SystemState<(
        SRes<EffectsMeta>,
        SRes<EffectBindGroups>,
        SRes<PipelineCache>,
        SQuery<Read<ViewUniformOffset>>,
        SQuery<Read<EffectBatch>>,
    )>,
}

impl DrawEffects {
    pub fn new(world: &mut World) -> Self {
        Self {
            params: SystemState::new(world),
        }
    }
}

#[cfg(feature = "2d")]
impl Draw<Transparent2d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Transparent2d,
    ) {
        trace!("Draw<Transparent2d>: view={:?}", view);
        let (effects_meta, effect_bind_groups, specialized_render_pipelines, views, effects) =
            self.params.get(world);
        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();
        if let Some(pipeline) = specialized_render_pipelines
            .into_inner()
            .get_render_pipeline(item.pipeline)
        {
            trace!("render pass");

            //let effect_group = &effects_meta.effect_cache.buffers()[0]; // TODO

            let render_particle_buffer_bind_group = match effect_bind_groups
                .group_from_index
                .get(&effect_batch.buffer_index)
            {
                Some(bind_groups) => &bind_groups.render_particle_buffer,
                None => {
                    return;
                }
            };

            pass.set_render_pipeline(pipeline);

            // Vertex buffer containing the particle model to draw. Generally a quad.
            pass.set_vertex_buffer(0, effects_meta.vertices.buffer().unwrap().slice(..));

            // group(0): Particles buffer
            pass.set_bind_group(0, render_particle_buffer_bind_group, &[]);

            // group(1): View properties (camera matrix, etc.)
            pass.set_bind_group(
                1,
                effects_meta.view_bind_group.as_ref().unwrap(),
                &[view_uniform.offset],
            );

            // group(2): Particle texture
            if effect_batch
                .layout_flags
                .contains(LayoutFlags::PARTICLE_TEXTURE)
            {
                let image_handle = Handle::weak(effect_batch.image_handle_id);
                if let Some(bind_group) = effect_bind_groups.images.get(&image_handle) {
                    pass.set_bind_group(2, bind_group, &[]);
                } else {
                    // Texture not ready; skip this drawing for now
                    trace!(
                        "Particle texture bind group not available for batch buf={} slice={:?}. Skipping draw call.",
                        effect_batch.buffer_index,
                        effect_batch.slice
                    );
                    return; //continue;
                }
            }

            let vertex_count = effects_meta.vertices.len() as u32;
            let particle_count = effect_batch.slice.end - effect_batch.slice.start;

            trace!(
                "Draw {} particles with {} vertices per particle for batch from buffer #{}.",
                particle_count,
                vertex_count,
                effect_batch.buffer_index
            );
            pass.draw(0..vertex_count, 0..particle_count);
        }
    }
}

#[cfg(feature = "3d")]
impl Draw<Transparent3d> for DrawEffects {
    fn draw<'w>(
        &mut self,
        world: &'w World,
        pass: &mut TrackedRenderPass<'w>,
        view: Entity,
        item: &Transparent3d,
    ) {
        trace!("Draw<Transparent3d>: view={:?}", view);
        let (effects_meta, effect_bind_groups, specialized_render_pipelines, views, effects) =
            self.params.get(world);
        let view_uniform = views.get(view).unwrap();
        let effects_meta = effects_meta.into_inner();
        let effect_bind_groups = effect_bind_groups.into_inner();
        let effect_batch = effects.get(item.entity).unwrap();
        if let Some(pipeline) = specialized_render_pipelines
            .into_inner()
            .get_render_pipeline(item.pipeline)
        {
            trace!("render pass");
            //let effect_group = &effects_meta.effect_cache.buffers()[0]; // TODO

            pass.set_render_pipeline(pipeline);

            // Vertex buffer containing the particle model to draw. Generally a quad.
            pass.set_vertex_buffer(0, effects_meta.vertices.buffer().unwrap().slice(..));

            // View properties (camera matrix, etc.)
            pass.set_bind_group(
                0,
                effects_meta.view_bind_group.as_ref().unwrap(),
                &[view_uniform.offset],
            );

            // Particles buffer
            pass.set_bind_group(
                1,
                effect_bind_groups
                    .render_particle_buffers
                    .get(&effect_batch.buffer_index)
                    .unwrap(),
                &[],
            );

            // Particle texture
            if effect_batch
                .layout_flags
                .contains(LayoutFlags::PARTICLE_TEXTURE)
            {
                let image_handle = Handle::weak(effect_batch.image_handle_id);
                if let Some(bind_group) = effect_bind_groups.images.get(&image_handle) {
                    pass.set_bind_group(2, bind_group, &[]);
                } else {
                    // Texture not ready; skip this drawing for now
                    trace!(
                        "Particle texture bind group not available for batch buf={}. Skipping draw call.",
                        effect_batch.buffer_index,
                    );
                    return;
                }
            }

            let vertex_count = effects_meta.vertices.len() as u32;

            for slice in &effect_batch.slices {
                let particle_count = slice.slice.end - slice.slice.start;

                trace!(
                    "Draw {} particles with {} vertices per particle for batch from buffer #{} slice {:?}.",
                    particle_count,
                    vertex_count,
                    effect_batch.buffer_index,
                    slice.slice
                );
                pass.draw(0..vertex_count, 0..particle_count);
            }
        }
    }
}

/// A render node to update the particles of all particle efects.
pub struct ParticleUpdateNode {
    /// Query to retrieve the list of entities holding an extracted particle effect to update.
    entity_query: QueryState<&'static ExtractedEffectEntities>,
    /// Query to retrieve the
    effect_query: QueryState<&'static EffectBatch>,
}

impl ParticleUpdateNode {
    /// Input entity marking the view.
    pub const IN_VIEW: &'static str = "view";
    /// Output particle buffer for that view. TODO - how to handle multiple buffers?! Should use Entity instead??
    //pub const OUT_PARTICLE_BUFFER: &'static str = "particle_buffer";

    pub fn new(world: &mut World) -> Self {
        Self {
            entity_query: QueryState::new(world),
            effect_query: QueryState::new(world),
        }
    }
}

impl Node for ParticleUpdateNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(ParticleUpdateNode::IN_VIEW, SlotType::Entity)]
    }

    // fn output(&self) -> Vec<SlotInfo> {
    //     vec![SlotInfo::new(
    //         ParticleUpdateNode::OUT_PARTICLE_BUFFER,
    //         SlotType::Buffer,
    //     )]
    // }

    fn update(&mut self, world: &mut World) {
        trace!("ParticleUpdateNode::update()");
        self.entity_query.update_archetypes(world);
        self.effect_query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        trace!("ParticleUpdateNode::run()");

        // Get the Entity containing the ViewEffectsEntity component used as container
        // for the input data for this node.
        //let view_entity = graph.get_input_entity(Self::IN_VIEW)?;

        // Begin encoder for init pass
        // trace!(
        //     "begin compute init pass... (world={:?} ents={:?} comps={:?})",
        //     world,
        //     world.entities(),
        //     world.components()
        // );

        // Compute prepare pass
        trace!("begin compute prepare pass...");
        {
            let mut compute_pass =
                render_context
                    .command_encoder
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi_prepare"),
                    });

            let effects_meta = world.get_resource::<EffectsMeta>().unwrap();
            let effect_bind_groups = world.get_resource::<EffectBindGroups>().unwrap();

            trace!("loop over effect batches...");
            for batch in self.effect_query.iter_manual(world) {
                if batch.slices.is_empty() {
                    continue;
                }

                let bind_groups = match effect_bind_groups.group_from_index.get(&batch.buffer_index)
                {
                    Some(bind_groups) => bind_groups,
                    None => continue,
                };

                //let item_size = batch.item_size;

                compute_pass.set_pipeline(&batch.prepare_pipeline);

                for slice in &batch.slices {
                    let buffer_offset = slice.slice.start;

                    trace!(
                        "record prepare commands for pipeline of effect {:?} slice={:?} buffer_offset={}...",
                        batch.handle,
                        slice.slice,
                        buffer_offset,
                    );

                    // Dispatch prepare compute pass
                    let dead_list_buffer_offset =
                        batch.buffer_index * std::mem::size_of::<u32>() as u32;
                    compute_pass.set_bind_group(
                        1,
                        &bind_groups.common_dead_list,
                        &[dead_list_buffer_offset],
                    );
                    compute_pass.dispatch(1, 1, 1);

                    trace!("prepare compute dispatched");
                }
            }
        }
        trace!("compute prepare pass done");

        // Compute init pass
        trace!("begin compute init pass...");
        {
            let mut compute_pass =
                render_context
                    .command_encoder
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi_init"),
                    });

            let effects_meta = world.get_resource::<EffectsMeta>().unwrap();
            let effect_bind_groups = world.get_resource::<EffectBindGroups>().unwrap();

            // Retrieve the ExtractedEffectEntities component itself
            //if let Ok(extracted_effect_entities) = self.entity_query.get_manual(world, view_entity)
            //if let Ok(effect_batches) = self.effect_query.get_manual(world, )
            {
                // Loop on all entities recorded inside the ExtractedEffectEntities input
                trace!("loop over effect batches...");
                //for effect_entity in extracted_effect_entities.entities.iter().copied() {

                for batch in self.effect_query.iter_manual(world) {
                    let compute_pipeline = match &batch.init_pipeline {
                        Some(compute_pipeline) => compute_pipeline,
                        None => continue,
                    };

                    //for (effect_entity, effect_slice) in effects_meta.entity_map.iter() {
                    // Retrieve the ExtractedEffect from the entity
                    //trace!("effect_entity={:?} effect_slice={:?}", effect_entity, effect_slice);
                    //let effect = self.effect_query.get_manual(world, *effect_entity).unwrap();

                    // Get the slice to update
                    //let effect_slice = effects_meta.get(&effect_entity);
                    // let effect_group =
                    //     &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];

                    let bind_groups =
                        match effect_bind_groups.group_from_index.get(&batch.buffer_index) {
                            Some(bind_groups) => bind_groups,
                            None => continue,
                        };

                    //let item_size = batch.item_size;

                    let mut spawner_index = batch.spawner_base;
                    for slice in &batch.slices {
                        let spawn_count = slice.spawn_count;
                        if spawn_count > 0 {
                            let workgroup_count = (spawn_count + 63) / 64;

                            let buffer_offset = slice.slice.start;
                            let slice_index = slice.slice_index;

                            trace!(
                                "record init commands for pipeline of effect {:?} ({} spawn / 64 = {} workgroups) spawner_index={} slice={:?} slice_index={} buffer_offset={}...",
                                batch.handle,
                                spawn_count,
                                workgroup_count,
                                spawner_index,
                                slice.slice,
                                slice_index,
                                buffer_offset,
                            );

                            // Setup init compute pass
                            compute_pass.set_pipeline(&compute_pipeline);
                            compute_pass.set_bind_group(
                                0,
                                &bind_groups.common_particle_buffer,
                                &[],
                            );
                            let dead_list_offset =
                                spawner_index * std::mem::size_of::<u32>() as u32;
                            compute_pass.set_bind_group(
                                1,
                                &bind_groups.common_dead_list,
                                &[dead_list_offset],
                            );
                            let spawner_params_offset =
                                spawner_index * SpawnerParamsUniform::std140_size_static() as u32;
                            compute_pass.set_bind_group(
                                2,
                                effects_meta.spawner_bind_group.as_ref().unwrap(),
                                &[spawner_params_offset],
                            );
                            let dispatch_buffer_offset =
                                batch.buffer_index * DispatchBuffer::std430_size_static() as u32;
                            compute_pass.set_bind_group(
                                3,
                                effects_meta.dispach_buffer_bind_group.as_ref().unwrap(),
                                &[dispatch_buffer_offset],
                            );
                            let slice_offset = slice_index * GpuSlice::std430_size_static() as u32;
                            compute_pass.set_bind_group(
                                4,
                                &bind_groups.common_slice_list,
                                &[slice_offset],
                            );
                            compute_pass.dispatch(workgroup_count, 1, 1);

                            trace!("init compute dispatched");
                        } else {
                            trace!(
                                "skipped init dispatch for effect {:?} with no spawn this frame.",
                                batch.handle
                            );
                        }

                        spawner_index += 1;
                    }
                }
            }
        }
        trace!("compute init pass done");

        // Copy final dead particle count to dispatch buffer
        for _batch in self.effect_query.iter_manual(world) {
            // TODO
            //render_context.command_encoder.copy_buffer_to_buffer(batch.dead_list_buffer, batch.slice.start, batch.dispatch_buffer, batch.slice.start, 4);
        }

        // Begin encoder for update pass
        // trace!(
        //     "begin compute update pass... (world={:?} ents={:?} comps={:?})",
        //     world,
        //     world.entities(),
        //     world.components()
        // );

        // Compute update pass
        trace!("begin compute update pass...");
        {
            let mut compute_pass =
                render_context
                    .command_encoder
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hanabi_update"),
                    });

            let effects_meta = world.get_resource::<EffectsMeta>().unwrap();
            let effect_bind_groups = world.get_resource::<EffectBindGroups>().unwrap();

            // Retrieve the ExtractedEffectEntities component itself
            //if let Ok(extracted_effect_entities) = self.entity_query.get_manual(world, view_entity)
            //if let Ok(effect_batches) = self.effect_query.get_manual(world, )
            {
                // Loop on all entities recorded inside the ExtractedEffectEntities input
                trace!("loop over effect batches...");
                //for effect_entity in extracted_effect_entities.entities.iter().copied() {

                for batch in self.effect_query.iter_manual(world) {
                    if let Some(compute_pipeline) = &batch.update_pipeline {
                        //for (effect_entity, effect_slice) in effects_meta.entity_map.iter() {
                        // Retrieve the ExtractedEffect from the entity
                        //trace!("effect_entity={:?} effect_slice={:?}", effect_entity, effect_slice);
                        //let effect = self.effect_query.get_manual(world, *effect_entity).unwrap();

                        // Get the slice to update
                        //let effect_slice = effects_meta.get(&effect_entity);
                        // let effect_group =
                        //     &effects_meta.effect_cache.buffers()[batch.buffer_index as usize];

                        let bind_groups =
                            match effect_bind_groups.group_from_index.get(&batch.buffer_index) {
                                Some(bind_groups) => bind_groups,
                                None => {
                                    continue;
                                }
                            };

                        let item_size = batch.item_size;
                        let spawner_base = batch.spawner_base;

                        for slice in &batch.slices {
                            let item_count = slice.slice.end - slice.slice.start;
                            let workgroup_count = (item_count + 63) / 64;

                            let buffer_offset = slice.slice.start * item_size;

                            trace!(
                                "record update commands for pipeline of effect {:?} ({} items / {}B/item = {} workgroups) spawner_base={} buffer_offset={}...",
                                batch.handle,
                                item_count,
                                item_size,
                                workgroup_count,
                                spawner_base,
                                buffer_offset,
                            );

                            // Setup update compute pass
                            compute_pass.set_pipeline(&compute_pipeline);
                            compute_pass.set_bind_group(
                                0,
                                &bind_groups.common_particle_buffer,
                                &[],
                            );
                            let dead_list_offset = spawner_base * std::mem::size_of::<u32>() as u32;
                            compute_pass.set_bind_group(
                                1,
                                &bind_groups.common_dead_list,
                                &[dead_list_offset],
                            );
                            compute_pass.set_bind_group(
                                2,
                                &bind_groups.update_draw_indirect,
                                &[0], // TODO
                            );
                            compute_pass.set_bind_group(
                                3,
                                &bind_groups.update_indirect_buffer,
                                &[0], // TODO
                            );
                            compute_pass.set_bind_group(
                                4,
                                effects_meta.sim_params_bind_group.as_ref().unwrap(),
                                &[],
                            );
                            let effect_params_offset =
                                spawner_base * EffectParamsUniform::std140_size_static() as u32;
                            compute_pass.set_bind_group(
                                5,
                                effects_meta.effect_params_bind_group.as_ref().unwrap(),
                                &[effect_params_offset],
                            );
                            let slice_offset =
                                slice.slice_index * GpuSlice::std430_size_static() as u32;
                            compute_pass.set_bind_group(
                                6,
                                &bind_groups.common_slice_list,
                                &[slice_offset],
                            );
                            compute_pass.dispatch(workgroup_count, 1, 1);
                            trace!("update compute dispatched");
                        }
                    }
                }
            }
        }
        trace!("compute update pass done");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec4;

    #[test]
    fn layout_flags() {
        let flags = LayoutFlags::default();
        assert_eq!(flags, LayoutFlags::NONE);
    }

    #[test]
    fn to_shader_code() {
        let mut grad = Gradient::new();
        assert_eq!("", grad.to_shader_code());

        grad.add_key(0.0, Vec4::splat(0.0));
        assert_eq!(
            "// Gradient\nlet t0 = 0.;\nlet c0 = vec4<f32>(0., 0., 0., 0.);\nout.color = c0;\n",
            grad.to_shader_code()
        );

        grad.add_key(1.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(
            r#"// Gradient
let t0 = 0.;
let c0 = vec4<f32>(0., 0., 0., 0.);
let t1 = 1.;
let c1 = vec4<f32>(1., 0., 0., 1.);
let life = particle.age / particle.lifetime;
if (life <= t0) { out.color = c0; }
else if (life <= t1) { out.color = mix(c0, c1, (life - t0) / (t1 - t0)); }
else { out.color = c1; }
"#,
            grad.to_shader_code()
        );
    }
}
