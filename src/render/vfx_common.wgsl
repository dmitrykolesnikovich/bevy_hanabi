//#define_import_path bevy_hanabi::common

// Global effect-independent simulation parameters.
struct SimParams {
    // Delta time since last frame, in seconds.
    dt: f32;
    // Current frame time, in seconds.
    time: f32;
};

// Per-effect parameters.
struct EffectParams {
    // Global acceleration (gravity-like) to apply to all particles this frame.
    accel: vec3<f32>;
    // Offset to add to the particle index to access it in its GPU particle buffer.
    // Same as Spawner::particle_base, but for the update pass.
    particle_base: u32;
};

struct Spawner {
    // Spawner origin in world space.
    origin: vec4<f32>;
    // Number of particles to spawn this frame, if possible.
    // This will be capped to the number of dead particles available for recycling.
    spawn_count: u32;
    // Spawner PRNG seed.
    seed: u32;
    // Offset to add to the particle index to access it in its GPU particle buffer.
    // Same as Spawner::particle_base, but for the init pass.
    particle_base: u32;
    // Index of the slice into the SliceList.
    slice_index: u32;
    //
    __pad2: vec4<f32>;
    __pad3: vec4<f32>;
};

// Indirect dispatch buffer.
struct DispatchBuffer {
    x: atomic<u32>;
    y: u32;
    z: u32;
};

// List of indices of dead particles.
struct DeadList {
    // Indices of dead particles into the particle buffer. Only values at an
    // index < 'count' in this array are valid; other values are garbage.
    indices: [[stride(4)]] array<u32>;
};

struct Slice {
    // Base index into the ParticleBuffer and the DeadList.
    base_index: u32,
    // Number of consecutive items in the slice.
    count: u32,
    // Number of consecutive "available" dead particle indices in the DeadList indices array. This is the
    // number of particles actually dead that can be recycled this frame.
    dead_count: atomic<u32>;
    // Number of dead particles when the dispatch is submitted. Copied from `dead_count` before the update
    // pass. Used to limit the number of threads accessing the dead list buffer in parallel, so that they
    // don't drop its count below zero.
    max_spawn_count: u32;
}

// struct SliceList {
//     slices: [[stride(16)]] array<Slice>;
// }

// Single indirect draw call (via draw_indirect / multi_draw_indirect / multi_draw_indirect_count).
struct DrawIndirect {
    // The number of vertices to draw.
    vertex_count: u32;
    // The number of instances to draw.
    instance_count: atomic<u32>;
    // The Index of the first vertex to draw.
    base_vertex: u32;
    // The instance ID of the first instance to draw.
    base_instance: u32;
};

// Single indexed indirect draw call (via draw_indexed_indirect / multi_draw_indexed_indirect / multi_draw_indexed_indirect_count).
struct DrawIndexedIndirect {
    // The number of vertices to draw.
    vertex_count: u32;
    // The number of instances to draw.
    instance_count: atomic<u32>;
    // The base index within the index buffer.
    base_index: u32;
    // The value added to the vertex index before indexing into the vertex buffer.
    vertex_offset: i32;
    // The instance ID of the first instance to draw.
    base_instance: u32;
};

// Indirection buffer for particle sorting.
struct IndirectBuffer {
    // Indices into the particle buffer.
    indices: [[stride(4)]] array<u32>;
};

// Current PRNG seed.
var<private> seed : u32 = 0u;

// Rand: PCG
// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
fn pcg_hash(input: u32) -> u32 {
    var state: u32 = input * 747796405u + 2891336453u;
    var word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Convert a packed u32 to a f32 in [0:1].
fn to_float01(u: u32) -> f32 {
    // Note: could generate only 24 bits of randomness
    return bitcast<f32>((u & 0x007fffffu) | 0x3f800000u) - 1.;
}

// Random floating-point number in [0:1].
fn rand() -> f32 {
    seed = pcg_hash(seed);
    return to_float01(pcg_hash(seed));
}

// Random floating-point number in [0:1]^2.
fn rand2() -> vec2<f32> {
    seed = pcg_hash(seed);
    var x = to_float01(seed);
    seed = pcg_hash(seed);
    var y = to_float01(seed);
    return vec2<f32>(x, y);
}

// Random floating-point number in [0:1]^3.
fn rand3() -> vec3<f32> {
    seed = pcg_hash(seed);
    var x = to_float01(seed);
    seed = pcg_hash(seed);
    var y = to_float01(seed);
    seed = pcg_hash(seed);
    var z = to_float01(seed);
    return vec3<f32>(x, y, z);
}

// Random floating-point number in [0:1]^4.
fn rand4(input: u32) -> vec4<f32> {
    // Each rand() produces 32 bits, and we need 24 bits per component,
    // so can get away with only 3 calls.
    var r0 = pcg_hash(seed);
    var r1 = pcg_hash(r0);
    var r2 = pcg_hash(r1);
    seed = r2;
    var x = to_float01(r0);
    var r01 = (r0 & 0xff000000u) >> 8u | (r1 & 0x0000ffffu);
    var y = to_float01(r01);
    var r12 = (r1 & 0xffff0000u) >> 8u | (r2 & 0x000000ffu);
    var z = to_float01(r12);
    var r22 = r2 >> 8u;
    var w = to_float01(r22);
    return vec4<f32>(x, y, z, w);
}
