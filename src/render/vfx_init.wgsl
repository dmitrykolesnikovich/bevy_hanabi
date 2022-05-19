//#import bevy_hanabi::common

struct Particle {
    pos: vec3<f32>;
    age: f32;
    vel: vec3<f32>;
    lifetime: f32;
};

struct ParticleBuffer {
    particles: [[stride(32)]] array<Particle>;
};

let tau: f32 = 6.283185307179586476925286766559;

[[group(0), binding(0)]] var<storage, read_write> particle_buffer : ParticleBuffer;
[[group(1), binding(0)]] var<storage, read_write> dead_list : DeadList;
[[group(2), binding(0)]] var<uniform> spawner : Spawner;
[[group(3), binding(0)]] var<storage, read_write> dispatch : DispatchBuffer;
[[group(4), binding(0)]] var<storage, read_write> slice : Slice;

struct PosVel {
    pos: vec3<f32>;
    vel: vec3<f32>;
};

fn init_pos_vel(index: u32) -> PosVel {
    var ret : PosVel;
{{INIT_POS_VEL}}
    return ret;
}

fn init_lifetime() -> f32 {
    return 5.0;
}

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    // Clamp the current iteration to the number of particles to spawn, and to the number
    // of dead particles that are available for such spawning.
    let spawn_index = global_invocation_id.x;
    let spawn_count = min(spawner.spawn_count, dead_list.max_spawn_count);
    if (spawn_index >= spawn_count) {
        return;
    }

    // Recycle a dead particle
    let dead_index = atomicSub(&dead_list.count, 1u) - 1u;
    let local_index = dead_list.indices[dead_index];

    // Update PRNG seed for this particle
    seed = pcg_hash(local_index ^ spawner.seed);

    // Initialize new particle
    var posVel = init_pos_vel(local_index);
    var vPos = posVel.pos + spawner.origin.xyz;
    var vVel = posVel.vel;
    var vAge = 0.0;
    var vLifetime = init_lifetime();

    // Actual index in global buffer
    let global_index = spawner.particle_base + local_index;

    // Write the initialized particle
    particle_buffer.particles[global_index].pos = vPos;
    particle_buffer.particles[global_index].vel = vVel;
    particle_buffer.particles[global_index].age = vAge;
    particle_buffer.particles[global_index].lifetime = vLifetime;

    // Increment dispatch for update
    //
    // FIXME - This is currently unused because this increment the workgroup count as if it was the
    // thread count. Normally we should divide by 64 before the indirect dispatch, but this requires
    // an extra compute pass just for this.
    //
    // IDEA : Layout DispatchBuffer in such a way that we prepend another u32 before 'x', and increment
    // the combined u64 by some clever size (1 << (32 - 6), since 64 == 2^6) such that every 64 such
    // increments 1 bit overflows from the previous dummy u32 to 'x'.
    //
    // DispatchBuffer {
    //     u32 fract;
    //     u32 x;
    //     [...]
    // }
    // DispatchBuffer {
    //     u64 fract_and_x;
    //     [...]
    // }
    // fract_and_x += 0x04000000u; // effectively adds 1/64 to x, kind of
    //
    // Note that to account for the rounding we need to start at fract_and_x == 0x04000000u * 63, such
    // that even for a single increment (single particle) we dispatch at least 1 workgroup. This inital
    // offset has the same effect as rounding up, i.e. (N + 63) / 64.
    //
    // Note that all of this is slightly tricky because WGSL doesn't support u64, so not sure we can
    // atomically do that overflow operation...
    atomicAdd(&dispatch.x, 1u);
}