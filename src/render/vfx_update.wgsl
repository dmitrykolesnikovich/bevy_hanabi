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
[[group(2), binding(0)]] var<storage, read_write> draw_indirect : DrawIndirect;
[[group(3), binding(0)]] var<storage, read_write> indirect_buffer : IndirectBuffer;
[[group(4), binding(0)]] var<uniform> sim_params : SimParams;
[[group(5), binding(0)]] var<uniform> effect_params : EffectParams;
[[group(6), binding(0)]] var<storage, read_write> slice : Slice;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    // Clamp number of particles to update
    let max_particles : u32 = arrayLength(&particle_buffer.particles);
    let local_index = global_invocation_id.x;
    let global_index = effect_params.particle_base + local_index;
    if (global_index >= max_particles) {
        return;
    }

    // Age the particle
    var vAge : f32 = particle_buffer.particles[global_index].age;
    var vLifetime : f32 = particle_buffer.particles[global_index].lifetime;
    var wasDead : bool = (vAge >= vLifetime);
    vAge = vAge + sim_params.dt;
    if (vAge < vLifetime) {
        // Load current particle attributes
        var vPos : vec3<f32> = particle_buffer.particles[global_index].pos;
        var vVel : vec3<f32> = particle_buffer.particles[global_index].vel;

        // Euler integration
        vVel = vVel + (effect_params.accel * sim_params.dt);
        vPos = vPos + (vVel * sim_params.dt);

        // Write back particle itself
        particle_buffer.particles[global_index].pos = vPos;
        particle_buffer.particles[global_index].vel = vVel;
        particle_buffer.particles[global_index].age = vAge;

        // Write indirection for drawing
        var indirect_index = atomicAdd(&draw_indirect.instance_count, 1u);
        indirect_buffer.indices[indirect_index] = local_index;
    } else if (!wasDead) {
        // Write back age to ensure particle stays dead even if (dt == 0)
        particle_buffer.particles[global_index].age = vLifetime;

        // Save dead particle index
        let dead_index = atomicAdd(&dead_list.count, 1u);
        dead_list.indices[dead_index] = local_index;
    }
}