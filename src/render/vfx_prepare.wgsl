//#import bevy_hanabi::common

[[group(1), binding(0)]] var<storage, read_write> dead_list : DeadList;

[[stage(compute), workgroup_size(64)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    // Only run 1 thread per invocation; the copy is only needed once per buffer
    let local_index = global_invocation_id.x;
    if (local_index > 0) {
        return;
    }

    // Prepare max_spawn_count by copying the value of dead_list.count from the previous
    // frame. This ensures vfx_init.wgsl cannot decrement count below zero.
    dead_list.max_spawn_count = atomicLoad(&dead_list.count);
}