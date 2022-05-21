use bevy::{asset::HandleId, prelude::*};
use std::ops::Range;

use super::{
    EffectBatch, EffectBatchSlice, EffectSlice, ExtractedEffect, ForceFieldParam, LayoutFlags,
    FFNUM,
};
use crate::EffectAsset;

trait SliceEx {
    /// Check if two objects are adjacent to each other, without overlap.
    fn is_adjacent_to(&self, other: &Self) -> bool;
}

impl SliceEx for Range<u32> {
    /// Check if two ranges are adjacent to each other, without overlap.
    ///
    /// Ranges are adjacent if they can be merged into a single range covering
    /// exactly the elements of the two input ranges, and none other.
    #[inline]
    fn is_adjacent_to(&self, other: &Self) -> bool {
        self.start == other.end || other.start == self.end
    }
}

/// Temporary data structure holding the current open batch of effects.
pub struct CurrentBatch {
    ///
    spawner_base: u32,
    /// Slices.
    slices: Vec<EffectBatchSlice>,
    /// Size of each particle, in bytes.
    item_size: u32,
    /// Index of the GPU particle buffer in the effect cache.
    buffer_index: u32,
    /// Handle to the effect asset the batch was extracted from.
    /// TODO - Can we batch different assets together if they're compatible?!
    asset: Handle<EffectAsset>,
    /// Flags for the particle layout, to determine compatibility of vertex attributes.
    layout_flags: LayoutFlags,
    /// Position code for the init compute shader; part of the global "layout".
    /// TODO - Should be merged with `layout_flags`.
    position_code: String,
    force_field_code: String,
    /// Optional handle to an image asset for the particle rendering.
    /// TODO - Should probably be more generic than this, as other rendering methods could
    /// need other kind of textures etc.
    image_handle_id: HandleId,
    /// Handle to the rendering shader for the effect batch.
    /// TODO - Can we batch different assets together if they're compatible?!
    shader: Handle<Shader>,
}

impl CurrentBatch {
    /// Check if an `other` batch is compatible with self. Compatible batches can be merged
    /// together to form a single batch.
    pub fn is_compatible_with(&self, other: &CurrentBatch) -> bool {
        self.item_size == other.item_size
            && self.buffer_index == other.buffer_index
            && self.asset == other.asset
            && self.layout_flags == other.layout_flags
            && self.image_handle_id == other.image_handle_id
            && self.shader == other.shader
            && self.position_code == other.position_code
            && self.force_field_code == other.force_field_code
    }

    /// Merge another batch into the current batch. The other batch must be compatible
    /// with this batch.
    ///
    /// # Panics
    ///
    /// Panics if the `other` batch is not compatible with the `self` batch.
    pub fn merge(&mut self, mut other: CurrentBatch) {
        assert!(self.is_compatible_with(&other));
        self.slices.append(&mut other.slices);
    }

    /// Try to merge a batch with the current batch. On success, consume `other` and return
    /// `None`. On failure, return the input batch `Some(other)`.
    pub fn try_merge(&mut self, other: CurrentBatch) -> Option<CurrentBatch> {
        if self.is_compatible_with(&other) {
            self.merge(other);
            None
        } else {
            Some(other)
        }
    }
}

impl From<CurrentBatch> for EffectBatch {
    fn from(batch: CurrentBatch) -> EffectBatch {
        EffectBatch {
            buffer_index: batch.buffer_index,
            spawner_base: batch.spawner_base,
            slices: batch.slices,
            item_size: batch.item_size,
            handle: batch.asset,
            layout_flags: batch.layout_flags,
            image_handle_id: batch.image_handle_id,
            shader: batch.shader,
            position_code: batch.position_code,
            force_field_code: batch.force_field_code,
            prepare_pipeline: None,
            init_pipeline: None,
            update_pipeline: None,
        }
    }
}

/// Utility to batch several effects together, keeping track of the various
/// properties of each effect and when to split them into a new batch.
pub(crate) struct EffectBatcher {
    /// Collection of batches already closed.
    batches: Vec<EffectBatch>,
    /// Current open batch where new effect can be added if they're compatible.
    current_batch: Option<CurrentBatch>,
    /// Index into the spawner buffer of the next available spawner block.
    free_spawner_base: u32,
}

impl EffectBatcher {
    /// Create a new empty batcher.
    pub fn new(spawner_base: u32) -> Self {
        EffectBatcher {
            batches: vec![],
            current_batch: None,
            free_spawner_base: spawner_base,
        }
    }

    /// Insert a new extracted effect into the collection, potentially splitting
    /// it into a new batch if the new effect is not compatible with the current
    /// batch (or if there's no current batch, for the first one). For best result,
    /// effects should be inserted sorted by compatibility.
    pub fn insert(&mut self, effect: &ExtractedEffect, slice: &EffectSlice) {
        let new_batch = CurrentBatch {
            spawner_base: self.free_spawner_base,
            slices: vec![EffectBatchSlice {
                spawn_count: effect.spawn_count,
                slice: slice.slice.clone(),
                origin: effect.transform.col(3).truncate(),
                accel: effect.accel,
                slice_index: slice.slice_index,
            }],
            item_size: slice.item_size,
            buffer_index: slice.group_index,
            asset: effect.handle.clone_weak(),
            layout_flags: effect.layout_flags(),
            image_handle_id: effect.image_handle_id,
            shader: effect.shader.clone(),
            position_code: effect.position_code.clone(),
            force_field_code: effect.force_field_code.clone(),
        };
        self.free_spawner_base += 1;

        if let Some(mut current_batch) = self.current_batch.take() {
            self.current_batch = if let Some(new_batch) = current_batch.try_merge(new_batch) {
                self.batches.push(current_batch.into());
                Some(new_batch)
            } else {
                Some(current_batch)
            };
        } else {
            self.current_batch = Some(new_batch);
        }
    }

    /// Consume the batches generated by previous calls to `insert()`.
    pub fn into_batches(mut self) -> impl IntoIterator<Item = EffectBatch> {
        // Finalize by inserting the last pending batch, if any
        if let Some(current_batch) = self.current_batch.take() {
            self.batches.push(current_batch.into());
        }
        self.batches.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_ex() {
        let s0: Range<u32> = 4..32;
        assert!(s0.is_adjacent_to(&(0u32..4u32)));
        assert!(!s0.is_adjacent_to(&(0u32..3u32)));
        assert!(s0.is_adjacent_to(&(32u32..40u32)));
        assert!(!s0.is_adjacent_to(&(33u32..40u32)));
        assert!(!s0.is_adjacent_to(&(0u32..5u32)));
        assert!(!s0.is_adjacent_to(&(31u32..40u32)));
    }

    #[test]
    fn batcher_empty() {
        let batcher = EffectBatcher::new(33);
        assert!(batcher.batches.is_empty());
        assert!(batcher.current_batch.is_none());
        assert_eq!(batcher.free_spawner_base, 33);
    }

    #[test]
    fn batcher_insert() {
        let mut batcher = EffectBatcher::new(0);

        let origin = Vec3::new(1., -3.5, 0.8);
        let accel = Vec3::new(0., -9., 2.);

        // Insert a single slice
        let effect = ExtractedEffect {
            handle: Handle::<EffectAsset>::default(),
            spawn_count: 245,
            transform: Mat4::from_translation(origin),
            accel,
            force_field: [ForceFieldParam::default(); FFNUM],
            color: Color::WHITE,
            rect: bevy::math::Rect::default(),
            has_image: false,
            image_handle_id: HandleId::default::<Image>(),
            shader: Handle::<Shader>::default(),
            position_code: "".to_string(),
            force_field_code: "".to_string(),
        };
        let slice = EffectSlice {
            slice: 0..5,
            group_index: 42,
            item_size: 64,
            slice_index: 0,
        };
        batcher.insert(&effect, &slice);
        assert!(batcher.batches.is_empty()); // current_batch not yet finalized
        assert!(batcher.current_batch.is_some());
        assert_eq!(batcher.free_spawner_base, 1);

        // Check the slice is pending in the `current_batch` container
        {
            let current_batch = batcher.current_batch.as_ref().unwrap();
            assert_eq!(current_batch.spawner_base, 0);
            assert_eq!(current_batch.slices.len(), 1);
            assert_eq!(current_batch.item_size, 64);
            assert_eq!(current_batch.buffer_index, 42);
            assert_eq!(current_batch.asset, effect.handle);
            assert_eq!(current_batch.layout_flags, effect.layout_flags());
            assert_eq!(current_batch.position_code, effect.position_code);
            assert_eq!(current_batch.image_handle_id, effect.image_handle_id);
            assert_eq!(current_batch.shader, effect.shader);
            let slice = &current_batch.slices[0];
            assert_eq!(slice.spawn_count, 245);
            assert_eq!(slice.slice, 0..5);
            assert_eq!(slice.origin, origin);
            assert_eq!(slice.accel, accel);
        }

        // Finalize
        let mut iter = batcher.into_batches().into_iter();
        let batch = iter.next();
        assert!(iter.next().is_none());
        assert!(batch.is_some());
        let batch = batch.unwrap();

        // Check the batch
        assert_eq!(batch.buffer_index, 42);
        assert_eq!(batch.spawner_base, 0);
        assert_eq!(batch.slices.len(), 1);
        assert_eq!(batch.item_size, 64);
        assert_eq!(batch.handle, effect.handle);
        assert_eq!(batch.layout_flags, effect.layout_flags());
        assert_eq!(batch.image_handle_id, effect.image_handle_id);
        assert_eq!(batch.shader, effect.shader);
        assert_eq!(batch.position_code, effect.position_code);
        let slice = &batch.slices[0];
        assert_eq!(slice.spawn_count, 245);
        assert_eq!(slice.slice, 0..5);
        assert_eq!(slice.origin, origin);
        assert_eq!(slice.accel, accel);
    }

    #[test]
    fn batcher_merge() {
        let mut batcher = EffectBatcher::new(0);

        let origin = Vec3::new(1., -3.5, 0.8);
        let accel = Vec3::new(0., -9., 2.);
        let effect = ExtractedEffect {
            handle: Handle::<EffectAsset>::default(),
            spawn_count: 245,
            transform: Mat4::from_translation(origin),
            accel,
            force_field: [ForceFieldParam::default(); FFNUM],
            color: Color::WHITE,
            rect: bevy::math::Rect::default(),
            has_image: false,
            image_handle_id: HandleId::default::<Image>(),
            shader: Handle::<Shader>::default(),
            position_code: "".to_string(),
            force_field_code: "".to_string(),
        };

        // Insert 2 compatible and adjacent slices
        let slice1 = EffectSlice {
            slice: 0..5,
            group_index: 42,
            item_size: 64,
            slice_index: 0,
        };
        let slice2 = EffectSlice {
            slice: 5..30,
            group_index: 42,
            item_size: 64,
            slice_index: 0,
        };
        assert!(slice1.slice.is_adjacent_to(&slice2.slice));
        batcher.insert(&effect, &slice1);
        assert_eq!(batcher.free_spawner_base, 1);
        batcher.insert(&effect, &slice2);
        assert_eq!(batcher.free_spawner_base, 2);

        // Check merge
        assert!(batcher.batches.is_empty()); // merged, and current_batch not yet finalized
        assert!(batcher.current_batch.is_some());
        {
            let current_batch = batcher.current_batch.as_ref().unwrap();
            assert_eq!(current_batch.spawner_base, 0);
            assert_eq!(current_batch.slices.len(), 2); // TODO - merge adjacent slices
        }

        // Insert a non-adjacent slice
        let slice3 = EffectSlice {
            slice: 42..50,
            group_index: 42,
            item_size: 64,
            slice_index: 0,
        };
        assert!(!slice2.slice.is_adjacent_to(&slice3.slice));
        batcher.insert(&effect, &slice3);
        assert_eq!(batcher.free_spawner_base, 3);

        // Check merge, because we keep a collection of slices, not only adjacent ones.
        // TODO - we should also merge slices together...
        assert!(batcher.batches.is_empty()); // merged, and current_batch not yet finalized
        assert!(batcher.current_batch.is_some());
        {
            let current_batch = batcher.current_batch.as_ref().unwrap();
            assert_eq!(current_batch.spawner_base, 0);
            assert_eq!(current_batch.slices.len(), 3);
        }

        // Insert a non-compatible slice
        let slice4 = EffectSlice {
            slice: 50..62,
            group_index: 42,
            item_size: 32, // different item size
            slice_index: 0,
        };
        assert!(slice3.slice.is_adjacent_to(&slice4.slice));
        batcher.insert(&effect, &slice4);
        assert_eq!(batcher.free_spawner_base, 4);

        // Check not merged
        assert_eq!(batcher.batches.len(), 1); // first batch was closed
        assert!(batcher.current_batch.is_some());
        {
            let current_batch = batcher.current_batch.as_ref().unwrap();
            assert_eq!(current_batch.spawner_base, 3);
            assert_eq!(current_batch.slices.len(), 1); // only slice4
            let slice = &current_batch.slices[0];
            assert_eq!(slice.slice, 50..62);
        }
    }
}
