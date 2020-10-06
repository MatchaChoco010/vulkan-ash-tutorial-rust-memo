use std::rc::Rc;

use anyhow::Result;

use crate::vk_wrapper_object::{DeviceObject, InstanceObject, PhysicalDeviceObject};

pub struct VkMemAllocatorObject {
    allocator: Rc<vk_mem::Allocator>,
}
impl VkMemAllocatorObject {
    /// vk-mem-rsのAllocatorを作成する。
    pub fn new(
        instance: &InstanceObject,
        physical_device: &PhysicalDeviceObject,
        device: &DeviceObject,
    ) -> Result<Self> {
        let allocator_create_info = vk_mem::AllocatorCreateInfo {
            physical_device: physical_device.physical_device(),
            device: device.device_as_ref().clone(),
            instance: instance.instance_as_ref().clone(),
            flags: vk_mem::AllocatorCreateFlags::empty(),
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };
        let allocator = Rc::new(vk_mem::Allocator::new(&allocator_create_info)?);
        Ok(Self { allocator })
    }

    /// allocatorを取得する。
    pub fn allocator(&self) -> Rc<vk_mem::Allocator> {
        Rc::clone(&self.allocator)
    }

    /// allocatorを借用する。
    pub fn allocator_as_ref(&self) -> &vk_mem::Allocator {
        self.allocator.as_ref()
    }
}
impl Drop for VkMemAllocatorObject {
    fn drop(&mut self) {
        match Rc::get_mut(&mut self.allocator) {
            Some(alloc) => alloc.destroy(),
            None => panic!("Failed to destroy vk-mem allocator."),
        }
    }
}
