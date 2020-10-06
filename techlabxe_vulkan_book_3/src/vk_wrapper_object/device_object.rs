use std::{collections::HashSet, rc::Rc};

use anyhow::Result;
use ash::{
    extensions::khr::Swapchain,
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device,
};

use crate::vk_wrapper_object::{InstanceObject, PhysicalDeviceObject};

pub struct DeviceObject {
    device: Rc<Device>,
}
impl DeviceObject {
    /// 論理デバイスの作成。
    pub fn new(instance: &InstanceObject, physical_device: &PhysicalDeviceObject) -> Result<Self> {
        let instance = instance.instance_as_ref();

        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(physical_device.graphics_queue_index());
        unique_queue_families.insert(physical_device.present_queue_index());

        let queue_priorities = [1.0_f32];
        let mut queue_create_infos = vec![];
        for &queue_family in unique_queue_families.iter() {
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities)
                .build();
            queue_create_infos.push(queue_create_info);
        }

        // anisotropyを有効にする。
        let physical_device_features =
            vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

        let enabled_extension_names = [Swapchain::name().as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos.as_slice())
            .enabled_extension_names(&enabled_extension_names)
            .enabled_features(&physical_device_features);

        let device = unsafe {
            instance.create_device(physical_device.physical_device(), &device_create_info, None)?
        };

        Ok(Self {
            device: Rc::new(device),
        })
    }

    /// Queueの取得。
    pub fn get_queue(&self, queue_index: u32) -> vk::Queue {
        unsafe { self.device.get_device_queue(queue_index, 0) }
    }

    /// deviceの取得。
    pub fn device(&self) -> Rc<Device> {
        Rc::clone(&self.device)
    }

    /// deviceの参照の取得。
    pub fn device_as_ref(&self) -> &Device {
        self.device.as_ref()
    }
}
impl Drop for DeviceObject {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
