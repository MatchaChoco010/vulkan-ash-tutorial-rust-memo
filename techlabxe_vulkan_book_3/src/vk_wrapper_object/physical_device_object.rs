use std::collections::HashSet;

use anyhow::Result;
use ash::{version::InstanceV1_0, vk};

use crate::vk_wrapper_object::{
    constants::DEVICE_EXTENSIONS, utils::vk_to_string, InstanceObject, SurfaceObject,
};

pub struct PhysicalDeviceObject {
    physical_device: vk::PhysicalDevice,
    graphics_queue_index: u32,
    present_queue_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}
impl PhysicalDeviceObject {
    /// PhysicalDeviceを選択する。
    /// Graphics用Queueのindexとpresent用Queueのindexも返す。
    /// vk::PhysicalDeviceMemoryPropertiesも返す
    pub fn new(instance: &InstanceObject, surface: &SurfaceObject) -> Result<Self> {
        let instance = instance.instance_as_ref();

        // Physical Deviceの中で条件を満たすものを抽出する
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let mut physical_devices = physical_devices.iter().filter_map(|&physical_device| {
            // 必要なキューファミリーに対応しているかどうか。
            // graphicsとpresentに対応しているか確認。
            // 両者が同じインデックスの場合もある。
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let mut graphics_queue_index = None;
            let mut present_queue_index = None;
            for (i, queue_family) in queue_families.iter().enumerate() {
                if queue_family.queue_count > 0
                    && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    graphics_queue_index = Some(i as u32);
                }

                let is_present_support = unsafe {
                    surface
                        .surface_loader()
                        .get_physical_device_surface_support(
                            physical_device,
                            i as u32,
                            surface.surface(),
                        )
                        .unwrap()
                };
                if queue_family.queue_count > 0 && is_present_support {
                    present_queue_index = Some(i as u32);
                }

                if graphics_queue_index.is_some() && present_queue_index.is_some() {
                    break;
                }
            }
            let is_queue_families_supported =
                graphics_queue_index.is_some() && present_queue_index.is_some();

            // DEVICE_EXTENSIONSで指定した拡張が対応しているかを確認
            let is_device_extension_supported = {
                let available_extensions = unsafe {
                    instance
                        .enumerate_device_extension_properties(physical_device)
                        .unwrap()
                };
                let mut available_extension_names = vec![];
                for extension in available_extensions.iter() {
                    let extension_name = vk_to_string(&extension.extension_name);
                    available_extension_names.push(extension_name);
                }
                let mut required_extensions = HashSet::new();
                for extension in DEVICE_EXTENSIONS.iter() {
                    required_extensions.insert(extension.to_string());
                }
                for extension in available_extension_names.iter() {
                    required_extensions.remove(extension);
                }
                required_extensions.is_empty()
            };

            // Swapchainをサポートしているかを確認
            let is_swapchain_supported = if is_device_extension_supported {
                let formats = unsafe {
                    surface
                        .surface_loader()
                        .get_physical_device_surface_formats(physical_device, surface.surface())
                        .unwrap()
                };
                let present_modes = unsafe {
                    surface
                        .surface_loader()
                        .get_physical_device_surface_present_modes(
                            physical_device,
                            surface.surface(),
                        )
                        .unwrap()
                };
                !formats.is_empty() && !present_modes.is_empty()
            } else {
                false
            };

            // AnisotropyなSamplerに対応しているかを確認
            let supported_features =
                unsafe { instance.get_physical_device_features(physical_device) };
            let is_supported_sampler_anisotropy = supported_features.sampler_anisotropy != 0;

            if is_queue_families_supported
                && is_device_extension_supported
                && is_swapchain_supported
                && is_supported_sampler_anisotropy
            {
                Some((
                    physical_device,
                    graphics_queue_index.unwrap(),
                    present_queue_index.unwrap(),
                ))
            } else {
                None
            }
        });

        // 条件を満たすうち最初を選択する
        let (physical_device, graphics_queue_index, present_queue_index) = physical_devices
            .next()
            .expect("There is no physical device that meets the requirements");

        // Memory Propertiesを取得しておく
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok(Self {
            physical_device,
            graphics_queue_index,
            present_queue_index,
            memory_properties,
        })
    }

    /// physical_deviceを取得する。
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// graphics_queue_indexを取得する。
    pub fn graphics_queue_index(&self) -> u32 {
        self.graphics_queue_index
    }

    /// present_queue_indexを取得する。
    pub fn present_queue_index(&self) -> u32 {
        self.present_queue_index
    }

    /// physical_device_memory_propertiesを取得する。
    pub fn memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        self.memory_properties
    }
}
