use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

pub struct DescriptorSetLayoutObject {
    descriptor_set_layout: vk::DescriptorSetLayout,
    device: Rc<Device>,
}
impl DescriptorSetLayoutObject {
    /// DescriptorSetLayoutObjectを生成する。
    pub fn new(device: Rc<Device>, bindings: &[vk::DescriptorSetLayoutBinding]) -> Result<Self> {
        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings),
                None,
            )?
        };
        Ok(Self {
            descriptor_set_layout: layout,
            device,
        })
    }

    /// descriptor_set_layoutを取得する。
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}
impl Drop for DescriptorSetLayoutObject {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
