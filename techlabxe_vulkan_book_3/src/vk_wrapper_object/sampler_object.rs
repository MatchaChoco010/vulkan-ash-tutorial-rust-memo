use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

pub struct SamplerObject {
    sampler: vk::Sampler,
    device: Rc<Device>,
}
impl SamplerObject {
    /// SamplerObjectを作成する。
    pub fn new(device: Rc<Device>, sampler_create_info: &vk::SamplerCreateInfo) -> Result<Self> {
        let sampler = unsafe { device.create_sampler(sampler_create_info, None)? };
        Ok(Self {
            sampler,
            device: Rc::clone(&device),
        })
    }

    /// samplerを取得する。
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }
}
impl Drop for SamplerObject {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}
