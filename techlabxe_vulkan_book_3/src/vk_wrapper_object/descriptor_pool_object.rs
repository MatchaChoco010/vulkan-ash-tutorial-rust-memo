use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

pub struct DescriptorPoolObject {
    descriptor_pool: vk::DescriptorPool,
    device: Rc<Device>,
}
impl DescriptorPoolObject {
    /// DescriptorPoolObjectを生成する。
    pub fn new(
        device: Rc<Device>,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
    ) -> Result<Self> {
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(max_sets)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        Ok(Self {
            descriptor_pool: pool,
            device,
        })
    }

    /// descriptor_poolを取得する。
    pub fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }
}
impl Drop for DescriptorPoolObject {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
