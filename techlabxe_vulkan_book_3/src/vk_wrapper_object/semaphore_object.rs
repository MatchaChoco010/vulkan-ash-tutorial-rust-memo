use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

use crate::vk_wrapper_object::DeviceObject;

pub struct SemaphoreObject {
    semaphore: vk::Semaphore,
    device: Rc<Device>,
}
impl SemaphoreObject {
    /// semaphoreを作成する。
    pub fn new(device: &DeviceObject) -> Result<Self> {
        let device = device.device();
        let semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)? };
        Ok(Self { semaphore, device })
    }

    /// semaphoreを取得する。
    pub fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }
}
impl Drop for SemaphoreObject {
    fn drop(&mut self) {
        unsafe { self.device.destroy_semaphore(self.semaphore, None) };
    }
}
