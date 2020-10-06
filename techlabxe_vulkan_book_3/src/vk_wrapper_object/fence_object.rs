use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

use crate::vk_wrapper_object::DeviceObject;

pub struct FenceObject {
    fence: vk::Fence,
    device: Rc<Device>,
}
impl FenceObject {
    /// fenceを作成する。
    pub fn new(device: &DeviceObject) -> Result<Self> {
        let device = device.device();
        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };
        Ok(Self { fence, device })
    }

    /// fenceを待機する。
    pub fn wait_for_fence(&self) -> Result<()> {
        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, std::u64::MAX)?
        };
        Ok(())
    }

    /// fenceをリセットする。
    pub fn reset_fence(&self) -> Result<()> {
        unsafe { self.device.reset_fences(&[self.fence])? };
        Ok(())
    }

    /// fenceを取得する。
    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}
impl Drop for FenceObject {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.fence, None) };
    }
}
