use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

use crate::vk_wrapper_object::DeviceObject;

pub struct RenderPassObject {
    render_pass: vk::RenderPass,
    device: Rc<Device>,
}
impl RenderPassObject {
    /// RenderPassを作成する。
    pub fn new(
        device: &DeviceObject,
        render_pass_create_info: &vk::RenderPassCreateInfo,
    ) -> Result<Self> {
        let device = device.device();
        let render_pass = unsafe { device.create_render_pass(render_pass_create_info, None)? };
        Ok(Self {
            render_pass,
            device,
        })
    }

    /// RenderPassの取得。
    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }
}
impl Drop for RenderPassObject {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
