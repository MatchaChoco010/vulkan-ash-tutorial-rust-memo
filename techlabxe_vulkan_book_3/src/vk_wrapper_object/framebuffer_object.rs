use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

use crate::vk_wrapper_object::{DeviceObject, RenderPassObject};

pub struct FramebufferObject {
    framebuffer: vk::Framebuffer,
    device: Rc<Device>,
}
impl FramebufferObject {
    /// framebufferを作成する。
    pub fn new(
        device: &DeviceObject,
        render_pass: &RenderPassObject,
        attachments: &[vk::ImageView],
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let device = device.device();

        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass.render_pass())
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )?
        };

        Ok(Self {
            framebuffer,
            device,
        })
    }

    /// framebufferを取得する。
    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.framebuffer
    }
}
impl Drop for FramebufferObject {
    fn drop(&mut self) {
        unsafe { self.device.destroy_framebuffer(self.framebuffer, None) };
    }
}
