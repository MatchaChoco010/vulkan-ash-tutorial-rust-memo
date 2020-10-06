use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

use crate::vk_wrapper_object::DeviceObject;

pub struct CommandPoolObject {
    device: Rc<Device>,
    command_pool: vk::CommandPool,
}
impl CommandPoolObject {
    /// CommandPoolを作成する。
    /// Reset Command Buffer指定。
    pub fn new(device: &DeviceObject, queue_index: u32) -> Result<Self> {
        let device = device.device();
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };
        Ok(Self {
            device,
            command_pool,
        })
    }

    /// command_poolを取得する。
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// コマンドバッファを準備する。
    /// Primaryコマンドバッファをcount個用意する。
    pub fn prepare_command_buffers(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        // 描画用コマンドバッファの作成
        Ok(unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(count),
            )?
        })
    }
}
impl Drop for CommandPoolObject {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
