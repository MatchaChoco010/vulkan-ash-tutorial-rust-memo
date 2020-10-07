use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk};

use crate::vk_wrapper_object::{CommandPoolObject, DeviceObject, VkMemAllocatorObject};

pub struct DeviceLocalBufferObject {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocator: Rc<vk_mem::Allocator>,
}
impl DeviceLocalBufferObject {
    /// ステージングバッファを使ってDeviceLocalなvk::Bufferを作成しアロケートする。
    pub fn new<T>(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        graphics_command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let device = device.device();
        let allocator = allocator.allocator();

        // サイズの計算
        let buffer_size = data.len() as u64 * std::mem::size_of::<T>() as u64;

        // 一時バッファの確保
        let (tmp_buffer, tmp_buffer_allocation, _info) = allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuToGpu,
                ..Default::default()
            },
        )?;
        // 一時バッファに転写
        unsafe {
            let mapped_memory = allocator.map_memory(&tmp_buffer_allocation)? as *mut T;
            mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());
            allocator.unmap_memory(&tmp_buffer_allocation)?;
        }
        // バッファの作成
        let (buffer, buffer_allocation, _info) = allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST | usage),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;
        // 一時バッファからバッファへのコピーコマンドの発行
        unsafe {
            let copy_cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(graphics_command_pool.command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                copy_cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_copy_buffer(
                copy_cmd,
                tmp_buffer,
                buffer,
                &[vk::BufferCopy::builder().size(buffer_size).build()],
            );
            device.end_command_buffer(copy_cmd)?;

            // キューにサブミットし待機
            device.queue_submit(
                graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[copy_cmd])
                    .build()],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(graphics_queue)?;
        }

        // 一時バッファの削除
        allocator.destroy_buffer(tmp_buffer, &tmp_buffer_allocation)?;

        Ok(Self {
            buffer,
            allocation: buffer_allocation,
            allocator,
        })
    }

    /// vk::Bufferを取得する。
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
}
impl Drop for DeviceLocalBufferObject {
    fn drop(&mut self) {
        self.allocator
            .destroy_buffer(self.buffer, &self.allocation)
            .expect("Failed to destroy buffer.");
    }
}
