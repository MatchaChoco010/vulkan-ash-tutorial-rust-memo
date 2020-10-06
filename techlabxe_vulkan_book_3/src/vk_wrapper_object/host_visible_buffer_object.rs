use std::rc::Rc;

use anyhow::Result;
use ash::vk;

use crate::vk_wrapper_object::VkMemAllocatorObject;

pub struct HostVisibleBufferObject {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocator: Rc<vk_mem::Allocator>,
    buffer_size: u64,
}
impl HostVisibleBufferObject {
    /// HostVisibleなvk::Bufferを作成しアロケートする。
    pub fn new<T>(
        allocator: &VkMemAllocatorObject,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let allocator = allocator.allocator();

        // サイズの計算
        let buffer_size = data.len() as u64 * std::mem::size_of::<T>() as u64;

        // バッファの確保
        let (buffer, buffer_allocation, _info) = allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(usage),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuToGpu,
                ..Default::default()
            },
        )?;
        // バッファに転写
        unsafe {
            let mapped_memory = allocator.map_memory(&buffer_allocation)? as *mut T;
            mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());
            allocator.unmap_memory(&buffer_allocation)?;
        }

        Ok(Self {
            buffer,
            allocation: buffer_allocation,
            allocator,
            buffer_size,
        })
    }

    /// vk::Bufferを取得する。
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// データを転送する。
    /// newで作成したデータサイズと異なるサイズのデータを渡すとパニックする。
    pub fn map_data<T>(&self, data: &[T]) -> Result<()> {
        // サイズの計算
        let buffer_size = data.len() as u64 * std::mem::size_of::<T>() as u64;
        // サイズの確認
        assert_eq!(buffer_size, self.buffer_size);

        // バッファに転写
        unsafe {
            let mapped_memory = self.allocator.map_memory(&self.allocation)? as *mut T;
            mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());
            self.allocator.unmap_memory(&self.allocation)?;
        }

        Ok(())
    }
}
impl Drop for HostVisibleBufferObject {
    fn drop(&mut self) {
        self.allocator
            .destroy_buffer(self.buffer, &self.allocation)
            .expect("Failed to destroy buffer");
    }
}
