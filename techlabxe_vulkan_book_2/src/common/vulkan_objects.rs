//! Vulkanのオブジェクト用構造体。
//! ハンドルとメモリをまとめるなど。
#![allow(dead_code)]

use std::{ffi::CString, path::Path, rc::Rc};

use anyhow::Result;
use ash::{
    extensions::khr::{Surface, Swapchain},
    version::DeviceV1_0,
    vk, Device, Instance,
};
use image::{DynamicImage, GenericImageView};

/// デバイスローカルなバッファオブジェクト。
/// vk_memを利用してアロケートする。
/// Dropで自動的にメモリを開放する。
pub struct DeviceLocalBufferObject {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocator: Rc<vk_mem::Allocator>,
}
impl DeviceLocalBufferObject {
    /// ステージングバッファを使ってDeviceLocalなvk::Bufferを作成しアロケートする。
    pub fn new<T>(
        allocator: Rc<vk_mem::Allocator>,
        device: Rc<Device>,
        graphics_command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
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
                    .command_pool(*graphics_command_pool)
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
                *graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[copy_cmd])
                    .build()],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(*graphics_queue)?;
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
            .expect("Failed to destroy buffer");
    }
}

/// ホストビジブルなバッファオブジェクト。
/// map_data()でデータを転送できる。
/// vk_memを利用してアロケートする。
/// Dropで自動的にメモリを開放する。
pub struct HostVisibleBufferObject {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocator: Rc<vk_mem::Allocator>,
    buffer_size: u64,
}
impl HostVisibleBufferObject {
    /// HostVisibleなvk::Bufferを作成しアロケートする。
    pub fn new<T>(
        allocator: Rc<vk_mem::Allocator>,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
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

/// デバイスローカルなイメージのオブジェクト。
/// SHADER_READ_ONLY_OPTIMAL。
/// vk_memを利用してアロケートする。
/// Dropで自動的にメモリを開放する。
pub struct ShaderReadOnlyImageObject {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
    allocator: Rc<vk_mem::Allocator>,
    device: Rc<Device>,
}
impl ShaderReadOnlyImageObject {
    /// 画像を読み込む。
    pub fn load_image(
        allocator: Rc<vk_mem::Allocator>,
        device: Rc<Device>,
        graphics_command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
        image: DynamicImage,
    ) -> Result<Self> {
        let (image_width, image_height) = (image.width(), image.height());
        let image_data = image.to_rgba().into_raw();

        let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;

        // ステージングバッファの作成
        let (staging_buffer, staging_buffer_allocation, _info) = allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(image_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuOnly,
                ..Default::default()
            },
        )?;
        // ステージングバッファへのマップ
        unsafe {
            let mapped_memory = allocator.map_memory(&staging_buffer_allocation)? as *mut u8;
            mapped_memory.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());
            allocator.unmap_memory(&staging_buffer_allocation)?;
        }

        let format = vk::Format::R8G8B8A8_UNORM;

        // Imageの作成
        let (image, image_allocation, _info) = allocator.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;

        // バッファからImageにコピーしてフォーマットを整える
        unsafe {
            let command = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*graphics_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            // コマンドの構築開始
            device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // BarrierでフォーマットをTRANSFER DST OPTIMALにする
            let image_barriers = [vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .build()];
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );

            // BufferからImageへデータをコピーする
            let buffer_image_regions = [vk::BufferImageCopy::builder()
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_extent(vk::Extent3D {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                })
                .buffer_offset(0)
                .buffer_image_height(0)
                .buffer_row_length(0)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .build()];
            device.cmd_copy_buffer_to_image(
                command,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_image_regions,
            );

            // BarrierでフォーマットをSHADER READ ONLY OPTIMALにする
            let image_barriers = [vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .build()];
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );

            // コマンド終了
            device.end_command_buffer(command)?;

            // コマンドのサブミット
            device.queue_submit(
                *graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[command])
                    .build()],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(*graphics_queue)?;
        }

        let image_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(image)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?
        };

        allocator.destroy_buffer(staging_buffer, &staging_buffer_allocation)?;

        Ok(Self {
            image,
            allocation: image_allocation,
            view: image_view,
            allocator,
            device,
        })
    }

    /// vk::Imageを取得する。
    pub fn image(&self) -> &vk::Image {
        &self.image
    }

    /// vk::ImageViewを取得する。
    pub fn image_view(&self) -> &vk::ImageView {
        &self.view
    }
}
impl Drop for ShaderReadOnlyImageObject {
    fn drop(&mut self) {
        self.allocator
            .destroy_image(self.image, &self.allocation)
            .expect("Failed to destroy image");
        unsafe { self.device.destroy_image_view(self.view, None) };
    }
}

/// デプスバッファ用のImageObject。
/// Dropで自動的にメモリを開放する。
pub struct DepthImageObject {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
    allocator: Rc<vk_mem::Allocator>,
    device: Rc<Device>,
}
impl DepthImageObject {
    /// DepthBufferの準備。
    pub fn new(
        allocator: Rc<vk_mem::Allocator>,
        device: Rc<Device>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let depth_image_create_info = vk::ImageCreateInfo::builder()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        let depth_image_allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            flags: vk_mem::AllocationCreateFlags::empty(),
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            memory_type_bits: 0,
            pool: None,
            user_data: None,
        };
        let (depth_buffer_image, depth_buffer_allocation, _depth_image_allocation_info) =
            allocator.create_image(&depth_image_create_info, &depth_image_allocation_info)?;
        let depth_buffer_image_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(depth_buffer_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    ),
                None,
            )?
        };

        Ok(Self {
            image: depth_buffer_image,
            allocation: depth_buffer_allocation,
            view: depth_buffer_image_view,
            allocator,
            device,
        })
    }

    /// vk::Imageを取得する。
    pub fn image(&self) -> &vk::Image {
        &self.image
    }

    /// vk::ImageViewを取得する。
    pub fn image_view(&self) -> &vk::ImageView {
        &self.view
    }
}
impl Drop for DepthImageObject {
    fn drop(&mut self) {
        self.allocator
            .destroy_image(self.image, &self.allocation)
            .expect("Failed to destroy image");
        unsafe { self.device.destroy_image_view(self.view, None) };
    }
}

/// Swapchain周りをまとめたオブジェクト。
/// リサイズ時はresizeメソッドを呼び出すとスワップチェインの再生成を行う。
/// Dropによって自動的にリソースは破棄される。
pub struct SwapchainObject {
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    format: vk::Format,
    physical_device: vk::PhysicalDevice,
    device: Rc<Device>,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    graphics_queue_index: u32,
    present_queue_index: u32,
}
impl SwapchainObject {
    /// スワップチェーンを作成する。
    pub fn new(
        instance: &Instance,
        device: Rc<Device>,
        physical_device: vk::PhysicalDevice,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        graphics_queue_index: u32,
        present_queue_index: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let swapchain_loader = Swapchain::new(instance, device.as_ref());

        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
        };

        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM || f.format == vk::Format::R8G8B8A8_UNORM
            })
            .unwrap_or(&formats[0])
            .clone();
        let present_mode = present_modes
            .into_iter()
            .find(|&p| p == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let extent = {
            if capabilities.current_extent.width != u32::max_value() {
                capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: width
                        .max(capabilities.min_image_extent.width)
                        .min(capabilities.max_image_extent.width),
                    height: height
                        .max(capabilities.min_image_extent.height)
                        .min(capabilities.max_image_extent.height),
                }
            }
        };

        let image_count = capabilities.min_image_count + 1;
        let image_count = if capabilities.max_image_count != 0 {
            image_count.min(capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_indices) =
            if graphics_queue_index != present_queue_index {
                (
                    vk::SharingMode::CONCURRENT,
                    vec![graphics_queue_index, present_queue_index],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_family_indices.as_slice())
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|&image| {
                unsafe {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )
                }
                .unwrap()
            })
            .collect();

        Ok(Self {
            swapchain_loader,
            swapchain,
            images: swapchain_images,
            image_views: swapchain_image_views,
            extent,
            format: format.format,
            physical_device,
            device,
            surface_loader: surface_loader.clone(),
            surface,
            graphics_queue_index,
            present_queue_index,
        })
    }

    /// スワップチェーンを再生成する。
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        // 待機
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }

        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)
                .unwrap()
        };
        let present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
                .unwrap()
        };

        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM || f.format == vk::Format::R8G8B8A8_UNORM
            })
            .unwrap_or(&formats[0])
            .clone();
        let present_mode = present_modes
            .into_iter()
            .find(|&p| p == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let extent = {
            if capabilities.current_extent.width != u32::max_value() {
                capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: width
                        .max(capabilities.min_image_extent.width)
                        .min(capabilities.max_image_extent.width),
                    height: height
                        .max(capabilities.min_image_extent.height)
                        .min(capabilities.max_image_extent.height),
                }
            }
        };

        let image_count = capabilities.min_image_count + 1;
        let image_count = if capabilities.max_image_count != 0 {
            image_count.min(capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_indices) =
            if self.graphics_queue_index != self.present_queue_index {
                (
                    vk::SharingMode::CONCURRENT,
                    vec![self.graphics_queue_index, self.present_queue_index],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_family_indices.as_slice())
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(self.swapchain);

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&swapchain_create_info, None)?
        };

        let swapchain_images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain)? };

        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|&image| {
                unsafe {
                    self.device.create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )
                }
                .unwrap()
            })
            .collect();

        let old_swapchain = self.swapchain;
        let mut old_image_views = vec![];
        std::mem::swap(&mut self.image_views, &mut old_image_views);
        self.swapchain = swapchain;
        self.images = swapchain_images;
        self.image_views = swapchain_image_views;
        self.format = format.format;
        self.extent = extent;

        // 以前のスワップチェーンの破棄
        unsafe {
            for &image_view in old_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        Ok(())
    }

    /// スワップチェーンの枚数を返す。
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// acquire_next_imageを呼び出す。
    pub fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<usize> {
        unsafe {
            let (image_index, _is_suboptimal) = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                semaphore,
                vk::Fence::null(),
            )?;

            let image_index = image_index as usize;
            Ok(image_index)
        }
    }

    /// Imageを取得する。
    pub fn get_image(&self, index: usize) -> &vk::Image {
        &self.images[index]
    }

    /// ImageViewを取得する。
    pub fn get_image_view(&self, index: usize) -> &vk::ImageView {
        &self.image_views[index]
    }

    /// queue_presentを呼び出す。
    pub fn queue_present(
        &self,
        present_queue: vk::Queue,
        index: usize,
        wait_semaphore: vk::Semaphore,
    ) -> Result<()> {
        unsafe {
            self.swapchain_loader.queue_present(
                present_queue,
                &vk::PresentInfoKHR::builder()
                    .swapchains(&[self.swapchain])
                    .image_indices(&[index as u32])
                    .wait_semaphores(&[wait_semaphore]),
            )?;
        }

        Ok(())
    }

    /// formatを取得する
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// extentを取得する
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// アスペクト比を取得する。
    pub fn aspect(&self) -> f32 {
        self.extent.width as f32 / self.extent.height as f32
    }
}
impl Drop for SwapchainObject {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");

            for &image_view in self.image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

/// RenderPassのRAIIオブジェクト。
pub struct RenderPassObject {
    render_pass: vk::RenderPass,
    device: Rc<Device>,
}
impl RenderPassObject {
    /// RenderPassObjectの作成。
    /// カラーとDepthが一枚ずつでsubpassが一つ。
    pub fn new(device: Rc<Device>, format: vk::Format) -> Result<Self> {
        // Attachments
        let attachments = [
            vk::AttachmentDescription::builder()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .build(),
            vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        ];
        // color reference
        let color_reference = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];
        // depth reference
        let depth_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        // subpass descriptionを作成
        let subpasses = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_reference)
            .depth_stencil_attachment(&depth_reference)
            .build()];
        // render passの作成
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses);
        let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None)? };

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

/// FramebufferのRAIIラッパーオブジェクト。
pub struct FramebufferObject {
    framebuffer: vk::Framebuffer,
    device: Rc<Device>,
}
impl FramebufferObject {
    /// FramebufferObjectの生成。
    /// 与えられたImageViewとDepthとRenderPassのフレームバッファを作成する。
    pub fn new(
        device: Rc<Device>,
        image_view: vk::ImageView,
        depth_buffer: &DepthImageObject,
        render_pass: &RenderPassObject,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let attachments = [image_view, *depth_buffer.image_view()];

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
        unsafe {
            self.device.destroy_framebuffer(self.framebuffer, None);
        }
    }
}

/// DescriptorPoolのRAIIラッパーオブジェクト。
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

/// DescriptorSetLayoutのRAIIラッパーオブジェクト。
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

/// PipelineのRAIIラッパーオブジェクト。
pub struct PipelineObject {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    device: Rc<Device>,
}
impl PipelineObject {
    /// GraphicsPipelineObjectを生成する。
    /// 不透明の描画用パイプライン。
    pub fn new_opaque(
        device: Rc<Device>,
        vertex_input_bindings: &[vk::VertexInputBindingDescription],
        vertex_input_attributes: &[vk::VertexInputAttributeDescription],
        vertex_shader_pass: &Path,
        fragment_shader_pass: &Path,
        width: u32,
        height: u32,
        render_pass: &RenderPassObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
    ) -> Result<Self> {
        // 頂点入力設定
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attributes)
            .vertex_binding_descriptions(&vertex_input_bindings);

        // ビューポートの設定
        let viewports = [vk::Viewport::builder()
            .x(0.0)
            .y(height as f32)
            .width(width as f32)
            .height(-1.0 * height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];
        let scissors = [vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D { width, height })
            .build()];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        // プリミティブトポロジー設定
        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // ラスタライザーステート設定
        let rasterization_state_createa_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        // パイプラインレイアウト
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layout.descriptor_set_layout()]),
                None,
            )?
        };

        // マルチサンプル設定
        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // ブレンディングの設定
        let color_blend_attachments_state = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments_state);

        // デプスステンシルステート設定
        let stencil_op = vk::StencilOpState::builder()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS)
            .build();
        let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(stencil_op)
            .back(stencil_op);

        // シェーダバイナリの読み込み
        let vertex_shader_module = {
            use std::fs::File;
            use std::io::Read;

            let spv_file = File::open(vertex_shader_pass)?;
            let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr() as *const u32,
                ..Default::default()
            };
            unsafe { device.create_shader_module(&shader_module_create_info, None)? }
        };
        let fragment_shader_module = {
            use std::fs::File;
            use std::io::Read;

            let spv_file = File::open(fragment_shader_pass)?;
            let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr() as *const u32,
                ..Default::default()
            };
            unsafe { device.create_shader_module(&shader_module_create_info, None)? }
        };
        // main関数の名前
        let main_function_name = CString::new("main").unwrap();
        // shader stage create info
        let pipeline_shader_stage_create_info = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader_module)
                .name(&main_function_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader_module)
                .name(&main_function_name)
                .build(),
        ];

        // パイプラインの構築
        let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&pipeline_shader_stage_create_info)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&input_assembly_state_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_state_createa_info)
            .multisample_state(&multisample_state_create_info)
            .depth_stencil_state(&depth_stencil_state_create_info)
            .color_blend_state(&color_blend_state_create_info)
            .layout(pipeline_layout)
            .render_pass(render_pass.render_pass())
            .build()];
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_info, None)
                .unwrap()[0]
        };

        // ShaderModuleはもう不要のため破棄
        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);

            device.destroy_shader_module(fragment_shader_module, None);
        }

        Ok(Self {
            pipeline,
            pipeline_layout,
            device,
        })
    }

    /// pipelineを取得する。
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// pipeline_layoutを取得する。
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}
impl Drop for PipelineObject {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}
