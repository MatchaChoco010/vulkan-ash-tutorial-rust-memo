use std::{path::Path, rc::Rc};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};
use image::GenericImageView;

use crate::vk_wrapper_object::{CommandPoolObject, DeviceObject, VkMemAllocatorObject};

pub struct ImageObject {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
    allocator: Rc<vk_mem::Allocator>,
    device: Rc<Device>,
    width: u32,
    height: u32,
}
// 2D
impl ImageObject {
    /// Imageを作成する。
    pub fn new(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let allocator_ref = allocator.allocator_as_ref();
        let device_ref = device.device_as_ref();

        // Imageの作成
        let (image, image_allocation, _info) = allocator_ref.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: width,
                    height: height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(usage),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;
        let image_view = unsafe {
            device_ref.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(image)
                    .format(vk::Format::R8G8B8A8_UNORM)
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

        Ok(Self {
            image,
            allocation: image_allocation,
            view: image_view,
            allocator: allocator.allocator(),
            device: device.device(),
            width,
            height,
        })
    }

    /// DepthBuffer用Imageを作成する。
    pub fn new_depth_stencil(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let allocator_ref = allocator.allocator_as_ref();
        let device_ref = device.device_as_ref();

        // Imageの作成
        let (image, image_allocation, _info) = allocator_ref.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT_S8_UINT)
                .extent(vk::Extent3D {
                    width: width,
                    height: height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;
        let image_view = unsafe {
            device_ref.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(image)
                    .format(vk::Format::D32_SFLOAT_S8_UINT)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )?
        };

        Ok(Self {
            image,
            allocation: image_allocation,
            view: image_view,
            allocator: allocator.allocator(),
            device: device.device(),
            width,
            height,
        })
    }

    /// データをアップロードしShaderReadOnlyなImageを作成する。
    pub fn new_with_data(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<Self> {
        let image_size = (std::mem::size_of::<u8>() as u32 * width * height * 4) as vk::DeviceSize;

        let allocator_ref = allocator.allocator_as_ref();

        // ステージングバッファの作成
        let (staging_buffer, staging_buffer_allocation, _info) = allocator_ref.create_buffer(
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
            let mapped_memory = allocator_ref.map_memory(&staging_buffer_allocation)? as *mut u8;
            mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());
            allocator_ref.unmap_memory(&staging_buffer_allocation)?;
        }

        let format = vk::Format::R8G8B8A8_UNORM;

        // Imageの作成
        let (image, image_allocation, _info) = allocator_ref.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: width,
                    height: height,
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
        let device_ref = device.device_as_ref();
        unsafe {
            let command = device_ref.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool.command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            // コマンドの構築開始
            device_ref.begin_command_buffer(
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
            device_ref.cmd_pipeline_barrier(
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
                    width: width,
                    height: height,
                    depth: 1,
                })
                .buffer_offset(0)
                .buffer_image_height(0)
                .buffer_row_length(0)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .build()];
            device_ref.cmd_copy_buffer_to_image(
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
            device_ref.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );

            // コマンド終了
            device_ref.end_command_buffer(command)?;

            // コマンドのサブミット
            device_ref.queue_submit(
                graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[command])
                    .build()],
                vk::Fence::null(),
            )?;
            device_ref.queue_wait_idle(graphics_queue)?;
        }

        let image_view = unsafe {
            device_ref.create_image_view(
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

        allocator_ref.destroy_buffer(staging_buffer, &staging_buffer_allocation)?;

        Ok(Self {
            image,
            allocation: image_allocation,
            view: image_view,
            allocator: allocator.allocator(),
            device: device.device(),
            width,
            height,
        })
    }

    /// 画像を読み込みShaderReadeOnlyなImageを作成する。
    pub fn load_image(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        filename: impl AsRef<Path>,
    ) -> Result<Self> {
        let image = image::open(filename)?;
        let (image_width, image_height) = (image.width(), image.height());
        let image_data = image.to_rgba().into_raw();
        Self::new_with_data(
            device,
            allocator,
            command_pool,
            graphics_queue,
            image_width,
            image_height,
            image_data.as_slice(),
        )
    }
}
// Cubemap
impl ImageObject {
    /// Cubemapを作成する。
    pub fn new_cubemap_with_data(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        width: u32,
        height: u32,
        data_xp: &[u8],
        data_xm: &[u8],
        data_yp: &[u8],
        data_ym: &[u8],
        data_zp: &[u8],
        data_zm: &[u8],
    ) -> Result<Self> {
        let image_size = (std::mem::size_of::<u8>() as u32 * width * height * 4) as vk::DeviceSize;

        let allocator_ref = allocator.allocator_as_ref();

        // ステージングバッファの作成
        let mut staging_buffers = vec![];
        for _ in 0..6 {
            staging_buffers.push(
                allocator_ref.create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(image_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::CpuOnly,
                        ..Default::default()
                    },
                )?,
            );
        }

        let format = vk::Format::R8G8B8A8_UNORM;

        // Imageの作成
        let (image, image_allocation, _info) = allocator_ref.create_image(
            &vk::ImageCreateInfo::builder()
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: width,
                    height: height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(6)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;

        let device_ref = device.device_as_ref();
        let data = [data_xp, data_xm, data_yp, data_ym, data_zp, data_zm];
        unsafe {
            let command = device_ref.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool.command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            // コマンドの構築開始
            device_ref.begin_command_buffer(
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
                        .layer_count(6)
                        .build(),
                )
                .build()];
            device_ref.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );

            for i in 0..6 {
                let (staging_buffer, staging_buffer_allocation, _) = staging_buffers[i];

                // ステージングバッファへのマップ
                let mapped_memory =
                    allocator_ref.map_memory(&staging_buffer_allocation)? as *mut u8;
                mapped_memory.copy_from_nonoverlapping(data[i].as_ptr(), data[i].len());
                allocator_ref.unmap_memory(&staging_buffer_allocation)?;

                // BufferからImageへデータをコピーする
                let buffer_image_regions = [vk::BufferImageCopy::builder()
                    .image_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(i as u32)
                            .layer_count(1)
                            .build(),
                    )
                    .image_extent(vk::Extent3D {
                        width: width,
                        height: height,
                        depth: 1,
                    })
                    .buffer_offset(0)
                    .buffer_image_height(0)
                    .buffer_row_length(0)
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .build()];
                device_ref.cmd_copy_buffer_to_image(
                    command,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &buffer_image_regions,
                );
            }

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
                        .layer_count(6)
                        .build(),
                )
                .build()];
            device_ref.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );

            // コマンド終了
            device_ref.end_command_buffer(command)?;

            // コマンドのサブミット
            device_ref.queue_submit(
                graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[command])
                    .build()],
                vk::Fence::null(),
            )?;
            device_ref.queue_wait_idle(graphics_queue)?;
        }

        let image_view = unsafe {
            device_ref.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::CUBE)
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
                        layer_count: 6,
                    }),
                None,
            )?
        };

        for (staging_buffer, staging_buffer_allocation, _) in staging_buffers {
            allocator_ref.destroy_buffer(staging_buffer, &staging_buffer_allocation)?;
        }

        Ok(Self {
            image,
            allocation: image_allocation,
            view: image_view,
            allocator: allocator.allocator(),
            device: device.device(),
            width,
            height,
        })
    }

    /// Cubemap画像を読み込む。
    pub fn load_cubemap_images(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        filename_xp: impl AsRef<Path>,
        filename_xm: impl AsRef<Path>,
        filename_yp: impl AsRef<Path>,
        filename_ym: impl AsRef<Path>,
        filename_zp: impl AsRef<Path>,
        filename_zm: impl AsRef<Path>,
    ) -> Result<Self> {
        let image_xp = image::open(filename_xp)?;
        let (image_width_xp, image_height_xp) = (image_xp.width(), image_xp.height());
        let image_data_xp = image_xp.to_rgba().into_raw();

        let image_xm = image::open(filename_xm)?;
        let (image_width_xm, image_height_xm) = (image_xm.width(), image_xm.height());
        let image_data_xm = image_xm.to_rgba().into_raw();

        let image_yp = image::open(filename_yp)?;
        let (image_width_yp, image_height_yp) = (image_yp.width(), image_yp.height());
        let image_data_yp = image_yp.to_rgba().into_raw();

        let image_ym = image::open(filename_ym)?;
        let (image_width_ym, image_height_ym) = (image_ym.width(), image_ym.height());
        let image_data_ym = image_ym.to_rgba().into_raw();

        let image_zp = image::open(filename_zp)?;
        let (image_width_zp, image_height_zp) = (image_zp.width(), image_zp.height());
        let image_data_zp = image_zp.to_rgba().into_raw();

        let image_zm = image::open(filename_zm)?;
        let (image_width_zm, image_height_zm) = (image_zm.width(), image_zm.height());
        let image_data_zm = image_zm.to_rgba().into_raw();

        assert_eq!(image_height_xm, image_height_xp);
        assert_eq!(image_height_xp, image_height_ym);
        assert_eq!(image_height_ym, image_height_yp);
        assert_eq!(image_height_yp, image_height_zm);
        assert_eq!(image_height_zm, image_height_zp);
        assert_eq!(image_width_xm, image_width_xp);
        assert_eq!(image_width_xp, image_width_ym);
        assert_eq!(image_width_ym, image_width_yp);
        assert_eq!(image_width_yp, image_width_zm);
        assert_eq!(image_width_zm, image_width_zp);

        Self::new_cubemap_with_data(
            device,
            allocator,
            command_pool,
            graphics_queue,
            image_width_xp,
            image_height_xp,
            image_data_xp.as_slice(),
            image_data_xm.as_slice(),
            image_data_yp.as_slice(),
            image_data_ym.as_slice(),
            image_data_zp.as_slice(),
            image_data_zm.as_slice(),
        )
    }
}
// Common
impl ImageObject {
    /// vk::Imageを取得する。
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// vk::ImageViewを取得する。
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// 画像の幅を取得する。
    pub fn width(&self) -> u32 {
        self.width
    }

    /// 画像の高さを取得する。
    pub fn height(&self) -> u32 {
        self.height
    }
}
// Drop
impl Drop for ImageObject {
    fn drop(&mut self) {
        self.allocator
            .destroy_image(self.image, &self.allocation)
            .expect("Failed to destroy image");
        unsafe { self.device.destroy_image_view(self.view, None) };
    }
}
