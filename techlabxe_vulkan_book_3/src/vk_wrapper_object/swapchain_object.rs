use std::rc::Rc;

use anyhow::Result;
use ash::{
    extensions::khr::{Surface, Swapchain},
    version::DeviceV1_0,
    vk, Device,
};

use crate::vk_wrapper_object::{DeviceObject, InstanceObject, PhysicalDeviceObject, SurfaceObject};

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
    fn create_swapchain(
        swapchain_loader: &Swapchain,
        physical_device: vk::PhysicalDevice,
        device: Rc<Device>,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        graphics_queue_index: u32,
        present_queue_index: u32,
        width: u32,
        height: u32,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Self> {
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

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
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

        if let Some(old_swapchain) = old_swapchain {
            swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain);
        }

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
            swapchain_loader: swapchain_loader.clone(),
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

    /// スワップチェーンを作成する。
    pub fn new(
        instance: &InstanceObject,
        device: &DeviceObject,
        physical_device: &PhysicalDeviceObject,
        surface: &SurfaceObject,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let swapchain_loader = Swapchain::new(instance.instance_as_ref(), device.device_as_ref());

        Self::create_swapchain(
            &swapchain_loader,
            physical_device.physical_device(),
            device.device(),
            surface.surface_loader(),
            surface.surface(),
            physical_device.graphics_queue_index(),
            physical_device.present_queue_index(),
            width,
            height,
            None,
        )
    }

    /// スワップチェーンを再生成する。
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        // 待機
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }

        let mut tmp_swapchain_object = Self::create_swapchain(
            &self.swapchain_loader,
            self.physical_device,
            Rc::clone(&self.device),
            &self.surface_loader,
            self.surface,
            self.graphics_queue_index,
            self.present_queue_index,
            width,
            height,
            Some(self.swapchain),
        )?;

        std::mem::swap(&mut self.swapchain, &mut tmp_swapchain_object.swapchain);
        std::mem::swap(&mut self.images, &mut tmp_swapchain_object.images);
        std::mem::swap(&mut self.image_views, &mut tmp_swapchain_object.image_views);
        std::mem::swap(&mut self.format, &mut tmp_swapchain_object.format);
        std::mem::swap(&mut self.extent, &mut tmp_swapchain_object.extent);

        Ok(())
    }

    /// スワップチェーンの枚数を返す。
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// acquire_next_imageを呼び出す。
    pub fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<(usize, bool)> {
        unsafe {
            let (image_index, is_suboptimal) = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                semaphore,
                vk::Fence::null(),
            )?;

            let image_index = image_index as usize;
            Ok((image_index, is_suboptimal))
        }
    }

    /// Imageを取得する。
    pub fn get_image(&self, index: usize) -> &vk::Image {
        &self.images[index]
    }

    /// ImageViewを取得する。
    pub fn get_image_view(&self, index: usize) -> vk::ImageView {
        self.image_views[index]
    }

    /// queue_presentを呼び出す。
    pub fn queue_present(
        &self,
        present_queue: vk::Queue,
        index: usize,
        wait_semaphore: vk::Semaphore,
    ) -> Result<bool> {
        let is_suboptimal = unsafe {
            self.swapchain_loader.queue_present(
                present_queue,
                &vk::PresentInfoKHR::builder()
                    .swapchains(&[self.swapchain])
                    .image_indices(&[index as u32])
                    .wait_semaphores(&[wait_semaphore]),
            )?
        };
        Ok(is_suboptimal)
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
