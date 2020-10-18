use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use winit::{dpi::PhysicalSize, event::Event, window::Window};

use crate::{
    app_base::{VulkanAppBase, VulkanAppBaseBuilder},
    cs_post_effect::{
        compute_sobel_pass::ComputeSobelPass, model_pass::ModelPass, quad_pass::QuadPass,
    },
    vk_wrapper_object::{
        CommandPoolObject, DebugUtilsObject, DeviceObject, EntryObject, FenceObject, ImageObject,
        InstanceObject, PhysicalDeviceObject, SemaphoreObject, SurfaceObject, SwapchainObject,
        VkMemAllocatorObject,
    },
};

pub struct AppBuilder {
    title: String,
    window_size: (u32, u32),
    version: (u32, u32, u32),
}
impl AppBuilder {
    pub fn window_size(self, width: u32, height: u32) -> Self {
        Self {
            window_size: (width, height),
            ..self
        }
    }

    pub fn title(self, title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..self
        }
    }

    pub fn version(self, major: u32, minor: u32, patch: u32) -> Self {
        Self {
            version: (major, minor, patch),
            ..self
        }
    }
}
impl VulkanAppBaseBuilder for AppBuilder {
    type Item = App;

    fn new() -> Self {
        Self {
            title: "Compute Shader App".to_string(),
            window_size: (800, 600),
            version: (0, 1, 0),
        }
    }

    fn window_size(&self) -> (u32, u32) {
        self.window_size
    }

    fn title(&self) -> &str {
        &self.title
    }

    fn build(self, window: &Window) -> Result<Self::Item> {
        let (width, height) = self.window_size;

        let entry = EntryObject::new()?;
        let instance = InstanceObject::new(&entry, window, self.title, self.version)?;
        let debug_utils = DebugUtilsObject::new(&entry, &instance);
        let surface = SurfaceObject::new(&entry, &instance, window)?;
        let physical_device = PhysicalDeviceObject::new(&instance, &surface)?;
        let device = DeviceObject::new(&instance, &physical_device)?;

        let allocator = VkMemAllocatorObject::new(&instance, &physical_device, &device)?;
        let graphics_command_pool =
            CommandPoolObject::new(&device, physical_device.graphics_queue_index())?;
        let graphics_queue = device.get_queue(physical_device.graphics_queue_index());
        let present_queue = device.get_queue(physical_device.present_queue_index());

        let command_buffer = unsafe {
            device.device_as_ref().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(graphics_command_pool.command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0]
        };

        let swapchain = SwapchainObject::new(
            &instance,
            &device,
            &physical_device,
            &surface,
            width,
            height,
        )?;
        let depth_buffer = ImageObject::new_depth_stencil(&device, &allocator, width, height)?;

        let fence = FenceObject::new(&device)?;
        let image_available_semaphore = SemaphoreObject::new(&device)?;
        let render_finish_semaphore = SemaphoreObject::new(&device)?;

        let src_image = ImageObject::new(
            &device,
            &allocator,
            800,
            600,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
        )?;
        let src_image_depth = ImageObject::new_depth_stencil(&device, &allocator, 800, 600)?;
        let dst_image = ImageObject::new(
            &device,
            &allocator,
            800,
            600,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
        )?;

        let model_pass = ModelPass::new(
            &device,
            &allocator,
            &graphics_command_pool,
            graphics_queue,
            &src_image,
            &src_image_depth,
            "assets/alicia-solid.vrm",
        )?;
        let quad_pass = QuadPass::new(
            &device,
            &allocator,
            &graphics_command_pool,
            graphics_queue,
            &swapchain,
            &depth_buffer,
            &src_image,
            &dst_image,
        )?;
        let compute_sobel_pass = ComputeSobelPass::new(&device, &src_image, &dst_image)?;

        Ok(Self::Item {
            render_finish_semaphore,
            image_available_semaphore,
            fence,
            compute_sobel_pass,
            quad_pass,
            model_pass,
            dst_image,
            src_image,
            _src_image_depth: src_image_depth,
            depth_buffer,
            swapchain,
            command_buffer,
            present_queue,
            graphics_queue,
            _graphics_command_pool: graphics_command_pool,
            allocator,
            device,
            _physical_device: physical_device,
            _surface: surface,
            _debug_utils: debug_utils,
            _instance: instance,
            _entry: entry,
        })
    }
}
pub struct App {
    render_finish_semaphore: SemaphoreObject,
    image_available_semaphore: SemaphoreObject,
    fence: FenceObject,

    compute_sobel_pass: ComputeSobelPass,
    quad_pass: QuadPass,
    model_pass: ModelPass,
    dst_image: ImageObject,
    src_image: ImageObject,
    _src_image_depth: ImageObject,

    depth_buffer: ImageObject,
    swapchain: SwapchainObject,
    command_buffer: vk::CommandBuffer,
    present_queue: vk::Queue,
    graphics_queue: vk::Queue,
    _graphics_command_pool: CommandPoolObject,
    allocator: VkMemAllocatorObject,
    device: DeviceObject,
    _physical_device: PhysicalDeviceObject,
    _surface: SurfaceObject,
    _debug_utils: DebugUtilsObject,
    _instance: InstanceObject,
    _entry: EntryObject,
}
impl VulkanAppBase for App {
    fn on_window_size_changed(&mut self, width: u32, height: u32) -> Result<()> {
        self.swapchain.resize(width, height)?;
        self.depth_buffer =
            ImageObject::new_depth_stencil(&self.device, &self.allocator, width, height)?;
        self.quad_pass.resize(&self.swapchain, &self.depth_buffer)?;

        Ok(())
    }

    fn on_new_events(&mut self) {}
    fn on_main_events_cleared(&mut self, _window: &Window) {}
    fn handle_event(&mut self, _window: &Window, _event: &Event<()>) {}

    fn render(&mut self, window: &Window) -> Result<()> {
        let device = self.device.device_as_ref();

        let (image_index, is_suboptimal) = self
            .swapchain
            .acquire_next_image(self.image_available_semaphore.semaphore())?;

        if is_suboptimal {
            let PhysicalSize { width, height } = window.inner_size();
            self.on_window_size_changed(width, height)?;
            return Ok(());
        }

        // Fenceを待機
        let fence = self.fence.fence();
        unsafe {
            device.wait_for_fences(&[fence], true, std::u64::MAX)?;
            device.reset_fences(&[fence])?;
        }

        // コマンドの構築
        let command = self.command_buffer;
        unsafe {
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::builder())?;
        }

        self.model_pass.cmd_draw(command);

        // ImageBarrier
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.src_image.image())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build(),
                    vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(self.dst_image.image())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build(),
                ],
            )
        }

        self.compute_sobel_pass.cmd_draw(command);

        // ImageBarrier
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier::builder()
                        .src_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(self.src_image.image())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build(),
                    vk::ImageMemoryBarrier::builder()
                        .src_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(self.dst_image.image())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .build(),
                ],
            )
        }

        self.quad_pass.cmd_draw(command, image_index);

        unsafe {
            device.end_command_buffer(command)?;

            device.queue_submit(
                self.graphics_queue,
                &[vk::SubmitInfo::builder()
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[command])
                    .wait_semaphores(&[self.image_available_semaphore.semaphore()])
                    .signal_semaphores(&[self.render_finish_semaphore.semaphore()])
                    .build()],
                fence,
            )?;
        }

        let is_suboptimal = self.swapchain.queue_present(
            self.present_queue,
            image_index,
            self.render_finish_semaphore.semaphore(),
        )?;
        if is_suboptimal {
            let PhysicalSize { width, height } = window.inner_size();
            self.on_window_size_changed(width, height)?;
        }

        Ok(())
    }
}
impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_as_ref()
                .device_wait_idle()
                .expect("Failed to wait idle.");
        }
    }
}
