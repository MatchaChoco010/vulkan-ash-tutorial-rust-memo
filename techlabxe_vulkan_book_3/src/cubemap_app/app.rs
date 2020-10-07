use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use cgmath::{Deg, Matrix4, Point3, Vector3, Vector4};
use winit::{dpi::PhysicalSize, event::Event, window::Window};

use crate::{
    app_base::{VulkanAppBase, VulkanAppBaseBuilder},
    cubemap_app::{
        camera::Camera,
        mesh_pass::MeshPass,
        scene::{Scene, SceneBuilder},
        sky_pass::SkyPass,
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
            title: "Cubemap App".to_string(),
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

        let camera = Camera::new(
            &allocator,
            Point3::new(0.0, -2.0, 10.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Deg(45.0),
            width as f32 / height as f32,
            0.01,
            100.0,
        )?;
        let scene = SceneBuilder::new()
            .add_camera(camera)
            .add_cubemap_textures(
                "assets/cubemap/posx.jpg",
                "assets/cubemap/negx.jpg",
                "assets/cubemap/posy.jpg",
                "assets/cubemap/negy.jpg",
                "assets/cubemap/posz.jpg",
                "assets/cubemap/negz.jpg",
            )
            .add_entity(
                "assets/monkey.obj",
                Matrix4::from_translation(Vector3::new(2.0, 0.0, 0.0)),
            )
            .add_entity(
                "assets/monkey.obj",
                Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.0)),
            )
            .build(&device, &allocator, &graphics_command_pool, graphics_queue)?;

        let mesh_pass = MeshPass::new(device.device(), &swapchain, &depth_buffer, &scene)?;
        let sky_pass = SkyPass::new(device.device(), &swapchain, &depth_buffer, &scene)?;

        Ok(Self::Item {
            cursor_pos: (0.0, 0.0),
            cursor_prev_pos: (0.0, 0.0),
            button_pressed: false,
            rotate_y: 0.0,
            rotate_x: 0.0,
            render_finish_semaphore,
            image_available_semaphore,
            fence,
            sky_pass,
            mesh_pass,
            scene,
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
    cursor_pos: (f64, f64),
    cursor_prev_pos: (f64, f64),
    button_pressed: bool,
    rotate_y: f32,
    rotate_x: f32,

    render_finish_semaphore: SemaphoreObject,
    image_available_semaphore: SemaphoreObject,
    fence: FenceObject,

    sky_pass: SkyPass,
    mesh_pass: MeshPass,
    scene: Scene,

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
        self.mesh_pass.resize(&self.swapchain, &self.depth_buffer)?;
        self.sky_pass.resize(&self.swapchain, &self.depth_buffer)?;

        self.scene
            .camera_mut()
            .set_proj(Deg(45.0), width as f32 / height as f32, 0.01, 100.0);

        Ok(())
    }

    fn on_mouse_button_down(&mut self, button: i32) {
        if button == 0 {
            self.button_pressed = true;
        }
    }
    fn on_mouse_button_up(&mut self, button: i32) {
        if button == 0 {
            self.button_pressed = false;
        }
    }
    fn on_mouse_move(&mut self, x: f64, y: f64) {
        self.cursor_pos = (x, y);
        if self.button_pressed {
            let dx = self.cursor_pos.0 - self.cursor_prev_pos.0;
            let dy = self.cursor_pos.1 - self.cursor_prev_pos.1;
            let dx = (dx * 0.1) as f32;
            let dy = (dy * 0.1) as f32;
            self.rotate_y -= dx;
            self.rotate_x = (self.rotate_x + dy).max(-90.0).min(90.0);
        }
        self.cursor_prev_pos = self.cursor_pos;
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

        // CameraのUniformの更新
        let camera_rotation =
            Matrix4::from_angle_y(Deg(self.rotate_y)) * Matrix4::from_angle_x(Deg(self.rotate_x));
        let eye = Point3::from_homogeneous(camera_rotation * Vector4::new(0.0, 0.0, 10.0, 1.0));
        let up = camera_rotation * Vector4::new(0.0, 1.0, 0.0, 0.0);
        let up = up.xyz();
        self.scene
            .camera_mut()
            .set_view(eye, Point3::new(0.0, 0.0, 0.0), up);
        self.scene.camera().update_uniform()?;

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

        self.mesh_pass.cmd_draw(command, image_index, &self.scene);
        self.sky_pass.cmd_draw(command, image_index);

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
