#![allow(dead_code)]

use std::{path::Path, rc::Rc, time::Instant};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device, Instance};
use cgmath::{Deg, Matrix4, Point3, Vector2, Vector3};
use imgui::*;
use imgui_rs_vulkan_renderer::{Renderer, RendererVkContext};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use memoffset::offset_of;
use winit::{dpi::PhysicalSize, event::Event, window::Window};

use crate::common::{
    cgmath_ext::Matrix4Ext,
    default_vulkan_app_base::{DefaultVulkanAppBase, DefaultVulkanAppBaseBuilder},
    vulkan_app_base::{VulkanAppBase, VulkanAppBaseBuilder},
    vulkan_objects::{
        DepthImageObject, DescriptorPoolObject, DescriptorSetLayoutObject, DeviceLocalBufferObject,
        FramebufferObject, HostVisibleBufferObject, PipelineObject, RenderPassObject,
        SamplerObject, SwapchainObject, TextureRenderingImageObject,
    },
};
#[repr(C)]
#[derive(Debug)]
struct TeapotVertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
}
impl TeapotVertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, position) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, normal) as u32)
                .build(),
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct MVPMatricesObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

struct VkContext<'a> {
    instance: &'a Instance,
    physical_device: vk::PhysicalDevice,
    device: Rc<Device>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
}
impl<'a> RendererVkContext for VkContext<'a> {
    fn instance(&self) -> &Instance {
        self.instance
    }
    fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    fn device(&self) -> &Device {
        &*self.device
    }
    fn queue(&self) -> vk::Queue {
        self.queue
    }
    fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

pub struct TextureRenderingAppBuilder {
    window_size: (u32, u32),
    title: String,
    base: DefaultVulkanAppBaseBuilder,
}
impl TextureRenderingAppBuilder {
    /// 初期Windowサイズを指定する。
    pub fn window_size(self, width: u32, height: u32) -> Self {
        Self {
            window_size: (width, height),
            ..self
        }
    }
    /// Windowタイトルを指定する。
    pub fn title(self, title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..self
        }
    }

    /// モデルを読み込む。
    fn load_teapot_model(
        allocator: Rc<vk_mem::Allocator>,
        device: Rc<Device>,
        graphics_command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
    ) -> Result<(DeviceLocalBufferObject, u32)> {
        let model_obj =
            tobj::load_obj(&Path::new("assets/teapot.obj"), true).expect("Failed to open model");
        let mut vertices = vec![];
        let (models, _) = model_obj;
        for m in models.iter() {
            let mesh = &m.mesh;

            for &i in mesh.indices.iter() {
                let i = i as usize;
                let vertex = TeapotVertex {
                    position: Vector3::new(
                        mesh.positions[3 * i],
                        mesh.positions[3 * i + 1],
                        mesh.positions[3 * i + 2],
                    ),
                    normal: Vector3::new(
                        mesh.normals[3 * i],
                        mesh.normals[3 * i + 1],
                        mesh.normals[3 * i + 2],
                    ),
                };
                vertices.push(vertex);
            }
        }

        let vertex_buffer = DeviceLocalBufferObject::new(
            allocator,
            device,
            graphics_command_pool,
            graphics_queue,
            vertices.as_slice(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        Ok((vertex_buffer, vertices.len() as u32))
    }

    /// UniformBufferを準備する。
    fn prepare_teapot_uniform_buffer(
        allocator: Rc<vk_mem::Allocator>,
        mvp_matrices_object: MVPMatricesObject,
    ) -> Result<HostVisibleBufferObject> {
        let teapot_uniform_buffer = HostVisibleBufferObject::new(
            Rc::clone(&allocator),
            &[mvp_matrices_object],
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        Ok(teapot_uniform_buffer)
    }

    /// RenderPassを準備する。
    fn prepare_teapot_render_pass(
        device: Rc<Device>,
        format: vk::Format,
    ) -> Result<RenderPassObject> {
        RenderPassObject::new_color_attachment_optimal(device, format)
    }

    /// DescriptorPoolを準備する。
    fn prepare_teapot_descriptor_pool(
        device: Rc<Device>,
        count: u32,
        max_count: u32,
    ) -> Result<DescriptorPoolObject> {
        DescriptorPoolObject::new(
            device,
            &[vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(count)
                .build()],
            max_count,
        )
    }

    /// DescriptorSetLayoutを準備する。
    fn prepare_teapot_descriptor_set_layout(
        device: Rc<Device>,
    ) -> Result<DescriptorSetLayoutObject> {
        DescriptorSetLayoutObject::new(
            device,
            &[vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .descriptor_count(1)
                .build()],
        )
    }

    /// DescriptorSetを準備する。
    fn prepare_teapot_descriptor_sets(
        device: Rc<Device>,
        pool: &DescriptorPoolObject,
        layout: &DescriptorSetLayoutObject,
        teapot_uniform_buffers: &HostVisibleBufferObject,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let count = 1;

        let mut layouts = vec![];
        for _ in 0..count {
            layouts.push(layout.descriptor_set_layout());
        }

        // ディスクリプタセットの生成
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(pool.descriptor_pool())
                    .set_layouts(layouts.as_slice()),
            )?
        };

        // ディスクリプタセットへの書き込み
        for i in 0..count {
            let descripto_buffer_infos = [vk::DescriptorBufferInfo::builder()
                .buffer(teapot_uniform_buffers.buffer())
                .offset(0)
                .range(std::mem::size_of::<MVPMatricesObject>() as u64)
                .build()];

            let write_descriptor_sets = [vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descripto_buffer_infos)
                .build()];

            unsafe {
                device.update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }

        Ok(descriptor_sets)
    }

    /// Pipelineを準備する。
    fn prepare_teapot_pipeline(
        device: Rc<Device>,
        width: u32,
        height: u32,
        render_pass: &RenderPassObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
    ) -> Result<PipelineObject> {
        PipelineObject::new_opaque(
            device,
            &TeapotVertex::get_binding_descriptions(),
            &TeapotVertex::get_attribute_descriptions(),
            &Path::new("shaders/spv/resizable.vert.spv"),
            &Path::new("shaders/spv/resizable.frag.spv"),
            width,
            height,
            render_pass,
            descriptor_set_layout,
        )
    }

    /// Framebufferを準備する。
    fn prepare_teapot_framebuffer(
        device: Rc<Device>,
        texture: &TextureRenderingImageObject,
        depth_buffer: &DepthImageObject,
        render_pass: &RenderPassObject,
    ) -> Result<FramebufferObject> {
        let framebuffer = FramebufferObject::new(
            Rc::clone(&device),
            texture.image_view(),
            depth_buffer,
            render_pass,
            texture.image_size().0,
            texture.image_size().1,
        )?;
        Ok(framebuffer)
    }

    /// UniformBufferを準備する。
    fn prepare_quad_uniform_buffers(
        allocator: Rc<vk_mem::Allocator>,
        mvp_matrices_object: MVPMatricesObject,
    ) -> Result<HostVisibleBufferObject> {
        let quad_uniform_buffer = HostVisibleBufferObject::new(
            Rc::clone(&allocator),
            &[mvp_matrices_object],
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        Ok(quad_uniform_buffer)
    }

    /// RenderPassを準備する。
    fn prepare_quad_render_pass(
        device: Rc<Device>,
        format: vk::Format,
    ) -> Result<RenderPassObject> {
        RenderPassObject::new(device, format)
    }

    /// DescriptorPoolを準備する。
    fn prepare_quad_descriptor_pool(
        device: Rc<Device>,
        count: u32,
        max_count: u32,
    ) -> Result<DescriptorPoolObject> {
        DescriptorPoolObject::new(
            device,
            &[
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(count)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(count)
                    .build(),
            ],
            max_count,
        )
    }

    /// DescriptorSetLayoutを準備する。
    fn prepare_quad_descriptor_set_layout(device: Rc<Device>) -> Result<DescriptorSetLayoutObject> {
        DescriptorSetLayoutObject::new(
            device,
            &[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_count(1)
                    .build(),
            ],
        )
    }

    /// Samplerを準備する。
    fn prepare_quad_sampler(device: Rc<Device>) -> Result<SamplerObject> {
        SamplerObject::new(device)
    }

    /// DescriptorSetを準備する。
    fn prepare_quad_descriptor_sets(
        device: Rc<Device>,
        pool: &DescriptorPoolObject,
        layout: &DescriptorSetLayoutObject,
        quad_uniform_buffers: &HostVisibleBufferObject,
        quad_texture: &TextureRenderingImageObject,
        quad_sampler: &SamplerObject,
        count: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let mut layouts = vec![];
        for _ in 0..count {
            layouts.push(layout.descriptor_set_layout());
        }

        // ディスクリプタセットの生成
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(pool.descriptor_pool())
                    .set_layouts(layouts.as_slice()),
            )?
        };

        // ディスクリプタセットへの書き込み
        for i in 0..count {
            let descripto_buffer_infos = [vk::DescriptorBufferInfo::builder()
                .buffer(quad_uniform_buffers.buffer())
                .offset(0)
                .range(std::mem::size_of::<MVPMatricesObject>() as u64)
                .build()];
            let descriptor_image_infos = [vk::DescriptorImageInfo::builder()
                .image_view(quad_texture.image_view())
                .sampler(quad_sampler.sampler())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&descripto_buffer_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&descriptor_image_infos)
                    .build(),
            ];

            unsafe {
                device.update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }

        Ok(descriptor_sets)
    }

    /// Pipelineを準備する。
    fn prepare_quad_pipeline(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        render_pass: &RenderPassObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
    ) -> Result<PipelineObject> {
        PipelineObject::new_opaque_no_vertex(
            device,
            &Path::new("shaders/spv/texture_rendering.vert.spv"),
            &Path::new("shaders/spv/texture_rendering.frag.spv"),
            swapchain.extent().width,
            swapchain.extent().height,
            render_pass,
            descriptor_set_layout,
        )
    }

    /// Framebufferを準備する。
    fn prepare_quad_framebuffers(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        depth_buffer: &DepthImageObject,
        render_pass: &RenderPassObject,
    ) -> Result<Vec<FramebufferObject>> {
        let mut framebuffers = vec![];
        for i in 0..swapchain.len() {
            framebuffers.push(FramebufferObject::new(
                Rc::clone(&device),
                *swapchain.get_image_view(i),
                depth_buffer,
                render_pass,
                swapchain.extent().width,
                swapchain.extent().height,
            )?);
        }
        Ok(framebuffers)
    }

    /// Imgui用のリソースを用意する。
    fn prepare_imgui_resource(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        width: u32,
        height: u32,
    ) -> Result<(RenderPassObject, Vec<FramebufferObject>)> {
        let imgui_render_pass =
            RenderPassObject::new_no_depth(Rc::clone(&device), swapchain.format())?;
        let imgui_framebuffers = (0..swapchain.len())
            .map(|index| {
                FramebufferObject::new_no_depth(
                    Rc::clone(&device),
                    *swapchain.get_image_view(index),
                    &imgui_render_pass,
                    width,
                    height,
                )
                .expect("Failed to create imgui framebuffer")
            })
            .collect::<Vec<_>>();

        Ok((imgui_render_pass, imgui_framebuffers))
    }
}
impl VulkanAppBaseBuilder for TextureRenderingAppBuilder {
    type Item = TextureRenderingApp;

    fn new() -> Self {
        Self {
            window_size: (800, 600),
            title: "Use Imgui App".into(),
            base: DefaultVulkanAppBaseBuilder::new(),
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

        let base = self
            .base
            .title(self.title)
            .window_size(width, height)
            .version(0, 1, 0)
            .build(window)?;

        // teapot用
        let teapot_texture_rendering_image =
            TextureRenderingImageObject::new(base.device(), base.allocator(), 800, 600)?;
        let teapot_depth_buffer_image =
            DepthImageObject::new(base.allocator(), base.device(), 800, 600)?;
        let (teapot_vertex_buffer, teapot_vertex_count) = Self::load_teapot_model(
            base.allocator(),
            base.device(),
            &base.graphics_command_pool(),
            &base.graphics_queue(),
        )?;
        let teapot_mvp_matrices_object = MVPMatricesObject {
            model: Matrix4::from_scale(1.0),
            view: Matrix4::look_at(
                Point3::new(0.0, 5.0, 10.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: Matrix4::perspective(Deg(45.0), 800.0 / 600.0, 0.01, 100.0),
        };
        let teapot_uniform_buffer =
            Self::prepare_teapot_uniform_buffer(base.allocator(), teapot_mvp_matrices_object)?;
        let teapot_render_pass =
            Self::prepare_teapot_render_pass(base.device(), vk::Format::R8G8B8A8_UNORM)?;
        let teapot_descriptor_pool = Self::prepare_teapot_descriptor_pool(base.device(), 100, 1)?;
        let teapot_descriptor_set_layout =
            Self::prepare_teapot_descriptor_set_layout(base.device())?;
        let teapot_descriptor_sets = Self::prepare_teapot_descriptor_sets(
            base.device(),
            &teapot_descriptor_pool,
            &teapot_descriptor_set_layout,
            &teapot_uniform_buffer,
        )?;
        let teapot_framebuffer = Self::prepare_teapot_framebuffer(
            base.device(),
            &teapot_texture_rendering_image,
            &teapot_depth_buffer_image,
            &teapot_render_pass,
        )?;
        let teapot_pipeline = Self::prepare_teapot_pipeline(
            base.device(),
            800,
            600,
            &teapot_render_pass,
            &teapot_descriptor_set_layout,
        )?;

        // quad用
        let quad_mvp_matrices_object = MVPMatricesObject {
            model: Matrix4::from_scale(1.0),
            view: Matrix4::look_at(
                Point3::new(12.0, 12.0, 12.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: Matrix4::perspective(Deg(45.0), base.swapchain().aspect(), 0.01, 100.0),
        };
        let quad_uniform_buffer =
            Self::prepare_quad_uniform_buffers(base.allocator(), quad_mvp_matrices_object)?;
        let quad_descriptor_pool =
            Self::prepare_quad_descriptor_pool(base.device(), 100, base.swapchain().len() as u32)?;
        let quad_descriptor_set_layout = Self::prepare_quad_descriptor_set_layout(base.device())?;
        let quad_sampler = Self::prepare_quad_sampler(base.device())?;
        let quad_descriptor_sets = Self::prepare_quad_descriptor_sets(
            base.device(),
            &quad_descriptor_pool,
            &quad_descriptor_set_layout,
            &quad_uniform_buffer,
            &teapot_texture_rendering_image,
            &quad_sampler,
            base.swapchain().len(),
        )?;
        let quad_render_pass =
            Self::prepare_quad_render_pass(base.device(), base.swapchain().format())?;
        let quad_framebuffers = Self::prepare_quad_framebuffers(
            base.device(),
            base.swapchain(),
            base.depth_buffer(),
            &quad_render_pass,
        )?;
        let quad_pipeline = Self::prepare_quad_pipeline(
            base.device(),
            base.swapchain(),
            &quad_render_pass,
            &quad_descriptor_set_layout,
        )?;

        // imguiリソースの作成
        let (imgui_render_pass, imgui_framebuffers) =
            Self::prepare_imgui_resource(base.device(), base.swapchain(), width, height)?;
        // imguiのセットアップ
        let mut imgui = Context::create();
        let mut platform = WinitPlatform::init(&mut imgui);
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);

        let vk_context = VkContext {
            instance: base.instance(),
            physical_device: base.physical_device(),
            device: base.device(),
            queue: base.graphics_queue(),
            command_pool: base.graphics_command_pool(),
        };
        let imgui_renderer =
            Renderer::new(&vk_context, 3, imgui_render_pass.render_pass(), &mut imgui)?;
        let last_frame = Instant::now();

        let teapot_background_color = [0.5, 0.25, 0.25, 1.0];
        let quad_background_color = [0.0, 0.0, 0.0, 1.0];

        Ok(Self::Item {
            teapot_background_color,
            teapot_rotate_x: 0.0,
            teapot_rotate_y: 0.0,
            quad_background_color,
            quad_rotate_x: 0.0,
            quad_rotate_y: 0.0,

            imgui,
            platform,
            imgui_renderer,
            last_frame,
            imgui_framebuffers,
            imgui_render_pass,

            quad_pipeline,
            quad_framebuffers,
            quad_render_pass,
            quad_descriptor_sets,
            quad_sampler,
            quad_descriptor_set_layout,
            quad_descriptor_pool,
            quad_uniform_buffer,
            quad_mvp_matrices_object,

            teapot_pipeline,
            teapot_framebuffer,
            teapot_descriptor_sets,
            teapot_descriptor_set_layout,
            teapot_descriptor_pool,
            teapot_render_pass,
            teapot_uniform_buffer,
            teapot_mvp_matrices_object,
            teapot_vertex_buffer,
            teapot_vertex_count,
            teapot_depth_buffer_image,
            teapot_texture_rendering_image,

            base,
        })
    }
}

pub struct TextureRenderingApp {
    teapot_background_color: [f32; 4],
    teapot_rotate_x: f32,
    teapot_rotate_y: f32,
    quad_background_color: [f32; 4],
    quad_rotate_x: f32,
    quad_rotate_y: f32,

    imgui: Context,
    platform: WinitPlatform,
    imgui_renderer: Renderer,
    last_frame: Instant,
    imgui_framebuffers: Vec<FramebufferObject>,
    imgui_render_pass: RenderPassObject,

    quad_pipeline: PipelineObject,
    quad_framebuffers: Vec<FramebufferObject>,
    quad_render_pass: RenderPassObject,
    quad_descriptor_sets: Vec<vk::DescriptorSet>,
    quad_sampler: SamplerObject,
    quad_descriptor_set_layout: DescriptorSetLayoutObject,
    quad_descriptor_pool: DescriptorPoolObject,
    quad_uniform_buffer: HostVisibleBufferObject,
    quad_mvp_matrices_object: MVPMatricesObject,

    teapot_pipeline: PipelineObject,
    teapot_framebuffer: FramebufferObject,
    teapot_descriptor_sets: Vec<vk::DescriptorSet>,
    teapot_descriptor_set_layout: DescriptorSetLayoutObject,
    teapot_descriptor_pool: DescriptorPoolObject,
    teapot_render_pass: RenderPassObject,
    teapot_uniform_buffer: HostVisibleBufferObject,
    teapot_mvp_matrices_object: MVPMatricesObject,
    teapot_vertex_buffer: DeviceLocalBufferObject,
    teapot_vertex_count: u32,
    teapot_depth_buffer_image: DepthImageObject,
    teapot_texture_rendering_image: TextureRenderingImageObject,

    base: DefaultVulkanAppBase,
}
impl TextureRenderingApp {}
impl Drop for TextureRenderingApp {
    fn drop(&mut self) {
        unsafe {
            self.base
                .device()
                .device_wait_idle()
                .expect("Failed to wait device idle");

            let vk_context = VkContext {
                instance: self.base.instance(),
                physical_device: self.base.physical_device(),
                device: self.base.device(),
                queue: self.base.graphics_queue(),
                command_pool: self.base.graphics_command_pool(),
            };
            self.imgui_renderer
                .destroy(&vk_context)
                .expect("Failed to destroy imgui renderer");
        }
    }
}
impl VulkanAppBase for TextureRenderingApp {
    fn on_window_size_changed(&mut self, width: u32, height: u32) -> Result<()> {
        self.base.on_window_size_changed(width, height)?;

        self.quad_render_pass = TextureRenderingAppBuilder::prepare_quad_render_pass(
            self.base.device(),
            self.base.swapchain().format(),
        )?;
        self.quad_descriptor_pool = TextureRenderingAppBuilder::prepare_quad_descriptor_pool(
            self.base.device(),
            100,
            self.base.swapchain().len() as u32,
        )?;
        self.quad_descriptor_sets = TextureRenderingAppBuilder::prepare_quad_descriptor_sets(
            self.base.device(),
            &self.quad_descriptor_pool,
            &self.quad_descriptor_set_layout,
            &self.quad_uniform_buffer,
            &self.teapot_texture_rendering_image,
            &self.quad_sampler,
            self.base.swapchain().len(),
        )?;
        self.quad_framebuffers = TextureRenderingAppBuilder::prepare_quad_framebuffers(
            self.base.device(),
            self.base.swapchain(),
            self.base.depth_buffer(),
            &self.quad_render_pass,
        )?;
        self.quad_pipeline = TextureRenderingAppBuilder::prepare_quad_pipeline(
            self.base.device(),
            self.base.swapchain(),
            &self.quad_render_pass,
            &self.quad_descriptor_set_layout,
        )?;

        // imguiリソースの作成
        let (imgui_render_pass, imgui_framebuffers) =
            TextureRenderingAppBuilder::prepare_imgui_resource(
                self.base.device(),
                self.base.swapchain(),
                width,
                height,
            )?;
        self.imgui_render_pass = imgui_render_pass;
        self.imgui_framebuffers = imgui_framebuffers;
        self.imgui_renderer.set_render_pass(
            &VkContext {
                instance: self.base.instance(),
                physical_device: self.base.physical_device(),
                device: self.base.device(),
                queue: self.base.graphics_queue(),
                command_pool: self.base.graphics_command_pool(),
            },
            self.imgui_render_pass.render_pass(),
        )?;

        // uniform bufferのアスペクト比を修正
        self.quad_mvp_matrices_object = MVPMatricesObject {
            proj: Matrix4::perspective(Deg(45.0), self.base.swapchain().aspect(), 0.01, 100.0),
            ..self.quad_mvp_matrices_object
        };
        self.quad_uniform_buffer
            .map_data(&[self.quad_mvp_matrices_object])?;

        Ok(())
    }

    /// イベント開始
    fn on_new_events(&mut self) {
        self.last_frame = self.imgui.io_mut().update_delta_time(self.last_frame);
    }

    /// メインイベントクリア
    fn on_main_events_cleared(&mut self, _window: &Window) {}

    /// handle event
    fn handle_event(&mut self, window: &Window, event: &Event<()>) {
        self.platform
            .handle_event(self.imgui.io_mut(), window, event);
    }

    fn render(&mut self, window: &Window) -> Result<()> {
        let device = self.base.device();

        let image_index = self
            .base
            .swapchain()
            .acquire_next_image(self.base.image_available_semaphore())?;

        // Fenceを待機
        let fence = self.base.fences()[image_index];
        unsafe {
            device.wait_for_fences(&[fence], true, std::u64::MAX)?;
            device.reset_fences(&[fence])?;
        }

        // Imguiの準備
        self.platform
            .prepare_frame(self.imgui.io_mut(), window)
            .expect("Failed to prepare frame");

        let teapot_color = &mut self.teapot_background_color;
        let teapot_rotate_x = &mut self.teapot_rotate_x;
        let teapot_rotate_y = &mut self.teapot_rotate_y;
        let quad_color = &mut self.quad_background_color;
        let quad_rotate_x = &mut self.quad_rotate_x;
        let quad_rotate_y = &mut self.quad_rotate_y;

        // UIの構築
        let ui = self.imgui.frame();
        imgui::Window::new(im_str!("Hello World"))
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(im_str!("Hello World!"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.separator();
                ColorPicker::new(im_str!("Teapot Background color"), teapot_color).build(&ui);
                Slider::new(im_str!("Teapot Rotate X"), -180.0..=180.0).build(&ui, teapot_rotate_x);
                Slider::new(im_str!("Teapot Rotate Y"), -180.0..=180.0).build(&ui, teapot_rotate_y);
                ui.separator();
                ColorPicker::new(im_str!("Background color"), quad_color).build(&ui);
                Slider::new(im_str!("Rotate X"), -180.0..=180.0).build(&ui, quad_rotate_x);
                Slider::new(im_str!("Rotate Y"), -180.0..=180.0).build(&ui, quad_rotate_y);
            });
        self.platform.prepare_render(&ui, &window);
        let draw_data = ui.render();

        // コマンドの構築
        let command = self.base.command_buffers()[image_index];
        unsafe {
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::builder())?;
        }

        // teapot
        {
            // クリア値
            let clear_value = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: self.teapot_background_color,
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            // Uniform Bufferの更新
            let uniform_buffer_object = MVPMatricesObject {
                model: Matrix4::from_angle_x(Deg(self.teapot_rotate_x))
                    * Matrix4::from_angle_y(Deg(self.teapot_rotate_y)),
                ..self.teapot_mvp_matrices_object
            };
            self.teapot_uniform_buffer
                .map_data(&[uniform_buffer_object])?;

            unsafe {
                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::builder()
                        .render_pass(self.teapot_render_pass.render_pass())
                        .framebuffer(self.teapot_framebuffer.framebuffer())
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: 800,
                                height: 600,
                            },
                        })
                        .clear_values(&clear_value),
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.teapot_pipeline.pipeline(),
                );
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.teapot_pipeline.pipeline_layout(),
                    0,
                    &[self.teapot_descriptor_sets[0]],
                    &[],
                );
                device.cmd_bind_vertex_buffers(
                    command,
                    0,
                    &[self.teapot_vertex_buffer.buffer()],
                    &[0],
                );
                device.cmd_draw(command, self.teapot_vertex_count, 1, 0, 0);
                device.cmd_end_render_pass(command);
            }
        }

        // ImageBarrier
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.teapot_texture_rendering_image.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build()],
            )
        }

        // quad
        {
            // クリア値
            let clear_value = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: self.quad_background_color,
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            // Uniform Bufferの更新
            let uniform_buffer_object = MVPMatricesObject {
                model: Matrix4::from_angle_x(Deg(self.quad_rotate_x))
                    * Matrix4::from_angle_y(Deg(self.quad_rotate_y)),
                ..self.quad_mvp_matrices_object
            };
            self.quad_uniform_buffer
                .map_data(&[uniform_buffer_object])?;

            unsafe {
                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::builder()
                        .render_pass(self.quad_render_pass.render_pass())
                        .framebuffer(self.quad_framebuffers[image_index].framebuffer())
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: self.base.swapchain().extent(),
                        })
                        .clear_values(&clear_value),
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.quad_pipeline.pipeline(),
                );
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.quad_pipeline.pipeline_layout(),
                    0,
                    &[self.quad_descriptor_sets[image_index]],
                    &[],
                );
                device.cmd_draw(command, 6, 1, 0, 0);
                device.cmd_end_render_pass(command);
            }
        }

        // imgui
        unsafe {
            device.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.imgui_render_pass.render_pass())
                    .framebuffer(self.imgui_framebuffers[image_index].framebuffer())
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.base.swapchain().extent(),
                    }),
                vk::SubpassContents::INLINE,
            );
            self.imgui_renderer.cmd_draw(
                &VkContext {
                    instance: self.base.instance(),
                    physical_device: self.base.physical_device(),
                    device: self.base.device(),
                    queue: self.base.graphics_queue(),
                    command_pool: self.base.graphics_command_pool(),
                },
                command,
                draw_data,
            )?;
            device.cmd_end_render_pass(command);
        }

        unsafe {
            device.end_command_buffer(command)?;

            device.queue_submit(
                self.base.graphics_queue(),
                &[vk::SubmitInfo::builder()
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[command])
                    .wait_semaphores(&[self.base.image_available_semaphore()])
                    .signal_semaphores(&[self.base.render_finish_semaphore()])
                    .build()],
                fence,
            )?;
        }

        let is_suboptimal = self.base.swapchain().queue_present(
            self.base.present_queue(),
            image_index,
            self.base.render_finish_semaphore(),
        )?;
        if is_suboptimal {
            let PhysicalSize { width, height } = window.inner_size();
            self.on_window_size_changed(width, height)?;
        }

        Ok(())
    }
}
