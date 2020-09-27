#![allow(dead_code)]

use std::{path::Path, rc::Rc, time::Instant};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device, Instance};
use cgmath::{Deg, InnerSpace, Matrix4, Point3, Vector2, Vector3, Vector4};
use imgui::*;
use imgui_rs_vulkan_renderer::{Renderer, RendererVkContext};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use memoffset::offset_of;
use rand::prelude::*;
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
struct Vertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
}
impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
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
#[derive(Debug)]
struct InstanceData {
    offset: Vector3<f32>,
    color: Vector4<f32>,
}
impl InstanceData {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(1)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE)
            .build()
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(1)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, offset) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(1)
                .location(3)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(offset_of!(Self, color) as u32)
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

#[repr(C)]
#[derive(Copy, Clone)]
struct PostEffectParameterObject {
    window_size: Vector2<f32>,
    block_size: f32,
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

pub struct PostEffectAppBuilder {
    window_size: (u32, u32),
    title: String,
    base: DefaultVulkanAppBaseBuilder,
}
impl PostEffectAppBuilder {
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
                let vertex = Vertex {
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

    /// Instanceごとのデータを準備する。
    fn prepare_instance_data(
        device: Rc<Device>,
        allocator: Rc<vk_mem::Allocator>,
        graphics_command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
    ) -> Result<DeviceLocalBufferObject> {
        let mut data = vec![];
        let color_set = [
            Vector4::new(1.0, 1.0, 1.0, 1.0),
            Vector4::new(1.0, 0.65, 1.0, 1.0),
            Vector4::new(0.1, 0.5, 1.0, 1.0),
            Vector4::new(0.6, 1.0, 0.8, 1.0),
            Vector4::new(0.8, 1.0, 0.6, 1.0),
            Vector4::new(0.6, 0.6, 0.8, 1.0),
        ];
        for i in 0..25 {
            for j in 0..32 {
                let instance_data = InstanceData {
                    offset: Vector3::new(
                        i as f32 * 5.0 - 12.5 * 5.0,
                        0.0,
                        j as f32 * 5.0 - 16.0 * 5.0,
                    ),
                    color: color_set[(i * 20 + j) % 6],
                };
                data.push(instance_data);
            }
        }

        let instance_data_buffer = DeviceLocalBufferObject::new(
            allocator,
            device,
            graphics_command_pool,
            graphics_queue,
            data.as_slice(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        Ok(instance_data_buffer)
    }

    /// UniformBufferを準備する。
    fn prepare_mvp_matrices_uniform_buffer(
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

    /// Instance用UniformBufferを準備する。
    fn prepare_instance_uniform_bufferf(
        allocator: Rc<vk_mem::Allocator>,
    ) -> Result<HostVisibleBufferObject> {
        let mut data = vec![];
        let mut rng = rand::thread_rng();
        for _ in 0..800 {
            let x = rng.gen::<f32>() * 2.0 - 1.0;
            let z = rng.gen::<f32>() * 2.0 - 1.0;
            let axis = Vector3::<f32>::new(x, 1.0, z).normalize();
            let angle = rng.gen::<f32>() * 360.0 - 180.0;
            data.push(Matrix4::from_axis_angle(axis, Deg(angle)));
        }
        let instance_uniform_buffer = HostVisibleBufferObject::new(
            allocator,
            data.as_slice(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        Ok(instance_uniform_buffer)
    }

    /// RenderPassを準備する。
    fn prepare_render_pass(device: Rc<Device>, format: vk::Format) -> Result<RenderPassObject> {
        RenderPassObject::new_color_attachment_optimal(device, format)
    }

    /// DescriptorPoolを準備する。
    fn prepare_descriptor_pool(
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
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(count)
                    .build(),
            ],
            max_count,
        )
    }

    /// DescriptorSetLayoutを準備する。
    fn prepare_descriptor_set_layout(device: Rc<Device>) -> Result<DescriptorSetLayoutObject> {
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
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .descriptor_count(1)
                    .build(),
            ],
        )
    }

    /// DescriptorSetを準備する。
    fn prepare_descriptor_set(
        device: Rc<Device>,
        pool: &DescriptorPoolObject,
        layout: &DescriptorSetLayoutObject,
        teapot_uniform_buffer: &HostVisibleBufferObject,
        instance_uniform_buffer: &HostVisibleBufferObject,
    ) -> Result<vk::DescriptorSet> {
        let layout = layout.descriptor_set_layout();

        // ディスクリプタセットの生成
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(pool.descriptor_pool())
                    .set_layouts(&[layout]),
            )?[0]
        };

        // ディスクリプタセットへの書き込み
        let descriptor_buffer_infos = [vk::DescriptorBufferInfo::builder()
            .buffer(teapot_uniform_buffer.buffer())
            .offset(0)
            .range(std::mem::size_of::<MVPMatricesObject>() as u64)
            .build()];
        let descriptor_instance_buffer_infos = [vk::DescriptorBufferInfo::builder()
            .buffer(instance_uniform_buffer.buffer())
            .offset(0)
            .range(std::mem::size_of::<Matrix4<f32>>() as u64 * 800)
            .build()];

        let write_descriptor_sets = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_instance_buffer_infos)
                .build(),
        ];

        unsafe {
            device.update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        Ok(descriptor_set)
    }

    /// Pipelineを準備する。
    fn prepare_pipeline(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        render_pass: &RenderPassObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
    ) -> Result<PipelineObject> {
        let attrib_descriptions = Vertex::get_attribute_descriptions()
            .iter()
            .cloned()
            .chain(InstanceData::get_attribute_descriptions().iter().cloned())
            .collect::<Vec<_>>();
        PipelineObject::new_opaque(
            device,
            &[
                Vertex::get_binding_description(),
                InstanceData::get_binding_description(),
            ],
            attrib_descriptions.as_slice(),
            &Path::new("shaders/spv/instancing.vert.spv"),
            &Path::new("shaders/spv/instancing.frag.spv"),
            swapchain.extent().width,
            swapchain.extent().height,
            render_pass,
            descriptor_set_layout,
        )
    }

    /// Framebufferを準備する。
    fn prepare_framebuffer(
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
    fn prepare_posteffect_uniform_buffers(
        allocator: Rc<vk_mem::Allocator>,
        posteffect_parameter_object: PostEffectParameterObject,
    ) -> Result<HostVisibleBufferObject> {
        let posteffect_uniform_buffer = HostVisibleBufferObject::new(
            Rc::clone(&allocator),
            &[posteffect_parameter_object],
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        Ok(posteffect_uniform_buffer)
    }

    /// RenderPassを準備する。
    fn prepare_posteffect_render_pass(
        device: Rc<Device>,
        format: vk::Format,
    ) -> Result<RenderPassObject> {
        RenderPassObject::new_posteffect(device, format)
    }

    /// DescriptorPoolを準備する。
    fn prepare_posteffect_descriptor_pool(
        device: Rc<Device>,
        count: u32,
        max_count: u32,
    ) -> Result<DescriptorPoolObject> {
        DescriptorPoolObject::new(
            device,
            &[
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(count)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(count)
                    .build(),
            ],
            max_count,
        )
    }

    /// DescriptorSetLayoutを準備する。
    fn prepare_posteffect_descriptor_set_layout(
        device: Rc<Device>,
    ) -> Result<DescriptorSetLayoutObject> {
        DescriptorSetLayoutObject::new(
            device,
            &[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_count(1)
                    .build(),
            ],
        )
    }

    /// Samplerを準備する。
    fn prepare_posteffect_sampler(device: Rc<Device>) -> Result<SamplerObject> {
        SamplerObject::new_clamp_to_edge(device)
    }

    /// DescriptorSetを準備する。
    fn prepare_posteffect_descriptor_sets(
        device: Rc<Device>,
        pool: &DescriptorPoolObject,
        layout: &DescriptorSetLayoutObject,
        render_texture: &TextureRenderingImageObject,
        sampler: &SamplerObject,
        posteffect_parameter_buffer: &HostVisibleBufferObject,
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
            let descriptor_image_infos = [vk::DescriptorImageInfo::builder()
                .image_view(render_texture.image_view())
                .sampler(sampler.sampler())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()];

            let descriptor_buffer_infos = [vk::DescriptorBufferInfo::builder()
                .buffer(posteffect_parameter_buffer.buffer())
                .offset(0)
                .range(std::mem::size_of::<PostEffectParameterObject>() as u64)
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&descriptor_image_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&descriptor_buffer_infos)
                    .build(),
            ];

            unsafe {
                device.update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }

        Ok(descriptor_sets)
    }

    /// Pipelineを準備する。
    fn prepare_posteffect_pipeline(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        render_pass: &RenderPassObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
    ) -> Result<PipelineObject> {
        PipelineObject::new_opaque_no_vertex(
            device,
            &Path::new("shaders/spv/posteffect.vert.spv"),
            &Path::new("shaders/spv/posteffect.frag.spv"),
            swapchain.extent().width,
            swapchain.extent().height,
            render_pass,
            descriptor_set_layout,
        )
    }

    /// Framebufferを準備する。
    fn prepare_posteffect_framebuffers(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        render_pass: &RenderPassObject,
    ) -> Result<Vec<FramebufferObject>> {
        let mut framebuffers = vec![];
        for i in 0..swapchain.len() {
            framebuffers.push(FramebufferObject::new_no_depth(
                Rc::clone(&device),
                *swapchain.get_image_view(i),
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
impl VulkanAppBaseBuilder for PostEffectAppBuilder {
    type Item = PostEffectApp;

    fn new() -> Self {
        Self {
            window_size: (800, 600),
            title: "PostEffect App".into(),
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

        // teapot
        let render_texture = TextureRenderingImageObject::new(
            base.device(),
            base.allocator(),
            base.swapchain().extent().width,
            base.swapchain().extent().height,
        )?;
        let (vertex_buffer, vertex_count) = Self::load_teapot_model(
            base.allocator(),
            base.device(),
            &base.graphics_command_pool(),
            &base.graphics_queue(),
        )?;
        let instance_data_buffer = Self::prepare_instance_data(
            base.device(),
            base.allocator(),
            &base.graphics_command_pool(),
            &base.graphics_queue(),
        )?;
        let mvp_matrices_object = MVPMatricesObject {
            model: Matrix4::from_scale(1.0),
            view: Matrix4::look_at(
                Point3::new(0.0, 15.0, 0.0),
                Point3::new(50.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: Matrix4::perspective(Deg(45.0), base.swapchain().aspect(), 0.01, 100.0),
        };
        let mvp_matrices_uniform_buffer =
            Self::prepare_mvp_matrices_uniform_buffer(base.allocator(), mvp_matrices_object)?;
        let instance_uniform_buffer = Self::prepare_instance_uniform_bufferf(base.allocator())?;
        let render_pass = Self::prepare_render_pass(base.device(), vk::Format::R8G8B8A8_UNORM)?;
        let descriptor_pool =
            Self::prepare_descriptor_pool(base.device(), 100, base.swapchain().len() as u32)?;
        let descriptor_set_layout = Self::prepare_descriptor_set_layout(base.device())?;
        let descriptor_set = Self::prepare_descriptor_set(
            base.device(),
            &descriptor_pool,
            &descriptor_set_layout,
            &mvp_matrices_uniform_buffer,
            &instance_uniform_buffer,
        )?;
        let framebuffer = Self::prepare_framebuffer(
            base.device(),
            &render_texture,
            &base.depth_buffer(),
            &render_pass,
        )?;
        let pipeline = Self::prepare_pipeline(
            base.device(),
            base.swapchain(),
            &render_pass,
            &descriptor_set_layout,
        )?;

        // PostEffect
        let posteffect_parameter_object = PostEffectParameterObject {
            window_size: Vector2::new(
                base.swapchain().extent().width as f32,
                base.swapchain().extent().height as f32,
            ),
            block_size: 5.0,
        };
        let posteffect_parameter_buffer = Self::prepare_posteffect_uniform_buffers(
            base.allocator(),
            posteffect_parameter_object,
        )?;
        let posteffect_descriptor_pool = Self::prepare_posteffect_descriptor_pool(
            base.device(),
            100,
            base.swapchain().len() as u32,
        )?;
        let posteffect_descriptor_set_layout =
            Self::prepare_posteffect_descriptor_set_layout(base.device())?;
        let sampler = Self::prepare_posteffect_sampler(base.device())?;
        let posteffect_descriptor_sets = Self::prepare_posteffect_descriptor_sets(
            base.device(),
            &posteffect_descriptor_pool,
            &posteffect_descriptor_set_layout,
            &render_texture,
            &sampler,
            &posteffect_parameter_buffer,
            base.swapchain().len(),
        )?;
        let posteffect_render_pass =
            Self::prepare_posteffect_render_pass(base.device(), base.swapchain().format())?;
        let posteffect_framebuffers = Self::prepare_posteffect_framebuffers(
            base.device(),
            base.swapchain(),
            &posteffect_render_pass,
        )?;
        let posteffect_pipeline = Self::prepare_posteffect_pipeline(
            base.device(),
            base.swapchain(),
            &posteffect_render_pass,
            &posteffect_descriptor_set_layout,
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

        let background_color = [0.5, 0.25, 0.25, 1.0];

        Ok(Self::Item {
            background_color,
            camera_rotate: 0.0,
            block_size: 1,

            imgui,
            platform,
            imgui_renderer,
            last_frame,
            imgui_framebuffers,
            imgui_render_pass,

            posteffect_pipeline,
            posteffect_framebuffers,
            posteffect_render_pass,
            posteffect_descriptor_sets,
            sampler,
            posteffect_descriptor_set_layout,
            posteffect_descriptor_pool,
            posteffect_parameter_buffer,
            posteffect_parameter_object,

            pipeline,
            framebuffer,
            descriptor_set,
            descriptor_set_layout,
            descriptor_pool,
            render_pass,
            instance_uniform_buffer,
            mvp_matrices_uniform_buffer,
            mvp_matrices_object,
            instance_data_buffer,
            vertex_buffer,
            vertex_count,
            render_texture,

            base,
        })
    }
}

pub struct PostEffectApp {
    background_color: [f32; 4],
    camera_rotate: f32,
    block_size: u32,

    imgui: Context,
    platform: WinitPlatform,
    imgui_renderer: Renderer,
    last_frame: Instant,
    imgui_framebuffers: Vec<FramebufferObject>,
    imgui_render_pass: RenderPassObject,

    posteffect_pipeline: PipelineObject,
    posteffect_framebuffers: Vec<FramebufferObject>,
    posteffect_render_pass: RenderPassObject,
    posteffect_descriptor_sets: Vec<vk::DescriptorSet>,
    sampler: SamplerObject,
    posteffect_descriptor_set_layout: DescriptorSetLayoutObject,
    posteffect_descriptor_pool: DescriptorPoolObject,
    posteffect_parameter_buffer: HostVisibleBufferObject,
    posteffect_parameter_object: PostEffectParameterObject,

    pipeline: PipelineObject,
    framebuffer: FramebufferObject,
    descriptor_set: vk::DescriptorSet,
    descriptor_set_layout: DescriptorSetLayoutObject,
    descriptor_pool: DescriptorPoolObject,
    render_pass: RenderPassObject,
    instance_uniform_buffer: HostVisibleBufferObject,
    mvp_matrices_uniform_buffer: HostVisibleBufferObject,
    mvp_matrices_object: MVPMatricesObject,
    instance_data_buffer: DeviceLocalBufferObject,
    vertex_buffer: DeviceLocalBufferObject,
    vertex_count: u32,
    render_texture: TextureRenderingImageObject,

    base: DefaultVulkanAppBase,
}
impl PostEffectApp {}
impl Drop for PostEffectApp {
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
impl VulkanAppBase for PostEffectApp {
    fn on_window_size_changed(&mut self, width: u32, height: u32) -> Result<()> {
        self.base.on_window_size_changed(width, height)?;

        // teapot
        self.render_texture = TextureRenderingImageObject::new(
            self.base.device(),
            self.base.allocator(),
            self.base.swapchain().extent().width,
            self.base.swapchain().extent().height,
        )?;
        self.render_pass = PostEffectAppBuilder::prepare_render_pass(
            self.base.device(),
            vk::Format::R8G8B8A8_UNORM,
        )?;
        self.descriptor_pool = PostEffectAppBuilder::prepare_descriptor_pool(
            self.base.device(),
            100,
            self.base.swapchain().len() as u32,
        )?;
        self.descriptor_set = PostEffectAppBuilder::prepare_descriptor_set(
            self.base.device(),
            &self.descriptor_pool,
            &self.descriptor_set_layout,
            &self.mvp_matrices_uniform_buffer,
            &self.instance_uniform_buffer,
        )?;
        self.framebuffer = PostEffectAppBuilder::prepare_framebuffer(
            self.base.device(),
            &self.render_texture,
            self.base.depth_buffer(),
            &self.render_pass,
        )?;
        self.pipeline = PostEffectAppBuilder::prepare_pipeline(
            self.base.device(),
            self.base.swapchain(),
            &self.render_pass,
            &self.descriptor_set_layout,
        )?;

        // posteffect
        self.posteffect_parameter_object = PostEffectParameterObject {
            window_size: Vector2::new(
                self.base.swapchain().extent().width as f32,
                self.base.swapchain().extent().height as f32,
            ),
            ..self.posteffect_parameter_object
        };
        self.posteffect_parameter_buffer
            .map_data(&[self.posteffect_parameter_object])?;
        self.posteffect_descriptor_pool = PostEffectAppBuilder::prepare_posteffect_descriptor_pool(
            self.base.device(),
            100,
            self.base.swapchain().len() as u32,
        )?;
        self.posteffect_descriptor_sets = PostEffectAppBuilder::prepare_posteffect_descriptor_sets(
            self.base.device(),
            &self.posteffect_descriptor_pool,
            &self.posteffect_descriptor_set_layout,
            &self.render_texture,
            &self.sampler,
            &self.posteffect_parameter_buffer,
            self.base.swapchain().len(),
        )?;
        self.posteffect_render_pass = PostEffectAppBuilder::prepare_posteffect_render_pass(
            self.base.device(),
            self.base.swapchain().format(),
        )?;
        self.posteffect_framebuffers = PostEffectAppBuilder::prepare_posteffect_framebuffers(
            self.base.device(),
            self.base.swapchain(),
            &self.posteffect_render_pass,
        )?;
        self.posteffect_pipeline = PostEffectAppBuilder::prepare_posteffect_pipeline(
            self.base.device(),
            self.base.swapchain(),
            &self.posteffect_render_pass,
            &self.posteffect_descriptor_set_layout,
        )?;

        // imguiリソースの作成
        let (imgui_render_pass, imgui_framebuffers) = PostEffectAppBuilder::prepare_imgui_resource(
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
        self.mvp_matrices_object = MVPMatricesObject {
            proj: Matrix4::perspective(Deg(45.0), self.base.swapchain().aspect(), 0.01, 100.0),
            ..self.mvp_matrices_object
        };
        self.mvp_matrices_uniform_buffer
            .map_data(&[self.mvp_matrices_object])?;

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

        let color = &mut self.background_color;
        let rotate = &mut self.camera_rotate;
        let block = &mut self.block_size;
        let max_block_size = 50;

        // UIの構築
        let ui = self.imgui.frame();
        imgui::Window::new(im_str!("Hello World"))
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(&ui, || {
                ColorPicker::new(im_str!("Teapot Background color"), color).build(&ui);
                Slider::new(im_str!("Camera Rotate"), -180.0..=180.0).build(&ui, rotate);

                ui.separator();

                Slider::new(im_str!("Block Size"), 1..=max_block_size).build(&ui, block);
            });
        self.platform.prepare_render(&ui, &window);
        let draw_data = ui.render();

        // コマンドの構築
        let command = self.base.command_buffers()[image_index];
        unsafe {
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::builder())?;
        }

        {
            // クリア値
            let clear_value = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: self.background_color,
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
                view: Matrix4::look_at(
                    Point3::new(0.0, 15.0, 0.0),
                    Point3::from_homogeneous(
                        Matrix4::from_angle_y(Deg(self.camera_rotate))
                            * Vector4::new(50.0, 0.0, 0.0, 1.0),
                    ),
                    Vector3::new(0.0, 1.0, 0.0),
                ),
                ..self.mvp_matrices_object
            };
            self.mvp_matrices_uniform_buffer
                .map_data(&[uniform_buffer_object])?;

            unsafe {
                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::builder()
                        .render_pass(self.render_pass.render_pass())
                        .framebuffer(self.framebuffer.framebuffer())
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
                    self.pipeline.pipeline(),
                );
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline_layout(),
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                device.cmd_bind_vertex_buffers(
                    command,
                    0,
                    &[
                        self.vertex_buffer.buffer(),
                        self.instance_data_buffer.buffer(),
                    ],
                    &[0, 0],
                );
                device.cmd_draw(command, self.vertex_count, 800, 0, 0);
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
                    .image(self.render_texture.image())
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

        // PostEffect
        {
            // クリア値
            let clear_value = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            }];

            // Uniform Bufferの更新
            let uniform_buffer_object = PostEffectParameterObject {
                block_size: self.block_size as f32,
                ..self.posteffect_parameter_object
            };
            self.posteffect_parameter_buffer
                .map_data(&[uniform_buffer_object])?;

            unsafe {
                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::builder()
                        .render_pass(self.posteffect_render_pass.render_pass())
                        .framebuffer(self.posteffect_framebuffers[image_index].framebuffer())
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
                    self.posteffect_pipeline.pipeline(),
                );
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.posteffect_pipeline.pipeline_layout(),
                    0,
                    &[self.posteffect_descriptor_sets[image_index]],
                    &[],
                );
                device.cmd_draw(command, 3, 1, 0, 0);
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
