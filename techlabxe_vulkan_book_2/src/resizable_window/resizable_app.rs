#![allow(dead_code)]

use std::{path::Path, rc::Rc};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};
use cgmath::{Deg, Matrix4, Point3, Vector3};
use memoffset::offset_of;
use winit::{dpi::PhysicalSize, window::Window};

use crate::common::{
    cgmath_ext::Matrix4Ext,
    default_vulkan_app_base::{DefaultVulkanAppBase, DefaultVulkanAppBaseBuilder},
    vulkan_app_base::{VulkanAppBase, VulkanAppBaseBuilder},
    vulkan_objects::{
        DepthImageObject, DescriptorPoolObject, DescriptorSetLayoutObject, DeviceLocalBufferObject,
        FramebufferObject, HostVisibleBufferObject, PipelineObject, RenderPassObject,
        SwapchainObject,
    },
};
#[repr(C)]
#[derive(Debug)]
struct Vertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
}
impl Vertex {
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
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

pub struct ResizableAppBuilder {
    window_size: (u32, u32),
    title: String,
    base: DefaultVulkanAppBaseBuilder,
}
impl ResizableAppBuilder {
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
    fn load_model(
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

    /// UniformBufferを準備する。
    fn prepare_uniform_buffer(
        allocator: Rc<vk_mem::Allocator>,
        aspect: f32,
        count: u32,
    ) -> Result<Vec<HostVisibleBufferObject>> {
        let unitorm_buffer_object = UniformBufferObject {
            model: Matrix4::from_scale(1.0),
            view: Matrix4::look_at(
                Point3::new(0.0, 5.0, 10.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: Matrix4::perspective(Deg(45.0), aspect, 0.01, 100.0),
        };

        let mut uniform_buffers = vec![];
        for _ in 0..count {
            uniform_buffers.push(HostVisibleBufferObject::new(
                Rc::clone(&allocator),
                &[unitorm_buffer_object],
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?);
        }

        Ok(uniform_buffers)
    }

    /// RenderPassを準備する。
    fn prepare_render_pass(device: Rc<Device>, format: vk::Format) -> Result<RenderPassObject> {
        RenderPassObject::new(device, format)
    }

    /// DescriptorPoolを準備する。
    fn prepare_descriptor_pool(
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
    fn prepare_descriptor_set_layout(device: Rc<Device>) -> Result<DescriptorSetLayoutObject> {
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
    fn prepare_descriptor_sets(
        device: Rc<Device>,
        pool: &DescriptorPoolObject,
        layout: &DescriptorSetLayoutObject,
        uniform_buffers: &Vec<HostVisibleBufferObject>,
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
                .buffer(uniform_buffers[i].buffer())
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as u64)
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

    /// Framebufferを準備する。
    fn prepare_framebuffers(
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

    /// Pipelineを準備する。
    fn prepare_pipeline(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        render_pass: &RenderPassObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
    ) -> Result<PipelineObject> {
        PipelineObject::new_opaque(
            device,
            &Vertex::get_binding_descriptions(),
            &Vertex::get_attribute_descriptions(),
            &Path::new("shaders/spv/resizable.vert.spv"),
            &Path::new("shaders/spv/resizable.frag.spv"),
            swapchain.extent().width,
            swapchain.extent().height,
            render_pass,
            descriptor_set_layout,
        )
    }
}
impl VulkanAppBaseBuilder for ResizableAppBuilder {
    type Item = ResizableApp;

    fn new() -> Self {
        Self {
            window_size: (800, 600),
            title: "Resizable Window App".into(),
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
        let (vertex_buffer, vertex_count) = Self::load_model(
            base.allocator(),
            base.device(),
            &base.graphics_command_pool(),
            &base.graphics_queue(),
        )?;
        let uniform_buffers = Self::prepare_uniform_buffer(
            base.allocator(),
            base.swapchain().aspect(),
            base.swapchain().len() as u32,
        )?;
        let render_pass = Self::prepare_render_pass(base.device(), base.swapchain().format())?;
        let descriptor_pool =
            Self::prepare_descriptor_pool(base.device(), 100, base.swapchain().len() as u32)?;
        let descriptor_set_layout = Self::prepare_descriptor_set_layout(base.device())?;
        let descriptor_sets = Self::prepare_descriptor_sets(
            base.device(),
            &descriptor_pool,
            &descriptor_set_layout,
            &uniform_buffers,
            base.swapchain().len(),
        )?;
        let framebuffers = Self::prepare_framebuffers(
            base.device(),
            base.swapchain(),
            base.depth_buffer(),
            &render_pass,
        )?;
        let pipeline = Self::prepare_pipeline(
            base.device(),
            base.swapchain(),
            &render_pass,
            &descriptor_set_layout,
        )?;

        Ok(Self::Item {
            pipeline,
            framebuffers,
            descriptor_sets,
            descriptor_set_layout,
            descriptor_pool,
            render_pass,
            uniform_buffers,
            vertex_buffer,
            vertex_count,
            base,
        })
    }
}

pub struct ResizableApp {
    pipeline: PipelineObject,
    framebuffers: Vec<FramebufferObject>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_set_layout: DescriptorSetLayoutObject,
    descriptor_pool: DescriptorPoolObject,
    render_pass: RenderPassObject,
    uniform_buffers: Vec<HostVisibleBufferObject>,
    vertex_buffer: DeviceLocalBufferObject,
    vertex_count: u32,
    base: DefaultVulkanAppBase,
}
impl ResizableApp {}
impl Drop for ResizableApp {
    fn drop(&mut self) {
        unsafe {
            self.base
                .device()
                .device_wait_idle()
                .expect("Failed to wait device idle");
        }
    }
}
impl VulkanAppBase for ResizableApp {
    fn on_window_size_changed(&mut self, width: u32, height: u32) -> Result<()> {
        self.base.on_window_size_changed(width, height)?;

        self.render_pass = ResizableAppBuilder::prepare_render_pass(
            self.base.device(),
            self.base.swapchain().format(),
        )?;
        self.descriptor_pool = ResizableAppBuilder::prepare_descriptor_pool(
            self.base.device(),
            100,
            self.base.swapchain().len() as u32,
        )?;
        self.descriptor_sets = ResizableAppBuilder::prepare_descriptor_sets(
            self.base.device(),
            &self.descriptor_pool,
            &self.descriptor_set_layout,
            &self.uniform_buffers,
            self.base.swapchain().len(),
        )?;
        self.framebuffers = ResizableAppBuilder::prepare_framebuffers(
            self.base.device(),
            self.base.swapchain(),
            self.base.depth_buffer(),
            &self.render_pass,
        )?;
        self.pipeline = ResizableAppBuilder::prepare_pipeline(
            self.base.device(),
            self.base.swapchain(),
            &self.render_pass,
            &self.descriptor_set_layout,
        )?;

        let unitorm_buffer_object = UniformBufferObject {
            model: Matrix4::from_scale(1.0),
            view: Matrix4::look_at(
                Point3::new(0.0, 5.0, 10.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: Matrix4::perspective(Deg(45.0), self.base.swapchain().aspect(), 0.01, 100.0),
        };
        for uniform_buffer in self.uniform_buffers.iter() {
            uniform_buffer.map_data(&[unitorm_buffer_object])?;
        }

        Ok(())
    }

    fn prepare(&mut self) -> Result<()> {
        self.base.prepare()?;
        Ok(())
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

        // クリア値
        let clear_value = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.5, 0.25, 0.25, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        // コマンドの構築
        unsafe {
            let command = self.base.command_buffers()[image_index];

            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::builder())?;

            device.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass.render_pass())
                    .framebuffer(self.framebuffers[image_index].framebuffer())
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

            device.cmd_bind_vertex_buffers(command, 0, &[self.vertex_buffer.buffer()], &[0]);

            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout(),
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );

            device.cmd_draw(command, self.vertex_count, 1, 0, 0);

            device.cmd_end_render_pass(command);

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
