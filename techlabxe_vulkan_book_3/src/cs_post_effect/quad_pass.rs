use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};
use cgmath::{Deg, Matrix4, Point3, Vector2, Vector3};
use crevice::std140::{AsStd140, Std140};
use memoffset::offset_of;

use crate::{
    common::Matrix4Ext,
    vk_wrapper_object::{
        CommandPoolObject, DescriptorPoolObject, DescriptorSetLayoutObject,
        DeviceLocalBufferObject, DeviceObject, FramebufferObject, HostVisibleBufferObject,
        ImageObject, PipelineObject, RenderPassObject, SamplerObject, SwapchainObject,
        VkMemAllocatorObject,
    },
};

#[repr(C)]
struct QuadVertex {
    position: Vector3<f32>,
    uv: Vector2<f32>,
}

#[derive(AsStd140)]
struct UniformBufferObject {
    view: mint::ColumnMatrix4<f32>,
    proj: mint::ColumnMatrix4<f32>,
}

pub struct QuadPass {
    device: Rc<Device>,
    quad_vertex_buffer: DeviceLocalBufferObject,
    render_pass: RenderPassObject,
    _sampler: SamplerObject,
    uniform_buffer: HostVisibleBufferObject,
    _descriptor_pool: DescriptorPoolObject,
    descriptor_set_layout: DescriptorSetLayoutObject,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline: PipelineObject,
    framebuffers: Vec<FramebufferObject>,
    width: u32,
    height: u32,
}
impl QuadPass {
    pub fn new(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        graphics_command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        swapchain: &SwapchainObject,
        depth_buffer: &ImageObject,
        src_image: &ImageObject,
        dst_image: &ImageObject,
    ) -> Result<Self> {
        let quad_vertices = [
            QuadVertex {
                position: Vector3::new(-3.0, -2.0, 0.0),
                uv: Vector2::new(0.0, 0.0),
            },
            QuadVertex {
                position: Vector3::new(-3.0, 2.0, 0.0),
                uv: Vector2::new(0.0, 1.0),
            },
            QuadVertex {
                position: Vector3::new(3.0, 2.0, 0.0),
                uv: Vector2::new(1.0, 1.0),
            },
            QuadVertex {
                position: Vector3::new(-3.0, -2.0, 0.0),
                uv: Vector2::new(0.0, 0.0),
            },
            QuadVertex {
                position: Vector3::new(3.0, 2.0, 0.0),
                uv: Vector2::new(1.0, 1.0),
            },
            QuadVertex {
                position: Vector3::new(3.0, -2.0, 0.0),
                uv: Vector2::new(1.0, 0.0),
            },
        ];
        let quad_vertex_buffer = DeviceLocalBufferObject::new(
            device,
            allocator,
            graphics_command_pool,
            graphics_queue,
            &quad_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        let render_pass = {
            // Attachments
            let attachments = [
                vk::AttachmentDescription::builder()
                    .format(swapchain.format())
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .build(),
                vk::AttachmentDescription::builder()
                    .format(vk::Format::D32_SFLOAT_S8_UINT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
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
            let render_pass_create_info = &vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);
            RenderPassObject::new(device.device(), render_pass_create_info)?
        };

        let sampler = {
            let sampler_create_info = &vk::SamplerCreateInfo::builder()
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(1.0)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .unnormalized_coordinates(false);
            SamplerObject::new(device.device(), sampler_create_info)?
        };

        let uniform_buffer = {
            let view: [[f32; 4]; 4] = Matrix4::look_at(
                Point3::new(0.0, 0.0, 15.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            )
            .into();
            let proj: [[f32; 4]; 4] = Matrix4::perspective(
                Deg(45.0),
                src_image.width() as f32 / src_image.height() as f32,
                0.01,
                100.0,
            )
            .into();
            let data = UniformBufferObject {
                view: view.into(),
                proj: proj.into(),
            }
            .as_std140();
            HostVisibleBufferObject::new(
                allocator,
                data.as_bytes(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?
        };

        let descriptor_pool = {
            let pool_sizes = &[
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(2)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(2)
                    .build(),
            ];
            DescriptorPoolObject::new(device.device(), pool_sizes, 2)?
        };

        let descriptor_set_layout = {
            let bindings = &[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
            ];
            DescriptorSetLayoutObject::new(device.device(), bindings)?
        };

        let descriptor_sets = {
            let layouts = &[
                descriptor_set_layout.descriptor_set_layout(),
                descriptor_set_layout.descriptor_set_layout(),
            ];
            let descriptor_sets = unsafe {
                device.device().allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool.descriptor_pool())
                        .set_layouts(layouts),
                )?
            };

            let descriptor_uniform_buffer_infos = [vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffer.buffer())
                .offset(0)
                .range(uniform_buffer.buffer_size())
                .build()];
            let descriptor_image_infos = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(src_image.view())
                .sampler(sampler.sampler())
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&descriptor_uniform_buffer_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[0])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&descriptor_image_infos)
                    .build(),
            ];

            unsafe {
                device
                    .device()
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }

            let descriptor_image_infos = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(dst_image.view())
                .sampler(sampler.sampler())
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[1])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&descriptor_uniform_buffer_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_sets[1])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&descriptor_image_infos)
                    .build(),
            ];

            unsafe {
                device
                    .device()
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }

            descriptor_sets
        };

        let pipeline = {
            let vertex_attribute_descriptions = [
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(QuadVertex, position) as u32)
                    .build(),
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(QuadVertex, uv) as u32)
                    .build(),
            ];
            let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(std::mem::size_of::<QuadVertex>() as u32)
                .build()];
            let vertex_input_state_create_info = &vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&vertex_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_binding_descriptions);

            let input_assembly_state_create_info =
                &vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let viewports = &[vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(swapchain.extent().width as f32)
                .height(swapchain.extent().height as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .build()];
            let scissors = &[vk::Rect2D::builder()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(swapchain.extent())
                .build()];
            let viewport_state_create_info = &vk::PipelineViewportStateCreateInfo::builder()
                .viewports(viewports)
                .scissors(scissors);

            let rasterization_state_create_info =
                &vk::PipelineRasterizationStateCreateInfo::builder()
                    .polygon_mode(vk::PolygonMode::FILL)
                    .cull_mode(vk::CullModeFlags::NONE)
                    .line_width(1.0);

            let multisample_state_create_info = &vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let color_blend_attachments = &[vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(false)
                .build()];
            let color_blend_state_create_info = &vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .attachments(color_blend_attachments);

            let stencil_op = vk::StencilOpState::builder()
                .reference(1)
                .write_mask(1)
                .compare_op(vk::CompareOp::ALWAYS)
                .depth_fail_op(vk::StencilOp::KEEP)
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::REPLACE)
                .build();
            let depth_stencil_state_create_info =
                &vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS)
                    .stencil_test_enable(true)
                    .front(stencil_op)
                    .back(stencil_op);

            let push_constant_range = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(std::mem::size_of::<Matrix4<f32>>() as u32)
                .build();

            PipelineObject::new(
                device.device(),
                "shaders/spv/quad_pass.vert.spv",
                "shaders/spv/quad_pass.frag.spv",
                vertex_input_state_create_info,
                input_assembly_state_create_info,
                viewport_state_create_info,
                rasterization_state_create_info,
                multisample_state_create_info,
                color_blend_state_create_info,
                depth_stencil_state_create_info,
                Some(push_constant_range),
                &render_pass,
                &descriptor_set_layout,
            )?
        };

        let framebuffers = {
            let mut buffers = vec![];
            for i in 0..swapchain.len() {
                buffers.push(FramebufferObject::new(
                    device.device(),
                    &render_pass,
                    &[swapchain.get_image_view(i), depth_buffer.view()],
                    swapchain.extent().width,
                    swapchain.extent().height,
                )?);
            }
            buffers
        };

        Ok(Self {
            device: device.device(),
            quad_vertex_buffer,
            render_pass,
            _sampler: sampler,
            uniform_buffer,
            _descriptor_pool: descriptor_pool,
            descriptor_set_layout,
            descriptor_sets,
            pipeline,
            framebuffers,
            width: swapchain.extent().width,
            height: swapchain.extent().height,
        })
    }

    pub fn resize(
        &mut self,
        swapchain: &SwapchainObject,
        depth_buffer: &ImageObject,
    ) -> Result<()> {
        self.render_pass = {
            // Attachments
            let attachments = [
                vk::AttachmentDescription::builder()
                    .format(swapchain.format())
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .build(),
                vk::AttachmentDescription::builder()
                    .format(vk::Format::D32_SFLOAT_S8_UINT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
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
            let render_pass_create_info = &vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);
            RenderPassObject::new(Rc::clone(&self.device), render_pass_create_info)?
        };

        self.pipeline = {
            let vertex_attribute_descriptions = [
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(QuadVertex, position) as u32)
                    .build(),
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(QuadVertex, uv) as u32)
                    .build(),
            ];
            let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(std::mem::size_of::<QuadVertex>() as u32)
                .build()];
            let vertex_input_state_create_info = &vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&vertex_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_binding_descriptions);

            let input_assembly_state_create_info =
                &vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let viewports = &[vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(swapchain.extent().width as f32)
                .height(swapchain.extent().height as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .build()];
            let scissors = &[vk::Rect2D::builder()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(swapchain.extent())
                .build()];
            let viewport_state_create_info = &vk::PipelineViewportStateCreateInfo::builder()
                .viewports(viewports)
                .scissors(scissors);

            let rasterization_state_create_info =
                &vk::PipelineRasterizationStateCreateInfo::builder()
                    .polygon_mode(vk::PolygonMode::FILL)
                    .cull_mode(vk::CullModeFlags::NONE)
                    .line_width(1.0);

            let multisample_state_create_info = &vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let color_blend_attachments = &[vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(false)
                .build()];
            let color_blend_state_create_info = &vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .attachments(color_blend_attachments);

            let stencil_op = vk::StencilOpState::builder()
                .reference(1)
                .write_mask(1)
                .compare_op(vk::CompareOp::ALWAYS)
                .depth_fail_op(vk::StencilOp::KEEP)
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::REPLACE)
                .build();
            let depth_stencil_state_create_info =
                &vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS)
                    .stencil_test_enable(true)
                    .front(stencil_op)
                    .back(stencil_op);

            let push_constant_range = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(std::mem::size_of::<Matrix4<f32>>() as u32)
                .build();

            PipelineObject::new(
                Rc::clone(&self.device),
                "shaders/spv/quad_pass.vert.spv",
                "shaders/spv/quad_pass.frag.spv",
                vertex_input_state_create_info,
                input_assembly_state_create_info,
                viewport_state_create_info,
                rasterization_state_create_info,
                multisample_state_create_info,
                color_blend_state_create_info,
                depth_stencil_state_create_info,
                Some(push_constant_range),
                &self.render_pass,
                &self.descriptor_set_layout,
            )?
        };

        self.framebuffers = {
            let mut buffers = vec![];
            for i in 0..swapchain.len() {
                buffers.push(FramebufferObject::new(
                    Rc::clone(&self.device),
                    &self.render_pass,
                    &[swapchain.get_image_view(i), depth_buffer.view()],
                    swapchain.extent().width,
                    swapchain.extent().height,
                )?);
            }
            buffers
        };

        {
            let view: [[f32; 4]; 4] = Matrix4::look_at(
                Point3::new(0.0, 0.0, 15.0),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            )
            .into();
            let proj: [[f32; 4]; 4] =
                Matrix4::perspective(Deg(45.0), swapchain.aspect(), 0.01, 100.0).into();
            let data = UniformBufferObject {
                view: view.into(),
                proj: proj.into(),
            }
            .as_std140();
            self.uniform_buffer.map_data(&[data])?;
        };

        self.width = swapchain.extent().width;
        self.height = swapchain.extent().height;

        Ok(())
    }

    pub fn cmd_draw(&self, command: vk::CommandBuffer, image_index: usize) {
        let device = self.device.as_ref();
        unsafe {
            // クリア値
            let clear_value = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 0.6, 0.6, 0.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            device.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass.render_pass())
                    .framebuffer(self.framebuffers[image_index].framebuffer())
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.width,
                            height: self.height,
                        },
                    })
                    .clear_values(&clear_value),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline(),
            );

            // 左のquad
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout(),
                0,
                &[self.descriptor_sets[0]],
                &[],
            );
            device.cmd_push_constants(
                command,
                self.pipeline.pipeline_layout(),
                vk::ShaderStageFlags::VERTEX,
                0,
                Matrix4::from_translation(Vector3::new(-4.0, 0.0, 0.0)).to_u8_slice(),
            );
            device.cmd_bind_vertex_buffers(command, 0, &[self.quad_vertex_buffer.buffer()], &[0]);
            device.cmd_draw(command, 6, 1, 0, 0);

            // 右のquad
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout(),
                0,
                &[self.descriptor_sets[1]],
                &[],
            );
            device.cmd_push_constants(
                command,
                self.pipeline.pipeline_layout(),
                vk::ShaderStageFlags::VERTEX,
                0,
                Matrix4::from_translation(Vector3::new(4.0, 0.0, 0.0)).to_u8_slice(),
            );
            device.cmd_bind_vertex_buffers(command, 0, &[self.quad_vertex_buffer.buffer()], &[0]);
            device.cmd_draw(command, 6, 1, 0, 0);

            device.cmd_end_render_pass(command);
        }
    }
}
