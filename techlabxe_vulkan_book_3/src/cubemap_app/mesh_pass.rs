use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};
use cgmath::Matrix4;
use memoffset::offset_of;

use crate::{
    common::Matrix4Ext,
    cubemap_app::{camera::CameraUniformObject, mesh::MeshVertex, scene::Scene},
    vk_wrapper_object::{
        DescriptorPoolObject, DescriptorSetLayoutObject, FramebufferObject, ImageObject,
        PipelineObject, RenderPassObject, SamplerObject, SwapchainObject,
    },
};

pub struct MeshPass {
    device: Rc<Device>,
    render_pass: RenderPassObject,
    _sampler: SamplerObject,
    _descriptor_pool: DescriptorPoolObject,
    descriptor_set_layout: DescriptorSetLayoutObject,
    descriptor_set: vk::DescriptorSet,
    pipeline: PipelineObject,
    framebuffers: Vec<FramebufferObject>,
    width: u32,
    height: u32,
}
impl MeshPass {
    pub fn new(
        device: Rc<Device>,
        swapchain: &SwapchainObject,
        depth_buffer: &ImageObject,
        scene: &Scene,
    ) -> Result<Self> {
        let device_ref = device.as_ref();

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
            RenderPassObject::new(Rc::clone(&device), render_pass_create_info)?
        };

        let sampler = {
            let sampler_create_info = &vk::SamplerCreateInfo::builder()
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
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
            SamplerObject::new(Rc::clone(&device), sampler_create_info)?
        };

        let descriptor_pool = {
            let pool_sizes = &[
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .build(),
            ];
            DescriptorPoolObject::new(Rc::clone(&device), pool_sizes, 1)?
        };

        let descriptor_set_layout = {
            let bindings = &[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
            ];
            DescriptorSetLayoutObject::new(Rc::clone(&device), bindings)?
        };

        let descriptor_set = {
            let layouts = &[descriptor_set_layout.descriptor_set_layout()];
            let descriptor_set = unsafe {
                device_ref.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool.descriptor_pool())
                        .set_layouts(layouts),
                )?
            }[0];

            let descriptor_camera_uniform_buffer_infos = [vk::DescriptorBufferInfo::builder()
                .buffer(scene.camera().uniform().buffer())
                .offset(0)
                .range(std::mem::size_of::<CameraUniformObject>() as u64)
                .build()];
            let descriptor_cubemap_infos = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(scene.cubemap().view())
                .sampler(sampler.sampler())
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&descriptor_camera_uniform_buffer_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&descriptor_cubemap_infos)
                    .build(),
            ];

            unsafe {
                device_ref.update_descriptor_sets(&write_descriptor_sets, &[]);
            }

            descriptor_set
        };

        let pipeline = {
            let vertex_attribute_descriptions = [
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(MeshVertex, position) as u32)
                    .build(),
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(MeshVertex, normal) as u32)
                    .build(),
            ];
            let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(std::mem::size_of::<MeshVertex>() as u32)
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
                Rc::clone(&device),
                "shaders/spv/mesh_pass.vert.spv",
                "shaders/spv/mesh_pass.frag.spv",
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
                    Rc::clone(&device),
                    &render_pass,
                    &[swapchain.get_image_view(i), depth_buffer.view()],
                    swapchain.extent().width,
                    swapchain.extent().height,
                )?);
            }
            buffers
        };

        Ok(Self {
            device: device,
            render_pass,
            _sampler: sampler,
            _descriptor_pool: descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
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
                    .offset(offset_of!(MeshVertex, position) as u32)
                    .build(),
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(MeshVertex, normal) as u32)
                    .build(),
            ];
            let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(std::mem::size_of::<MeshVertex>() as u32)
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
                "shaders/spv/mesh_pass.vert.spv",
                "shaders/spv/mesh_pass.frag.spv",
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

        self.width = swapchain.extent().width;
        self.height = swapchain.extent().height;

        Ok(())
    }

    pub fn cmd_draw(&self, command: vk::CommandBuffer, image_index: usize, scene: &Scene) {
        let device = self.device.as_ref();
        unsafe {
            // クリア値
            let clear_value = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
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
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline_layout(),
                0,
                &[self.descriptor_set],
                &[],
            );

            for entity in scene.entities().iter() {
                let model_matrix = entity.get_model_matrix();
                let mesh = entity.get_mesh();
                device.cmd_push_constants(
                    command,
                    self.pipeline.pipeline_layout(),
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    model_matrix.to_u8_slice(),
                );
                device.cmd_bind_vertex_buffers(command, 0, &[mesh.buffer().buffer()], &[0]);
                device.cmd_draw(command, mesh.vertex_count(), 1, 0, 0);
            }
            device.cmd_end_render_pass(command);
        }
    }
}
