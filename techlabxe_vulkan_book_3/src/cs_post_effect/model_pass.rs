use std::{path::Path, rc::Rc};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};
use cgmath::{prelude::*, Deg, Matrix4, Point3, Vector2, Vector3};
use crevice::std140::{AsStd140, Std140};
use gltf::{image::Source, material::AlphaMode};
use image::{
    GenericImageView,
    ImageFormat::{Jpeg, Png},
};
use memoffset::offset_of;

use crate::{
    common::Matrix4Ext,
    vk_wrapper_object::{
        CommandPoolObject, DescriptorPoolObject, DescriptorSetLayoutObject,
        DeviceLocalBufferObject, DeviceObject, FramebufferObject, HostVisibleBufferObject,
        ImageObject, PipelineObject, RenderPassObject, SamplerObject, VkMemAllocatorObject,
    },
};

#[repr(C)]
struct ModelVertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
    uv: Vector2<f32>,
}

#[derive(AsStd140)]
struct UniformBufferObject {
    world: mint::ColumnMatrix4<f32>,
    view: mint::ColumnMatrix4<f32>,
    proj: mint::ColumnMatrix4<f32>,
}

struct ModelMeshBuilder {
    vertex_buffer: DeviceLocalBufferObject,
    index_buffer: DeviceLocalBufferObject,
    vertex_count: u32,
    index_count: u32,
    material_index: usize,
}
impl ModelMeshBuilder {
    fn build(self, descriptor_set: vk::DescriptorSet) -> ModelMesh {
        ModelMesh {
            vertex_buffer: self.vertex_buffer,
            index_buffer: self.index_buffer,
            _vertex_count: self.vertex_count,
            index_count: self.index_count,
            material_index: self.material_index,
            descriptor_set,
        }
    }
}
struct ModelMesh {
    vertex_buffer: DeviceLocalBufferObject,
    index_buffer: DeviceLocalBufferObject,
    _vertex_count: u32,
    index_count: u32,
    material_index: usize,
    descriptor_set: vk::DescriptorSet,
}
struct ModelMaterial {
    texture: ImageObject,
    alpha_mode: AlphaMode,
}

pub struct ModelPass {
    device: Rc<Device>,
    meshes: Vec<ModelMesh>,
    materials: Vec<ModelMaterial>,
    _uniform_buffer: HostVisibleBufferObject,
    render_pass: RenderPassObject,
    _sampler: SamplerObject,
    _descriptor_pool: DescriptorPoolObject,
    _descriptor_set_layout: DescriptorSetLayoutObject,
    pipeline_opaque: PipelineObject,
    pipeline_transparent: PipelineObject,
    framebuffer: FramebufferObject,
    width: u32,
    height: u32,
}
impl ModelPass {
    pub fn new(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        graphics_command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        image: &ImageObject,
        depth_image: &ImageObject,
        model_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let (mesh_builders, materials) = {
            let mut mesh_builders = vec![];
            let mut materials = vec![];

            let (document, buffers, _images) = gltf::import(model_path)?;

            for mesh in document.meshes() {
                for prim in mesh.primitives() {
                    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                    let positions = reader
                        .read_positions()
                        .expect("the mesh needs position attributes")
                        .collect::<Vec<_>>();

                    let normals = reader
                        .read_normals()
                        .expect("the mesh needs normal attributes")
                        .collect::<Vec<_>>();

                    let uvs = reader
                        .read_tex_coords(0)
                        .expect("the mesh needs uv0 attributes")
                        .into_f32()
                        .collect::<Vec<_>>();

                    assert!(positions.len() == normals.len());
                    assert!(positions.len() == uvs.len());
                    assert!(normals.len() == uvs.len());

                    let mut vertices = vec![];
                    for i in 0..positions.len() {
                        vertices.push(ModelVertex {
                            position: Vector3::from(positions[i]),
                            normal: Vector3::from(normals[i]),
                            uv: Vector2::from(uvs[i]),
                        });
                    }

                    let vertex_buffer = DeviceLocalBufferObject::new(
                        device,
                        allocator,
                        graphics_command_pool,
                        graphics_queue,
                        vertices.as_slice(),
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    )?;
                    let vertex_count = vertices.len() as u32;

                    let indices = if let Some(indices) = reader.read_indices() {
                        indices.into_u32().collect::<Vec<_>>()
                    } else {
                        (0..positions.len()).map(|i| i as u32).collect::<Vec<_>>()
                    };

                    let index_buffer = DeviceLocalBufferObject::new(
                        device,
                        allocator,
                        graphics_command_pool,
                        graphics_queue,
                        indices.as_slice(),
                        vk::BufferUsageFlags::INDEX_BUFFER,
                    )?;
                    let index_count = indices.len() as u32;

                    let material_index = prim.material().index().expect(
                        "some mesh does not assign a material. \
                    this app does not support default material.",
                    );

                    mesh_builders.push(ModelMeshBuilder {
                        vertex_buffer,
                        index_buffer,
                        vertex_count,
                        index_count,
                        material_index,
                    });
                }

                for material in document.materials() {
                    let alpha_mode = material.alpha_mode();

                    let texture = material
                        .pbr_metallic_roughness()
                        .base_color_texture()
                        .expect(
                            "No base color texture.\
            this app does not support model which has no base color texture",
                        )
                        .texture()
                        .source()
                        .source();
                    let image = match texture {
                        Source::View { view, mime_type } => {
                            let parent_buffer_data = &buffers[view.buffer().index()].0;
                            let begin = view.offset();
                            let end = begin + view.length();
                            let data = &parent_buffer_data[begin..end];
                            match mime_type {
                                "image/jpeg" => image::load_from_memory_with_format(data, Jpeg)?,
                                "image/png" => image::load_from_memory_with_format(data, Png)?,
                                _ => panic!("Not supported type texture"),
                            }
                        }
                        Source::Uri { .. } => panic!("Not supported uri texture"),
                    };
                    let texture = ImageObject::new_with_data(
                        device,
                        allocator,
                        graphics_command_pool,
                        graphics_queue,
                        image.width(),
                        image.height(),
                        image.to_rgba().into_raw().as_slice(),
                    )?;

                    materials.push(ModelMaterial {
                        texture,
                        alpha_mode,
                    });
                }
            }

            (mesh_builders, materials)
        };

        let uniform_buffer = {
            let world: [[f32; 4]; 4] = Matrix4::identity().into();
            let view: [[f32; 4]; 4] = Matrix4::look_at(
                Point3::new(0.0, 2.0, -2.0),
                Point3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            )
            .into();
            let proj: [[f32; 4]; 4] = Matrix4::perspective(
                Deg(45.0),
                image.width() as f32 / image.height() as f32,
                0.01,
                100.0,
            )
            .into();
            let data = UniformBufferObject {
                world: world.into(),
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

        let render_pass = {
            // Attachments
            let attachments = [
                vk::AttachmentDescription::builder()
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
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

        let descriptor_pool = {
            let count = mesh_builders.len() as u32;
            let pool_sizes = &[
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(count)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(count)
                    .build(),
            ];
            DescriptorPoolObject::new(device.device(), pool_sizes, count)?
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

        let meshes = {
            let mut descriptor_sets = vec![];
            let layouts = &[descriptor_set_layout.descriptor_set_layout()];
            for index in 0..mesh_builders.len() {
                // ディスクリプタセットの確保
                let descriptor_set = unsafe {
                    device.device().allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::builder()
                            .descriptor_pool(descriptor_pool.descriptor_pool())
                            .set_layouts(layouts),
                    )?
                }[0];

                // ディスクリプタセットへの書き込み
                let material = &materials[mesh_builders[index].material_index];
                let descripto_buffer_infos = [vk::DescriptorBufferInfo::builder()
                    .buffer(uniform_buffer.buffer())
                    .offset(0)
                    .range(uniform_buffer.buffer_size())
                    .build()];

                let descriptor_image_infos = [vk::DescriptorImageInfo::builder()
                    .image_view(material.texture.view())
                    .sampler(sampler.sampler())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build()];

                let write_descriptor_sets = [
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&descripto_buffer_infos)
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
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

                descriptor_sets.push(descriptor_set);
            }

            mesh_builders
                .into_iter()
                .zip(descriptor_sets.into_iter())
                .map(|(builder, descriptor_set)| builder.build(descriptor_set))
                .collect::<Vec<_>>()
        };

        let (pipeline_opaque, pipeline_transparent) = {
            let vertex_attribute_descriptions = [
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(ModelVertex, position) as u32)
                    .build(),
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(1)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(offset_of!(ModelVertex, normal) as u32)
                    .build(),
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .location(2)
                    .format(vk::Format::R32G32_SFLOAT)
                    .offset(offset_of!(ModelVertex, uv) as u32)
                    .build(),
            ];
            let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(std::mem::size_of::<ModelVertex>() as u32)
                .build()];
            let vertex_input_state_create_info = &vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&vertex_attribute_descriptions)
                .vertex_binding_descriptions(&vertex_binding_descriptions);

            let input_assembly_state_create_info =
                &vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let viewports = &[vk::Viewport::builder()
                .x(0.0)
                .y(image.height() as f32)
                .width(image.width() as f32)
                .height(-1.0 * image.height() as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .build()];
            let scissors = &[vk::Rect2D::builder()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(vk::Extent2D {
                    width: image.width(),
                    height: image.height(),
                })
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

            let color_blend_attachments_opaque =
                &[vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(
                        vk::ColorComponentFlags::R
                            | vk::ColorComponentFlags::G
                            | vk::ColorComponentFlags::B
                            | vk::ColorComponentFlags::A,
                    )
                    .blend_enable(false)
                    .build()];
            let color_blend_state_create_info_opaque =
                &vk::PipelineColorBlendStateCreateInfo::builder()
                    .logic_op_enable(false)
                    .attachments(color_blend_attachments_opaque);

            let color_blend_attachments_transparent =
                &[vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(
                        vk::ColorComponentFlags::R
                            | vk::ColorComponentFlags::G
                            | vk::ColorComponentFlags::B
                            | vk::ColorComponentFlags::A,
                    )
                    .blend_enable(true)
                    .build()];
            let color_blend_state_create_info_transparent =
                &vk::PipelineColorBlendStateCreateInfo::builder()
                    .logic_op_enable(false)
                    .attachments(color_blend_attachments_transparent);

            let stencil_op = vk::StencilOpState::builder()
                .compare_op(vk::CompareOp::ALWAYS)
                .depth_fail_op(vk::StencilOp::KEEP)
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .build();
            let depth_stencil_state_create_info_opaque =
                &vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS)
                    .stencil_test_enable(true)
                    .front(stencil_op)
                    .back(stencil_op);
            let depth_stencil_state_create_info_transparent =
                &vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(false)
                    .depth_compare_op(vk::CompareOp::LESS)
                    .stencil_test_enable(true)
                    .front(stencil_op)
                    .back(stencil_op);

            let pipeline_opaque = PipelineObject::new(
                device.device(),
                "shaders/spv/model_pass.vert.spv",
                "shaders/spv/model_pass_opaque.frag.spv",
                vertex_input_state_create_info,
                input_assembly_state_create_info,
                viewport_state_create_info,
                rasterization_state_create_info,
                multisample_state_create_info,
                color_blend_state_create_info_opaque,
                depth_stencil_state_create_info_opaque,
                None,
                &render_pass,
                &descriptor_set_layout,
            )?;
            let pipeline_transparent = PipelineObject::new(
                device.device(),
                "shaders/spv/model_pass.vert.spv",
                "shaders/spv/model_pass_transparent.frag.spv",
                vertex_input_state_create_info,
                input_assembly_state_create_info,
                viewport_state_create_info,
                rasterization_state_create_info,
                multisample_state_create_info,
                color_blend_state_create_info_transparent,
                depth_stencil_state_create_info_transparent,
                None,
                &render_pass,
                &descriptor_set_layout,
            )?;
            (pipeline_opaque, pipeline_transparent)
        };

        let framebuffer = FramebufferObject::new(
            device.device(),
            &render_pass,
            &[image.view(), depth_image.view()],
            image.width(),
            image.height(),
        )?;

        Ok(ModelPass {
            device: device.device(),
            meshes,
            materials,
            _uniform_buffer: uniform_buffer,
            render_pass,
            _sampler: sampler,
            _descriptor_pool: descriptor_pool,
            _descriptor_set_layout: descriptor_set_layout,
            pipeline_opaque,
            pipeline_transparent,
            framebuffer,
            width: image.width(),
            height: image.height(),
        })
    }

    pub fn cmd_draw(&self, command: vk::CommandBuffer) {
        unsafe {
            // クリア値
            let clear_value = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.6, 0.6, 1.0, 0.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            self.device.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass.render_pass())
                    .framebuffer(self.framebuffer.framebuffer())
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

            // Mesh毎の描画処理
            for mode in [AlphaMode::Opaque, AlphaMode::Mask, AlphaMode::Blend].iter() {
                let pipeline = match mode {
                    AlphaMode::Opaque | AlphaMode::Mask => &self.pipeline_opaque,
                    AlphaMode::Blend => &self.pipeline_transparent,
                };
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline(),
                );
                for mesh in self.meshes.iter() {
                    if mode != &self.materials[mesh.material_index].alpha_mode {
                        continue;
                    }
                    // 各バッファオブジェクトのセット
                    self.device.cmd_bind_vertex_buffers(
                        command,
                        0,
                        &[mesh.vertex_buffer.buffer()],
                        &[0],
                    );
                    self.device.cmd_bind_index_buffer(
                        command,
                        mesh.index_buffer.buffer(),
                        0,
                        vk::IndexType::UINT32,
                    );
                    // ディスクリプタセットのセット
                    self.device.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline_layout(),
                        0,
                        &[mesh.descriptor_set],
                        &[],
                    );
                    // メッシュを描画
                    self.device
                        .cmd_draw_indexed(command, mesh.index_count, 1, 0, 0, 0);
                }
            }

            self.device.cmd_end_render_pass(command);
        }
    }
}
