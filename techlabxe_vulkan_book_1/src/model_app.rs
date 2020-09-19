#![allow(dead_code)]

use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use cgmath::{Deg, Matrix4, Point3, Vector2, Vector3};
use gltf::{image::Source, material::AlphaMode};
use image::{
    DynamicImage, GenericImageView,
    ImageFormat::{Jpeg, Png},
};
use memoffset::offset_of;
use std::{ffi::CString, path::Path};
use winit::window::Window;

use crate::vulkan_app_base::{DefaultVulkanAppBase, VulkanAppBase};

/// perspectiveを生やすための拡張用トレイト。
trait Matrix4Ext {
    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspecf32: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32>;
}
impl Matrix4Ext for Matrix4<f32> {
    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        use cgmath::{Angle, Rad};
        let f: Rad<f32> = fovy.into();
        let f = f / 2.0;
        let f = Rad::cot(f);
        Matrix4::<f32>::new(
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            (near * far) / (near - far),
            0.0,
        )
    }
}

/// BufferとAllocationをまとめた構造体。
struct BufferObject {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
}

/// ImageとImageView、Allocationをまとめた構造体
struct TextureObject {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
}

/// Vertex用構造体
struct Vertex {
    pos: Vector3<f32>,
    normal: Vector3<f32>,
    uv: Vector2<f32>,
}

/// UniformBuffer用構造体
struct UniformBufferObject {
    world: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

/// モデルのMeshの情報
struct ModelMesh {
    vertex_buffer: BufferObject,
    index_buffer: BufferObject,
    vertex_count: u32,
    index_count: u32,
    material_index: usize,
    descriptor_sets: Option<Vec<vk::DescriptorSet>>,
}

/// モデルのMaterialの情報
struct Material {
    texture: TextureObject,
    alpha_mode: AlphaMode,
}

/// モデルの情報
struct Model {
    meshes: Vec<ModelMesh>,
    materials: Vec<Material>,
}

/// モデルを読み込み表示するアプリケーション
pub struct ModelApp {
    base: DefaultVulkanAppBase,
    model: Model,
    uniform_buffers: Option<Vec<BufferObject>>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    sampler: Option<vk::Sampler>,
    pipeline_layout: Option<vk::PipelineLayout>,
    pipeline_opaque: Option<vk::Pipeline>,
    pipeline_transparent: Option<vk::Pipeline>,
}

impl ModelApp {
    pub fn new() -> Self {
        Self {
            base: DefaultVulkanAppBase::new(),
            model: Model {
                meshes: vec![],
                materials: vec![],
            },
            uniform_buffers: None,
            descriptor_set_layout: None,
            descriptor_pool: None,
            sampler: None,
            pipeline_layout: None,
            pipeline_opaque: None,
            pipeline_transparent: None,
        }
    }

    /// GPUのデバイスローカルのバッファを作成する。
    /// ステージングバッファを作成してからデバイスローカルにコピーする。
    fn create_buffer<T>(&self, data: &[T], usage: vk::BufferUsageFlags) -> Result<BufferObject> {
        // サイズの計算
        let buffer_size = data.len() as u64 * std::mem::size_of::<T>() as u64;

        // 一時バッファの確保
        let (tmp_buffer, tmp_buffer_allocation, _info) = self.base.allocator().create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuToGpu,
                ..Default::default()
            },
        )?;
        // 一時バッファに転写
        unsafe {
            let mapped_memory = self.base.allocator().map_memory(&tmp_buffer_allocation)? as *mut T;
            mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());
            self.base.allocator().unmap_memory(&tmp_buffer_allocation)?;
        }
        // バッファの作成
        let (buffer, buffer_allocation, _info) = self.base.allocator().create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST | usage),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            },
        )?;
        // 一時バッファからバッファへのコピーコマンドの発行
        unsafe {
            let device = self.base.device();
            let copy_cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*self.base.graphics_command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                copy_cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_copy_buffer(
                copy_cmd,
                tmp_buffer,
                buffer,
                &[vk::BufferCopy::builder().size(buffer_size).build()],
            );
            device.end_command_buffer(copy_cmd)?;

            // キューにサブミットし待機
            device.queue_submit(
                *self.base.graphics_queue(),
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[copy_cmd])
                    .build()],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(*self.base.graphics_queue())?;
        }

        // 一時バッファの削除
        self.base
            .allocator()
            .destroy_buffer(tmp_buffer, &tmp_buffer_allocation)?;

        Ok(BufferObject {
            buffer,
            allocation: buffer_allocation,
        })
    }

    /// Textureを読み込み準備する。
    fn load_texture(&self, image: DynamicImage) -> Result<TextureObject> {
        let (image_width, image_height) = (image.width(), image.height());
        let image_data = image.to_rgba().into_raw();

        let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;

        // ステージングバッファの作成
        let (staging_buffer, staging_buffer_allocation, _info) =
            self.base.allocator().create_buffer(
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
            let mapped_memory = self
                .base
                .allocator()
                .map_memory(&staging_buffer_allocation)? as *mut u8;
            mapped_memory.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());
            self.base
                .allocator()
                .unmap_memory(&staging_buffer_allocation)?;
        }

        let format = vk::Format::R8G8B8A8_UNORM;

        // Imageの作成
        let (image, image_allocation, _info) = self.base.allocator().create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: image_width,
                    height: image_height,
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
        unsafe {
            let device = self.base.device();

            let command = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*self.base.graphics_command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            // コマンドの構築開始
            device.begin_command_buffer(
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
            device.cmd_pipeline_barrier(
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
                    width: image_width,
                    height: image_height,
                    depth: 1,
                })
                .buffer_offset(0)
                .buffer_image_height(0)
                .buffer_row_length(0)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .build()];
            device.cmd_copy_buffer_to_image(
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
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );

            // コマンド終了
            device.end_command_buffer(command)?;

            // コマンドのサブミット
            device.queue_submit(
                *self.base.graphics_queue(),
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[command])
                    .build()],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(*self.base.graphics_queue())?;
        }

        let image_view = unsafe {
            self.base.device().create_image_view(
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

        self.base
            .allocator()
            .destroy_buffer(staging_buffer, &staging_buffer_allocation)?;
        Ok(TextureObject {
            image,
            allocation: image_allocation,
            view: image_view,
        })
    }

    /// VRMモデルを読み込みメッシュとマテリアルを生成する
    pub fn load_model(&mut self) -> Result<()> {
        let (document, buffers, _images) = gltf::import("assets/alicia-solid.vrm")?;

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
                    vertices.push(Vertex {
                        pos: Vector3::from(positions[i]),
                        normal: Vector3::from(normals[i]),
                        uv: Vector2::from(uvs[i]),
                    });
                }

                let vertex_buffer =
                    self.create_buffer(&vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;
                let vertex_count = vertices.len() as u32;

                let indices = if let Some(indices) = reader.read_indices() {
                    indices.into_u32().collect::<Vec<_>>()
                } else {
                    (0..positions.len()).map(|i| i as u32).collect::<Vec<_>>()
                };

                let index_buffer =
                    self.create_buffer(&indices, vk::BufferUsageFlags::INDEX_BUFFER)?;
                let index_count = indices.len() as u32;

                let material_index = prim.material().index().expect(
                    "some mesh does not assign a material. \
                    this app does not support default material.",
                );

                self.model.meshes.push(ModelMesh {
                    vertex_buffer,
                    index_buffer,
                    vertex_count,
                    index_count,
                    material_index,
                    descriptor_sets: None,
                });
            }
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
            let texture = self.load_texture(image)?;

            self.model.materials.push(Material {
                texture,
                alpha_mode,
            });
        }

        Ok(())
    }

    /// UniformBufferObjectの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn uniform_buffers(&self) -> &Vec<BufferObject> {
        self.uniform_buffers.as_ref().expect("Not initialized")
    }

    /// UniformBufferを作成。
    fn prepare_uniform_buffers(&mut self) -> Result<()> {
        let mut uniform_buffers = vec![];
        for _ in 0..self.base.swapchain_images().len() {
            // サイズの計算
            let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;

            // バッファの確保
            let (buffer, buffer_allocation, _info) = self.base.allocator().create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(buffer_size)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::CpuToGpu,
                    ..Default::default()
                },
            )?;

            uniform_buffers.push(BufferObject {
                buffer,
                allocation: buffer_allocation,
            });
        }

        self.uniform_buffers = Some(uniform_buffers);

        Ok(())
    }

    /// DescriptorPoolの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn descriptor_pool(&self) -> &vk::DescriptorPool {
        self.descriptor_pool.as_ref().expect("Not initialized")
    }

    /// DescriptorPoolを用意する
    fn prepare_descriptor_pool(&mut self) -> Result<()> {
        let count = 100;
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(count)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(count)
                .build(),
        ];

        let pool = unsafe {
            self.base.device().create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .max_sets(
                        self.base.swapchain_images().len() as u32 * self.model.meshes.len() as u32,
                    )
                    .pool_sizes(&descriptor_pool_size),
                None,
            )?
        };

        self.descriptor_pool = Some(pool);

        Ok(())
    }

    /// DescriptorSetLayoutの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn descriptor_set_layout(&self) -> &vk::DescriptorSetLayout {
        self.descriptor_set_layout
            .as_ref()
            .expect("Not initialized")
    }

    /// DescriptorSetLayoutを準備する。
    fn prepare_descriptor_set_layout(&mut self) -> Result<()> {
        let bindings = [
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
        ];

        let layout = unsafe {
            self.base.device().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings),
                None,
            )?
        };
        self.descriptor_set_layout = Some(layout);

        Ok(())
    }

    /// Samplerの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn sampler(&self) -> &vk::Sampler {
        self.sampler.as_ref().expect("Not initialized")
    }

    /// Sampleの準備。
    fn prepare_sampler(&mut self) -> Result<()> {
        let sampler = unsafe {
            self.base.device().create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
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
                    .unnormalized_coordinates(false),
                None,
            )?
        };

        self.sampler = Some(sampler);

        Ok(())
    }

    /// DescriptorSetの準備。
    /// 各メッシュに対してスワップチェーンの数だけ準備する。
    fn prepare_descriptor_set(&mut self) -> Result<()> {
        let mut layouts = vec![];
        for _ in 0..self.uniform_buffers().len() {
            layouts.push(*self.descriptor_set_layout());
        }

        let descriptor_pool = *self.descriptor_pool();
        let sampler = *self.sampler();

        // 各メッシュについて準備する
        for index in 0..self.model.meshes.len() {
            // ディスクリプタセットの確保
            let descriptor_sets = unsafe {
                self.base.device().allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(layouts.as_slice()),
                )?
            };

            // ディスクリプタセットへの書き込み
            let material = &self.model.materials[self.model.meshes[index].material_index];
            for i in 0..self.uniform_buffers().len() {
                let descripto_buffer_infos = [vk::DescriptorBufferInfo::builder()
                    .buffer(self.uniform_buffers()[i].buffer)
                    .offset(0)
                    .range(std::mem::size_of::<UniformBufferObject>() as u64)
                    .build()];

                let descriptor_image_infos = [vk::DescriptorImageInfo::builder()
                    .image_view(material.texture.view)
                    .sampler(sampler)
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
                    self.base
                        .device()
                        .update_descriptor_sets(&write_descriptor_sets, &[]);
                }
            }

            self.model.meshes[index].descriptor_sets = Some(descriptor_sets);
        }

        Ok(())
    }

    /// Pipeline Layoutの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn pipeline_layout(&self) -> &vk::PipelineLayout {
        self.pipeline_layout.as_ref().expect("Not initialized")
    }

    /// PipelineOpaqueの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn pipeline_opaque(&self) -> &vk::Pipeline {
        self.pipeline_opaque.as_ref().expect("Not initialized")
    }

    /// PipelineOpaqueの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn pipeline_transparent(&self) -> &vk::Pipeline {
        self.pipeline_transparent.as_ref().expect("Not initialized")
    }

    /// pipelineの準備
    fn prepare_pipelines(&mut self) -> Result<()> {
        // 頂点入力設定
        let vertex_input_bindings = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];
        let vertex_input_attributes = [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, normal) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, uv) as u32)
                .build(),
        ];
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attributes)
            .vertex_binding_descriptions(&vertex_input_bindings);

        // ビューポートの設定
        let extent = self.base.swapchain_extent();
        let viewports = [vk::Viewport::builder()
            .x(0.0)
            .y(extent.height as f32)
            .width(extent.width as f32)
            .height(-1.0 * extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];
        let scissors = [vk::Rect2D::builder()
            .offset(vk::Offset2D::builder().x(0).y(0).build())
            .extent(*extent)
            .build()];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        // プリミティブトポロジー設定
        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // ラスタライザーステート設定
        let rasterization_state_createa_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        // パイプラインレイアウト
        let pipeline_layout = unsafe {
            self.base.device().create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[*self.descriptor_set_layout()]),
                None,
            )?
        };
        self.pipeline_layout = Some(pipeline_layout);

        // マルチサンプル設定
        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Pipeline Opaque
        {
            // ブレンディングの設定
            let color_blend_attachments_state = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .build()];
            let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments_state);

            // デプスステンシルステート設定
            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_state_create_info =
                vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                    .depth_bounds_test_enable(false)
                    .stencil_test_enable(false)
                    .front(stencil_op)
                    .back(stencil_op);

            // シェーダバイナリの読み込み
            let vertex_shader_module = {
                use std::fs::File;
                use std::io::Read;

                let spv_file = File::open(&Path::new("shaders/spv/model.vert.spv"))?;
                let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.base
                        .device()
                        .create_shader_module(&shader_module_create_info, None)?
                }
            };
            let fragment_shader_module = {
                use std::fs::File;
                use std::io::Read;

                let spv_file = File::open(&Path::new("shaders/spv/model_opaque.frag.spv"))?;
                let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.base
                        .device()
                        .create_shader_module(&shader_module_create_info, None)?
                }
            };
            // main関数の名前
            let main_function_name = CString::new("main").unwrap();
            // shader stage create info
            let pipeline_shader_stage_create_info = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name)
                    .build(),
            ];

            // パイプラインの構築
            let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
                .stages(&pipeline_shader_stage_create_info)
                .vertex_input_state(&vertex_input_state_create_info)
                .input_assembly_state(&input_assembly_state_create_info)
                .viewport_state(&viewport_state_create_info)
                .rasterization_state(&rasterization_state_createa_info)
                .multisample_state(&multisample_state_create_info)
                .depth_stencil_state(&depth_stencil_state_create_info)
                .color_blend_state(&color_blend_state_create_info)
                .layout(pipeline_layout)
                .render_pass(*self.base.render_pass())
                .build()];
            let pipeline = unsafe {
                self.base
                    .device()
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &pipeline_create_info,
                        None,
                    )
                    .unwrap()[0]
            };

            // ShaderModuleはもう不要のため破棄
            unsafe {
                self.base
                    .device()
                    .destroy_shader_module(vertex_shader_module, None);
                self.base
                    .device()
                    .destroy_shader_module(fragment_shader_module, None);
            }

            self.pipeline_opaque = Some(pipeline);
        }

        // Pipeline Transparent
        {
            // ブレンディングの設定
            let color_blend_attachments_state = [vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .build()];
            let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments_state);

            // デプスステンシルステート設定
            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_state_create_info =
                vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(false)
                    .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                    .depth_bounds_test_enable(false)
                    .stencil_test_enable(false)
                    .front(stencil_op)
                    .back(stencil_op);

            // シェーダバイナリの読み込み
            let vertex_shader_module = {
                use std::fs::File;
                use std::io::Read;

                let spv_file = File::open(&Path::new("shaders/spv/model.vert.spv"))?;
                let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.base
                        .device()
                        .create_shader_module(&shader_module_create_info, None)?
                }
            };
            let fragment_shader_module = {
                use std::fs::File;
                use std::io::Read;

                let spv_file = File::open(&Path::new("shaders/spv/model_transparent.frag.spv"))?;
                let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.base
                        .device()
                        .create_shader_module(&shader_module_create_info, None)?
                }
            };
            // main関数の名前
            let main_function_name = CString::new("main").unwrap();
            // shader stage create info
            let pipeline_shader_stage_create_info = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name)
                    .build(),
            ];

            // パイプラインの構築
            let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
                .stages(&pipeline_shader_stage_create_info)
                .vertex_input_state(&vertex_input_state_create_info)
                .input_assembly_state(&input_assembly_state_create_info)
                .viewport_state(&viewport_state_create_info)
                .rasterization_state(&rasterization_state_createa_info)
                .multisample_state(&multisample_state_create_info)
                .depth_stencil_state(&depth_stencil_state_create_info)
                .color_blend_state(&color_blend_state_create_info)
                .layout(pipeline_layout)
                .render_pass(*self.base.render_pass())
                .build()];
            let pipeline = unsafe {
                self.base
                    .device()
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &pipeline_create_info,
                        None,
                    )
                    .unwrap()[0]
            };

            // ShaderModuleはもう不要のため破棄
            unsafe {
                self.base
                    .device()
                    .destroy_shader_module(vertex_shader_module, None);
                self.base
                    .device()
                    .destroy_shader_module(fragment_shader_module, None);
            }

            self.pipeline_transparent = Some(pipeline);
        }

        Ok(())
    }
}

impl VulkanAppBase for ModelApp {
    fn init(&mut self, window: &Window) -> Result<()> {
        self.base.init(window)?;
        self.load_model()?;
        self.prepare_uniform_buffers()?;
        self.prepare_descriptor_set_layout()?;
        self.prepare_descriptor_pool()?;
        self.prepare_sampler()?;
        self.prepare_descriptor_set()?;
        self.prepare_pipelines()?;

        Ok(())
    }

    fn render(&mut self) -> Result<()> {
        let device = self.base.device();

        unsafe {
            let (image_index, _is_suboptimal) = self.base.swapchain_loader().acquire_next_image(
                *self.base.swapchain(),
                std::u64::MAX,
                *self.base.image_available_semaphore(),
                vk::Fence::null(),
            )?;
            let image_index = image_index as usize;

            // Fenceを待機
            let fence = self.base.fences()[image_index];
            device.wait_for_fences(&[fence], true, std::u64::MAX)?;
            device.reset_fences(&[fence])?;

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
            let command_buffer = self.base.command_buffers()[image_index];

            device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder())?;

            device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(*self.base.render_pass())
                    .framebuffer(self.base.framebuffers()[image_index])
                    .render_area(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D::builder().x(0).y(0).build())
                            .extent(*self.base.swapchain_extent())
                            .build(),
                    )
                    .clear_values(&clear_value),
                vk::SubpassContents::INLINE,
            );

            // UniformBufferの中身を更新する
            let uniform_buffer_object = UniformBufferObject {
                world: Matrix4::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Deg(0.0)),
                view: Matrix4::look_at(
                    Point3::new(0.0, 1.5, -1.0),
                    Point3::new(0.0, 1.25, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                ),
                proj: Matrix4::perspective(Deg(45.0), 800.0 / 600.0, 0.01, 100.0),
            };
            {
                let mapped_memory = self
                    .base
                    .allocator()
                    .map_memory(&self.uniform_buffers()[image_index].allocation)?
                    as *mut UniformBufferObject;
                mapped_memory.copy_from_nonoverlapping([uniform_buffer_object].as_ptr(), 1);
                self.base
                    .allocator()
                    .unmap_memory(&self.uniform_buffers()[image_index].allocation)?;
            }
            // Mesh毎の描画処理
            for mode in [AlphaMode::Opaque, AlphaMode::Mask, AlphaMode::Blend].iter() {
                match mode {
                    AlphaMode::Opaque | AlphaMode::Mask => {
                        // 作成したパイプラインをセット
                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            *self.pipeline_opaque(),
                        );
                    }
                    AlphaMode::Blend => {
                        // 作成したパイプラインをセット
                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            *self.pipeline_transparent(),
                        )
                    }
                }
                for mesh in self.model.meshes.iter() {
                    if mode != &self.model.materials[mesh.material_index].alpha_mode {
                        continue;
                    }
                    // 各バッファオブジェクトのセット
                    device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &[mesh.vertex_buffer.buffer],
                        &[0],
                    );
                    device.cmd_bind_index_buffer(
                        command_buffer,
                        mesh.index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    // ディスクリプタセットのセット
                    let descriptor_sets = mesh
                        .descriptor_sets
                        .as_ref()
                        .expect("Not prepared descriptor set");
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        *self.pipeline_layout(),
                        0,
                        &[descriptor_sets[image_index]],
                        &[],
                    );
                    // メッシュを描画
                    device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
                }
            }

            device.cmd_end_render_pass(command_buffer);

            device.end_command_buffer(command_buffer)?;

            // コマンドを送信
            device.queue_submit(
                *self.base.graphics_queue(),
                &[vk::SubmitInfo::builder()
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[command_buffer])
                    .wait_semaphores(&[*self.base.image_available_semaphore()])
                    .signal_semaphores(&[*self.base.render_finish_semaphore()])
                    .build()],
                fence,
            )?;

            // Present処理
            self.base.swapchain_loader().queue_present(
                *self.base.present_queue(),
                &vk::PresentInfoKHR::builder()
                    .swapchains(&[*self.base.swapchain()])
                    .image_indices(&[image_index as u32])
                    .wait_semaphores(&[*self.base.render_finish_semaphore()]),
            )?;
        }

        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        let device = self.base.device();
        let allocator = self.base.allocator();

        unsafe {
            device.queue_wait_idle(*self.base.graphics_queue())?;
            device.queue_wait_idle(*self.base.present_queue())?;

            device.destroy_pipeline_layout(*self.pipeline_layout(), None);
            device.destroy_pipeline(*self.pipeline_opaque(), None);
            device.destroy_pipeline(*self.pipeline_transparent(), None);

            device.destroy_sampler(*self.sampler(), None);

            for uniform_buffer in self.uniform_buffers().iter() {
                allocator.destroy_buffer(uniform_buffer.buffer, &uniform_buffer.allocation)?;
            }

            device.destroy_descriptor_set_layout(*self.descriptor_set_layout(), None);
            device.destroy_descriptor_pool(*self.descriptor_pool(), None);

            for mesh in self.model.meshes.iter() {
                self.base
                    .allocator()
                    .destroy_buffer(mesh.vertex_buffer.buffer, &mesh.vertex_buffer.allocation)?;
                self.base
                    .allocator()
                    .destroy_buffer(mesh.index_buffer.buffer, &mesh.index_buffer.allocation)?;
            }
            for material in self.model.materials.iter() {
                self.base
                    .allocator()
                    .destroy_image(material.texture.image, &material.texture.allocation)?;
                device.destroy_image_view(material.texture.view, None);
            }
        }

        self.base.cleanup()
    }
}
