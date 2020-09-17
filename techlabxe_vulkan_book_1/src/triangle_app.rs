#![allow(dead_code)]

use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use cgmath::Vector3;
use memoffset::offset_of;
use std::{ffi::CString, path::Path};
use winit::window::Window;

use crate::vulkan_app_base::{DefaultVulkanAppBase, VulkanAppBase};

/// BufferとDeviceMemoryをまとめた構造体。
struct BufferObject {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
}

/// 三角形を表示するアプリケーション。
pub struct TriangleApp {
    base: DefaultVulkanAppBase,
    vertex_buffer: Option<BufferObject>,
    index_buffer: Option<BufferObject>,
    index_count: Option<u32>,
    pipeline_layout: Option<vk::PipelineLayout>,
    pipeline: Option<vk::Pipeline>,
}

impl TriangleApp {
    /// 未初期化のインスタンスを返す。
    pub fn new() -> Self {
        Self {
            base: DefaultVulkanAppBase::new(),
            vertex_buffer: None,
            index_buffer: None,
            index_count: None,
            pipeline_layout: None,
            pipeline: None,
        }
    }

    /// vertex bufferの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn vertex_buffer(&self) -> &BufferObject {
        self.vertex_buffer.as_ref().expect("Not initialized")
    }

    /// index bufferの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn index_buffer(&self) -> &BufferObject {
        self.index_buffer.as_ref().expect("Not initialized")
    }

    /// index bufferの数を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn index_count(&self) -> u32 {
        self.index_count.expect("Not initialized")
    }

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

    /// vertex bufferとindex bufferを作成する
    /// ステージングバッファを作成しGPUデバイスローカルのバッファに転送している。
    fn create_buffers(&mut self) -> Result<()> {
        // データの用意
        let vertices = [
            Vertex {
                pos: Vector3::new(-1.0, 0.0, 0.0),
                color: Vector3::new(1.0, 0.0, 0.0),
            },
            Vertex {
                pos: Vector3::new(1.0, 0.0, 0.0),
                color: Vector3::new(0.0, 1.0, 0.0),
            },
            Vertex {
                pos: Vector3::new(0.0, 1.0, 0.0),
                color: Vector3::new(0.0, 0.0, 1.0),
            },
        ];
        let indices: [u32; 3] = [0, 1, 2];

        self.vertex_buffer =
            Some(self.create_buffer(&vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?);
        self.index_buffer = Some(self.create_buffer(&indices, vk::BufferUsageFlags::INDEX_BUFFER)?);
        self.index_count = Some(indices.len() as u32);

        Ok(())
    }

    /// Pipeline Layoutの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn pipeline_layout(&self) -> &vk::PipelineLayout {
        self.pipeline_layout.as_ref().expect("Not initialized")
    }

    /// Pipelineの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    fn pipeline(&self) -> &vk::Pipeline {
        self.pipeline.as_ref().expect("Not initialized")
    }
}

struct Vertex {
    pos: Vector3<f32>,
    color: Vector3<f32>,
}

impl VulkanAppBase for TriangleApp {
    /// アプリケーションの初期化コード。
    fn init(&mut self, window: &Window) -> Result<()> {
        self.base.init(window)?;
        self.create_buffers()?;

        // pipelineの準備
        {
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
                    .offset(offset_of!(Vertex, color) as u32)
                    .build(),
            ];
            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&vertex_input_attributes)
                .vertex_binding_descriptions(&vertex_input_bindings);

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
            let input_assembly_state_create_info =
                vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            // ラスタライザーステート設定
            let rasterization_state_createa_info =
                vk::PipelineRasterizationStateCreateInfo::builder()
                    .polygon_mode(vk::PolygonMode::FILL)
                    .cull_mode(vk::CullModeFlags::NONE)
                    .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                    .line_width(1.0);

            // マルチサンプル設定
            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

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

                let spv_file = File::open(&Path::new("shaders/spv/triangle.vert.spv"))?;
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

                let spv_file = File::open(&Path::new("shaders/spv/triangle.frag.spv"))?;
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

            // パイプラインレイアウト
            let pipeline_layout = unsafe {
                self.base
                    .device()
                    .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder(), None)?
            };

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

            self.pipeline_layout = Some(pipeline_layout);
            self.pipeline = Some(pipeline);
        }

        Ok(())
    }

    /// レンダリングのコード。
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

            // 作成したパイプラインをセット
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                *self.pipeline(),
            );
            // 各バッファオブジェクトのセット
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer().buffer], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer().buffer,
                0,
                vk::IndexType::UINT32,
            );
            // 三角形描画
            device.cmd_draw_indexed(command_buffer, self.index_count(), 1, 0, 0, 0);

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

    /// 後片付けのコード。
    fn cleanup(&mut self) -> Result<()> {
        unsafe {
            let device = self.base.device();
            let allocator = self.base.allocator();

            device.queue_wait_idle(*self.base.graphics_queue())?;
            device.queue_wait_idle(*self.base.present_queue())?;

            device.destroy_pipeline_layout(*self.pipeline_layout(), None);
            device.destroy_pipeline(*self.pipeline(), None);
            allocator
                .destroy_buffer(self.index_buffer().buffer, &self.index_buffer().allocation)?;
            allocator.destroy_buffer(
                self.vertex_buffer().buffer,
                &self.vertex_buffer().allocation,
            )?;
        }

        self.base.cleanup()
    }
}
