use std::{ffi::CString, path::Path, rc::Rc};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

use crate::vk_wrapper_object::{DescriptorSetLayoutObject, DeviceObject, RenderPassObject};

pub struct PipelineObject {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    device: Rc<Device>,
}
impl PipelineObject {
    /// GraphicsPipelineObjectを生成する。
    pub fn new(
        device: &DeviceObject,
        vertex_shader_pass: impl AsRef<Path>,
        fragment_shader_pass: impl AsRef<Path>,
        vertex_input_state_create_info: vk::PipelineVertexInputStateCreateInfo,
        input_assembly_state_create_info: vk::PipelineInputAssemblyStateCreateInfo,
        viewport_state_create_info: vk::PipelineViewportStateCreateInfo,
        rasterization_state_create_info: vk::PipelineRasterizationStateCreateInfo,
        multisample_state_create_info: vk::PipelineMultisampleStateCreateInfo,
        color_blend_state_create_info: vk::PipelineColorBlendStateCreateInfo,
        depth_stencil_state_create_info: vk::PipelineDepthStencilStateCreateInfo,
        render_pass: &RenderPassObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
    ) -> Result<Self> {
        let device_ref = device.device_as_ref();

        // パイプラインレイアウト
        let pipeline_layout = unsafe {
            device_ref.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layout.descriptor_set_layout()]),
                None,
            )?
        };

        // シェーダバイナリの読み込み
        let vertex_shader_module = {
            use std::fs::File;
            use std::io::Read;

            let spv_file = File::open(vertex_shader_pass)?;
            let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr() as *const u32,
                ..Default::default()
            };
            unsafe { device_ref.create_shader_module(&shader_module_create_info, None)? }
        };
        let fragment_shader_module = {
            use std::fs::File;
            use std::io::Read;

            let spv_file = File::open(fragment_shader_pass)?;
            let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr() as *const u32,
                ..Default::default()
            };
            unsafe { device_ref.create_shader_module(&shader_module_create_info, None)? }
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
            .rasterization_state(&rasterization_state_create_info)
            .multisample_state(&multisample_state_create_info)
            .depth_stencil_state(&depth_stencil_state_create_info)
            .color_blend_state(&color_blend_state_create_info)
            .layout(pipeline_layout)
            .render_pass(render_pass.render_pass())
            .build()];
        let pipeline = unsafe {
            device_ref
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_info, None)
                .unwrap()[0]
        };

        // ShaderModuleはもう不要のため破棄
        unsafe {
            device_ref.destroy_shader_module(vertex_shader_module, None);

            device_ref.destroy_shader_module(fragment_shader_module, None);
        }

        Ok(Self {
            pipeline,
            pipeline_layout,
            device: device.device(),
        })
    }

    /// pipelineを取得する。
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// pipeline_layoutを取得する。
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}
impl Drop for PipelineObject {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}
