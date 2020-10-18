use std::rc::Rc;

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device};

use crate::vk_wrapper_object::{
    DescriptorPoolObject, DescriptorSetLayoutObject, DeviceObject, ImageObject, PipelineObject,
    SamplerObject,
};

pub struct ComputeSobelPass {
    device: Rc<Device>,
    _sampler: SamplerObject,
    _descriptor_pool: DescriptorPoolObject,
    _descriptor_set_layout: DescriptorSetLayoutObject,
    descriptor_set: vk::DescriptorSet,
    pipeline: PipelineObject,
}
impl ComputeSobelPass {
    pub fn new(
        device: &DeviceObject,
        src_image: &ImageObject,
        dst_image: &ImageObject,
    ) -> Result<Self> {
        let sampler = {
            let sampler_create_info = &vk::SamplerCreateInfo::builder()
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(1.0)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .unnormalized_coordinates(false);
            SamplerObject::new(device.device(), sampler_create_info)?
        };

        let descriptor_pool = {
            let pool_sizes = &[
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .build(),
            ];
            DescriptorPoolObject::new(device.device(), pool_sizes, 1)?
        };

        let descriptor_set_layout = {
            let bindings = &[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];
            DescriptorSetLayoutObject::new(device.device(), bindings)?
        };

        let descriptor_set = {
            let layouts = &[descriptor_set_layout.descriptor_set_layout()];
            let descriptor_set = unsafe {
                device.device().allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool.descriptor_pool())
                        .set_layouts(layouts),
                )?
            }[0];

            let descriptor_src_image_infos = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(src_image.view())
                .sampler(sampler.sampler())
                .build()];

            let descriptor_dst_image_infos = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(dst_image.view())
                .sampler(sampler.sampler())
                .build()];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&descriptor_src_image_infos)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&descriptor_dst_image_infos)
                    .build(),
            ];

            unsafe {
                device
                    .device()
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }

            descriptor_set
        };

        let pipeline = PipelineObject::new_compute(
            device.device(),
            "shaders/spv/compute_sobel_pass.comp.spv",
            &descriptor_set_layout,
        )?;

        Ok(Self {
            device: device.device(),
            _sampler: sampler,
            _descriptor_pool: descriptor_pool,
            _descriptor_set_layout: descriptor_set_layout,
            descriptor_set,
            pipeline,
        })
    }

    pub fn cmd_draw(&self, command: vk::CommandBuffer) {
        let device = self.device.as_ref();
        unsafe {
            device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline(),
            );

            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline_layout(),
                0,
                &[self.descriptor_set],
                &[],
            );

            let group_count_x = 800 / 16 + 1;
            let group_count_y = 600 / 16 + 1;
            device.cmd_dispatch(command, group_count_x, group_count_y, 1);
        }
    }
}
