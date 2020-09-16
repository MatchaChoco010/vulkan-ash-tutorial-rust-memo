#![allow(dead_code)]

use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use winit::window::Window;

use crate::vulkan_app_base::{DefaultVulkanAppBase, VulkanAppBase};

/// 画面をクリアするアプリケーション
pub struct ClearScreenApp {
    base: DefaultVulkanAppBase,
}

impl ClearScreenApp {
    /// 未初期化のインスタンスの作成。
    pub fn new() -> Self {
        Self {
            base: DefaultVulkanAppBase::new(),
        }
    }
}

impl VulkanAppBase for ClearScreenApp {
    /// アプリケーションの初期化コード。
    fn init(&mut self, window: &Window) -> Result<()> {
        self.base.init(window)?;
        Ok(())
    }

    /// レンダリングのコード。
    fn render(&mut self) -> Result<()> {
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
            self.base
                .device()
                .wait_for_fences(&[fence], true, std::u64::MAX)?;
            self.base.device().reset_fences(&[fence])?;

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
            self.base
                .device()
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::builder())?;

            self.base.device().cmd_begin_render_pass(
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

            self.base.device().cmd_end_render_pass(command_buffer);

            self.base.device().end_command_buffer(command_buffer)?;

            // コマンドを送信
            self.base.device().queue_submit(
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
        self.base.cleanup()
    }
}
