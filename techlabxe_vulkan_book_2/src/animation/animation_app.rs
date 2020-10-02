#![allow(dead_code)]

use std::{path::Path, rc::Rc, time::Instant};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device, Instance};
use cgmath::{Deg, InnerSpace, Matrix4, Point3, Vector2, Vector3, Vector4};
use imgui::*;
use imgui_rs_vulkan_renderer::{Renderer, RendererVkContext};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use memoffset::offset_of;
use winit::{dpi::PhysicalSize, event::Event, window::Window};

use crate::{
    animation::{
        animator::Animator,
        model::{Model, SceneParameter},
    },
    common::{
        cgmath_ext::Matrix4Ext,
        default_vulkan_app_base::{DefaultVulkanAppBase, DefaultVulkanAppBaseBuilder},
        vulkan_app_base::{VulkanAppBase, VulkanAppBaseBuilder},
        vulkan_objects::{
            DepthImageObject, DescriptorPoolObject, DescriptorSetLayoutObject,
            DeviceLocalBufferObject, FramebufferObject, HostVisibleBufferObject, PipelineObject,
            RenderPassObject, SamplerObject, SwapchainObject, TextureRenderingImageObject,
        },
    },
};

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

pub struct AnimationAppBuilder {
    window_size: (u32, u32),
    title: String,
    base: DefaultVulkanAppBaseBuilder,
}
impl AnimationAppBuilder {
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

    /// Framebufferを準備する。
    fn prepare_model_framebuffers(
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
impl VulkanAppBaseBuilder for AnimationAppBuilder {
    type Item = AnimationApp;

    fn new() -> Self {
        Self {
            window_size: (800, 600),
            title: "Animation App".into(),
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

        // Model
        let model = Model::load("assets/Model/初音ミク.pmd", &base)?;
        let model_framebuffers = Self::prepare_model_framebuffers(
            base.device(),
            base.swapchain(),
            base.depth_buffer(),
            &model.render_pass_default,
        )?;
        let shadow_framebuffer = FramebufferObject::new_depth_only(
            base.device(),
            model.shadow_map.image_view(),
            &model.render_pass_shadow,
            1024,
            1024,
        )?;

        let animator = Animator::new("assets/Motion/ごまえミク.vmd")?;

        // imguiリソースの作成
        let (imgui_render_pass, imgui_framebuffers) =
            Self::prepare_imgui_resource(base.device(), base.swapchain(), width, height)?;
        // imguiのセットアップ
        let mut imgui = Context::create();
        let mut platform = WinitPlatform::init(&mut imgui);
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.fonts().add_font(&[
            FontSource::DefaultFontData {
                config: Some(FontConfig {
                    size_pixels: font_size,
                    ..FontConfig::default()
                }),
            },
            FontSource::TtfData {
                data: include_bytes!("../../assets/mplus-1p-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    rasterizer_multiply: 1.75,
                    glyph_ranges: FontGlyphRanges::japanese(),
                    ..FontConfig::default()
                }),
            },
        ]);

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

        let model_name = model.model_name.clone();
        let model_version = model.model_version;
        let model_comment = model.model_comment.clone();

        let frame_period = animator.get_frame_period();

        Ok(Self::Item {
            current_frame: 0,
            frame_period,
            model_name,
            model_version,
            model_comment,

            imgui,
            platform,
            imgui_renderer,
            last_frame,
            imgui_framebuffers,
            imgui_render_pass,

            animator,

            shadow_framebuffer,
            model_framebuffers,
            model,

            base,
        })
    }
}

pub struct AnimationApp {
    current_frame: u32,
    frame_period: u32,
    model_name: String,
    model_version: f32,
    model_comment: String,

    imgui: Context,
    platform: WinitPlatform,
    imgui_renderer: Renderer,
    last_frame: Instant,
    imgui_framebuffers: Vec<FramebufferObject>,
    imgui_render_pass: RenderPassObject,

    animator: Animator,

    shadow_framebuffer: FramebufferObject,
    model_framebuffers: Vec<FramebufferObject>,
    model: Model,

    base: DefaultVulkanAppBase,
}
impl AnimationApp {}
impl Drop for AnimationApp {
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
impl VulkanAppBase for AnimationApp {
    fn on_window_size_changed(&mut self, width: u32, height: u32) -> Result<()> {
        self.base.on_window_size_changed(width, height)?;

        self.model_framebuffers = AnimationAppBuilder::prepare_model_framebuffers(
            self.base.device(),
            self.base.swapchain(),
            self.base.depth_buffer(),
            &self.model.render_pass_default,
        )?;

        // Modelに紐付いたリソースをどうにかする。
        self.model.set_scene_parameter(SceneParameter {
            view: Matrix4::look_at(
                Point3::new(0.0, 10.0, 50.0),
                Point3::new(0.0, 10.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: Matrix4::perspective(Deg(45.0), self.base.swapchain().aspect(), 0.01, 100.0),
            light_direction: Vector4::new(1.0, 1.0, 1.0, 0.0).normalize(),
            eye_position: Vector4::new(0.0, 10.0, 50.0, 1.0),
            light_view_proj: Matrix4::ortho(-20.0, 20.0, -20.0, 20.0, 0.0, 500.0)
                * Matrix4::look_at(
                    Point3::new(50.0, 50.0, 50.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                ),
            light_view_proj_bias: Matrix4::from_translation(Vector3::new(0.5, 0.5, 0.0))
                * Matrix4::from_nonuniform_scale(0.5, 0.5, 1.0),
            resolution: Vector2::new(
                self.base.swapchain().extent().width as f32,
                self.base.swapchain().extent().height as f32,
            ),
        });
        self.model.update_command_buffers_aspect(&self.base)?;

        // imguiリソースの作成
        let (imgui_render_pass, imgui_framebuffers) = AnimationAppBuilder::prepare_imgui_resource(
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

        // UIの構築
        let ui = self.imgui.frame();

        if ui.is_key_down(ui.key_index(imgui::Key::RightArrow)) {
            self.current_frame += 1;
        }
        if ui.is_key_down(ui.key_index(imgui::Key::LeftArrow)) && self.current_frame > 0 {
            self.current_frame -= 1;
        }
        self.current_frame = self.current_frame.min(self.frame_period);

        let model_name = &self.model_name;
        let model_version = &self.model_version;
        let model_comment = &self.model_comment;
        let frame_period = self.frame_period;
        let current_frame = &mut self.current_frame;

        imgui::Window::new(im_str!("Hello World"))
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(unsafe {
                    imgui::ImStr::from_utf8_with_nul_unchecked(model_name.as_bytes())
                });
                ui.text(unsafe {
                    imgui::ImStr::from_utf8_with_nul_unchecked(model_version.to_string().as_bytes())
                });
                ui.text(unsafe {
                    imgui::ImStr::from_utf8_with_nul_unchecked(model_comment.as_bytes())
                });
                ui.separator();
                Slider::new(im_str!("frame"), 0..=frame_period).build(&ui, current_frame);
            });
        self.platform.prepare_render(&ui, &window);
        let draw_data = ui.render();

        // コマンドの構築
        let command = self.base.command_buffers()[image_index];
        unsafe {
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::builder())?;
        }

        // Model描画
        {
            self.animator
                .update_animation(&mut self.model, self.current_frame);

            self.model.update(image_index)?;

            // クリア値
            let clear_value = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.5, 0.25, 0.25, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            let clear_depth = [vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            }];

            unsafe {
                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::builder()
                        .render_pass(self.model.render_pass_shadow.render_pass())
                        .framebuffer(self.shadow_framebuffer.framebuffer())
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: 1024,
                                height: 1024,
                            },
                        })
                        .clear_values(&clear_depth),
                    vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
                );
                let subcommands = self.model.get_command_buffers_shadow(image_index);
                device.cmd_execute_commands(command, subcommands.as_slice());
                device.cmd_end_render_pass(command);

                // BarrierでフォーマットをSHADER READ ONLY OPTIMALにする
                let image_barriers = [vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.model.shadow_map.image())
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build()];
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_barriers,
                );

                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::builder()
                        .render_pass(self.model.render_pass_default.render_pass())
                        .framebuffer(self.model_framebuffers[image_index].framebuffer())
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: self.base.swapchain().extent(),
                        })
                        .clear_values(&clear_value),
                    vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
                );

                let subcommands = self.model.get_command_buffers(image_index);
                device.cmd_execute_commands(command, subcommands.as_slice());

                let subcommands = self.model.get_command_buffers_outline(image_index);
                device.cmd_execute_commands(command, subcommands.as_slice());

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
