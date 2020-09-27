#![allow(unused_imports)]

mod common;
mod instancing;
mod posteffect;
mod resizable_window;
mod texture_rendering;
mod use_imgui;

use common::{
    default_vulkan_app_base::DefaultVulkanAppBaseBuilder, vulkan_app_base::VulkanAppBaseBuilder,
    window_event_loop::WindowEventLoop,
};
use instancing::instancing_app::InstancingAppBuilder;
use posteffect::posteffect_app::PostEffectAppBuilder;
use resizable_window::resizable_app::ResizableAppBuilder;
use texture_rendering::texture_rendering_app::TextureRenderingAppBuilder;
use use_imgui::use_imgui_app::UseImguiAppBuilder;

fn main() {
    // let app_builder = DefaultVulkanAppBaseBuilder::new().title("Vulkan App!");
    // let app_builder = ResizableAppBuilder::new();
    // let app_builder = UseImguiAppBuilder::new();
    // let app_builder = TextureRenderingAppBuilder::new();
    // let app_builder = InstancingAppBuilder::new();
    let app_builder = PostEffectAppBuilder::new();
    WindowEventLoop::run(app_builder)
}
