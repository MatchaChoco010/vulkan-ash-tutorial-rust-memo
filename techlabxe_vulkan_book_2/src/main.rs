#![allow(unused_imports)]

mod common;
mod resizable_window;

use common::{
    default_vulkan_app_base::DefaultVulkanAppBaseBuilder, vulkan_app_base::VulkanAppBaseBuilder,
    window_event_loop::WindowEventLoop,
};
use resizable_window::resizable_app::ResizableAppBuilder;

fn main() {
    // let app_builder = DefaultVulkanAppBaseBuilder::new().title("Vulkan App!");
    let app_builder = ResizableAppBuilder::new();
    WindowEventLoop::run(app_builder)
}
