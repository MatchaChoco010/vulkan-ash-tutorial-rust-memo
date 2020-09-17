#![allow(unused_imports)]

mod clear_screen_app;
mod constants;
mod triangle_app;
mod utils;
mod vulkan_app_base;
mod window_event_loop;

use clear_screen_app::ClearScreenApp;
use triangle_app::TriangleApp;
use vulkan_app_base::DefaultVulkanAppBase;
use window_event_loop::WindowEventLoop;

fn main() {
    // WindowEventLoop::run(DefaultVulkanAppBase::new());
    // WindowEventLoop::run(ClearScreenApp::new());
    WindowEventLoop::run(TriangleApp::new());
}
