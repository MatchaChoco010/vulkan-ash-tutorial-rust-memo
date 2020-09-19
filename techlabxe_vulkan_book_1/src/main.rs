#![allow(unused_imports)]

mod clear_screen_app;
mod constants;
mod model_app;
mod textured_cube_app;
mod triangle_app;
mod utils;
mod vulkan_app_base;
mod window_event_loop;

use clear_screen_app::ClearScreenApp;
use model_app::ModelApp;
use textured_cube_app::TexturedCubeApp;
use triangle_app::TriangleApp;
use vulkan_app_base::DefaultVulkanAppBase;
use window_event_loop::WindowEventLoop;

fn main() {
    // WindowEventLoop::run(DefaultVulkanAppBase::new());
    // WindowEventLoop::run(ClearScreenApp::new());
    // WindowEventLoop::run(TriangleApp::new());
    // WindowEventLoop::run(TexturedCubeApp::new());
    WindowEventLoop::run(ModelApp::new());
}
