use techlabxe_vulkan_book_3::{
    app_base::{VulkanAppBaseBuilder, WindowEventLoop},
    cubemap_app::app::AppBuilder,
};

fn main() {
    let app_builder = AppBuilder::new();
    WindowEventLoop::run(app_builder);
}
