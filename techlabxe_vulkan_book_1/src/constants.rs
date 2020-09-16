//! プログラムに使う定数たち。

pub const WINDOW_TITLE: &'static str = "Vulkan App";
pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;

pub const APP_NAME: &'static str = "Vulkan App";

#[cfg(all(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

pub const VALIDATION: &[&str] = &[
    "VK_LAYER_KHRONOS_validation",
    "VK_LAYER_LUNARG_standard_validation",
];

pub const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];
