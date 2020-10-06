#[cfg(all(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

pub const VALIDATION: &[&str] = &["VK_LAYER_KHRONOS_validation"];

pub const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];
