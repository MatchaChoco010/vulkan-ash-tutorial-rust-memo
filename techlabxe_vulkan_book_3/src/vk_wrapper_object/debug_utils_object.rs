use std::ffi::{c_void, CStr};

use ash::{extensions::ext::DebugUtils, vk};

use crate::vk_wrapper_object::{constants::ENABLE_VALIDATION_LAYERS, EntryObject, InstanceObject};

pub struct DebugUtilsObject {
    debug_utils_loader: DebugUtils,
    util_messenger: vk::DebugUtilsMessengerEXT,
}
impl DebugUtilsObject {
    // DebugUtilsMessengerCreateInfoExtの構築をする関数。
    pub fn populate_debug_utils_messenger_create_info_ext() -> vk::DebugUtilsMessengerCreateInfoEXT
    {
        // デバッグのコールバック関数
        unsafe extern "system" fn callback(
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT,
            p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
            _p_user_data: *mut c_void,
        ) -> vk::Bool32 {
            let severity = match message_severity {
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
                _ => "[Unknown]",
            };
            let types = match message_type {
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
                vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
                _ => "[Unknown]",
            };
            let message = CStr::from_ptr((*p_callback_data).p_message);
            println!("[Debug]{}{}{:?}", severity, types, message);

            vk::FALSE
        }
        let debug_utils_messenger_create_info_ext = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(callback));

        debug_utils_messenger_create_info_ext.build()
    }

    /// debug utilsをセットアップする。
    /// Debugモード時はメッセンジャーを作成し、
    /// Releaseモード時はメッセンジャーを作成しない。
    pub fn new(entry: &EntryObject, instance: &InstanceObject) -> Self {
        let entry = entry.entry_as_ref();
        let instance = instance.instance_as_ref();
        let debug_utils_loader = DebugUtils::new(entry, instance);

        if ENABLE_VALIDATION_LAYERS == false {
            Self {
                debug_utils_loader,
                util_messenger: vk::DebugUtilsMessengerEXT::null(),
            }
        } else {
            let messenger_ci = Self::populate_debug_utils_messenger_create_info_ext();
            let util_messenger = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&messenger_ci, None)
                    .expect("Failed to create Debug Utils messenger.")
            };
            Self {
                debug_utils_loader,
                util_messenger,
            }
        }
    }
}
impl Drop for DebugUtilsObject {
    fn drop(&mut self) {
        if ENABLE_VALIDATION_LAYERS {
            unsafe {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.util_messenger, None);
            }
        }
    }
}
