use std::{ffi::CString, rc::Rc};

use anyhow::Result;
use ash::{
    extensions::ext::DebugUtils,
    version::{EntryV1_0, InstanceV1_0},
    vk, Instance,
};
use ash_window::enumerate_required_extensions;
use winit::window::Window;

use crate::vk_wrapper_object::{
    constants::{ENABLE_VALIDATION_LAYERS, VALIDATION},
    DebugUtilsObject, EntryObject,
};

pub struct InstanceObject {
    instance: Rc<Instance>,
}
impl InstanceObject {
    /// Instanceの作成。
    /// debugモードかreleaseモードかによって処理内容が違う。
    /// debugモードの場合はデバッグのコールバックと
    /// バリデーションレイヤーを有効にしたインスタンスを返す。
    /// releaseモードの場合はプレーンなインスタンスを返す。
    /// どちらの場合もSurfaceを作成するのに必要なextensionを要求する。
    pub fn new(
        entry: &EntryObject,
        window: &Window,
        app_name: impl Into<String>,
        version: (u32, u32, u32),
    ) -> Result<Self> {
        let entry = entry.entry_as_ref();

        // applicationi info
        let app_name = CString::new(app_name.into()).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_version(1, 2, 0))
            .application_version(vk::make_version(version.0, version.1, version.2))
            .application_name(&app_name)
            .engine_version(vk::make_version(version.0, version.1, version.2))
            .engine_name(&engine_name);

        // Surface作成に必要なextensionの取得
        let extension_names = enumerate_required_extensions(window)?;
        let mut extension_names: Vec<_> = extension_names
            .iter()
            .map(|extension_name| extension_name.as_ptr())
            .collect();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugUtils::name().as_ptr());
        }

        // Validationに必要なレイヤー
        let enabled_layer_names: Vec<CString> = VALIDATION
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let enabled_layer_names: Vec<_> = enabled_layer_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let mut debug_utils_messenger_create_info_ext =
            DebugUtilsObject::populate_debug_utils_messenger_create_info_ext();

        // instance create info
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&enabled_layer_names);
        let create_info = if ENABLE_VALIDATION_LAYERS {
            create_info.push_next(&mut debug_utils_messenger_create_info_ext)
        } else {
            create_info
        };

        // instanceの作成
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(Self {
            instance: Rc::new(instance),
        })
    }

    /// Instanceを取得する。
    pub fn instance(&self) -> Rc<Instance> {
        Rc::clone(&self.instance)
    }

    /// Instanceの参照を取得する。
    pub fn instance_as_ref(&self) -> &Instance {
        self.instance.as_ref()
    }
}
impl Drop for InstanceObject {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}
