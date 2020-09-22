//! アプリケーションのベーストレイトのデフォルト実装。
#![allow(dead_code)]

use std::{mem::ManuallyDrop, rc::Rc};

use anyhow::Result;
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use ash_window::{create_surface, enumerate_required_extensions};
use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    os::raw::c_void,
};
use winit::window::Window;

use crate::common::{
    utils::vk_to_string,
    vulkan_app_base::{VulkanAppBase, VulkanAppBaseBuilder},
    vulkan_objects::{DepthImageObject, SwapchainObject},
};

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const VALIDATION: &[&str] = &[
    "VK_LAYER_KHRONOS_validation",
    "VK_LAYER_LUNARG_standard_validation",
];

const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

/// デフォルトのVulkanAppBaseのビルダー。
pub struct DefaultVulkanAppBaseBuilder {
    title: String,
    version: (u32, u32, u32),
    window_size: (u32, u32),
}

impl DefaultVulkanAppBaseBuilder {
    /// アプリケーションの名前の設定。
    pub fn title(self, title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..self
        }
    }

    /// アプリケーションのバージョンの設定。
    pub fn version(self, major: u32, patch: u32, minor: u32) -> Self {
        Self {
            version: (major, patch, minor),
            ..self
        }
    }

    /// アプリケーションの初期ウィンドウサイズの設定。
    pub fn window_size(self, width: u32, height: u32) -> Self {
        Self {
            window_size: (width, height),
            ..self
        }
    }

    // 以下build用。

    // DebugUtilsMessengerCreateInfoExtの構築をする関数。
    fn populate_debug_utils_messenger_create_info_ext() -> vk::DebugUtilsMessengerCreateInfoEXT {
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

    /// Instanceの作成。
    /// debugモードかreleaseモードかによって処理内容が違う。
    /// debugモードの場合はデバッグのコールバックと
    /// バリデーションレイヤーを有効にしたインスタンスを返す。
    /// releaseモードの場合はプレーンなインスタンスを返す。
    /// どちらの場合もSurfaceを作成するのに必要なextensionを要求する。
    fn create_instance(
        entry: &Entry,
        window: &Window,
        app_name: String,
        version: (u32, u32, u32),
    ) -> Result<Instance> {
        // applicationi info
        let app_name = CString::new(app_name).unwrap();
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
            Self::populate_debug_utils_messenger_create_info_ext();

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
        Ok(unsafe { entry.create_instance(&create_info, None)? })
    }

    /// debug utilsをセットアップする。
    fn setup_debug_utils(
        entry: &Entry,
        instance: &Instance,
    ) -> (DebugUtils, vk::DebugUtilsMessengerEXT) {
        let debug_utils_loader = DebugUtils::new(entry, instance);

        if ENABLE_VALIDATION_LAYERS == false {
            (debug_utils_loader, vk::DebugUtilsMessengerEXT::null())
        } else {
            let messenger_ci = Self::populate_debug_utils_messenger_create_info_ext();
            let util_messenger = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&messenger_ci, None)
                    .expect("Debug Utils Callback")
            };
            (debug_utils_loader, util_messenger)
        }
    }

    /// Surfaceの作成を行う。
    fn init_surface(
        entry: &Entry,
        instance: &Instance,
        window: &Window,
    ) -> Result<(Surface, vk::SurfaceKHR)> {
        let surface_loader = Surface::new(entry, instance);
        let surface = unsafe { create_surface(entry, instance, window, None)? };

        Ok((surface_loader, surface))
    }

    /// PhysicalDeviceを選択する。
    /// Graphics用Queueのindexとpresent用Queueのindexも返す。
    /// vk::PhysicalDeviceMemoryPropertiesも返す
    fn select_physical_device(
        instance: &Instance,
        surface_loader: Surface,
        surface: vk::SurfaceKHR,
    ) -> Result<(
        vk::PhysicalDevice,
        u32,
        u32,
        vk::PhysicalDeviceMemoryProperties,
    )> {
        // Physical Deviceの中で条件を満たすものを抽出する
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let mut physical_devices = physical_devices.iter().filter_map(|&physical_device| {
            // 必要なキューファミリーに対応しているかどうか。
            // graphicsとpresentに対応しているか確認。
            // 両者が同じインデックスの場合もある。
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let mut graphics_queue_index = None;
            let mut present_queue_index = None;
            for (i, queue_family) in queue_families.iter().enumerate() {
                if queue_family.queue_count > 0
                    && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    graphics_queue_index = Some(i as u32);
                }

                let is_present_support = unsafe {
                    surface_loader
                        .get_physical_device_surface_support(physical_device, i as u32, surface)
                        .unwrap()
                };
                if queue_family.queue_count > 0 && is_present_support {
                    present_queue_index = Some(i as u32);
                }

                if graphics_queue_index.is_some() && present_queue_index.is_some() {
                    break;
                }
            }
            let is_queue_families_supported =
                graphics_queue_index.is_some() && present_queue_index.is_some();

            // DEVICE_EXTENSIONSで指定した拡張が対応しているかを確認
            let is_device_extension_supported = {
                let available_extensions = unsafe {
                    instance
                        .enumerate_device_extension_properties(physical_device)
                        .unwrap()
                };
                let mut available_extension_names = vec![];
                for extension in available_extensions.iter() {
                    let extension_name = vk_to_string(&extension.extension_name);
                    available_extension_names.push(extension_name);
                }
                let mut required_extensions = HashSet::new();
                for extension in DEVICE_EXTENSIONS.iter() {
                    required_extensions.insert(extension.to_string());
                }
                for extension in available_extension_names.iter() {
                    required_extensions.remove(extension);
                }
                required_extensions.is_empty()
            };

            // Swapchainをサポートしているかを確認
            let is_swapchain_supported = if is_device_extension_supported {
                let formats = unsafe {
                    surface_loader
                        .get_physical_device_surface_formats(physical_device, surface)
                        .unwrap()
                };
                let present_modes = unsafe {
                    surface_loader
                        .get_physical_device_surface_present_modes(physical_device, surface)
                        .unwrap()
                };
                !formats.is_empty() && !present_modes.is_empty()
            } else {
                false
            };

            // AnisotropyなSamplerに対応しているかを確認
            let supported_features =
                unsafe { instance.get_physical_device_features(physical_device) };
            let is_supported_sampler_anisotropy = supported_features.sampler_anisotropy != 0;

            if is_queue_families_supported
                && is_device_extension_supported
                && is_swapchain_supported
                && is_supported_sampler_anisotropy
            {
                Some((
                    physical_device,
                    graphics_queue_index.unwrap(),
                    present_queue_index.unwrap(),
                ))
            } else {
                None
            }
        });

        // 条件を満たすうち最初を選択する
        let (physical_device, graphics_queue_index, present_queue_index) = physical_devices
            .next()
            .expect("There is no physical device that meets the requirements");

        // Memory Propertiesを取得しておく
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok((
            physical_device,
            graphics_queue_index,
            present_queue_index,
            memory_properties,
        ))
    }

    /// 論理デバイスの作成
    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        graphics_queue_index: u32,
        present_queue_index: u32,
    ) -> Result<Rc<Device>> {
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(graphics_queue_index);
        unique_queue_families.insert(present_queue_index);

        let queue_priorities = [1.0_f32];
        let mut queue_create_infos = vec![];
        for &queue_family in unique_queue_families.iter() {
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities)
                .build();
            queue_create_infos.push(queue_create_info);
        }

        let physical_device_features =
            vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

        let enabled_extension_names = [Swapchain::name().as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(queue_create_infos.as_slice())
            .enabled_extension_names(&enabled_extension_names)
            .enabled_features(&physical_device_features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        Ok(Rc::new(device))
    }

    /// queueを取得する。
    fn get_queues(
        device: Rc<Device>,
        graphics_queue_index: u32,
        present_queue_index: u32,
    ) -> (vk::Queue, vk::Queue) {
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_queue_index, 0) };
        (graphics_queue, present_queue)
    }

    /// Graphicsのコマンドプールを準備する。
    fn prepare_graphics_command_pool(
        device: Rc<Device>,
        graphics_queue_index: u32,
    ) -> Result<vk::CommandPool> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_queue_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        Ok(unsafe { device.create_command_pool(&command_pool_create_info, None)? })
    }

    /// vk-mem-rsのAllocatorを作成する。
    fn setup_vk_mem_allocator(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: Rc<Device>,
    ) -> Result<Rc<vk_mem::Allocator>> {
        let allocator_create_info = vk_mem::AllocatorCreateInfo {
            physical_device: physical_device,
            device: device.as_ref().clone(),
            instance: instance.clone(),
            flags: vk_mem::AllocatorCreateFlags::empty(),
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };
        Ok(Rc::new(vk_mem::Allocator::new(&allocator_create_info)?))
    }

    /// コマンドバッファを準備する
    fn prepare_command_buffers(
        device: Rc<Device>,
        graphics_command_pool: vk::CommandPool,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>> {
        // 描画用コマンドバッファの作成
        Ok(unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(graphics_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(count),
            )?
        })
    }

    /// フェンスを準備する
    fn prepare_fences(device: Rc<Device>, count: usize) -> Result<Vec<vk::Fence>> {
        let mut fences = vec![];
        for _ in 0..count {
            unsafe {
                fences.push(device.create_fence(
                    &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
            }
        }
        Ok(fences)
    }

    /// セマフォを準備する。
    fn prepare_semaphores(device: Rc<Device>) -> Result<(vk::Semaphore, vk::Semaphore)> {
        unsafe {
            let image_available_semaphore =
                device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?;
            let render_finish_semaphore =
                device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?;
            Ok((image_available_semaphore, render_finish_semaphore))
        }
    }
}

impl VulkanAppBaseBuilder for DefaultVulkanAppBaseBuilder {
    type Item = DefaultVulkanAppBase;

    fn new() -> Self {
        Self {
            title: "".to_string(),
            version: (0, 1, 0),
            window_size: (800, 600),
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

        let entry = Entry::new()?;
        let instance = Self::create_instance(&entry, window, self.title, self.version)?;
        let (debug_utils_loader, debug_messenger) = Self::setup_debug_utils(&entry, &instance);
        let (surface_loader, surface) = Self::init_surface(&entry, &instance, window)?;
        let (physical_device, graphics_queue_index, present_queue_index, memory_properties) =
            Self::select_physical_device(&instance, surface_loader.clone(), surface.clone())?;
        let device = Self::create_device(
            &instance,
            physical_device,
            graphics_queue_index,
            present_queue_index,
        )?;
        let (graphics_queue, present_queue) =
            Self::get_queues(device.clone(), graphics_queue_index, present_queue_index);
        let graphics_command_pool =
            Self::prepare_graphics_command_pool(device.clone(), graphics_queue_index)?;
        let swapchain = SwapchainObject::new(
            &instance,
            device.clone(),
            physical_device,
            &surface_loader,
            surface,
            graphics_queue_index,
            present_queue_index,
            width,
            height,
        )?;
        let allocator = Self::setup_vk_mem_allocator(&instance, physical_device, device.clone())?;
        let depth_buffer = DepthImageObject::new(allocator.clone(), device.clone(), width, height)?;
        let command_buffers = Self::prepare_command_buffers(
            device.clone(),
            graphics_command_pool,
            swapchain.len() as u32,
        )?;
        let fences = Self::prepare_fences(device.clone(), swapchain.len())?;
        let (image_available_semaphore, render_finish_semaphore) =
            Self::prepare_semaphores(device.clone())?;

        Ok(Self::Item {
            entry,
            instance,
            debug_utils_loader,
            debug_messenger,
            surface_loader,
            surface,
            physical_device,
            graphics_queue_index,
            present_queue_index,
            memory_properties,
            device,
            graphics_queue,
            present_queue,
            graphics_command_pool,
            swapchain: ManuallyDrop::new(swapchain),
            allocator,
            depth_buffer: ManuallyDrop::new(depth_buffer),
            command_buffers,
            fences,
            image_available_semaphore,
            render_finish_semaphore,
        })
    }
}

/// デフォルトのVulkanAppBase。
pub struct DefaultVulkanAppBase {
    entry: Entry,
    instance: Instance,
    debug_utils_loader: DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_queue_index: u32,
    present_queue_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: Rc<Device>,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    graphics_command_pool: vk::CommandPool,
    swapchain: ManuallyDrop<SwapchainObject>,
    allocator: Rc<vk_mem::Allocator>,
    depth_buffer: ManuallyDrop<DepthImageObject>,
    command_buffers: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    image_available_semaphore: vk::Semaphore,
    render_finish_semaphore: vk::Semaphore,
}
impl DefaultVulkanAppBase {
    /// Instanceを取得する
    pub fn instance(&self) -> &Instance {
        &self.instance
    }
    /// surface_loaderを取得する。
    pub fn surface_loader(&self) -> &Surface {
        &self.surface_loader
    }
    /// surfaceを取得する。
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }
    /// physical_deviceを取得する。
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    /// graphics_queue_indexを取得する。
    pub fn graphics_queue_index(&self) -> u32 {
        self.graphics_queue_index
    }
    /// present_queue_indexを取得する。
    pub fn present_queue_index(&self) -> u32 {
        self.present_queue_index
    }
    /// memory_propertiesを取得する。
    pub fn memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        self.memory_properties
    }
    /// deviceを取得する。
    pub fn device(&self) -> Rc<Device> {
        self.device.clone()
    }
    /// graphics_queueを取得する。
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }
    /// present_queueを取得する。
    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }
    /// graphics_command_poolを取得する。
    pub fn graphics_command_pool(&self) -> vk::CommandPool {
        self.graphics_command_pool
    }
    /// swapchainを取得する。
    pub fn swapchain(&self) -> &SwapchainObject {
        &self.swapchain
    }
    /// allocatorを取得する。
    pub fn allocator(&self) -> Rc<vk_mem::Allocator> {
        self.allocator.clone()
    }
    /// depth_bufferを取得する。
    pub fn depth_buffer(&self) -> &DepthImageObject {
        &self.depth_buffer
    }
    /// command_buffersを取得する。
    pub fn command_buffers(&self) -> &Vec<vk::CommandBuffer> {
        &self.command_buffers
    }
    /// fencesを取得する。
    pub fn fences(&self) -> &Vec<vk::Fence> {
        &self.fences
    }
    /// image_available_semaphoreを取得する。
    pub fn image_available_semaphore(&self) -> vk::Semaphore {
        self.image_available_semaphore
    }
    /// render_finish_semaphoreを取得する。
    pub fn render_finish_semaphore(&self) -> vk::Semaphore {
        self.render_finish_semaphore
    }
}
impl VulkanAppBase for DefaultVulkanAppBase {
    /// ウィンドウサイズ変化時のイベント関数。
    fn on_window_size_changed(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };

        self.swapchain.resize(width, height)?;

        unsafe {
            ManuallyDrop::drop(&mut self.depth_buffer);
        }

        self.depth_buffer = ManuallyDrop::new(DepthImageObject::new(
            self.allocator.clone(),
            self.device.clone(),
            width,
            height,
        )?);

        Ok(())
    }

    /// 描画の準備を行う。
    fn prepare(&mut self) -> Result<()> {
        Ok(())
    }

    /// 描画処理を行う。
    fn render(&mut self) -> Result<()> {
        Ok(())
    }
}
impl Drop for DefaultVulkanAppBase {
    fn drop(&mut self) {
        unsafe {
            self.device
                .queue_wait_idle(self.graphics_queue)
                .expect("Failed to wait idle");
            self.device
                .queue_wait_idle(self.present_queue)
                .expect("Failed to wait idle");

            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finish_semaphore, None);
            for &fence in self.fences.iter() {
                self.device.destroy_fence(fence, None);
            }

            ManuallyDrop::drop(&mut self.swapchain);
            ManuallyDrop::drop(&mut self.depth_buffer);

            if let Some(alloc) = Rc::get_mut(&mut self.allocator) {
                alloc.destroy();
            }

            self.device
                .destroy_command_pool(self.graphics_command_pool, None);
            self.device.destroy_device(None);

            self.surface_loader.destroy_surface(self.surface, None);

            if ENABLE_VALIDATION_LAYERS {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
