//! アプリケーションのベース。
//! Vulkanのリソースの管理。

#![allow(dead_code)]

use anyhow::Result;
use ash::{
    extensions::khr::{Surface, Swapchain},
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

use crate::constants::{
    APP_NAME, DEVICE_EXTENSIONS, ENABLE_VALIDATION_LAYERS, VALIDATION, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use crate::utils::vk_to_string;

/// WindowEventLoopに渡すために必要なアプリケーションのベースとなるトレイト。
/// WindowEventLoopに渡すと適切なタイミングで各種メソッドが呼ばれる。
pub trait VulkanAppBase {
    /// アプリケーションの初期化コード。
    fn init(&mut self, window: &Window) -> Result<()>;

    /// 描画のための準備を行う。
    fn prepare(&mut self) -> Result<()> {
        Ok(())
    }

    /// レンダリングのコード。
    fn render(&mut self) -> Result<()> {
        Ok(())
    }

    /// 後片付けのコード。
    fn cleanup(&mut self) -> Result<()>;
}

/// VulkanAppBaseのデフォルト実装。
///
/// フレームバッファなどの描画に必要なVulkanのリソースを確保し、
/// アプリ終了時にそれらのリソースを破棄する。
/// レンダリングは何もしない。
///
/// 独自のVulkanAppBaseを実装する場合はこの構造体をメンバに持つと便利。
/// 初期化にはWindowが必要になるためnewで初期化できない。
/// WindowEventLoopの内部であとからWindowを渡して初期化することになる。
/// 初期化を遅延するためにすべてのメンバをOptionにくるんでいる。
/// 初期化後はフィールドと同名のメソッドでunwrapして取得できる。
pub struct DefaultVulkanAppBase {
    entry: Option<Entry>,
    instance: Option<Instance>,
    surface_loader: Option<Surface>,
    surface: Option<vk::SurfaceKHR>,
    physical_device: Option<vk::PhysicalDevice>,
    graphics_queue_index: Option<u32>,
    present_queue_index: Option<u32>,
    memory_properties: Option<vk::PhysicalDeviceMemoryProperties>,
    device: Option<Device>,
    graphics_queue: Option<vk::Queue>,
    present_queue: Option<vk::Queue>,
    graphics_command_pool: Option<vk::CommandPool>,
    swapchain_loader: Option<Swapchain>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_images: Option<Vec<vk::Image>>,
    swapchain_image_views: Option<Vec<vk::ImageView>>,
    swapchain_format: Option<vk::SurfaceFormatKHR>,
    swapchain_extent: Option<vk::Extent2D>,
    allocator: Option<vk_mem::Allocator>,
    depth_buffer_image: Option<vk::Image>,
    depth_buffer_allocation: Option<vk_mem::Allocation>,
    depth_buffer_image_view: Option<vk::ImageView>,
    render_pass: Option<vk::RenderPass>,
    framebuffers: Option<Vec<vk::Framebuffer>>,
    command_buffers: Option<Vec<vk::CommandBuffer>>,
    fences: Option<Vec<vk::Fence>>,
    image_available_semaphore: Option<vk::Semaphore>,
    render_finish_semaphore: Option<vk::Semaphore>,
}
impl DefaultVulkanAppBase {
    /// 未初期化のインスタンスを返す。
    pub fn new() -> Self {
        Self {
            entry: None,
            instance: None,
            surface_loader: None,
            surface: None,
            physical_device: None,
            graphics_queue_index: None,
            present_queue_index: None,
            memory_properties: None,
            device: None,
            graphics_queue: None,
            present_queue: None,
            graphics_command_pool: None,
            swapchain_loader: None,
            swapchain: None,
            swapchain_images: None,
            swapchain_image_views: None,
            swapchain_format: None,
            swapchain_extent: None,
            allocator: None,
            depth_buffer_image: None,
            depth_buffer_allocation: None,
            depth_buffer_image_view: None,
            render_pass: None,
            framebuffers: None,
            command_buffers: None,
            fences: None,
            image_available_semaphore: None,
            render_finish_semaphore: None,
        }
    }

    /// Entryの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn entry(&self) -> &Entry {
        self.entry.as_ref().expect("Not initialized")
    }

    /// Instanceの参照を取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn instance(&self) -> &Instance {
        self.instance.as_ref().expect("Not initialized")
    }

    /// Instanceを作成する。
    /// debugモードかreleaseモードかによって処理内容が違う。
    /// debugモードの場合はデバッグのコールバックと
    /// バリデーションレイヤーを有効にしたインスタンスを返す。
    /// releaseモードの場合はプレーンなインスタンスを返す。
    /// どちらの場合もSurfaceを作成するのに必要なextensionを要求する。
    fn init_instance(&mut self, entry: &Entry, window: &Window) -> Result<()> {
        // applicationi info
        let app_name = CString::new(APP_NAME).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_version(1, 2, 0))
            .application_version(vk::make_version(0, 1, 0))
            .application_name(&app_name)
            .engine_version(vk::make_version(0, 1, 0))
            .engine_name(&engine_name);

        // Surface作成に必要なextensionの取得
        let extension_names = enumerate_required_extensions(window)?;
        let extension_names: Vec<_> = extension_names
            .iter()
            .map(|extension_name| extension_name.as_ptr())
            .collect();

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
        let mut debug_utils_messenger_create_info_ext =
            vk::DebugUtilsMessengerCreateInfoEXT::builder()
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

        // Validationに必要なレイヤー
        let enabled_layer_names: Vec<CString> = VALIDATION
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let enabled_layer_names: Vec<_> = enabled_layer_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

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
        self.instance = Some(unsafe { entry.create_instance(&create_info, None)? });

        Ok(())
    }

    /// surface_loaderを取得する。
    /// device由来の関数のうちSurfaceに関係するものはこちらから呼び出す。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn surface_loader(&self) -> &Surface {
        self.surface_loader.as_ref().expect("Not initialized")
    }

    /// surfaceを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn surface(&self) -> &vk::SurfaceKHR {
        self.surface.as_ref().expect("Not initialized")
    }

    /// Surfaceの作成を行う。
    fn init_surface(&mut self, entry: &Entry, window: &Window) -> Result<()> {
        let surface_loader = Surface::new(entry, self.instance());
        let surface = unsafe { create_surface(entry, self.instance(), window, None)? };

        self.surface_loader = Some(surface_loader);
        self.surface = Some(surface);

        Ok(())
    }

    /// PhysicalDeviceを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn physical_device(&self) -> &vk::PhysicalDevice {
        self.physical_device.as_ref().expect("Not initialized")
    }

    /// Graphics Queueのindexを取得する。
    /// Present用のQueueと同じindexの可能性がある。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn graphics_queue_index(&self) -> &u32 {
        self.graphics_queue_index.as_ref().expect("Not initialized")
    }

    /// Present用Queueのindexを取得する。
    /// Graphics Queueと同じindexの可能性がある。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn present_queue_index(&self) -> &u32 {
        self.present_queue_index.as_ref().expect("Not initialized")
    }

    /// 物理デバイスのメモリプロパティを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        self.memory_properties.as_ref().expect("Not initialized")
    }

    /// 物理デバイスを選択し、GraphicsとPresent用のQueueのindexを返す。
    fn select_physical_device(&mut self) -> Result<()> {
        // Physical Deviceの中で条件を満たすものを抽出する
        let physical_devices = unsafe { self.instance().enumerate_physical_devices()? };
        let mut physical_devices = physical_devices.iter().filter_map(|&physical_device| {
            // 必要なキューファミリーに対応しているかどうか。
            // graphicsとpresentに対応しているか確認。
            // 両者が同じインデックスの場合もある。
            let queue_families = unsafe {
                self.instance()
                    .get_physical_device_queue_family_properties(physical_device)
            };
            let mut graphics_queue_index = None;
            let mut present_queue_index = None;
            for (i, queue_family) in queue_families.iter().enumerate() {
                if queue_family.queue_count > 0
                    && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    graphics_queue_index = Some(i as u32);
                }

                let is_present_support = unsafe {
                    self.surface_loader()
                        .get_physical_device_surface_support(
                            physical_device,
                            i as u32,
                            *self.surface(),
                        )
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
                    self.instance()
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
                    self.surface_loader()
                        .get_physical_device_surface_formats(physical_device, *self.surface())
                        .unwrap()
                };
                let present_modes = unsafe {
                    self.surface_loader()
                        .get_physical_device_surface_present_modes(physical_device, *self.surface())
                        .unwrap()
                };
                !formats.is_empty() && !present_modes.is_empty()
            } else {
                false
            };

            // AnisotropyなSamplerに対応しているかを確認
            let supported_features = unsafe {
                self.instance()
                    .get_physical_device_features(physical_device)
            };
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

        self.physical_device = Some(physical_device);
        self.graphics_queue_index = Some(graphics_queue_index);
        self.present_queue_index = Some(present_queue_index);

        // Memory Propertiesを取得しておく
        self.memory_properties = Some(unsafe {
            self.instance()
                .get_physical_device_memory_properties(physical_device)
        });

        Ok(())
    }

    /// Deviceを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn device(&self) -> &Device {
        self.device.as_ref().expect("Not initialized")
    }

    /// 論理デバイスの作成
    fn create_device(&mut self) -> Result<()> {
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(*self.graphics_queue_index());
        unique_queue_families.insert(*self.present_queue_index());

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

        self.device = Some(unsafe {
            self.instance()
                .create_device(*self.physical_device(), &device_create_info, None)?
        });

        Ok(())
    }

    /// Graphics Queueを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn graphics_queue(&self) -> &vk::Queue {
        self.graphics_queue.as_ref().expect("Not initialized")
    }

    /// Present用Queueを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn present_queue(&self) -> &vk::Queue {
        self.present_queue.as_ref().expect("Not initialized")
    }

    /// queueを取得する。
    fn get_queues(&mut self) {
        let graphics_queue = unsafe {
            self.device()
                .get_device_queue(*self.graphics_queue_index(), 0)
        };
        let present_queue = unsafe {
            self.device()
                .get_device_queue(*self.present_queue_index(), 0)
        };
        self.graphics_queue = Some(graphics_queue);
        self.present_queue = Some(present_queue);
    }

    /// Graphicsのコマンドプールを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn graphics_command_pool(&self) -> &vk::CommandPool {
        self.graphics_command_pool
            .as_ref()
            .expect("Not initialized")
    }

    /// Graphicsのコマンドプールを準備する。
    fn prepare_graphics_command_pool(&mut self) -> Result<()> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(*self.graphics_queue_index())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        self.graphics_command_pool = Some(unsafe {
            self.device()
                .create_command_pool(&command_pool_create_info, None)?
        });

        Ok(())
    }

    /// swapchain_loaderを取得する。
    /// deviceに由来する関数のうちswapchain関係はこちらから呼び出す。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn swapchain_loader(&self) -> &Swapchain {
        self.swapchain_loader.as_ref().expect("Not initialized")
    }

    /// swapchainを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn swapchain(&self) -> &vk::SwapchainKHR {
        self.swapchain.as_ref().expect("Not initialized")
    }

    /// swapchain_imagesを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn swapchain_images(&self) -> &Vec<vk::Image> {
        self.swapchain_images.as_ref().expect("Not initialized")
    }

    /// swapchain_image_viewsを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn swapchain_image_views(&self) -> &Vec<vk::ImageView> {
        self.swapchain_image_views
            .as_ref()
            .expect("Not initialized")
    }

    /// swapchain_formatを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn swapchain_format(&self) -> &vk::SurfaceFormatKHR {
        self.swapchain_format.as_ref().expect("Not initialized")
    }

    /// swapchain_extentを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn swapchain_extent(&self) -> &vk::Extent2D {
        self.swapchain_extent.as_ref().expect("Not initialized")
    }

    /// Swapchainを作成する。
    /// FormatはB8G8R8A8_UNORMかR8G8B8A8_UNORMを優先し、
    /// 対応していなければ一番最初に対応しているフォーマットを選ぶ。
    /// PresentモードはMAILBOXを優先で、対応していなければFIFOを選ぶ。
    /// SharingModeはgraphics queueとpresent queueが同じならばEXCLUSIVEを
    /// 違う場合はCONCURRENTを選ぶ。
    fn create_swapchain(&mut self) -> Result<()> {
        let swapchain_loader = Swapchain::new(self.instance(), self.device());

        let capabilities = unsafe {
            self.surface_loader()
                .get_physical_device_surface_capabilities(
                    *self.physical_device(),
                    *self.surface(),
                )?
        };
        let formats = unsafe {
            self.surface_loader()
                .get_physical_device_surface_formats(*self.physical_device(), *self.surface())
                .unwrap()
        };
        let present_modes = unsafe {
            self.surface_loader()
                .get_physical_device_surface_present_modes(*self.physical_device(), *self.surface())
                .unwrap()
        };

        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM || f.format == vk::Format::R8G8B8A8_UNORM
            })
            .unwrap_or(&formats[0])
            .clone();
        let present_mode = present_modes
            .into_iter()
            .find(|&p| p == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let extent = {
            if capabilities.current_extent.width != u32::max_value() {
                capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: WINDOW_WIDTH
                        .max(capabilities.min_image_extent.width)
                        .min(capabilities.max_image_extent.width),
                    height: WINDOW_HEIGHT
                        .max(capabilities.min_image_extent.height)
                        .min(capabilities.max_image_extent.height),
                }
            }
        };

        let image_count = capabilities.min_image_count + 1;
        let image_count = if capabilities.max_image_count != 0 {
            image_count.min(capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_indices) =
            if *self.graphics_queue_index() != *self.present_queue_index() {
                (
                    vk::SharingMode::CONCURRENT,
                    vec![*self.graphics_queue_index(), *self.present_queue_index()],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(*self.surface())
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_family_indices.as_slice())
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|&image| {
                unsafe {
                    self.device().create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )
                }
                .unwrap()
            })
            .collect();

        self.swapchain_loader = Some(swapchain_loader);
        self.swapchain = Some(swapchain);
        self.swapchain_images = Some(swapchain_images);
        self.swapchain_image_views = Some(swapchain_image_views);
        self.swapchain_format = Some(format);
        self.swapchain_extent = Some(extent);

        Ok(())
    }

    /// vk-mem-rsのAllocatorを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn allocator(&self) -> &vk_mem::Allocator {
        self.allocator.as_ref().expect("Not initialized")
    }

    /// vk-mem-rsのAllocatorを作成する。
    fn setup_vk_mem_allocator(&mut self) -> Result<()> {
        let allocator_create_info = vk_mem::AllocatorCreateInfo {
            physical_device: *self.physical_device(),
            device: self.device().clone(),
            instance: self.instance().clone(),
            flags: vk_mem::AllocatorCreateFlags::empty(),
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };
        self.allocator = Some(vk_mem::Allocator::new(&allocator_create_info)?);

        Ok(())
    }

    /// vk-mem-rsのAllocatorを破棄する。
    fn destroy_vk_mem_allocator(&mut self) {
        self.allocator.as_mut().expect("Not initialized").destroy();
    }

    /// DepthBufferのImageを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn depth_buffer_image(&self) -> &vk::Image {
        self.depth_buffer_image.as_ref().expect("Not initialized")
    }

    /// DepthBufferのAllocationを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn depth_buffer_allocation(&self) -> &vk_mem::Allocation {
        self.depth_buffer_allocation
            .as_ref()
            .expect("Not initialized")
    }

    /// DepthBufferのImageViewを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn depth_buffer_image_view(&self) -> &vk::ImageView {
        self.depth_buffer_image_view
            .as_ref()
            .expect("Not initialized")
    }

    /// DepthBufferのImageとImageViewとを作成する。
    fn create_depth_buffer(&mut self) -> Result<()> {
        let depth_image_create_info = vk::ImageCreateInfo::builder()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: self.swapchain_extent().width,
                height: self.swapchain_extent().height,
                depth: 1,
            });
        let depth_image_allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            flags: vk_mem::AllocationCreateFlags::empty(),
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            memory_type_bits: 0,
            pool: None,
            user_data: None,
        };
        let (depth_buffer_image, depth_buffer_allocation, _depth_image_allocation_info) = self
            .allocator()
            .create_image(&depth_image_create_info, &depth_image_allocation_info)?;
        let depth_buffer_image_view = unsafe {
            self.device().create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(depth_buffer_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    ),
                None,
            )?
        };

        self.depth_buffer_image = Some(depth_buffer_image);
        self.depth_buffer_allocation = Some(depth_buffer_allocation);
        self.depth_buffer_image_view = Some(depth_buffer_image_view);

        Ok(())
    }

    /// RenderPassを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn render_pass(&self) -> &vk::RenderPass {
        self.render_pass.as_ref().expect("Not initialized")
    }

    /// RenderPassを作成する。
    fn create_render_pass(&mut self) -> Result<()> {
        // Attachments
        let attachments = [
            vk::AttachmentDescription::builder()
                .format(self.swapchain_format().format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .build(),
            vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build(),
        ];
        // color reference
        let color_reference = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];
        // depth reference
        let depth_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        // subpass descriptionを作成
        let subpasses = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_reference)
            .depth_stencil_attachment(&depth_reference)
            .build()];
        // render passの作成
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses);
        self.render_pass = Some(unsafe {
            self.device()
                .create_render_pass(&render_pass_create_info, None)?
        });

        Ok(())
    }

    /// Framebuffersを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn framebuffers(&self) -> &Vec<vk::Framebuffer> {
        self.framebuffers.as_ref().expect("Not initialized")
    }

    /// Framebuffersを作成する。
    /// Swapchainの枚数だけフレームバッファを作成している。
    fn create_framebuffers(&mut self) -> Result<()> {
        let mut framebuffers = vec![];
        for &image_view in self.swapchain_image_views().iter() {
            let attachments = [image_view, *self.depth_buffer_image_view()];

            framebuffers.push(unsafe {
                self.device().create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(*self.render_pass())
                        .attachments(&attachments)
                        .width(self.swapchain_extent().width)
                        .height(self.swapchain_extent().height)
                        .layers(1),
                    None,
                )?
            });
        }

        self.framebuffers = Some(framebuffers);

        Ok(())
    }

    /// コマンドバッファを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn command_buffers(&self) -> &Vec<vk::CommandBuffer> {
        self.command_buffers.as_ref().expect("Not initialized")
    }

    /// コマンドバッファを準備する
    fn prepare_command_buffers(&mut self) -> Result<()> {
        // 描画用コマンドバッファの作成
        self.command_buffers = Some(unsafe {
            self.device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*self.graphics_command_pool())
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(self.swapchain_images().len() as u32),
            )?
        });
        Ok(())
    }

    /// Fenceを取得する
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn fences(&self) -> &Vec<vk::Fence> {
        self.fences.as_ref().expect("Not initialized")
    }

    /// フェンスを準備する
    fn prepare_fences(&mut self) -> Result<()> {
        let mut fences = vec![];
        for _ in 0..self.swapchain_images().len() {
            unsafe {
                fences.push(self.device().create_fence(
                    &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
            }
        }
        self.fences = Some(fences);
        Ok(())
    }

    /// 描画の同期に使うセマフォを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn image_available_semaphore(&self) -> &vk::Semaphore {
        self.image_available_semaphore
            .as_ref()
            .expect("Not initialized")
    }

    /// 描画の同期に使うセマフォを取得する。
    /// 構造体が初期化済みであることを要求するので注意。
    pub fn render_finish_semaphore(&self) -> &vk::Semaphore {
        self.render_finish_semaphore
            .as_ref()
            .expect("Not initialized")
    }

    /// セマフォを準備する。
    fn prepare_semaphores(&mut self) -> Result<()> {
        unsafe {
            self.image_available_semaphore = Some(
                self.device()
                    .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?,
            );
            self.render_finish_semaphore = Some(
                self.device()
                    .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?,
            );
        }
        Ok(())
    }
}

impl VulkanAppBase for DefaultVulkanAppBase {
    fn init(&mut self, window: &Window) -> Result<()> {
        let entry = Entry::new()?;
        self.init_instance(&entry, window)?;
        self.init_surface(&entry, window)?;
        self.select_physical_device()?;
        self.create_device()?;
        self.get_queues();
        self.prepare_graphics_command_pool()?;
        self.create_swapchain()?;
        self.setup_vk_mem_allocator()?;
        self.create_depth_buffer()?;
        self.create_render_pass()?;
        self.create_framebuffers()?;
        self.prepare_command_buffers()?;
        self.prepare_fences()?;
        self.prepare_semaphores()?;

        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        unsafe {
            self.device().queue_wait_idle(*self.graphics_queue())?;
            self.device().queue_wait_idle(*self.present_queue())?;

            self.device()
                .destroy_semaphore(*self.image_available_semaphore(), None);
            self.device()
                .destroy_semaphore(*self.render_finish_semaphore(), None);
            for &fence in self.fences() {
                self.device().destroy_fence(fence, None);
            }
            for &swapchain_image_view in self.swapchain_image_views() {
                self.device().destroy_image_view(swapchain_image_view, None);
            }
            self.device()
                .destroy_image_view(*self.depth_buffer_image_view(), None);
            self.device().destroy_render_pass(*self.render_pass(), None);
            for &framebuffer in self.framebuffers().iter() {
                self.device().destroy_framebuffer(framebuffer, None);
            }
            self.allocator()
                .destroy_image(*self.depth_buffer_image(), self.depth_buffer_allocation())?;
            self.destroy_vk_mem_allocator();
            self.swapchain_loader()
                .destroy_swapchain(*self.swapchain(), None);
            self.device()
                .destroy_command_pool(*self.graphics_command_pool(), None);
            self.device().destroy_device(None);
            self.surface_loader().destroy_surface(*self.surface(), None);
            self.instance().destroy_instance(None);
        }
        Ok(())
    }
}
