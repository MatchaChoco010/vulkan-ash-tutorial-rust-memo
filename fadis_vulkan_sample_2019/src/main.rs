use anyhow::Result;
use ash::{
    extensions::khr::{Surface, Swapchain},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};
use ash_window::{create_surface, enumerate_required_extensions};
use cgmath::{Deg, Matrix4, Point3, Vector3, Vector4};
use memoffset::offset_of;
use num::NumCast;
use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
    path::Path,
};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WINDOW_TITLE: &'static str = "Vulkan";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;
const VALIDATION: &[&str] = &[
    "VK_LAYER_KHRONOS_validation",
    "VK_LAYER_LUNARG_standard_validation",
];

const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

const VERTEX_SHADER_PASS: &'static str = "shaders/spv/vert.spv";
const FRAGMENT_SHADER_PASS: &'static str = "shaders/spv/frag.spv";
const MODEL_PATH: &'static str = "assets/sponza.obj";

fn vk_to_string(raw_string_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

trait Matrix4Ext {
    fn to_u8_slice(&self) -> &[u8];
    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspecf32: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32>;
}
impl Matrix4Ext for Matrix4<f32> {
    fn to_u8_slice(&self) -> &[u8] {
        unsafe {
            ::std::slice::from_raw_parts(
                (self as *const Matrix4<f32>) as *const u8,
                ::std::mem::size_of::<Matrix4<f32>>(),
            )
        }
    }

    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        use cgmath::{Angle, Rad};
        let f: Rad<f32> = fovy.into();
        let tow: f32 = NumCast::from(2.0).unwrap();
        let f = f / tow;
        let f = Rad::cot(f);
        Matrix4::<f32>::new(
            f / aspect,
            NumCast::from(0.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            f,
            NumCast::from(0.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            far / (near - far),
            NumCast::from(-1.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            NumCast::from(0.0).unwrap(),
            (near * far) / (near - far),
            NumCast::from(0.0).unwrap(),
        )
    }
}

trait Vector3Ext {
    fn to_u8_slice(&self) -> &[u8];
}
impl Vector3Ext for Vector3<f32> {
    fn to_u8_slice(&self) -> &[u8] {
        unsafe {
            ::std::slice::from_raw_parts(
                (self as *const Vector3<f32>) as *const u8,
                ::std::mem::size_of::<Vector3<f32>>(),
            )
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
struct Vertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
}
impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, position) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, normal) as u32)
                .build(),
        ]
    }
}

fn main() -> Result<()> {
    // windowの作成
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .with_decorations(false)
        .with_inner_size(LogicalSize::new(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32))
        .build(&event_loop)?;

    // Entryの作成
    let entry = Entry::new()?;

    // Instanceの作成
    let instance: Instance = {
        // application info
        let app_name = CString::new(WINDOW_TITLE).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_version(1, 2, 0))
            .application_version(vk::make_version(0, 1, 0))
            .application_name(&app_name)
            .engine_version(vk::make_version(0, 1, 0))
            .engine_name(&engine_name);

        // Surface作成に必要なextensionsの取得
        let extension_names = enumerate_required_extensions(&window)?;
        let extension_names: Vec<_> = extension_names
            .iter()
            .map(|extension_name| extension_name.as_ptr())
            .collect();

        // コールバック関数
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
        unsafe { entry.create_instance(&create_info, None)? }
    };

    // Surfaceの作成
    let surface_loader = Surface::new(&entry, &instance);
    let surface = unsafe { create_surface(&entry, &instance, &window, None)? };

    // Physical deviceの選択
    let (physical_device, graphics_queue_index, present_queue_index) = {
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
        physical_devices.next().unwrap()
    };

    // 論理デバイスを作成する
    let device: Device = {
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

        unsafe { instance.create_device(physical_device, &device_create_info, None)? }
    };

    // キューを作成する
    let graphics_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };
    let present_queue = unsafe { device.get_device_queue(present_queue_index, 0) };

    // Graphics Command Poolを作成する
    let graphics_command_pool = {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_queue_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        unsafe { device.create_command_pool(&command_pool_create_info, None)? }
    };

    // swapchainを作成する
    let swapchain_loader = Swapchain::new(&instance, &device);
    let (swapchain, format, extent) = {
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
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
            if graphics_queue_index != present_queue_index {
                (
                    vk::SharingMode::EXCLUSIVE,
                    vec![graphics_queue_index, present_queue_index],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
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
        (swapchain, format, extent)
    };
    let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

    // Uniform用のDescriptor周りを作成する
    // DescriptorPoolの作成
    let descriptor_pool = {
        let pool_sizes = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(swapchain_images.len() as u32)
            .build()];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(20)
            .pool_sizes(&pool_sizes);
        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None)? }
    };
    // Descriptor Set Layout Bindingsの作成
    let descriptor_set_layout_bindings = vec![vk::DescriptorSetLayoutBinding::builder()
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .binding(0)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build()];
    // Descriptor Set Layoutを作成
    let descriptor_set_layouts: Vec<_> = (0..swapchain_images.len())
        .map(|_| {
            let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(descriptor_set_layout_bindings.as_slice());
            unsafe {
                device
                    .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                    .unwrap()
            }
        })
        .collect();
    // Descriptor Setsを作成する
    let descriptor_sets = {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(descriptor_set_layouts.as_slice());
        unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
    }?;
    // Update Descriptor sets
    // 今回はUniformを使わないので無し

    // RenderPassを作成する
    let render_pass = {
        // Attachments
        let attachments = [
            vk::AttachmentDescription::builder()
                .format(format.format)
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
        unsafe { device.create_render_pass(&render_pass_create_info, None)? }
    };

    // vk-memのセットアップ
    let mut allocator = {
        let allocator_create_info = vk_mem::AllocatorCreateInfo {
            physical_device,
            device: device.clone(),
            instance: instance.clone(),
            flags: vk_mem::AllocatorCreateFlags::empty(),
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };
        vk_mem::Allocator::new(&allocator_create_info)?
    };

    // Framebufferの作成
    // depthバッファも同時に作成している
    let (framebuffers, depth_images_and_allocations, color_image_views, depth_image_views) = {
        let mut framebuffers = vec![];
        let mut depth_images_and_allocations = vec![];
        let mut color_image_views = vec![];
        let mut depth_image_views = vec![];
        for &image in swapchain_images.iter() {
            let mut attachments = vec![];

            let color_attachment = unsafe {
                device.create_image_view(
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
                )?
            };
            attachments.push(color_attachment);
            color_image_views.push(color_attachment);
            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
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
            let (depth_image, depth_image_allocation, _depth_image_allocation_info) =
                allocator.create_image(&depth_image_create_info, &depth_image_allocation_info)?;
            let depth_attachment = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(depth_image)
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
            attachments.push(depth_attachment);
            depth_image_views.push(depth_attachment);
            framebuffers.push(unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass)
                        .attachments(attachments.as_slice())
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )?
            });
            depth_images_and_allocations.push((depth_image, depth_image_allocation));
        }
        (
            framebuffers,
            depth_images_and_allocations,
            color_image_views,
            depth_image_views,
        )
    };

    // Graphics Pipelineを構築する
    let (graphics_pipeline, pipeline_layout) = {
        // シェーダの読み込み
        let vertex_shader_module = {
            use std::fs::File;
            use std::io::Read;

            let spv_file = File::open(&Path::new(VERTEX_SHADER_PASS))?;
            let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr() as *const u32,
                ..Default::default()
            };
            unsafe { device.create_shader_module(&shader_module_create_info, None)? }
        };
        let fragment_shader_module = {
            use std::fs::File;
            use std::io::Read;

            let spv_file = File::open(&Path::new(FRAGMENT_SHADER_PASS))?;
            let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

            let shader_module_create_info = vk::ShaderModuleCreateInfo {
                code_size: bytes_code.len(),
                p_code: bytes_code.as_ptr() as *const u32,
                ..Default::default()
            };
            unsafe { device.create_shader_module(&shader_module_create_info, None)? }
        };
        // main関数の名前
        let main_function_name = CString::new("main").unwrap();
        // shader stages infoの用意
        let pipeline_shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader_module)
                .name(&main_function_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader_module)
                .name(&main_function_name)
                .build(),
        ];
        //push constant rangeの用意
        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(
                std::mem::size_of::<Matrix4<f32>>() as u32 * 3
                    + std::mem::size_of::<Vector4<f32>>() as u32 * 2,
            )
            .build()];
        // Pipeline Layout
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&descriptor_set_layouts)
                    .push_constant_ranges(&push_constant_ranges),
                None,
            )?
        };
        // VertexInputBinding
        let vertex_input_binding = Vertex::get_binding_descriptions();
        // Vertex Input Attribute Descriptions
        let vertex_input_attribute = Vertex::get_attribute_descriptions();
        // input_assembly info
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        // viewport info
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);
        // rasterization_info
        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);
        // stencil op
        let stencil_op = vk::StencilOpState::builder()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS)
            .build();
        // depth stencil op
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(stencil_op)
            .back(stencil_op);
        // color blend attachments
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let color_blend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);
        // dynamic state
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
        // Vertex input state create info
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute)
            .vertex_binding_descriptions(&vertex_input_binding);
        // multi sample info
        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // pipeline create info
        let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&pipeline_shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .build()];
        // Graphics Pipeline
        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_info, None)
                .unwrap()[0]
        };
        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        (graphics_pipeline, pipeline_layout)
    };

    // 頂点の準備
    let vertices = {
        let model_obj = tobj::load_obj(MODEL_PATH, true)?;
        let mut vertices = vec![];
        let (models, _) = model_obj;
        for m in models.iter() {
            let mesh = &m.mesh;

            for &i in mesh.indices.iter() {
                let i = i as usize;
                let vertex = Vertex {
                    position: Vector3::new(
                        mesh.positions[3 * i],
                        mesh.positions[3 * i + 1],
                        mesh.positions[3 * i + 2],
                    ),
                    normal: Vector3::new(
                        mesh.normals[3 * i],
                        mesh.normals[3 * i + 1],
                        mesh.normals[3 * i + 2],
                    ),
                };
                vertices.push(vertex);
            }
        }

        vertices
    };
    let vertex_buffer_size = vertices.len() as u64 * std::mem::size_of::<Vertex>() as u64;

    // 頂点の一時バッファ
    let (
        temporary_vertex_buffer,
        temporary_vertex_buffer_allocation,
        _temporary_vertex_buffer_allocation_info,
    ) = allocator.create_buffer(
        &vk::BufferCreateInfo::builder()
            .size(vertex_buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC),
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuOnly,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        },
    )?;
    unsafe {
        let mapped_memory =
            allocator.map_memory(&temporary_vertex_buffer_allocation)? as *mut Vertex;
        mapped_memory.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
        allocator.unmap_memory(&temporary_vertex_buffer_allocation)?;
    }
    // 頂点バッファ
    let (vertex_buffer, vertex_buffer_allocation, _vertex_buffer_allocation_info) = allocator
        .create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(vertex_buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
        )?;

    // 同期に必要なオブジェクトを生成
    let (fences, image_acquired_semaphores, draw_complete_semaphores) = {
        let mut fences = vec![];
        let mut image_acquired_semaphores = vec![];
        let mut draw_complete_semaphores = vec![];
        for _ in 0..swapchain_images.len() {
            unsafe {
                fences.push(device.create_fence(
                    &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
                image_acquired_semaphores
                    .push(device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?);
                draw_complete_semaphores
                    .push(device.create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?);
            }
        }
        (fences, image_acquired_semaphores, draw_complete_semaphores)
    };

    // 頂点バッファを転送する
    {
        // 一時頂点バッファから頂点バッファへの頂点情報転送コマンド
        let copy_cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(graphics_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        unsafe {
            device.begin_command_buffer(
                copy_cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_copy_buffer(
                copy_cmd,
                temporary_vertex_buffer,
                vertex_buffer,
                &[vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(vertex_buffer_size)
                    .build()],
            );
            device.end_command_buffer(copy_cmd)?;

            device.queue_submit(
                graphics_queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[copy_cmd])
                    .build()],
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(graphics_queue)?;

            // コマンドバッファを解放
            device.free_command_buffers(graphics_command_pool, &[copy_cmd]);
        }

        // 一時バッファの削除
        allocator.destroy_buffer(temporary_vertex_buffer, &temporary_vertex_buffer_allocation)?;
    }

    // 描画用コマンドバッファの作成
    let graphics_command_buffers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(graphics_command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(swapchain_images.len() as u32),
        )?
    };
    for i in 0..swapchain_images.len() {
        let command_buffer = graphics_command_buffers[i];
        unsafe {
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
            )?;
            device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffers[i])
                    .render_area(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D::builder().x(0).y(0).build())
                            .extent(extent)
                            .build(),
                    )
                    .clear_values(&[
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ]),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::builder()
                    .width(extent.width as f32)
                    .height(extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)
                    .build()],
            );
            device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().build())
                    .extent(extent)
                    .build()],
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[descriptor_sets[i]],
                &[],
            );

            // push constants
            //   model matrix
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                Matrix4::from_scale(1.0).to_u8_slice(),
            );
            //    view matrix
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                std::mem::size_of::<Matrix4<f32>>() as u32,
                Matrix4::look_at(
                    Point3::new(500.0, -100.0, 0.0),
                    Point3::new(0.0, -400.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                )
                .to_u8_slice(),
            );
            //    projection matrix
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                std::mem::size_of::<Matrix4<f32>>() as u32 * 2,
                Matrix4::perspective(
                    Deg(90.0),
                    extent.width as f32 / extent.height as f32,
                    0.1,
                    5000.0,
                )
                .to_u8_slice(),
            );
            //    camera position
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                std::mem::size_of::<Matrix4<f32>>() as u32 * 3,
                Vector3::new(500.0, -30.0, 0.0).to_u8_slice(),
            );
            //    light position
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                std::mem::size_of::<Matrix4<f32>>() as u32 * 3
                    + std::mem::size_of::<Vector4<f32>>() as u32,
                Vector3::new(0.0, -800.0, 0.0).to_u8_slice(),
            );

            device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
            device.cmd_draw(command_buffer, vertices.len() as u32, 1, 0, 0);

            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;
        }
    }

    // メインループ
    let mut current_frame = 0;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => match input {
                KeyboardInput {
                    virtual_keycode,
                    state,
                    ..
                } => match (virtual_keycode, state) {
                    (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                },
            },
            Event::RedrawRequested(_window_id) => unsafe {
                // 描画処理
                let fence = fences[current_frame];
                device
                    .wait_for_fences(&[fence], true, std::u64::MAX)
                    .unwrap();
                device.reset_fences(&[fence]).unwrap();

                let (image_index, _is_suboptimal) = swapchain_loader
                    .acquire_next_image(
                        swapchain,
                        std::u64::MAX,
                        image_acquired_semaphores[current_frame],
                        vk::Fence::null(),
                    )
                    .unwrap();

                device
                    .queue_submit(
                        graphics_queue,
                        &[vk::SubmitInfo::builder()
                            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                            .wait_semaphores(&[image_acquired_semaphores[current_frame]])
                            .command_buffers(&[graphics_command_buffers[image_index as usize]])
                            .signal_semaphores(&[draw_complete_semaphores[current_frame]])
                            .build()],
                        fence,
                    )
                    .unwrap();

                swapchain_loader
                    .queue_present(
                        present_queue,
                        &vk::PresentInfoKHR::builder()
                            .wait_semaphores(&[draw_complete_semaphores[current_frame]])
                            .swapchains(&[swapchain])
                            .image_indices(&[image_index]),
                    )
                    .unwrap();

                current_frame = (current_frame + 1) % swapchain_images.len();
            },
            Event::LoopDestroyed => unsafe {
                device.queue_wait_idle(graphics_queue).unwrap();
                device.queue_wait_idle(present_queue).unwrap();

                // 後片付け
                for &image_acquired_semaphore in image_acquired_semaphores.iter() {
                    device.destroy_semaphore(image_acquired_semaphore, None);
                }
                for &draw_complete_semaphore in draw_complete_semaphores.iter() {
                    device.destroy_semaphore(draw_complete_semaphore, None);
                }
                for &fence in fences.iter() {
                    device.destroy_fence(fence, None);
                }
                allocator
                    .destroy_buffer(vertex_buffer, &vertex_buffer_allocation)
                    .unwrap();
                for &color_image_view in color_image_views.iter() {
                    device.destroy_image_view(color_image_view, None);
                }
                for &depth_image_view in depth_image_views.iter() {
                    device.destroy_image_view(depth_image_view, None);
                }
                device.destroy_pipeline(graphics_pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_render_pass(render_pass, None);
                for &framebuffer in framebuffers.iter() {
                    device.destroy_framebuffer(framebuffer, None);
                }
                for (depth_image, allocation) in depth_images_and_allocations.iter() {
                    allocator.destroy_image(*depth_image, allocation).unwrap();
                }

                allocator.destroy();
                for &layout in descriptor_set_layouts.iter() {
                    device.destroy_descriptor_set_layout(layout, None);
                }
                device.destroy_descriptor_pool(descriptor_pool, None);
                swapchain_loader.destroy_swapchain(swapchain, None);
                device.destroy_command_pool(graphics_command_pool, None);
                device.destroy_device(None);
                surface_loader.destroy_surface(surface, None);
                instance.destroy_instance(None);
            },
            _ => (),
        }
    });
}
