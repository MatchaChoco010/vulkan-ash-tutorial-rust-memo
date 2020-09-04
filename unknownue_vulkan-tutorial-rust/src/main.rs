use std::{
    ptr,
    ffi::CString,
};
use ash::{
    Entry,
    Instance,
    vk,
    version::{EntryV1_0, InstanceV1_0},
};
use winit::{
    dpi::LogicalSize,
    window::{Window, WindowBuilder},
    event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent},
    event_loop::{EventLoop, ControlFlow},
};

mod platforms;

const WINDOW_TITLE: &'static str = "Vulkan";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

struct VulkanApp{
    _entry: Entry,
    instance: Instance,
}

impl VulkanApp {
    pub fn new() -> Self {
        let entry = Entry::new().unwrap();
        let instance = Self::create_instance(&entry);

        Self {
            _entry: entry,
            instance
        }
    }

    fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(event_loop)
        .expect("Failed to create window.")
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_name = CString::new(WINDOW_TITLE).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_version(0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_version(0, 1, 0),
            api_version: vk::make_version(1, 2, 0),
        };

        let extension_names = platforms::required_extension_names();

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &app_info,
            pp_enabled_layer_names: ptr::null(),
            enabled_layer_count: 0,
            pp_enabled_extension_names: extension_names.as_ptr(),
            enabled_extension_count: extension_names.len() as u32,
        };

        let instance: Instance = unsafe {
            entry.create_instance(&create_info, None).expect("Failed to create instance!")
        };

        instance
    }

    fn draw_frame(&mut self) {
        // Drawing will be here
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) {
        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::WindowEvent {event: WindowEvent::KeyboardInput{input, ..}, ..} => {
                    match input {
                        KeyboardInput {virtual_keycode, state, ..} => {
                            match (virtual_keycode, state) {
                                (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                    *control_flow = ControlFlow::Exit;
                                }
                                _ => ()
                            }
                        }
                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    self.draw_frame();
                }
                _ => ()
            }
        })
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = VulkanApp::init_window(&event_loop);

    let vulkan_app = VulkanApp::new();
    vulkan_app.main_loop(event_loop, window);
}
