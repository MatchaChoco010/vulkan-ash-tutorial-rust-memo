use winit::{
    dpi::LogicalSize,
    window::{Window, WindowBuilder},
    event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent},
    event_loop::{EventLoop, ControlFlow},
};

const WINDOW_TITLE: &'static str = "Vulkan";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

struct VulkanApp;

impl VulkanApp {
    fn init_window(event_loop: &EventLoop<()>) -> Window {
        WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(event_loop)
        .expect("Failed to create window.")
    }

    pub fn main_loop(event_loop: EventLoop<()>) {
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
                _ => ()
            }
        })
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let _window = VulkanApp::init_window(&event_loop);

    VulkanApp::main_loop(event_loop);
}
