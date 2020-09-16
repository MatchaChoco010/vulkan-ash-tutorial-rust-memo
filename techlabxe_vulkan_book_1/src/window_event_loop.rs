//! Windowの作成とイベントループの管理を行うラッパー。

use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use crate::constants::{WINDOW_HEIGHT, WINDOW_TITLE, WINDOW_WIDTH};
use crate::vulkan_app_base::VulkanAppBase;

pub struct WindowEventLoop;

impl WindowEventLoop {
    /// VulkanAppBaseを実行するための関数。
    ///
    /// ```
    /// use vulkan_app_base::DefaultVulkanApp;
    /// WindowEventLoop::run(DefaultVulkanApp::new())
    /// ````
    pub fn run(mut app: impl VulkanAppBase + 'static) -> ! {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32))
            .with_resizable(false)
            .build(&event_loop)
            .expect("Failed to create window");

        app.init(&window).expect("Failed to init app");

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
                Event::MainEventsCleared => {
                    app.prepare().expect("Failed to call prepare");
                    window.request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    app.render().expect("Failed to call render");
                }
                Event::LoopDestroyed => {
                    app.cleanup().expect("Failed to call terminate");
                }
                _ => (),
            }
        })
    }
}
