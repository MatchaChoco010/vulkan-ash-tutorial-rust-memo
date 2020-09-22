//! Windowの作成とイベントループの管理を行うラッパー。

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use crate::common::vulkan_app_base::{VulkanAppBase, VulkanAppBaseBuilder};

pub struct WindowEventLoop;

impl WindowEventLoop {
    /// VulkanAppBaseを作成し実行するための関数。
    /// VulkanAPpBaseBuilderを渡すと内部でbuildして実行する。
    ///
    /// ```
    /// let builder = DefaultVulkanAppBuilder::new();
    /// WindowEventLoop::run(builder);
    /// ```
    pub fn run(app_builder: impl VulkanAppBaseBuilder + 'static) -> ! {
        let (width, height) = app_builder.window_size();
        let title = app_builder.title();

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(width, height))
            .with_resizable(true)
            .build(&event_loop)
            .expect("Failed to create window");

        let mut app = app_builder.build(&window).expect("Failed to init app");
        app.prepare().expect("Failed to call prepare");

        let mut mouse_button_pressed = [false, false, false];
        let mut is_minimize = false;

        event_loop.run(move |event, _, control_flow| match event {
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
            Event::WindowEvent {
                event: WindowEvent::Resized(PhysicalSize { width, height }),
                ..
            } => {
                is_minimize = width == 0 || height == 0;
                if is_minimize {
                    return;
                }
                app.on_window_size_changed(width, height)
                    .expect("Failed to resize app");
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { state, button, .. },
                ..
            } => {
                match state {
                    ElementState::Pressed => {
                        match button {
                            MouseButton::Left => {
                                if mouse_button_pressed[0] == false {
                                    // MouseDown
                                    app.on_mouse_button_down(0)
                                        .expect("Failed to handle mouse down");
                                }
                                mouse_button_pressed[0] = true;
                            }
                            MouseButton::Right => {
                                if mouse_button_pressed[1] == false {
                                    // MouseDown
                                    app.on_mouse_button_down(1)
                                        .expect("Failed to handle mouse down");
                                }
                                mouse_button_pressed[1] = true;
                            }
                            MouseButton::Middle => {
                                if mouse_button_pressed[2] == false {
                                    // MouseDown
                                    app.on_mouse_button_down(2)
                                        .expect("Failed to handle mouse down");
                                }
                                mouse_button_pressed[2] = true;
                            }
                            _ => (),
                        }
                    }
                    ElementState::Released => {
                        match button {
                            MouseButton::Left => {
                                // MouseUp
                                app.on_mouse_button_up(0)
                                    .expect("Failed to handle mouse down");
                                mouse_button_pressed[0] = false;
                            }
                            MouseButton::Right => {
                                // MouseUp
                                app.on_mouse_button_up(1)
                                    .expect("Failed to handle mouse down");
                                mouse_button_pressed[1] = false;
                            }
                            MouseButton::Middle => {
                                // MouseUp
                                app.on_mouse_button_up(2)
                                    .expect("Failed to handle mouse down");
                                mouse_button_pressed[2] = false;
                            }
                            _ => (),
                        }
                    }
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_window_id) => {
                if is_minimize {
                    return;
                }
                app.render().expect("Failed to call render");
            }
            Event::LoopDestroyed => (),
            _ => (),
        })
    }
}
