//! アプリケーションのベースのトレイト。
//! Vulkanのリソースの管理。
use anyhow::Result;
use winit::{event::Event, window::Window};

pub trait VulkanAppBaseBuilder {
    type Item: VulkanAppBase;

    /// ビルダーを新しく生成して返す。
    fn new() -> Self;

    /// window_sizeを取得する。
    fn window_size(&self) -> (u32, u32);

    /// ウィンドウタイトルを取得する。
    fn title(&self) -> &str;

    /// アプリケーションのビルドをする。
    fn build(self, window: &Window) -> Result<Self::Item>;
}

pub trait VulkanAppBase: Drop {
    /// ウィンドウサイズ変化時のイベント関数。
    fn on_window_size_changed(&mut self, width: u32, height: u32) -> Result<()>;
    /// マウスダウン。
    fn on_mouse_button_down(&mut self, _button: i32) {}
    /// マウスアップ。
    fn on_mouse_button_up(&mut self, _button: i32) {}
    /// マウスムーブ。
    fn on_mouse_move(&mut self, _x: f64, _y: f64) {}
    /// イベント開始
    fn on_new_events(&mut self) {}
    /// メインイベントクリア
    fn on_main_events_cleared(&mut self, _window: &Window) {}
    /// handle event
    fn handle_event(&mut self, _window: &Window, _event: &Event<()>) {}

    /// 描画処理を行う。
    fn render(&mut self, window: &Window) -> Result<()>;
}
