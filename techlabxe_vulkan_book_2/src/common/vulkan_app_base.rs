//! アプリケーションのベースのトレイト。
//! Vulkanのリソースの管理。
#![allow(dead_code)]

use anyhow::Result;
use winit::window::Window;

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
    fn on_mouse_button_down(&mut self, _button: i32) -> Result<()> {
        Ok(())
    }
    /// マウスアップ。
    fn on_mouse_button_up(&mut self, _button: i32) -> Result<()> {
        Ok(())
    }
    /// マウスムーブ。
    fn on_mouse_move(&mut self, _dx: i32, _dy: i32) -> Result<()> {
        Ok(())
    }

    /// 描画の準備を行う。
    fn prepare(&mut self) -> Result<()>;

    /// 描画処理を行う。
    fn render(&mut self) -> Result<()>;
}
