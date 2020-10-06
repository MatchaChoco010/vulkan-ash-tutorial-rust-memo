use anyhow::Result;
use ash::{extensions::khr::Surface, vk};
use ash_window::create_surface;
use winit::window::Window;

use crate::vk_wrapper_object::{EntryObject, InstanceObject};

pub struct SurfaceObject {
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
}
impl SurfaceObject {
    /// Surfaceの作成を行う。
    pub fn new(entry: &EntryObject, instance: &InstanceObject, window: &Window) -> Result<Self> {
        let entry = entry.entry_as_ref();
        let instance = instance.instance_as_ref();
        let surface_loader = Surface::new(entry, instance);
        let surface = unsafe { create_surface(entry, instance, window, None)? };

        Ok(Self {
            surface_loader,
            surface,
        })
    }

    /// surface_loaderを取得する。
    pub fn surface_loader(&self) -> &Surface {
        &self.surface_loader
    }

    /// surfaceを取得する。
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }
}
impl Drop for SurfaceObject {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
