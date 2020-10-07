use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

use anyhow::Result;
use ash::vk;
use cgmath::Matrix4;

use crate::{
    cubemap_app::{camera::Camera, mesh::Mesh, mesh_entity::MeshEntity},
    vk_wrapper_object::{CommandPoolObject, DeviceObject, ImageObject, VkMemAllocatorObject},
};

pub struct SceneBuilder {
    camera: Option<Camera>,
    entities: Vec<(String, Matrix4<f32>)>,
    cubemap_texture_xp: Option<String>,
    cubemap_texture_xm: Option<String>,
    cubemap_texture_yp: Option<String>,
    cubemap_texture_ym: Option<String>,
    cubemap_texture_zp: Option<String>,
    cubemap_texture_zm: Option<String>,
}
impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            camera: None,
            entities: vec![],
            cubemap_texture_xp: None,
            cubemap_texture_xm: None,
            cubemap_texture_yp: None,
            cubemap_texture_ym: None,
            cubemap_texture_zp: None,
            cubemap_texture_zm: None,
        }
    }

    pub fn add_camera(self, camera: Camera) -> Self {
        Self {
            camera: Some(camera),
            ..self
        }
    }

    pub fn add_entity(
        mut self,
        obj_filename: impl Into<String>,
        model_matrix: Matrix4<f32>,
    ) -> Self {
        self.entities.push((obj_filename.into(), model_matrix));
        self
    }

    pub fn add_cubemap_textures(
        self,
        cubemap_texture_xp: impl Into<String>,
        cubemap_texture_xm: impl Into<String>,
        cubemap_texture_yp: impl Into<String>,
        cubemap_texture_ym: impl Into<String>,
        cubemap_texture_zp: impl Into<String>,
        cubemap_texture_zm: impl Into<String>,
    ) -> Self {
        Self {
            cubemap_texture_xp: Some(cubemap_texture_xp.into()),
            cubemap_texture_xm: Some(cubemap_texture_xm.into()),
            cubemap_texture_yp: Some(cubemap_texture_yp.into()),
            cubemap_texture_ym: Some(cubemap_texture_ym.into()),
            cubemap_texture_zp: Some(cubemap_texture_zp.into()),
            cubemap_texture_zm: Some(cubemap_texture_zm.into()),
            ..self
        }
    }

    pub fn build(
        self,
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        graphics_command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
    ) -> Result<Scene> {
        let mut model_filenames = HashSet::<String>::new();
        for (filename, _) in self.entities.iter() {
            model_filenames.insert(filename.clone());
        }
        let mut meshes = vec![];
        for filename in model_filenames.iter() {
            meshes.push(Rc::new(Mesh::new(
                device,
                allocator,
                graphics_command_pool,
                graphics_queue,
                filename,
            )?));
        }
        let mut model_sets = HashMap::<String, Rc<Mesh>>::new();
        for (filename, model) in model_filenames.iter().zip(meshes.iter()) {
            model_sets.insert(filename.clone(), Rc::clone(model));
        }
        let mut entities = vec![];
        for (filename, model_matirx) in self.entities {
            let mesh = Rc::clone(model_sets.get(&filename).expect("Failed to get Mesh"));
            let mesh_entity = MeshEntity::new(mesh, model_matirx);
            entities.push(mesh_entity);
        }

        let cubemap = ImageObject::load_cubemap_images(
            device,
            allocator,
            graphics_command_pool,
            graphics_queue,
            self.cubemap_texture_xp.expect(
                "Failed to get cubemap texture. You may not have called add_cubemap_texture.",
            ),
            self.cubemap_texture_xm.expect(
                "Failed to get cubemap texture. You may not have called add_cubemap_texture.",
            ),
            self.cubemap_texture_yp.expect(
                "Failed to get cubemap texture. You may not have called add_cubemap_texture.",
            ),
            self.cubemap_texture_ym.expect(
                "Failed to get cubemap texture. You may not have called add_cubemap_texture.",
            ),
            self.cubemap_texture_zp.expect(
                "Failed to get cubemap texture. You may not have called add_cubemap_texture.",
            ),
            self.cubemap_texture_zm.expect(
                "Failed to get cubemap texture. You may not have called add_cubemap_texture.",
            ),
        )?;

        Ok(Scene {
            camera: self
                .camera
                .expect("Failed to get camera. You may not have called add_camera."),
            entities,
            cubemap,
        })
    }
}

pub struct Scene {
    camera: Camera,
    entities: Vec<MeshEntity>,
    cubemap: ImageObject,
}
impl Scene {
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn entities(&self) -> &Vec<MeshEntity> {
        &self.entities
    }

    pub fn cubemap(&self) -> &ImageObject {
        &self.cubemap
    }
}
