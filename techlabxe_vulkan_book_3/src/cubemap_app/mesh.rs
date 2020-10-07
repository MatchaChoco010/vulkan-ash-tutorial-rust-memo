use anyhow::Result;
use ash::vk;
use cgmath::Vector3;
use tobj;

use crate::vk_wrapper_object::{
    CommandPoolObject, DeviceLocalBufferObject, DeviceObject, VkMemAllocatorObject,
};

#[repr(C)]
pub struct MeshVertex {
    pub position: Vector3<f32>,
    pub normal: Vector3<f32>,
}

pub struct Mesh {
    buffer: DeviceLocalBufferObject,
    vertex_count: u32,
}
impl Mesh {
    pub fn new(
        device: &DeviceObject,
        allocator: &VkMemAllocatorObject,
        graphics_command_pool: &CommandPoolObject,
        graphics_queue: vk::Queue,
        filename: impl Into<String>,
    ) -> Result<Self> {
        let model_obj = tobj::load_obj(filename.into(), true)?;
        let mut vertices = vec![];
        let (models, _) = model_obj;
        for m in models.iter() {
            let mesh = &m.mesh;

            for &i in mesh.indices.iter() {
                let i = i as usize;
                let vertex = MeshVertex {
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

        let buffer = DeviceLocalBufferObject::new(
            device,
            allocator,
            graphics_command_pool,
            graphics_queue,
            vertices.as_slice(),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        Ok(Self {
            buffer,
            vertex_count: vertices.len() as u32,
        })
    }

    pub fn buffer(&self) -> &DeviceLocalBufferObject {
        &self.buffer
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
}
