use anyhow::Result;
use ash::vk;
use cgmath::{Matrix4, Point3, Rad, SquareMatrix, Vector3, Vector4};

use crate::{
    common::Matrix4Ext,
    vk_wrapper_object::{HostVisibleBufferObject, VkMemAllocatorObject},
};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CameraUniformObject {
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    rotation: Matrix4<f32>,
    eye: Vector4<f32>,
    fovy: f32,
    aspect: f32,
}

pub struct Camera {
    uniform_object: CameraUniformObject,
    uniform: HostVisibleBufferObject,
}
impl Camera {
    pub fn new<A: Into<Rad<f32>>>(
        allocator: &VkMemAllocatorObject,
        eye: Point3<f32>,
        center: Point3<f32>,
        up: Vector3<f32>,
        fovy: A,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Result<Self> {
        let fovy: Rad<_> = fovy.into();
        let view = Matrix4::look_at(eye, center, up);
        let proj = Matrix4::perspective(fovy, aspect, near, far);
        let rotation = view
            .invert()
            .expect("Failed to invert camera view matrix.")
            .rotate_scale_matrix();
        let fovy = fovy.0;
        let eye = eye.to_homogeneous();

        let uniform_object = CameraUniformObject {
            view,
            proj,
            rotation,
            eye,
            fovy,
            aspect,
        };

        let uniform = HostVisibleBufferObject::new(
            allocator,
            &[uniform_object],
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        Ok(Self {
            uniform_object,
            uniform,
        })
    }

    pub fn set_view(&mut self, eye: Point3<f32>, center: Point3<f32>, up: Vector3<f32>) {
        self.uniform_object.view = Matrix4::look_at(eye, center, up);
        self.uniform_object.rotation = self
            .uniform_object
            .view
            .invert()
            .expect("Failed to invert camera view matrix.")
            .rotate_scale_matrix();
        self.uniform_object.eye = eye.to_homogeneous();
    }

    pub fn set_proj<A: Into<Rad<f32>>>(&mut self, fovy: A, aspect: f32, near: f32, far: f32) {
        let fovy: Rad<_> = fovy.into();
        self.uniform_object.proj = Matrix4::perspective(fovy, aspect, near, far);
        self.uniform_object.fovy = fovy.0;
        self.uniform_object.aspect = aspect;
    }

    pub fn update_uniform(&self) -> Result<()> {
        self.uniform.map_data(&[self.uniform_object])?;
        Ok(())
    }

    pub fn uniform(&self) -> &HostVisibleBufferObject {
        &self.uniform
    }
}
