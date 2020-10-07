use std::rc::Rc;

use cgmath::Matrix4;

use crate::cubemap_app::mesh::Mesh;

pub struct MeshEntity {
    mesh: Rc<Mesh>,
    model: Matrix4<f32>,
}
impl MeshEntity {
    pub fn new(mesh: Rc<Mesh>, model_matrix: Matrix4<f32>) -> Self {
        Self {
            mesh,
            model: model_matrix,
        }
    }

    pub fn set_model_matrix(&mut self, model_matrix: Matrix4<f32>) {
        self.model = model_matrix;
    }

    pub fn get_model_matrix(&self) -> Matrix4<f32> {
        self.model
    }

    pub fn get_mesh(&self) -> &Mesh {
        self.mesh.as_ref()
    }
}
