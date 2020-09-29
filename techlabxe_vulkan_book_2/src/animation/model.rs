#![allow(dead_code)]

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
    rc::Rc,
};

use anyhow::Result;
use ash::{version::DeviceV1_0, vk, Device, Instance};
use cgmath::{
    Deg, InnerSpace, Matrix4, Point3, Quaternion, SquareMatrix, Vector2, Vector3, Vector4,
};
use image::GenericImageView;
use memoffset::offset_of;

use crate::{
    animation::pmd_loader,
    common::{
        cgmath_ext::Matrix4Ext,
        default_vulkan_app_base::DefaultVulkanAppBase,
        vulkan_objects::{
            DescriptorPoolObject, DescriptorSetLayoutObject, DeviceLocalBufferObject,
            HostVisibleBufferObject, PipelineObject, RenderPassObject, SamplerObject,
            ShaderReadOnlyImageObject, TextureRenderingDepthObject,
        },
    },
};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MaterialParameters {
    diffuse: Vector4<f32>,
    ambient: Vector4<f32>,
    specular: Vector4<f32>,
    use_texture: u8,
    edge_flag: u8,
}

pub struct Material {
    parameters: MaterialParameters,
    uniform_buffer: Option<HostVisibleBufferObject>,
    texture: Option<ShaderReadOnlyImageObject>,
    descriptor_sets: Vec<vk::DescriptorSet>,
}
impl Material {
    pub fn new(params: MaterialParameters) -> Self {
        Self {
            parameters: params,
            uniform_buffer: None,
            texture: None,
            descriptor_sets: vec![],
        }
    }

    pub fn has_texture(&self) -> bool {
        self.parameters.use_texture != 0
    }

    pub fn get_texture(&self) -> &ShaderReadOnlyImageObject {
        self.texture
            .as_ref()
            .expect("This material has no texture.")
    }

    pub fn set_texture(&mut self, texture: ShaderReadOnlyImageObject) {
        self.texture = Some(texture);
    }

    pub fn get_uniform_buffer(&self) -> &HostVisibleBufferObject {
        self.uniform_buffer
            .as_ref()
            .expect("This material has not set uniform buffer object.")
    }

    pub fn set_uniform_buffer(&mut self, uniform_buffer: HostVisibleBufferObject) {
        self.uniform_buffer = Some(uniform_buffer);
    }

    pub fn get_descriptor_set(&self, index: usize) -> vk::DescriptorSet {
        self.descriptor_sets[index]
    }

    pub fn set_descriptor_sets(&mut self, descriptor_sets: Vec<vk::DescriptorSet>) {
        self.descriptor_sets = descriptor_sets;
    }

    pub fn get_edge_flag(&self) -> u8 {
        self.parameters.edge_flag
    }

    pub fn update(&mut self) {
        self.get_uniform_buffer()
            .map_data(&[self.parameters])
            .expect("Failed to map material uniform buffer.")
    }
}

#[derive(Debug)]
pub struct Bone {
    name: String,

    translation: Vector3<f32>,
    rotation: Quaternion<f32>,
    initial_translation: Vector3<f32>,

    parent: Option<Rc<RefCell<Bone>>>,
    children: Vec<Rc<RefCell<Bone>>>,

    local: Matrix4<f32>,
    world: Matrix4<f32>,
    inv_bind: Matrix4<f32>,
}
impl Bone {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            translation: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            initial_translation: Vector3::new(0.0, 0.0, 0.0),
            parent: None,
            children: vec![],
            local: Matrix4::from_scale(1.0),
            world: Matrix4::from_scale(1.0),
            inv_bind: Matrix4::from_scale(1.0),
        }
    }

    pub fn set_translation(&mut self, trans: Vector3<f32>) {
        self.translation = trans;
    }
    pub fn set_rotation(&mut self, rot: Quaternion<f32>) {
        self.rotation = rot;
    }
    pub fn get_translation(&self) -> &Vector3<f32> {
        &self.translation
    }
    pub fn get_rotation(&self) -> &Quaternion<f32> {
        &self.rotation
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn get_local_matrix(&self) -> &Matrix4<f32> {
        &self.local
    }
    pub fn get_world_matrix(&self) -> &Matrix4<f32> {
        &self.world
    }
    pub fn get_inv_bind_matrix(&self) -> &Matrix4<f32> {
        &self.inv_bind
    }

    pub fn update_local_matrix(&mut self) {
        self.local =
            Matrix4::from_translation(self.translation) * Matrix4::<f32>::from(self.rotation);
    }
    pub fn update_world_matrix(&mut self) {
        self.update_local_matrix();
        let mut parent_matrix = Matrix4::from_scale(1.0);
        if let Some(p) = &self.parent {
            parent_matrix = *Rc::clone(p).borrow().get_world_matrix();
        }
        self.world = parent_matrix * self.local;
    }

    pub fn set_initial_translation(&mut self, trans: Vector3<f32>) {
        self.initial_translation = trans;
    }

    pub fn set_inv_bind_matrix(&mut self, inv_bind: Matrix4<f32>) {
        self.inv_bind = inv_bind;
    }
}
pub trait RcRefCellBoneExt {
    fn set_parent(&mut self, parent: Rc<RefCell<Bone>>);
    fn update_matrices(&mut self);
}
impl RcRefCellBoneExt for Rc<RefCell<Bone>> {
    fn set_parent(&mut self, parent: Rc<RefCell<Bone>>) {
        self.borrow_mut().parent = Some(parent);
        self.borrow_mut()
            .parent
            .as_ref()
            .expect("Failed to get parent.")
            .borrow_mut()
            .children
            .push(Rc::clone(self));
    }

    fn update_matrices(&mut self) {
        self.borrow_mut().update_world_matrix();
        for child in self.borrow().children.iter() {
            Rc::clone(child).update_matrices();
        }
    }
}

pub struct PMDBoneIK {
    effector: Rc<RefCell<Bone>>,
    target: Rc<RefCell<Bone>>,
    ik_chains: Vec<Rc<RefCell<Bone>>>,
    angle_limit: f32,
    iteration: i32,
}
impl PMDBoneIK {
    pub fn new(target: Rc<RefCell<Bone>>, eff: Rc<RefCell<Bone>>) -> Self {
        Self {
            target,
            effector: eff,
            ik_chains: vec![],
            angle_limit: 0.0,
            iteration: 0,
        }
    }

    pub fn get_effector(&self) -> Rc<RefCell<Bone>> {
        Rc::clone(&self.effector)
    }
    pub fn get_target(&self) -> Rc<RefCell<Bone>> {
        Rc::clone(&self.target)
    }
    pub fn get_angle_weight(&self) -> f32 {
        self.angle_limit
    }
    pub fn get_chains(&self) -> &Vec<Rc<RefCell<Bone>>> {
        &self.ik_chains
    }
    pub fn get_iteration_count(&self) -> i32 {
        self.iteration
    }

    pub fn set_angle_limit(&mut self, angle: f32) {
        self.angle_limit = angle;
    }
    pub fn set_iteration_count(&mut self, iteration_count: i32) {
        self.iteration = iteration_count;
    }
    pub fn set_ik_chains(&mut self, chains: Vec<Rc<RefCell<Bone>>>) {
        self.ik_chains = chains;
    }
}

pub struct Mesh {
    start_index_offset: u32,
    index_count: u32,
}
#[repr(C)]
pub struct PMDVertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
    texcoord: Vector2<f32>,
    bone_indices: Vector2<u32>,
    bone_weights: Vector2<f32>,
    edge_flag: u32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SceneParameter {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
    pub light_direction: Vector4<f32>,
    pub eye_position: Vector4<f32>,
    pub light_view_proj: Matrix4<f32>,
    pub light_view_proj_bias: Matrix4<f32>,
    pub resolution: Vector2<f32>,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BoneParameter {
    bone: [Matrix4<f32>; 512],
}
pub struct PMDFaceBaseInfo {
    indices: Vec<u32>,
    vertices_pos: Vec<Vector3<f32>>,
}
pub struct PMDFaceInfo {
    pub name: String,
    indices: Vec<u32>,
    vertices_offset: Vec<Vector3<f32>>,
}

pub struct Model {
    pub model_name: String,
    pub model_version: f32,
    pub model_comment: String,

    host_mem_vertices: Vec<PMDVertex>,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    scene_param: SceneParameter,
    bone_matrices: BoneParameter,

    vertex_buffers: Vec<HostVisibleBufferObject>,
    bone_ubo: Vec<HostVisibleBufferObject>,
    scene_param_ubo: Vec<HostVisibleBufferObject>,

    index_buffer: DeviceLocalBufferObject,

    command_buffers: Vec<Vec<vk::CommandBuffer>>,
    command_buffers_outline: Vec<Vec<vk::CommandBuffer>>,
    command_buffers_shadow: Vec<Vec<vk::CommandBuffer>>,

    pub shadow_map: TextureRenderingDepthObject,
    dummy_texture: ShaderReadOnlyImageObject,
    sampler: SamplerObject,

    descriptor_pool: DescriptorPoolObject,
    descriptor_set_layout: DescriptorSetLayoutObject,
    pub render_pass_default: RenderPassObject,
    pub render_pass_shadow: RenderPassObject,

    pipelines: HashMap<String, PipelineObject>,
    bones: Vec<Rc<RefCell<Bone>>>,

    face_base_info: PMDFaceBaseInfo,
    pub face_offset_info: Vec<PMDFaceInfo>,
    face_morph_weights: Vec<f32>,

    bone_ik_list: Vec<PMDBoneIK>,
}
impl Model {
    pub fn load(filename: impl Into<String>, app: &DefaultVulkanAppBase) -> Result<Self> {
        let filename: String = filename.into();
        let mut cur = Cursor::new(BufReader::new(File::open(&Path::new(&filename))?));
        let file = pmd_loader::PMDFile::new(cur.get_mut());

        let model_name = file.name().to_string();
        let model_version = file.version();
        let model_comment = file.comment().to_string();

        let vertex_count = file.vertex_count();
        let file_vertices = file.vertices();
        let mut host_mem_vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let v = &file_vertices[i];
            host_mem_vertices.push(PMDVertex {
                position: v.position(),
                normal: v.normal(),
                texcoord: v.texcoord(),
                bone_indices: Vector2::new(v.bone_index(0) as u32, v.bone_index(1) as u32),
                bone_weights: Vector2::new(v.bone_weight(0), v.bone_weight(1)),
                edge_flag: v.edge_flag() as u32,
            });
        }

        let index_count = file.index_count();
        let file_indices = file.indices();
        let mut indices = Vec::with_capacity(index_count);
        for i in 0..index_count {
            indices.push(file_indices[i]);
        }

        let index_buffer = DeviceLocalBufferObject::new(
            app.allocator(),
            app.device(),
            &app.graphics_command_pool(),
            &app.graphics_queue(),
            indices.as_slice(),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        let image_count = app.swapchain().len();
        let mut vertex_buffers = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            vertex_buffers.push(HostVisibleBufferObject::new(
                app.allocator(),
                host_mem_vertices.as_slice(),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?);
        }

        // マテリアル読み込み
        let material_count = file.material_count();
        let mut materials = Vec::with_capacity(material_count);
        for i in 0..material_count {
            let src = file.material(i);

            let mut material_params = MaterialParameters {
                diffuse: Vector4::new(
                    src.diffuse().x,
                    src.diffuse().y,
                    src.diffuse().z,
                    src.alpha(),
                ),
                ambient: Vector4::new(src.ambient().x, src.ambient().y, src.ambient().z, 0.0),
                specular: Vector4::new(
                    src.specular().x,
                    src.specular().y,
                    src.specular().z,
                    src.shininess(),
                ),
                use_texture: 0,
                edge_flag: src.edge_flag(),
            };

            let mut texture_filename = src.texture();
            let has_sphere_map = texture_filename.find("*").is_some();
            if has_sphere_map {
                texture_filename = texture_filename.split('*').collect::<Vec<_>>()[0];
            }
            if !texture_filename.is_empty() {
                material_params.use_texture = 1;
            }
            let mut material = Material::new(material_params);
            let uniform_buffer = HostVisibleBufferObject::new(
                app.allocator(),
                &[material_params],
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            material.set_uniform_buffer(uniform_buffer);

            if material_params.use_texture != 0 {
                let img = image::open(Path::new("assets/Model").join(Path::new(texture_filename)))?;
                let texture = ShaderReadOnlyImageObject::load_image(
                    app.allocator(),
                    app.device(),
                    &app.graphics_command_pool(),
                    &app.graphics_queue(),
                    img,
                )?;
                material.set_texture(texture);
            }

            material.update();
            materials.push(material);
        }

        // 描画用メッシュ情報構築
        let mut start_index_offset = 0;
        let mut meshes = Vec::with_capacity(material_count);
        for i in 0..material_count {
            let src = file.material(i);
            let index_count = src.number_of_polygons();
            meshes.push(Mesh {
                start_index_offset,
                index_count: index_count,
            });
            start_index_offset += index_count;
        }

        // ボーン情報構築
        let bone_count = file.bone_count();
        let mut bones = Vec::with_capacity(bone_count);
        for i in 0..bone_count {
            let bone_src = file.bone(i);
            let index = bone_src.parent();

            let mut bone = Bone::new(bone_src.name());
            let mut translation = bone_src.position();
            if index != 0xFFFF_u16 {
                let parent = file.bone(bone_src.parent() as usize);
                translation = translation - parent.position();
            }
            bone.set_translation(translation);
            bone.set_initial_translation(translation);

            // バインド逆行列をグローバル位置より求める
            let inv_bind = Matrix4::from_translation(bone_src.position())
                .invert()
                .expect("Failed to invert bind matrix");
            bone.set_inv_bind_matrix(inv_bind);

            bones.push(Rc::new(RefCell::new(bone)));
        }
        for i in 0..bone_count {
            let bone_src = file.bone(i);
            let mut bone: Rc<RefCell<Bone>> = Rc::clone(&bones[i]);
            let index = bone_src.parent();
            if index != 0xFFFF_u16 {
                bone.set_parent(Rc::clone(&bones[index as usize]));
            }
        }

        // 表情モーフ情報読み込み
        let (face_base_info, face_offset_info, face_morph_weights) = {
            // 表情ベース
            let base_face = file.face_base();
            let face_base_info = PMDFaceBaseInfo {
                vertices_pos: base_face.face_vertices().clone(),
                indices: base_face.face_indices().clone(),
            };

            // オフセット表情モーフ
            let face_count = file.face_count() - 1;
            let mut face_offset_info = Vec::with_capacity(face_count);
            for i in 0..face_count {
                let face_src = file.face(i + 1);
                let face = PMDFaceInfo {
                    name: face_src.name().to_string(),
                    vertices_offset: face_src.face_vertices().clone(),
                    indices: face_src.face_indices().clone(),
                };
                face_offset_info.push(face);
            }

            let face_morph_weights = vec![0.0; face_count];

            (face_base_info, face_offset_info, face_morph_weights)
        };

        // update Matricesに当たる処理

        let bone_matrices = BoneParameter {
            bone: [Matrix4::from_scale(1.0); 512],
        };

        // IKボーン情報を読み込む
        let ik_bone_count = file.ik_count();
        let mut bone_ik_list = Vec::with_capacity(ik_bone_count);
        for i in 0..ik_bone_count {
            let ik = file.ik(i);
            let target_bone = Rc::clone(&bones[ik.target_bone_id() as usize]);
            let effector_bone = Rc::clone(&bones[ik.bone_eff() as usize]);
            let mut bone_ik = PMDBoneIK::new(target_bone, effector_bone);
            bone_ik.set_angle_limit(ik.angle_limit());
            bone_ik.set_iteration_count(ik.iterations() as i32);

            let chains = ik.chains();
            let mut ik_chains = Vec::with_capacity(chains.len());
            for &id in chains.iter() {
                ik_chains.push(Rc::clone(&bones[id as usize]));
            }
            bone_ik.set_ik_chains(ik_chains);

            bone_ik_list.push(bone_ik);
        }

        let dummy_texture = Self::prepare_dummy_texture(app)?;

        let descriptor_pool =
            Self::prepare_descriptor_pool(app, (materials.len() * image_count) as u32)?;
        let descriptor_set_layout = Self::prepare_descriptor_set_layout(app)?;
        let (render_pass_default, render_pass_shadow) = Self::prepare_render_passes(app)?;

        let pipelines = Self::prepare_pipelines(
            app,
            &descriptor_set_layout,
            &render_pass_default,
            &render_pass_shadow,
        )?;

        let shadow_map =
            TextureRenderingDepthObject::new(app.device(), app.allocator(), 1024, 1024)?;

        let sampler = SamplerObject::new(app.device())?;

        let scene_param = SceneParameter {
            view: Matrix4::look_at(
                Point3::new(0.0, 10.0, 25.0),
                Point3::new(0.0, 10.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ),
            proj: Matrix4::perspective(Deg(45.0), app.swapchain().aspect(), 0.01, 100.0),
            light_direction: Vector4::new(1.0, 1.0, 1.0, 0.0).normalize(),
            eye_position: Vector4::new(0.0, 10.0, 25.0, 1.0),
            light_view_proj: Matrix4::ortho(-20.0, 20.0, -20.0, 20.0, 0.0, 500.0)
                * Matrix4::look_at(
                    Point3::new(50.0, 50.0, 50.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                ),
            light_view_proj_bias: Matrix4::from_translation(Vector3::new(0.5, 0.5, 0.0))
                * Matrix4::from_nonuniform_scale(0.5, 0.5, 1.0),
            resolution: Vector2::new(
                app.swapchain().extent().width as f32,
                app.swapchain().extent().height as f32,
            ),
        };
        let (scene_param_ubo, bone_ubo) =
            Self::prepare_model_uniform_buffers(image_count, app, scene_param, bone_matrices)?;

        Self::prepare_descriptor_sets(
            app,
            &descriptor_pool,
            &descriptor_set_layout,
            image_count,
            &mut materials,
            &scene_param_ubo,
            &bone_ubo,
            &sampler,
            &dummy_texture,
            &shadow_map,
        )?;

        let (command_buffers, command_buffers_outline, command_buffers_shadow) =
            Self::prepare_command_buffers(
                image_count,
                materials.len(),
                &materials,
                &meshes,
                app,
                &render_pass_default,
                &render_pass_shadow,
                &vertex_buffers,
                &index_buffer,
                &pipelines,
            )?;

        // set_scene_parameterとupdateに当たる処理？

        let mut instance = Self {
            model_name,
            model_version,
            model_comment,

            host_mem_vertices,
            meshes,
            materials,
            scene_param,
            bone_matrices,

            vertex_buffers,
            bone_ubo,
            scene_param_ubo,

            index_buffer,

            command_buffers,
            command_buffers_outline,
            command_buffers_shadow,

            shadow_map,
            dummy_texture,
            sampler,

            descriptor_pool,
            descriptor_set_layout,
            render_pass_default,
            render_pass_shadow,

            pipelines,
            bones,

            face_base_info,
            face_offset_info,
            face_morph_weights,

            bone_ik_list,
        };
        instance.update_matrices();

        Ok(instance)
    }
    fn prepare_dummy_texture(app: &DefaultVulkanAppBase) -> Result<ShaderReadOnlyImageObject> {
        ShaderReadOnlyImageObject::create_dummy_image(
            app.allocator(),
            app.device(),
            &app.graphics_command_pool(),
            &app.graphics_queue(),
        )
    }
    fn prepare_descriptor_pool(
        app: &DefaultVulkanAppBase,
        max_count: u32,
    ) -> Result<DescriptorPoolObject> {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .build(),
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .build(),
        ];

        DescriptorPoolObject::new(app.device(), &descriptor_pool_size, max_count)
    }
    fn prepare_descriptor_set_layout(
        app: &DefaultVulkanAppBase,
    ) -> Result<DescriptorSetLayoutObject> {
        DescriptorSetLayoutObject::new(
            app.device(),
            &[
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build(),
            ],
        )
    }
    fn prepare_render_passes(
        app: &DefaultVulkanAppBase,
    ) -> Result<(RenderPassObject, RenderPassObject)> {
        Ok((
            RenderPassObject::new(app.device(), app.swapchain().format())?,
            RenderPassObject::new_depth_only(app.device())?,
        ))
    }
    fn prepare_model_uniform_buffers(
        count: usize,
        app: &DefaultVulkanAppBase,
        scene_param: SceneParameter,
        bone_param: BoneParameter,
    ) -> Result<(Vec<HostVisibleBufferObject>, Vec<HostVisibleBufferObject>)> {
        let mut scene_param_ubo = vec![];
        let mut bone_param_ubo = vec![];
        for _ in 0..count {
            scene_param_ubo.push(HostVisibleBufferObject::new(
                app.allocator(),
                &[scene_param],
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?);
            bone_param_ubo.push(HostVisibleBufferObject::new(
                app.allocator(),
                &[bone_param],
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?);
        }
        Ok((scene_param_ubo, bone_param_ubo))
    }
    fn prepare_pipelines(
        app: &DefaultVulkanAppBase,
        descriptor_set_layout: &DescriptorSetLayoutObject,
        render_pass_default: &RenderPassObject,
        render_pass_shadow: &RenderPassObject,
    ) -> Result<HashMap<String, PipelineObject>> {
        let input_attribs = [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(PMDVertex, position) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(PMDVertex, normal) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(PMDVertex, texcoord) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(3)
                .format(vk::Format::R32G32_UINT)
                .offset(offset_of!(PMDVertex, bone_indices) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(4)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(PMDVertex, bone_weights) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(5)
                .format(vk::Format::R32_UINT)
                .offset(offset_of!(PMDVertex, edge_flag) as u32)
                .build(),
        ];
        let input_bindings = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<PMDVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];

        let mut pipelines = HashMap::new();

        let pipeline_default = PipelineObject::new_opaque(
            app.device(),
            &input_bindings,
            &input_attribs,
            &Path::new("shaders/spv/animation.vert.spv"),
            &Path::new("shaders/spv/animation.frag.spv"),
            app.swapchain().extent().width,
            app.swapchain().extent().height,
            render_pass_default,
            descriptor_set_layout,
        )?;
        pipelines.insert("normal_draw".to_string(), pipeline_default);

        let pipeline_outline = PipelineObject::new_opaque_cull_front(
            app.device(),
            &input_bindings,
            &input_attribs,
            &Path::new("shaders/spv/animation_outline.vert.spv"),
            &Path::new("shaders/spv/animation_outline.frag.spv"),
            app.swapchain().extent().width,
            app.swapchain().extent().height,
            render_pass_default,
            descriptor_set_layout,
        )?;
        pipelines.insert("outline_draw".to_string(), pipeline_outline);

        let pipeline_shadow = PipelineObject::new_shadow(
            app.device(),
            &input_bindings,
            &input_attribs,
            &Path::new("shaders/spv/animation_shadow.vert.spv"),
            1024,
            1024,
            render_pass_shadow,
            descriptor_set_layout,
        )?;
        pipelines.insert("shadow".to_string(), pipeline_shadow);

        Ok(pipelines)
    }
    fn prepare_descriptor_sets(
        app: &DefaultVulkanAppBase,
        descriptor_pool: &DescriptorPoolObject,
        descriptor_set_layout: &DescriptorSetLayoutObject,
        image_count: usize,
        materials: &mut Vec<Material>,
        scene_param_ubo: &Vec<HostVisibleBufferObject>,
        bone_ubo: &Vec<HostVisibleBufferObject>,
        sampler: &SamplerObject,
        dummy_texture: &ShaderReadOnlyImageObject,
        shadow_texture: &TextureRenderingDepthObject,
    ) -> Result<()> {
        for material in materials.iter_mut() {
            let layout = descriptor_set_layout.descriptor_set_layout();

            let layouts = (0..image_count).map(|_| layout).collect::<Vec<_>>();
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool.descriptor_pool())
                .set_layouts(layouts.as_slice());
            let descriptor_sets = unsafe {
                app.device()
                    .allocate_descriptor_sets(&descriptor_set_allocate_info)?
            };

            for i in 0..image_count {
                let scene_param_ubo = vk::DescriptorBufferInfo::builder()
                    .buffer(scene_param_ubo[i].buffer())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build();
                let bone_ubo = vk::DescriptorBufferInfo::builder()
                    .buffer(bone_ubo[i].buffer())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build();
                let material_ubo = vk::DescriptorBufferInfo::builder()
                    .buffer(material.get_uniform_buffer().buffer())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build();
                let mut diffuse_texture = vk::DescriptorImageInfo::builder()
                    .sampler(sampler.sampler())
                    .image_view(*dummy_texture.image_view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build();
                let shadow_texture = vk::DescriptorImageInfo::builder()
                    .sampler(sampler.sampler())
                    .image_view(shadow_texture.image_view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .build();

                if material.has_texture() {
                    diffuse_texture.image_view = *material.get_texture().image_view();
                }

                let write_descriptors = [
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[i])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[scene_param_ubo])
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[i])
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[bone_ubo])
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[i])
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[material_ubo])
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[i])
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&[diffuse_texture])
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[i])
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&[shadow_texture])
                        .build(),
                ];

                unsafe {
                    app.device().update_descriptor_sets(&write_descriptors, &[]);
                }
            }

            material.set_descriptor_sets(descriptor_sets);
        }

        Ok(())
    }
    fn prepare_command_buffers(
        image_count: usize,
        material_count: usize,
        materials: &Vec<Material>,
        meshes: &Vec<Mesh>,
        app: &DefaultVulkanAppBase,
        render_pass_default: &RenderPassObject,
        render_pass_shadow: &RenderPassObject,
        vertex_buffers: &Vec<HostVisibleBufferObject>,
        index_buffer: &DeviceLocalBufferObject,
        pipelines: &HashMap<String, PipelineObject>,
    ) -> Result<(
        Vec<Vec<vk::CommandBuffer>>,
        Vec<Vec<vk::CommandBuffer>>,
        Vec<Vec<vk::CommandBuffer>>,
    )> {
        let device = app.device();

        let inherit_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(render_pass_default.render_pass());
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .inheritance_info(&inherit_info)
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE);

        // 通常描画のコマンド構築
        let mut command_buffers = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(material_count as u32)
                        .command_pool(app.graphics_command_pool())
                        .level(vk::CommandBufferLevel::SECONDARY),
                )?
            };

            let vertex_buffer = &vertex_buffers[i];
            let pipeline = pipelines
                .get(&"normal_draw".to_string())
                .expect("failed to get normal_draw pipeline.");
            for index in 0..material_count {
                let descriptor_set = materials[index].get_descriptor_set(i);
                let pipeline_layout = pipeline.pipeline_layout();
                let mesh = &meshes[index];
                let command = buffers[index];

                unsafe {
                    device.begin_command_buffer(command, &begin_info)?;
                    device.cmd_bind_pipeline(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline(),
                    );
                    device.cmd_bind_index_buffer(
                        command,
                        index_buffer.buffer(),
                        0,
                        vk::IndexType::UINT16,
                    );
                    device.cmd_bind_vertex_buffers(command, 0, &[vertex_buffer.buffer()], &[0]);
                    device.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    device.cmd_draw_indexed(
                        command,
                        mesh.index_count,
                        1,
                        mesh.start_index_offset,
                        0,
                        0,
                    );
                    device.end_command_buffer(command)?;
                }
            }
            command_buffers.push(buffers);
        }

        // アウトライン描画のコマンド構築
        let mut command_buffers_outline = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(material_count as u32)
                        .command_pool(app.graphics_command_pool())
                        .level(vk::CommandBufferLevel::SECONDARY),
                )?
            };

            let vertex_buffer = &vertex_buffers[i];
            let pipeline = pipelines
                .get(&"outline_draw".to_string())
                .expect("failed to get outline_draw pipeline.");
            for index in 0..material_count {
                let descriptor_set = materials[index].get_descriptor_set(i);
                let pipeline_layout = pipeline.pipeline_layout();
                let mesh = &meshes[index];
                let command = buffers[index];

                if materials[index].get_edge_flag() == 0 {
                    unsafe {
                        device.begin_command_buffer(command, &begin_info)?;
                        device.end_command_buffer(command)?;
                    }
                    continue;
                }

                unsafe {
                    device.begin_command_buffer(command, &begin_info)?;
                    device.cmd_bind_pipeline(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline(),
                    );
                    device.cmd_bind_index_buffer(
                        command,
                        index_buffer.buffer(),
                        0,
                        vk::IndexType::UINT16,
                    );
                    device.cmd_bind_vertex_buffers(command, 0, &[vertex_buffer.buffer()], &[0]);
                    device.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    device.cmd_draw_indexed(
                        command,
                        mesh.index_count,
                        1,
                        mesh.start_index_offset,
                        0,
                        0,
                    );
                    device.end_command_buffer(command)?;
                }
            }
            command_buffers_outline.push(buffers);
        }

        // シャドウパス用のコマンド構築
        let inherit_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(render_pass_shadow.render_pass());
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .inheritance_info(&inherit_info)
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE);

        let mut command_buffers_shadow = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(material_count as u32)
                        .command_pool(app.graphics_command_pool())
                        .level(vk::CommandBufferLevel::SECONDARY),
                )?
            };

            let vertex_buffer = &vertex_buffers[i];
            let pipeline = pipelines
                .get(&"shadow".to_string())
                .expect("failed to get shadow pipeline.");
            for index in 0..material_count {
                let descriptor_set = materials[index].get_descriptor_set(i);
                let pipeline_layout = pipeline.pipeline_layout();
                let mesh = &meshes[index];
                let command = buffers[index];

                unsafe {
                    device.begin_command_buffer(command, &begin_info)?;
                    device.cmd_bind_pipeline(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline(),
                    );
                    device.cmd_bind_index_buffer(
                        command,
                        index_buffer.buffer(),
                        0,
                        vk::IndexType::UINT16,
                    );
                    device.cmd_bind_vertex_buffers(command, 0, &[vertex_buffer.buffer()], &[0]);
                    device.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    device.cmd_draw_indexed(
                        command,
                        mesh.index_count,
                        1,
                        mesh.start_index_offset,
                        0,
                        0,
                    );
                    device.end_command_buffer(command)?;
                }
            }
            command_buffers_shadow.push(buffers);
        }

        Ok((
            command_buffers,
            command_buffers_outline,
            command_buffers_shadow,
        ))
    }

    pub fn set_scene_parameter(&mut self, params: SceneParameter) {
        self.scene_param = params;
    }

    pub fn update_matrices(&mut self) {
        for bone in self.bones.iter_mut() {
            if let Some(_) = &bone.borrow().parent {
                continue;
            }
            bone.update_matrices();
        }
    }

    pub fn update_command_buffers_aspect(&mut self, app: &DefaultVulkanAppBase) -> Result<()> {
        self.pipelines = Self::prepare_pipelines(
            app,
            &self.descriptor_set_layout,
            &self.render_pass_default,
            &self.render_pass_shadow,
        )?;
        let (command_buffers, command_buffers_outline, command_buffers_shadow) =
            Self::prepare_command_buffers(
                app.swapchain().len(),
                self.materials.len(),
                &self.materials,
                &self.meshes,
                app,
                &self.render_pass_default,
                &self.render_pass_shadow,
                &self.vertex_buffers,
                &self.index_buffer,
                &self.pipelines,
            )?;
        self.command_buffers = command_buffers;
        self.command_buffers_outline = command_buffers_outline;
        self.command_buffers_shadow = command_buffers_shadow;
        Ok(())
    }

    pub fn update(&mut self, image_index: usize) -> Result<()> {
        self.scene_param_ubo[image_index].map_data(&[self.scene_param])?;

        for i in 0..self.bones.len() {
            let bone = &self.bones[i];
            let matrix = bone.borrow().get_world_matrix() * bone.borrow().get_inv_bind_matrix();
            self.bone_matrices.bone[i] = matrix;
        }
        self.bone_ubo[image_index].map_data(&[self.bone_matrices])?;

        // 頂点バッファの更新
        {
            let vertex_count = self.face_base_info.vertices_pos.len();
            for i in 0..vertex_count {
                let offset_index = self.face_base_info.indices[i];
                self.host_mem_vertices[offset_index as usize].position =
                    self.face_base_info.vertices_pos[i];
            }

            for face_index in 0..self.face_offset_info.len() {
                let face = &self.face_offset_info[face_index];
                let w = self.face_morph_weights[face_index];

                for i in 0..(face.indices.len()) {
                    let base_vertex_index = face.indices[i];
                    let displacement = face.vertices_offset[i];

                    let offset_index = self.face_base_info.indices[base_vertex_index as usize];
                    self.host_mem_vertices[offset_index as usize].position += displacement * w;
                    // self.host_mem_vertices[base_vertex_index as usize].position += displacement * w;
                }
            }

            self.vertex_buffers[image_index].map_data(self.host_mem_vertices.as_slice())?;
        }

        Ok(())
    }

    pub fn get_command_buffers(&self, index: usize) -> &Vec<vk::CommandBuffer> {
        &self.command_buffers[index]
    }
    pub fn get_command_buffers_outline(&self, index: usize) -> &Vec<vk::CommandBuffer> {
        &self.command_buffers_outline[index]
    }
    pub fn get_command_buffers_shadow(&self, index: usize) -> &Vec<vk::CommandBuffer> {
        &self.command_buffers_shadow[index]
    }

    pub fn set_shadow_map(&mut self, shadow_map: TextureRenderingDepthObject) {
        self.shadow_map = shadow_map;
    }

    pub fn get_bone_count(&self) -> u32 {
        self.bones.len() as u32
    }
    pub fn get_bone(&self, index: usize) -> Rc<RefCell<Bone>> {
        Rc::clone(&self.bones[index])
    }

    pub fn get_face_morph_count(&self) -> u32 {
        self.face_offset_info.len() as u32
    }
    pub fn get_morph_index(&self, name: impl Into<String>) -> Option<usize> {
        let name: String = name.into();
        for i in 0..self.face_offset_info.len() {
            let face = &self.face_offset_info[i];
            if face.name == name {
                return Some(i);
            }
        }
        None
    }
    pub fn set_morph_weight(&mut self, index: usize, weight: f32) {
        self.face_morph_weights[index] = weight;
    }

    pub fn get_bone_ik_count(&self) -> u32 {
        self.bone_ik_list.len() as u32
    }
    pub fn get_bone_ik(&self, index: usize) -> &PMDBoneIK {
        &self.bone_ik_list[index]
    }
}
