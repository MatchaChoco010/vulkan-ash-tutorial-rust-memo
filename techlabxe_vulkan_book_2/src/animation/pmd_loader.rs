#![allow(dead_code)]

use std::{
    ffi::CString,
    fs::File,
    io::{prelude::*, BufReader, Read, SeekFrom},
    mem,
    path::Path,
};

use cgmath::{BaseFloat, Vector2, Vector3, Vector4};
use encoding::{all::WINDOWS_31J, DecoderTrap, EncoderTrap, Encoding};

/// bytesからf32の値を読み込む関数。
fn read_f32(reader: &mut impl Read) -> f32 {
    let mut buf = [0_u8; 4];
    reader
        .read_exact(&mut buf)
        .expect("Failed to read f32 from bytes.");
    unsafe { mem::transmute::<[u8; 4], f32>(buf) }
}
/// bytesからu8の値を読み込む関数。
fn read_u8(reader: &mut impl Read) -> u8 {
    let mut buf = [0_u8; 1];
    reader
        .read_exact(&mut buf)
        .expect("Failed to read u8 from bytes.");
    buf[0]
}
/// bytesからu16の値を読み込む関数。
fn read_u16(reader: &mut impl Read) -> u16 {
    let mut buf = [0_u8; 2];
    reader
        .read_exact(&mut buf)
        .expect("Failed to read u16 from bytes.");
    unsafe { mem::transmute::<[u8; 2], u16>(buf) }
}
/// bytesからu32の値を読み込む関数。
fn read_u32(reader: &mut impl Read) -> u32 {
    let mut buf = [0_u8; 4];
    reader
        .read_exact(&mut buf)
        .expect("Failed to read u32 from bytes.");
    unsafe { mem::transmute::<[u8; 4], u32>(buf) }
}
/// bytesからVector2<f32>の値を読み込む関数。
fn read_vector2(reader: &mut impl Read) -> Vector2<f32> {
    let x = read_f32(reader);
    let y = read_f32(reader);
    Vector2::new(x, y)
}
/// bytesからVector3<f32>の値を読み込む関数。
fn read_vector3(reader: &mut impl Read) -> Vector3<f32> {
    let x = read_f32(reader);
    let y = read_f32(reader);
    let z = read_f32(reader);
    Vector3::new(x, y, z)
}
/// bytesからVector4<f32>の値を読み込む関数。
fn read_vector4(reader: &mut impl Read) -> Vector4<f32> {
    let x = read_f32(reader);
    let y = read_f32(reader);
    let z = read_f32(reader);
    let w = read_f32(reader);
    Vector4::new(x, y, z, w)
}

/// flip_to_rhを生やすための拡張トレイト。
trait Vector3Ext {
    fn flip_to_rh(&self) -> Self;
}
impl<T: BaseFloat> Vector3Ext for Vector3<T> {
    fn flip_to_rh(&self) -> Self {
        Vector3::new(self.x, self.y, -self.z)
    }
}
/// flip_to_rhを生やすための拡張トレイト。
trait Vector4Ext {
    fn flip_to_rh(&self) -> Self;
}
impl<T: BaseFloat> Vector4Ext for Vector4<T> {
    fn flip_to_rh(&self) -> Self {
        Vector4::new(self.x, self.y, -self.z, -self.w)
    }
}

/// PMDHeader情報の構造体。
struct PMDHeader {
    magic: [u8; 3],
    version: f32,
    name: String,
    comment: String,
}
impl PMDHeader {
    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 3];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read magic from bytes.");
        let magic = buf;

        let version = read_f32(reader);

        let mut buf = [0_u8; 20];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let name = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert name to String");

        let mut buf = [0_u8; 256];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read comment from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let comment = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert comment to String");

        Self {
            magic,
            version,
            name,
            comment,
        }
    }
}

/// PMDファイルの頂点情報を保持する構造体。
pub struct PMDVertex {
    position: Vector3<f32>,
    normal: Vector3<f32>,
    texcoord: Vector2<f32>,
    bone_num: [u16; 2],
    bone_weight: u8,
    edge_flag: u8,
}
impl PMDVertex {
    pub fn new(
        position: Vector3<f32>,
        normal: Vector3<f32>,
        texcoord: Vector2<f32>,
        bone_num: [u16; 2],
        bone_weight: u8,
        edge_flag: u8,
    ) -> Self {
        Self {
            position,
            normal,
            texcoord,
            bone_num,
            bone_weight,
            edge_flag,
        }
    }
    pub fn position(&self) -> Vector3<f32> {
        self.position
    }
    pub fn normal(&self) -> Vector3<f32> {
        self.normal
    }
    pub fn texcoord(&self) -> Vector2<f32> {
        self.texcoord
    }
    pub fn bone_index(&self, index: usize) -> u16 {
        self.bone_num[index]
    }
    pub fn bone_weight(&self, index: usize) -> f32 {
        if index == 0 {
            self.bone_weight as f32 / 100.0
        } else {
            (100 - self.bone_weight) as f32 / 100.0
        }
    }
    pub fn edge_flag(&self) -> u8 {
        self.edge_flag
    }

    fn load(reader: &mut impl Read) -> Self {
        let position = read_vector3(reader).flip_to_rh();
        let normal = read_vector3(reader).flip_to_rh();
        let texcoord = read_vector2(reader);
        let bone_num: [u16; 2] = [read_u16(reader), read_u16(reader)];
        let bone_weight = read_u8(reader);
        let edge_flag = read_u8(reader);
        Self {
            position,
            normal,
            texcoord,
            bone_num,
            bone_weight,
            edge_flag,
        }
    }
}

/// PMDのマテリアル情報を保持する構造体。
pub struct PMDMaterial {
    diffuse: Vector3<f32>,
    alpha: f32,
    shininess: f32,
    specular: Vector3<f32>,
    ambient: Vector3<f32>,
    toon_id: u8,
    edge_flag: u8,
    number_of_polygons: u32,
    texture: String,
}
impl PMDMaterial {
    pub fn diffuse(&self) -> Vector3<f32> {
        self.diffuse
    }
    pub fn ambient(&self) -> Vector3<f32> {
        self.ambient
    }
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
    pub fn shininess(&self) -> f32 {
        self.shininess
    }
    pub fn specular(&self) -> Vector3<f32> {
        self.specular
    }
    pub fn texture(&self) -> &str {
        &self.texture
    }
    pub fn edge_flag(&self) -> u8 {
        self.edge_flag
    }
    pub fn number_of_polygons(&self) -> u32 {
        self.number_of_polygons
    }

    fn load(reader: &mut impl Read) -> Self {
        let diffuse = read_vector3(reader);
        let alpha = read_f32(reader);
        let shininess = read_f32(reader);
        let specular = read_vector3(reader);
        let ambient = read_vector3(reader);
        let toon_id = read_u8(reader);
        let edge_flag = read_u8(reader);
        let number_of_polygons = read_u32(reader);

        let mut buf = [0_u8; 20];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read texture file name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let texture = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert texture filename to String");

        Self {
            diffuse,
            alpha,
            shininess,
            specular,
            ambient,
            toon_id,
            edge_flag,
            number_of_polygons,
            texture,
        }
    }
}

/// PMDのボーン情報を保持する構造体。
pub struct PMDBone {
    name: String,
    parent: u16,
    child: u16,
    ty: u8,
    target_bone: u16,
    position: Vector3<f32>,
}
impl PMDBone {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn parent(&self) -> u16 {
        self.parent
    }
    pub fn target(&self) -> u16 {
        self.target_bone
    }
    pub fn position(&self) -> Vector3<f32> {
        self.position
    }

    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 20];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read bone name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let name = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert bone name to String");

        let parent = read_u16(reader);
        let child = read_u16(reader);
        let ty = read_u8(reader);
        let target_bone = read_u16(reader);
        let position = read_vector3(reader).flip_to_rh();

        Self {
            name,
            parent,
            child,
            ty,
            target_bone,
            position,
        }
    }
}

/// PMDのIk情報を保持する構造体。
pub struct PMDIk {
    bone_index: u16,
    bone_target: u16,
    num_chains: u8,
    num_iterations: u16,
    angle_limit: f32,
    ik_bones: Vec<u16>,
}
impl PMDIk {
    pub fn target_bone_id(&self) -> u16 {
        self.bone_target
    }
    pub fn bone_eff(&self) -> u16 {
        self.bone_index
    }
    pub fn chains(&self) -> &Vec<u16> {
        &self.ik_bones
    }
    pub fn iterations(&self) -> u16 {
        self.num_iterations
    }
    pub fn angle_limit(&self) -> f32 {
        self.angle_limit
    }

    fn load(reader: &mut impl Read) -> Self {
        let bone_index = read_u16(reader);
        let bone_target = read_u16(reader);
        let num_chains = read_u8(reader);
        let num_iterations = read_u16(reader);
        let angle_limit = read_f32(reader) * std::f32::consts::PI;

        let mut ik_bones = vec![];
        for _ in 0..num_chains {
            ik_bones.push(read_u16(reader));
        }

        Self {
            bone_index,
            bone_target,
            num_chains,
            num_iterations,
            angle_limit,
            ik_bones,
        }
    }
}

/// PMDFace
pub enum FaceType {
    BASE,
    EYEBROW,
    EYE,
    LIP,
    OTHER,
}
impl FaceType {
    fn from_u8(id: u8) -> Self {
        match id {
            0 => FaceType::BASE,
            1 => FaceType::EYEBROW,
            2 => FaceType::EYE,
            3 => FaceType::LIP,
            _ => FaceType::OTHER,
        }
    }
}
pub struct PMDFace {
    name: String,
    num_vertices: u32,
    face_type: FaceType,
    face_indices: Vec<u32>,
    face_vertices: Vec<Vector3<f32>>,
}
impl PMDFace {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn ty(&self) -> &FaceType {
        &self.face_type
    }
    pub fn vertex_count(&self) -> u32 {
        self.face_vertices.len() as u32
    }
    pub fn index_count(&self) -> u32 {
        self.face_indices.len() as u32
    }
    pub fn face_vertices(&self) -> &Vec<Vector3<f32>> {
        &self.face_vertices
    }
    pub fn face_indices(&self) -> &Vec<u32> {
        &self.face_indices
    }

    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 20];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read face name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let name = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert face name to String");

        let num_vertices = read_u32(reader);
        let face_type = FaceType::from_u8(read_u8(reader));

        let mut face_vertices = vec![];
        let mut face_indices = vec![];

        for _ in 0..num_vertices {
            face_indices.push(read_u32(reader));
            face_vertices.push(read_vector3(reader).flip_to_rh());
        }

        Self {
            name,
            num_vertices,
            face_type,
            face_indices,
            face_vertices,
        }
    }
}

/// PMDRigidParam
pub enum ShapeType {
    SPHERE,
    BOX,
    CAPSULE,
}
impl ShapeType {
    fn from_u8(id: u8) -> Self {
        match id {
            0 => ShapeType::SPHERE,
            1 => ShapeType::BOX,
            2 => ShapeType::CAPSULE,
            _ => unreachable!(),
        }
    }
}
pub enum RigidBodyType {
    BONE,
    PHYSICS,
    #[allow(non_camel_case_types)]
    PHYSICS_BONE_CORRECT,
}
impl RigidBodyType {
    fn from_u8(id: u8) -> Self {
        match id {
            0 => RigidBodyType::BONE,
            1 => RigidBodyType::PHYSICS,
            2 => RigidBodyType::PHYSICS_BONE_CORRECT,
            _ => unreachable!(),
        }
    }
}
pub struct PMDRigidParam {
    name: String,
    bone_id: u16,
    group_id: u8,
    group_mask: u16,
    shape_type: ShapeType,
    body_type: RigidBodyType,

    shape_w: f32,
    shape_h: f32,
    shape_d: f32,
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    weight: f32,
    attenuation_pos: f32,
    attenuation_rot: f32,
    recoil: f32,
    friction: f32,
}
impl PMDRigidParam {
    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 20];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read rigid param name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let name = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert rigid name to String");

        let bone_id = read_u16(reader);
        let group_id = read_u8(reader);
        let group_mask = read_u16(reader);
        let shape_type = ShapeType::from_u8(read_u8(reader));
        let shape_w = read_f32(reader);
        let shape_h = read_f32(reader);
        let shape_d = read_f32(reader);
        let position = read_vector3(reader);
        let rotation = read_vector3(reader);
        let weight = read_f32(reader);
        let attenuation_pos = read_f32(reader);
        let attenuation_rot = read_f32(reader);
        let recoil = read_f32(reader);
        let friction = read_f32(reader);
        let body_type = RigidBodyType::from_u8(read_u8(reader));

        Self {
            name,
            bone_id,
            group_id,
            group_mask,
            shape_type,
            shape_w,
            shape_h,
            shape_d,
            position,
            rotation,
            weight,
            attenuation_pos,
            attenuation_rot,
            recoil,
            friction,
            body_type,
        }
    }
}

/// PMDJointParam
pub struct PMDJointParam {
    name: String,
    target_rigid_bodies: [u32; 2],
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    constraint_pos: [Vector3<f32>; 2],
    constraint_rot: [Vector3<f32>; 2],
    spring_pos: Vector3<f32>,
    spring_rot: Vector3<f32>,
}
impl PMDJointParam {
    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 20];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read joint name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let name = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert joint name to String");

        let target_rigid_bodies: [u32; 2] = [read_u32(reader), read_u32(reader)];
        let position = read_vector3(reader);
        let rotation = read_vector3(reader);
        let constraint_pos: [_; 2] = [read_vector3(reader), read_vector3(reader)];
        let constraint_rot: [_; 2] = [read_vector3(reader), read_vector3(reader)];
        let spring_pos = read_vector3(reader);
        let spring_rot = read_vector3(reader);

        Self {
            name,
            target_rigid_bodies,
            position,
            rotation,
            constraint_pos,
            constraint_rot,
            spring_pos,
            spring_rot,
        }
    }
}

/// PMDFile
pub struct PMDFile {
    version: f32,
    name: String,
    comment: String,
    vertices: Vec<PMDVertex>,
    indices: Vec<u16>,
    materials: Vec<PMDMaterial>,
    bones: Vec<PMDBone>,
    iks: Vec<PMDIk>,
    faces: Vec<PMDFace>,
    toon_textures: Vec<String>,
    rigid_bodies: Vec<PMDRigidParam>,
    joints: Vec<PMDJointParam>,
}
impl PMDFile {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn version(&self) -> f32 {
        self.version
    }
    pub fn comment(&self) -> &str {
        &self.comment
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }
    pub fn ik_count(&self) -> usize {
        self.iks.len()
    }
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }
    pub fn rigid_body_count(&self) -> usize {
        self.rigid_bodies.len()
    }
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    pub fn vertices(&self) -> &Vec<PMDVertex> {
        &self.vertices
    }
    pub fn indices(&self) -> &Vec<u16> {
        &self.indices
    }
    pub fn material(&self, index: usize) -> &PMDMaterial {
        &self.materials[index]
    }
    pub fn bone(&self, index: usize) -> &PMDBone {
        &self.bones[index]
    }
    pub fn ik(&self, index: usize) -> &PMDIk {
        &self.iks[index]
    }
    pub fn face(&self, index: usize) -> &PMDFace {
        &self.faces[index]
    }
    pub fn face_base(&self) -> &PMDFace {
        for face in self.faces.iter() {
            match face.face_type {
                FaceType::BASE => return face,
                _ => continue,
            }
        }
        unreachable!()
    }

    pub fn new(reader: &mut (impl Read + Seek)) -> Self {
        let header = PMDHeader::load(reader);
        let version = header.version;
        let name = header.name;
        let comment = header.comment;

        let vertex_count = read_u32(reader);
        let mut vertices = Vec::with_capacity(vertex_count as usize);
        for _ in 0..vertex_count {
            vertices.push(PMDVertex::load(reader));
        }

        let index_count = read_u32(reader);
        let polygon_count = index_count / 3;
        let mut indices = Vec::with_capacity(index_count as usize);
        for _ in 0..polygon_count {
            let idx0 = read_u16(reader);
            let idx2 = read_u16(reader);
            let idx1 = read_u16(reader);
            indices.push(idx0);
            indices.push(idx1);
            indices.push(idx2);
        }

        let material_count = read_u32(reader);
        let mut materials = Vec::with_capacity(material_count as usize);
        for _ in 0..material_count {
            materials.push(PMDMaterial::load(reader));
        }

        let bone_count = read_u16(reader);
        let mut bones = Vec::with_capacity(bone_count as usize);
        for _ in 0..bone_count {
            bones.push(PMDBone::load(reader));
        }

        let ik_list_count = read_u16(reader);
        let mut iks = Vec::with_capacity(ik_list_count as usize);
        for _ in 0..ik_list_count {
            iks.push(PMDIk::load(reader));
        }

        let face_count = read_u16(reader);
        let mut faces = Vec::with_capacity(face_count as usize);
        for _ in 0..face_count {
            faces.push(PMDFace::load(reader));
        }

        // 表情枠。Skip
        let face_disp_count = read_u8(reader);
        reader
            .seek(SeekFrom::Current(
                face_disp_count as i64 * mem::size_of::<u16>() as i64,
            ))
            .expect("Failed to seek face disp");

        // ボーン枠名前。Skip
        let bone_disp_name_count = read_u8(reader);
        reader
            .seek(SeekFrom::Current(
                bone_disp_name_count as i64 * mem::size_of::<[u8; 50]>() as i64,
            ))
            .expect("Failed to seek bone disp name.");

        // ボーン枠。Skip
        let bone_disp_count = read_u32(reader);
        reader
            .seek(SeekFrom::Current(
                bone_disp_count as i64 * mem::size_of::<[u8; 3]>() as i64,
            ))
            .expect("Failed to seek bone disp.");

        // 英語名ヘッダ。Skip
        let eng_name_count = read_u8(reader);
        reader
            .seek(SeekFrom::Current(
                eng_name_count as i64 * mem::size_of::<[u8; 20 + 256]>() as i64,
            ))
            .expect("Failed to seek eng name");

        // 英語名ボーン。Skip
        reader
            .seek(SeekFrom::Current(
                (bones.len() * mem::size_of::<[u8; 20]>()) as i64,
            ))
            .expect("Failed to seek eng bone name");

        // 英語名表情リスト。Skip
        reader
            .seek(SeekFrom::Current(
                ((faces.len() - 1) * mem::size_of::<[u8; 20]>()) as i64,
            ))
            .expect("Failed to seek eng face");

        // 英語名ボーン枠。Skip
        reader
            .seek(SeekFrom::Current(
                bone_disp_name_count as i64 * mem::size_of::<[u8; 50]>() as i64,
            ))
            .expect("Failed to seek eng bone disp name");

        // トゥーンテクスチャリスト。
        let mut toon_textures = Vec::with_capacity(10);
        for _ in 0..10 {
            let mut buf = [0_u8; 100];
            reader
                .read_exact(&mut buf)
                .expect("Failed to read toon texture name from bytes.");
            let end = buf
                .iter()
                .position(|&u| u == 0_u8)
                .expect("Failed to find null char.");
            let name = WINDOWS_31J
                .decode(&buf[0..end], DecoderTrap::Ignore)
                .expect("Failed to convert toon texture name to String");

            toon_textures.push(name);
        }

        // 物理演算・剛体。
        let rigid_body_count = read_u32(reader);
        let mut rigid_bodies = Vec::with_capacity(rigid_body_count as usize);
        for _ in 0..rigid_body_count {
            rigid_bodies.push(PMDRigidParam::load(reader));
        }

        // 物理演算・ジョイント。
        let joint_count = read_u32(reader);
        let mut joints = Vec::with_capacity(joint_count as usize);
        for _ in 0..joint_count {
            joints.push(PMDJointParam::load(reader));
        }

        Self {
            version,
            name,
            comment,
            vertices,
            indices,
            materials,
            bones,
            iks,
            faces,
            toon_textures,
            rigid_bodies,
            joints,
        }
    }
}
