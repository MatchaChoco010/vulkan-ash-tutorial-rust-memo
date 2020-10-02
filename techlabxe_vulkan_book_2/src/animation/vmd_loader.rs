#![allow(dead_code)]

use std::{
    collections::hash_map::HashMap,
    ffi::CString,
    fs::File,
    io::{prelude::*, BufReader, Read, SeekFrom},
    mem,
    path::Path,
};

use cgmath::{BaseFloat, Quaternion, Vector2, Vector3, Vector4};
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

pub struct VMDHeader {
    magic: [u8; 30],
    model_name: String,
}
impl VMDHeader {
    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 30];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read magic from bytes.");
        let magic = buf;

        let mut buf = [0_u8; 20];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let model_name = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert name to String");

        Self { magic, model_name }
    }
}

pub struct VMDNode {
    keyframe: u32,
    location: Vector3<f32>,
    rotation: Quaternion<f32>,
    name: String,
    interpolation: [u8; 64],
}
impl VMDNode {
    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn get_keyframe_number(&self) -> u32 {
        self.keyframe
    }
    pub fn get_location(&self) -> &Vector3<f32> {
        &self.location
    }
    pub fn get_rotation(&self) -> &Quaternion<f32> {
        &self.rotation
    }
    pub fn get_bezier_param(&self, index: usize) -> Vector4<f32> {
        let x = self.interpolation[4 * 0 + index] as f32 / 127.0;
        let y = self.interpolation[4 * 1 + index] as f32 / 127.0;
        let z = self.interpolation[4 * 2 + index] as f32 / 127.0;
        let w = self.interpolation[4 * 3 + index] as f32 / 127.0;
        Vector4::new(x, y, z, w)
    }

    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 15];
        reader
            .read_exact(&mut buf)
            .expect("Failed to read name from bytes.");
        let end = buf
            .iter()
            .position(|&u| u == 0_u8)
            .expect("Failed to find null char.");
        let node_name = WINDOWS_31J
            .decode(&buf[0..end], DecoderTrap::Ignore)
            .expect("Failed to convert name to String");

        let keyframe = read_u32(reader);

        let location = read_vector3(reader).flip_to_rh();
        let rotation = read_vector4(reader).flip_to_rh();
        let rotation = Quaternion::new(rotation.w, rotation.x, rotation.y, rotation.z);

        let mut interpolation = [0_u8; 64];
        reader
            .read_exact(&mut interpolation)
            .expect("Failed to read bezier param from bytes.");

        Self {
            keyframe,
            location,
            rotation,
            name: node_name,
            interpolation,
        }
    }
}

pub struct VMDMorph {
    name: String,
    keyframe: u32,
    weight: f32,
}
impl VMDMorph {
    pub fn get_name(&self) -> &str {
        &self.name
    }
    pub fn get_keyframe_number(&self) -> u32 {
        self.keyframe
    }
    pub fn get_weight(&self) -> f32 {
        self.weight
    }

    fn load(reader: &mut impl Read) -> Self {
        let mut buf = [0_u8; 15];
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

        let keyframe = read_u32(reader);

        let weight = read_f32(reader);

        Self {
            name,
            keyframe,
            weight,
        }
    }
}

pub struct VMDFile {
    keyframe_count: u32,
    animation_map: HashMap<String, Vec<VMDNode>>,
    node_name_list: Vec<String>,
    morph_map: HashMap<String, Vec<VMDMorph>>,
    morph_name_list: Vec<String>,
}
impl VMDFile {
    pub fn load(reader: &mut impl Read) -> Self {
        VMDHeader::load(reader);

        let data_num = read_u32(reader);
        let mut nodes = Vec::with_capacity(data_num as usize);
        for _ in 0..data_num {
            nodes.push(VMDNode::load(reader));
        }

        let mut animation_map = HashMap::<String, Vec<VMDNode>>::new();
        for node in nodes {
            let node_keyframes = animation_map
                .entry(node.get_name().to_string())
                .or_insert(vec![]);
            node_keyframes.push(node);
        }
        for keyframes in animation_map.values_mut() {
            keyframes.sort_unstable_by_key(|k| k.get_keyframe_number());
        }
        let mut keyframe_count = 0;
        for nodes in animation_map.values() {
            let k = nodes
                .last()
                .expect("Failed to get last node")
                .get_keyframe_number();
            keyframe_count = keyframe_count.max(k);
        }
        let mut node_name_list = vec![];
        for name in animation_map.keys() {
            node_name_list.push(name.clone());
        }

        let data_num = read_u32(reader);
        let mut morphs = Vec::with_capacity(data_num as usize);
        for _ in 0..data_num {
            morphs.push(VMDMorph::load(reader));
        }

        let mut morph_map = HashMap::<String, Vec<VMDMorph>>::new();
        for morph in morphs {
            let morph_keyframes = morph_map
                .entry(morph.get_name().to_string())
                .or_insert(vec![]);
            morph_keyframes.push(morph);
        }
        for keyframes in morph_map.values_mut() {
            keyframes.sort_unstable_by_key(|k| k.get_keyframe_number());
        }
        let mut morph_name_list = vec![];
        for name in morph_map.keys() {
            morph_name_list.push(name.clone());
        }

        Self {
            keyframe_count,
            animation_map,
            node_name_list,
            morph_map,
            morph_name_list,
        }
    }

    pub fn get_node_count(&self) -> usize {
        self.node_name_list.len()
    }
    pub fn get_node_name(&self, index: usize) -> &str {
        &self.node_name_list[index]
    }

    pub fn get_morph_count(&self) -> usize {
        self.morph_name_list.len()
    }
    pub fn get_morph_name(&self, index: usize) -> &str {
        &self.morph_name_list[index]
    }

    pub fn get_keyframes(&self, node_name: impl Into<String>) -> &Vec<VMDNode> {
        let node_name: String = node_name.into();
        self.animation_map
            .get(&node_name)
            .expect("No such name node.")
    }
    pub fn get_morph_keyframes(&self, node_name: impl Into<String>) -> &Vec<VMDMorph> {
        let node_name: String = node_name.into();
        self.morph_map.get(&node_name).expect("No such name node.")
    }

    pub fn get_keyframe_segment(
        &self,
        node_name: impl Into<String>,
        frame: u32,
    ) -> (&VMDNode, &VMDNode) {
        let node = self.find_node(node_name);
        let seg0 = node
            .iter()
            .rev()
            .find(|v| v.get_keyframe_number() <= frame)
            .expect("Failed to get seg0.");
        let seg1 = node
            .iter()
            .find(|v| frame < v.get_keyframe_number())
            .expect("Failed to get seg1.");
        (seg0, seg1)
    }

    pub fn get_translation(&self, node_name: impl Into<String>, frame: u32) -> Vector3<f32> {
        let node = self.find_node(node_name);
        let seg0 = node
            .iter()
            .rev()
            .find(|v| v.get_keyframe_number() <= frame)
            .expect("Failed to get seg0.");
        let seg1 = node
            .iter()
            .find(|v| frame < v.get_keyframe_number())
            .expect("Failed to get seg1.");
        let segment = (seg0.get_keyframe_number() - seg0.get_keyframe_number()) as f32;
        let t = (frame - seg0.get_keyframe_number()) as f32 / segment;

        let t0 = seg0.get_location();
        let t1 = seg1.get_location();

        t0 + t * (t1 - t0)
    }

    #[allow(unused_variables)]
    pub fn get_rotation(&self, node_name: impl Into<String>, frame: u32) -> Quaternion<f32> {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }

    pub fn get_keyframe_count(&self) -> u32 {
        self.keyframe_count
    }

    fn find_node(&self, node_name: impl Into<String>) -> &Vec<VMDNode> {
        let node_name: String = node_name.into();
        self.animation_map
            .get(&node_name)
            .expect("No such name node.")
    }
}
