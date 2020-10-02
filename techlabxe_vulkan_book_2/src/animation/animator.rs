#![allow(dead_code)]

use std::{
    cmp::Ordering,
    collections::hash_map::HashMap,
    fs::File,
    io::{prelude::*, BufReader, Cursor, Read, SeekFrom},
    path::Path,
    rc::Rc,
};

use anyhow::Result;
use cgmath::{
    ElementWise, Euler, InnerSpace, Quaternion, Rad, Rotation3, SquareMatrix, Vector2, Vector3,
    Vector4,
};

use crate::animation::{
    model::{Model, PMDBoneIK},
    vmd_loader::VMDFile,
};

trait Keyframe {
    fn keyframe(&self) -> u32;
}

struct Animation<T: Keyframe> {
    keyframes: Vec<T>,
}
impl<T: Keyframe> Animation<T> {
    fn new() -> Self {
        Self { keyframes: vec![] }
    }

    fn find_segment(&self, frame: u32) -> (&T, &T) {
        let last = self
            .keyframes
            .iter()
            .find(|k| frame < k.keyframe())
            .unwrap_or(self.keyframes.last().expect("Failed to get last keyframe"));
        let first = self
            .keyframes
            .iter()
            .rev()
            .find(|k| k.keyframe() < frame)
            .unwrap_or(
                self.keyframes
                    .first()
                    .expect("Failed to get first keyframe"),
            );
        (first, last)
    }
}

struct NodeAnimationFrame {
    frame: u32,
    translation: Vector3<f32>,
    rotation: Quaternion<f32>,
    interp_x: Vector4<f32>,
    interp_y: Vector4<f32>,
    interp_z: Vector4<f32>,
    interp_r: Vector4<f32>,
}
impl PartialOrd for NodeAnimationFrame {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NodeAnimationFrame {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frame.cmp(&other.frame)
    }
}
impl PartialEq for NodeAnimationFrame {
    fn eq(&self, other: &NodeAnimationFrame) -> bool {
        self.frame == other.frame
    }
}
impl Eq for NodeAnimationFrame {}
impl Keyframe for NodeAnimationFrame {
    fn keyframe(&self) -> u32 {
        self.frame
    }
}

struct MorphAnimationFrame {
    frame: u32,
    weight: f32,
}
impl PartialOrd for MorphAnimationFrame {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MorphAnimationFrame {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frame.cmp(&other.frame)
    }
}
impl PartialEq for MorphAnimationFrame {
    fn eq(&self, other: &MorphAnimationFrame) -> bool {
        self.frame == other.frame
    }
}
impl Eq for MorphAnimationFrame {}
impl Keyframe for MorphAnimationFrame {
    fn keyframe(&self) -> u32 {
        self.frame
    }
}

type NodeAnimation = Animation<NodeAnimationFrame>;
type MorphAnimation = Animation<MorphAnimationFrame>;

fn d_fx(ax: f32, ay: f32, t: f32) -> f32 {
    let s = 1.0 - t;
    let v = -6.0 * s * t * t * ax + 3.0 * s * s * ax - 3.0 * t * t * ay
        + 6.60 * s * t * ay
        + 3.0 * t * t;
    v
}
fn fx(ax: f32, ay: f32, t: f32, x0: f32) -> f32 {
    let s = 1.0 - t;
    3.0 * s * s * t * ax + 3.0 * s * t * t * ay + t * t * t - x0
}

fn func_bezier_x(k: &Vector4<f32>, t: f32) -> f32 {
    let s = 1.0 - t;
    3.0 * s * s * t * k.x + 3.0 * s * t * t * k.y + t * t * t
}
fn func_bezier_y(k: &Vector4<f32>, t: f32) -> f32 {
    let s = 1.0 - t;
    3.0 * s * s * t * k.z + 3.0 * s * t * t * k.w + t * t * t
}

pub struct Animator {
    node_map: HashMap<String, NodeAnimation>,
    morph_map: HashMap<String, MorphAnimation>,
    frame_period: u32,
}
impl Animator {
    pub fn new(filename: impl Into<String>) -> Result<Self> {
        let filename: String = filename.into();
        let mut cur = Cursor::new(BufReader::new(File::open(&Path::new(&filename))?));
        let loader = VMDFile::load(cur.get_mut());

        let frame_period = loader.get_keyframe_count();

        let node_count = loader.get_node_count();
        let mut node_map = HashMap::<String, NodeAnimation>::new();
        for i in 0..node_count {
            let name = loader.get_node_name(i);
            let keyframes = node_map
                .entry(name.to_string())
                .or_insert(NodeAnimation::new());
            let frame_src = loader.get_keyframes(name);

            let frame_count = frame_src.len();
            let mut frames = Vec::with_capacity(frame_count);
            for i in 0..frame_count {
                let src = &frame_src[i];
                frames.push(NodeAnimationFrame {
                    frame: src.get_keyframe_number(),
                    translation: *src.get_location(),
                    rotation: *src.get_rotation(),
                    interp_x: src.get_bezier_param(0),
                    interp_y: src.get_bezier_param(1),
                    interp_z: src.get_bezier_param(2),
                    interp_r: src.get_bezier_param(3),
                });
            }
            keyframes.keyframes = frames;
        }

        let morph_count = loader.get_morph_count();
        let mut morph_map = HashMap::<String, MorphAnimation>::new();
        for i in 0..morph_count {
            let name = loader.get_morph_name(i);
            let keyframes = morph_map
                .entry(name.to_string())
                .or_insert(MorphAnimation::new());
            let frame_src = loader.get_morph_keyframes(name);

            let frame_count = frame_src.len();
            let mut frames = Vec::with_capacity(frame_count);
            for i in 0..frame_count {
                let src = &frame_src[i];
                frames.push(MorphAnimationFrame {
                    frame: src.get_keyframe_number(),
                    weight: src.get_weight(),
                });
            }
            keyframes.keyframes = frames;
        }

        Ok(Self {
            node_map,
            morph_map,
            frame_period,
        })
    }

    pub fn update_animation(&self, model: &mut Model, anime_frame: u32) {
        self.update_node_animation(model, anime_frame);
        self.update_morph_animation(model, anime_frame);

        self.update_ik_chains(model);
    }
    fn update_node_animation(&self, model: &mut Model, anime_frame: u32) {
        let bone_count = model.get_bone_count() as usize;
        for i in 0..bone_count {
            let bone = model.get_bone(i);

            let node_animation = self.node_map.get(&bone.borrow().get_name().to_string());
            if node_animation.is_none() {
                continue;
            }
            let node_animation = node_animation.expect("Failed to unwrap node_animation");

            let (start, last) = node_animation.find_segment(anime_frame);

            // 線形補間
            let range = (last.frame - start.frame) as f32;
            // let mut translation = start.translation;
            // let mut rotation = start.rotation;
            // if range > 0.0 {
            //     let rate = (anime_frame - start.frame) as f32 / range;
            //     translation = start.translation;
            //     translation += (last.translation - start.translation) * rate;
            //     translation += *bone.borrow().get_initial_translation();
            //     rotation = start.rotation.slerp(last.rotation, rate).normalize();
            // }
            // if range == 0 {
            //     continue;
            // }
            // bone.borrow_mut().set_translation(translation);
            // bone.borrow_mut().set_rotation(rotation);

            // ベジェ補間
            let rate = (anime_frame - start.frame) as f32 / range;
            let mut bezier_k = Vector4::new(0.0, 0.0, 0.0, 0.0);
            bezier_k.x = Self::interporate_bezier(&start.interp_x, rate);
            bezier_k.y = Self::interporate_bezier(&start.interp_y, rate);
            bezier_k.z = Self::interporate_bezier(&start.interp_z, rate);
            bezier_k.w = Self::interporate_bezier(&start.interp_r, rate);

            let mut translation = start.translation;
            translation += (last.translation - start.translation)
                .mul_element_wise(Vector3::new(bezier_k.x, bezier_k.y, bezier_k.z));
            translation += *bone.borrow().get_initial_translation();
            bone.borrow_mut().set_translation(translation);

            let rotation = start.rotation.slerp(last.rotation, bezier_k.w);
            bone.borrow_mut().set_rotation(rotation);
        }

        model.update_matrices();
    }
    fn update_morph_animation(&self, model: &mut Model, anime_frame: u32) {
        for (name, anim) in self.morph_map.iter() {
            let (start, last) = anim.find_segment(anime_frame);

            let range = (last.frame - start.frame) as f32;
            let mut weight = start.weight;
            if range > 0.0 {
                let rate = (anime_frame - start.frame) as f32 / range;
                weight += (last.weight - start.weight) * rate;
            }

            let index = model.get_morph_index(name);
            if let Some(i) = index {
                model.set_morph_weight(i, weight);
            }
        }
    }
    fn update_ik_chains(&self, model: &Model) {
        let ik_count = model.get_bone_ik_count() as usize;
        for i in 0..ik_count {
            self.solve_ik(model.get_bone_ik(i));
        }
    }
    fn solve_ik(&self, bone_ik: &PMDBoneIK) {
        let target = bone_ik.get_target();
        let eff = bone_ik.get_effector();

        let chains = bone_ik.get_chains();
        for _ in 0..bone_ik.get_iteration_count() {
            for i in 0..chains.len() {
                let bone = Rc::clone(&chains[i]);
                let matrix_inv_bone = bone
                    .borrow()
                    .get_world_matrix()
                    .invert()
                    .expect("Failed to cal inverse of bone world matrix.");

                let effector_pos = matrix_inv_bone * eff.borrow().get_world_matrix().w;
                let effector_pos =
                    Vector3::new(effector_pos.x, effector_pos.y, effector_pos.z) / effector_pos.w;
                let target_pos = matrix_inv_bone * target.borrow().get_world_matrix().w;
                let target_pos =
                    Vector3::new(target_pos.x, target_pos.y, target_pos.z) / target_pos.w;

                let len = (target_pos - effector_pos).magnitude2();
                if len * len < 0.0001 {
                    return;
                }

                let vec_to_eff = effector_pos.normalize();
                let vec_to_target = target_pos.normalize();

                let dot = vec_to_eff.dot(vec_to_target);
                let dot = dot.max(-1.0).min(1.0);
                let radian = dot.acos();
                if radian < 0.0001 {
                    continue;
                }
                let limit_angle = bone_ik.get_angle_weight();
                let radian = radian.max(-limit_angle).min(limit_angle);

                let axis = vec_to_target.cross(vec_to_eff).normalize();

                if radian < 0.001 {
                    continue;
                }

                if bone.borrow().get_name().find("ひざ").is_some() {
                    let rotation = Quaternion::from_axis_angle(axis, Rad(radian));
                    let euler_angle: Euler<Rad<f32>> = rotation.into();
                    let angle_x = match euler_angle.x {
                        Rad(x) => x,
                    };
                    let rotation = Euler::new(
                        Rad(angle_x.max(0.002).min(std::f32::consts::PI)),
                        Rad(0.0),
                        Rad(0.0),
                    );
                    let rotation: Quaternion<f32> = rotation.into();
                    let rotation = (bone.borrow().get_rotation() * rotation).normalize();
                    bone.borrow_mut().set_rotation(rotation);
                } else {
                    let rotation = Quaternion::from_axis_angle(axis, Rad(radian));
                    let rotation = (bone.borrow().get_rotation() * rotation).normalize();
                    bone.borrow_mut().set_rotation(rotation);
                }

                for j in (0..=i).rev() {
                    chains[j].borrow_mut().update_world_matrix();
                }
                eff.borrow_mut().update_world_matrix();
                target.borrow_mut().update_world_matrix();
            }
        }
    }

    fn interporate_bezier(bezier: &Vector4<f32>, x: f32) -> f32 {
        let mut t = 0.5;
        let mut ft = fx(bezier.x, bezier.z, t, x);
        for _ in 0..32 {
            let dfx = d_fx(bezier.x, bezier.z, t);
            t = t - ft / dfx;
            ft = fx(bezier.x, bezier.z, t, x);
        }
        t = 0.0_f32.max(t).min(1.0);
        let dy = func_bezier_y(bezier, t);
        dy
    }

    pub fn get_frame_period(&self) -> u32 {
        self.frame_period
    }
}
