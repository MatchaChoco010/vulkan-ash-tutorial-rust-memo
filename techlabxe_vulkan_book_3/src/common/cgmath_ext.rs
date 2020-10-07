use cgmath::{Matrix4, Rad};

/// perspectiveを生やすための拡張用トレイト。
pub trait Matrix4Ext {
    fn perspective<A: Into<Rad<f32>>>(fovy: A, aspect: f32, near: f32, far: f32) -> Matrix4<f32>;

    fn ortho(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Matrix4<f32>;

    fn rotate_scale_matrix(&self) -> Matrix4<f32>;

    fn to_u8_slice(&self) -> &[u8];
}
impl Matrix4Ext for Matrix4<f32> {
    fn perspective<A: Into<Rad<f32>>>(fovy: A, aspect: f32, near: f32, far: f32) -> Matrix4<f32> {
        use cgmath::Angle;
        let f: Rad<f32> = fovy.into();
        let f = f / 2.0;
        let f = Rad::cot(f);
        Matrix4::<f32>::new(
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            (near * far) / (near - far),
            0.0,
        )
    }

    fn ortho(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Matrix4<f32> {
        Matrix4::<f32>::new(
            2.0 / (right - left),
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / (bottom - top),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / (near - far),
            0.0,
            -(right + left) / (right - left),
            -(bottom + top) / (bottom - top),
            near / (near - far),
            1.0,
        )
    }

    fn rotate_scale_matrix(&self) -> Matrix4<f32> {
        Matrix4::new(
            self.x.x, self.x.y, self.x.z, 0.0, self.y.x, self.y.y, self.y.z, 0.0, self.z.x,
            self.z.y, self.z.z, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    fn to_u8_slice(&self) -> &[u8] {
        unsafe {
            ::std::slice::from_raw_parts(
                (self as *const Matrix4<f32>) as *const u8,
                ::std::mem::size_of::<Matrix4<f32>>(),
            )
        }
    }
}
