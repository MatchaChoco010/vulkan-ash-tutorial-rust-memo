use cgmath::{Deg, Matrix4, Point3, Vector2, Vector3};

/// perspectiveを生やすための拡張用トレイト。
pub trait Matrix4Ext {
    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspecf32: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32>;

    fn ortho(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Matrix4<f32>;
}
impl Matrix4Ext for Matrix4<f32> {
    fn perspective<A: Into<cgmath::Rad<f32>>>(
        fovy: A,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Matrix4<f32> {
        use cgmath::{Angle, Rad};
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
}
