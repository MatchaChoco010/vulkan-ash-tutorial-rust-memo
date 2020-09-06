use std::time::Instant;

pub struct Timer {
    counter: Instant,
    delta_frame: u32,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            counter: Instant::now(),
            delta_frame: 0,
        }
    }

    // 毎フレーム呼び出す関数
    pub fn tick_frame(&mut self) {
        let time_elapsed = self.counter.elapsed();
        self.counter = Instant::now();
        self.delta_frame = time_elapsed.subsec_micros();
    }

    pub fn delta_time(&self) -> f32 {
        self.delta_frame as f32 / 1000_000.0_f32
    }
}
