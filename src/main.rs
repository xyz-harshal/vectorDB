#[derive(Clone, Debug)]
pub struct Vector {
    v: Vec<f32>,
}

impl Vector {
    fn new(data: Vec<f32>) -> Self {
        Self {
            v: data,
        }
    }

    fn len(&self) -> usize {
        self.v.len()
    }

    fn as_slice(&self) -> &[f32] {
        &self.v
    }

    fn normalize(&mut self) {
        let mut norm: f32 = 0.0;
        //here the val is an &f32 which is the reference so to dereference it we use &val
        //other option is to keep it val and inside the loop write (*val) instead of just val to dereference it
        for &val in &self.v { norm += val * val; }
        let norm = norm.sqrt();
        if norm == 0.0 { return; }
        //the &mut self.v vector when looped gives out element having &mut f32 type
        //You write it as val and then use it to update it, now it auto deref when applied to
        //operators which are syntactic sugar so it auto deref it self.
        //otherwise you have to check whether you get an error or not if you got an error then use
        //*val instead of val there
        for val in &mut self.v { val /= norm; }
    }
}

pub trait DistanceMetric {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32;
}

pub struct Euclidean;

impl DistanceMetric for Euclidean {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let mut s: f32 = 0;
        for i in 0..v1.len().min(v2.len()) {
            let a = v1[i] - v2[i];
            s += a * a;
        }
        s.sqrt()
    }
}

pub struct Cosine;

impl DistanceMetric for Cosine {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let mut s: f32 = 0;
        let mut m1: f32 = 0;
        let mut m2: f32 = 0;
        for i in 0..v1.len().min(v2.len()) {
            s += v1[i] * v2[i];
            m1 += v1[i] * v1[i];
            m2 += v2[i] * v2[i];
        }
        let den: f32 = (m1 * m2).sqrt();
        if den == 0.0 { return 0.0; }
        1.0 - (s / den)
    }
}

pub struct DotProduct;

impl DistanceMetric for DotProduct {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let mut s: f32 = 0;
        for i in 0..v1.len().min(v2.len()) {
            s += (v1[i] * v2[i]);
        }
        -s
    }
}

fn main() {

}
