use rand::Rng;

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

    //this is where the input is borrowed vector and it returns the borrowed string of vectors.
    fn as_slice(&self) -> &[f32] {
        &self.v
    }

    fn normalize(&mut self) {
        let mut norm: f32 = 0.0;
        //here the val is an &f32 which is the reference so to dereference it we use &val
        //other option is to keep it val and inside the loop write (*val) instead of just val to dereference it
        for val in &self.v { norm += (*val) * (*val); }
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

#[derive(Clone, Debug)]
pub struct Node {
    id: usize,
    data: Vector,
    neighbors: Vec<Vec<usize>>,
}

pub struct Index {
    //All the vectors present in the List
    nodes: Vec<Option<Node>>,
    //The start point of the HNSW graph
    start_point: Option<usize>,
    max_height: u32,
    //m is the maximum number of neighbors a single node is allowed to have at any given layer!
    m: usize,
    //The below attribute controls the quality vs. speed of inserting a new node!
    //I would call it an hyperparameter
    ef_construction: usize,
    //The metric is the pointer to the actual data which actually lives in the heap
    //The size of the pointer is 16 byte which is considered as an fat pointer
    //8 byte for the pointer of the data which metric is gonna hold:
    //Another 8 byte for the vtable which tells the rust which dist() function to run:
    //The box actually bounds the address where the data in the heap is actually living like the
    //actual data living in the heap can be of any data size but the pointer which Box will hold is
    //bound to have 8 bytes
    //The other 8 bytes stores the address of the vtable which contains the address of all the
    //traits of this particular struct, because rust doesn't know which dist function.
    metric: Box<dyn DistanceMetric>,
}

//Box<dyn DistanceMetric> is a data type btw
//Box is an heap allocated pointer;

impl Index {
    pub fn new(m: usize, ef_construction: usize, metric: Box<dyn DistanceMetric>) -> Self {
        Self {
            nodes: Vec::new(),
            start_point: None,
            max_height: 0,
            m,
            ef_construction,
            metric,
        }
    }

    //This function will basically roll a weighted die and decide, like in which layer the vector will fall.
    pub fn random_level(&self) -> usize {
        let r: f64 = rand::random();
        if r == 0.0 { return 0; }
        let ml: f64 = 1.0 / (self.m as f64).ln();
        let lev: usize = (-r.ln() * ml).floor() as usize;
        lev
    }

    pub fn insert_vec(&mut self, vec: Vec<f32>) {
        let data = Vector::new(vec);
        let level: usize = self.random_level();
        let id: usize = self.nodes.len();
        let neighbors = vec![vec![]; level + 1];
        let node = Node {
            id,
            data,
            neighbors,
        };
        self.nodes.push(Some(node));
        if self.start_point == None {
            self.start_point = Some(id);
            self.max_height = level as u32;
            return;
        }

        //To find the best neighborhood for your node at a layer.
        //At each layer, it looks at the current node's neighbors.
        //You kinda loook at all the neighbor node in the current node and only move with the
        //neighbor which is the closest one

        let mut start: usize = self.start_point.unwrap();
        for i in (0..=self.max_height).rev() {
            start = self.search_layer(id, i, start);
            if i <= level as u32 {
                //push is an method call, and method calls in Rust auto-deref automatically.
                let a: &mut Vec<usize> = self.nodes[id].as_mut().unwrap().neighbors[i as usize];
                if a.len() < self.m { a.push(start); }
                let b: &mut Vec<usize> = self.nodes[start].as_mut().unwrap().neighbors[i as usize];
                if b.len() < self.m { b.push(id); }
            }
        }
        self.max_height = self.max_height.max(level as u32);
    }

    //greedy search at single layer.
    pub fn search_layer(&self, id: usize, height: u32, start_node: usize) -> usize {
        //We used .as_ref() to conver the &Option<T> into Option<&T>
        //The &Option<T> is because of the &self at the beginning.
        //We use .unwrap() to resolve Option<&T> into &T
        //The & at the beginning is the explicit  borrow operator
        //After .unwrap() gives us &Node(), which gives us &Vec<f32>

        let input_node_data: &[f32] = &self.nodes[id].as_ref().unwrap().data.v;

        let mut current_node_index: usize = start_node;
        let mut current_node_sim: f32 = self.metric.dist(self.nodes[current_node_index].as_ref().unwrap().data.v, input_node_data);

        let mut temp_node_index: usize = current_node_index;
        let mut temp_node_sim: f32 = current_node_sim;

        loop {
            let neighbor_index: &[usize] = self.nodes[current_node_index].as_ref().unwrap().neighbors[height as usize];

            for neighbor in &neighbor_index {
                //The neighbor is of data type &usize but rust auto-derefs it when indexing so we don't need to deref it.
                let neighbor_data: &[f32] = self.nodes[neighbor].as_ref().unwrap().data.v;
                temp_node_sim = self.metric.dist(&neighbor_data, &input_node_data);
                if temp_node_sim < current_node_sim {
                    current_node_sim = temp_node_sim;
                    temp_node_index = *neighbor; //the neighbor is &usize so we have to deref it
                }
            }
            if current_node_index == temp_node_index {
                break;
            }
            current_node_index = temp_node_index;
        }
        current_node_index
    }
}

fn main() {

}
