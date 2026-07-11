//Mutability isn't a property the variable carries around. It's a property of the reference.
//Option and Result are an enums btw
//A ref to a struct let's us access the fields, but it doesn't gives us the references to them.
//Reaching a field just names us places - we don't own it.
//&path if mentioned explicitly will give us reference to the data till the path ends.
//the & given by the .as_ref() func will only give ref to it's target not it's childrens!
use std::collections::{HashSet, BinaryHeap};
use std::cmp::Reverse;
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

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
        //&self.v gives &f32 per element, and the &val pattern destructure it to a plain f32.
        //Or we can just deref it by putting * before val inside the scope.
        for &val in &self.v { norm += val * val; }
        let norm = norm.sqrt();
        if norm == 0.0 { return; }
        //&mut self.v gives out &mut f32 so we can't destructure because if we do it we will lose
        //the mutability and it would detach a copy from the vector so we just deref it in the 
        //scope to keep the mutability property and change the value.
        for val in &mut self.v { *val /= norm; }
    }
}

pub trait DistanceMetric {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32;
}

pub struct Euclidean;

impl DistanceMetric for Euclidean {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let mut s: f32 = 0.0;
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
        let mut s: f32 = 0.0;
        let mut m1: f32 = 0.0;
        let mut m2: f32 = 0.0;
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
        let mut s: f32 = 0.0;
        for i in 0..v1.len().min(v2.len()) {
            s += v1[i] * v2[i];
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
        if r == 0.0 || self.m == 1 || self.m == 0 { return 0; }
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

            let mut candidate: Vec<usize> = self.search_layer(&self.nodes[id].as_ref().unwrap().data.v, i, start, self.ef_construction);
            if candidate.is_empty() { continue; }
            start = candidate[0];
            //This will make sure to get the m best nodes which are closest to the input node.
            candidate.truncate(self.m);

            if i <= level as u32 {
                for node in candidate {
                    //id and node are both the indexes in consideration!
                    let d: f32 = self.metric.dist(&self.nodes[id].as_ref().unwrap().data.v, &self.nodes[node].as_ref().unwrap().data.v);
                    //The code to get the vector neighbor of both of the nodes
                    let id_neighbor: &Vec<usize> = &self.nodes[id].as_ref().unwrap().neighbors[i];
                    let node_neighbor: &Vec<usize> = &self.nodes[node].as_ref().unwrap().neighbors[i];
                    //To get the worst element of the neighborhood;
                    let mut id_addr: usize = 0;
                    let mut id_dist: f32 = f32::NEG_INFINITY;
                    for j in 0..id_neighbor.len() {
                        let new_dist: f32 = self.metric.dist(&self.nodes[id].as_ref().unwrap().data.v, &self.nodes[id_neighbor[j]].as_ref().unwrap().data.v);
                        if new_dist > id_dist {
                            id_addr = j;
                            id_dist = new_dist;
                        }
                    }
                    let mut node_addr: usize = 0;
                    let mut node_dist: f32 = f32::NEG_INFINITY;
                    for j in 0..node_neighbor.len() {
                        let new_dist: f32 = self.metric.dist(&self.nodes[node].as_ref().unwrap().data.v, &self.nodes[node_neighbor[j]].as_ref().unwrap().data.v);
                        if new_dist > node_dist {
                            node_addr = j;
                            node_dist = new_dist;
                        }
                    }

                    let id_full: bool = id_neighbor.len() >= self.m;
                    let node_full: bool = node_neighbor.len() >= self.m;

                    let should_connect = match (id_full, node_full) {
                        (false, false) => true,
                        (false, true) => node_dist > d,
                        (true, false) => id_dist > d,
                        (true, true) => id_dist > d && node_dist > d,
                    };

                    if should_connect {
                        {
                            let id_list: &mut Vec<usize> = &mut self.nodes[id].as_mut().unwrap().neighbors[i];
                            if id_full {
                                id_list[id_addr] = node;
                            }else {
                                id_list.push(node);
                            }
                        }
                        {
                            let node_list: &mut Vec<usize> = &mut self.nodes[node].as_mut().unwrap().neighbors[i];
                            if node_full {
                                node_list[node_addr] = id;
                            }else {
                                node_list.push(id);
                            }
                        }
                    }

                }
            }
        }
        if self.max_height < level as u32 {
            self.max_height = level as u32;
            self.start_point = Some(id);
        }
    }

    //greedy search at single layer.
    pub fn search_layer(&self, input_node_data: &[f32], height: u32, current_node_index: usize, ef: usize) -> Vec<usize> {
        //We used .as_ref() to conver the &Option<T> into Option<&T>
        //The &Option<T> is because of the &self at the beginning.
        //We use .unwrap() to resolve Option<&T> into &T
        //The & at the beginning is the explicit  borrow operator
        //After .unwrap() gives us &Node(), which gives us &Vec<f32>
        //A set to actually keep track of visited nodes while traversing in the graph.
        //The information of the current node is inserted in the binary heap!
        //The Vec<usize> is an growable vector whereas the [usize] is an slice which whose size is dynamically determined at runtime
        //Now i am using the .get() function which gives out the Option<&T> so we don't need to
        //put the & ref before the statement, we did this because when we did indexing on the
        //neighbors 2d vec, rust gives out the owned value when indexing so now it won't.
        //The binary heap requires T: Ord instead of f32 so that it can use the cmp function to
        //compare and sort because f32 can be NaN sometimes.
        
        let mut frontier: BinaryHeap<Reverse<(OrdF32, usize)>> = BinaryHeap::new();
        let mut board: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        
        //Now i am going to insert the current node in both of the heaps:
        let c0: OrdF32 = OrdF32(self.metric.dist(&self.nodes[current_node_index].as_ref().unwrap().data.v, input_node_data));
        frontier.push(Reverse((c0, current_node_index)));
        board.push((c0, current_node_index));
        visited.insert(current_node_index);

        loop {
            if frontier.is_empty() { break; }
            if let Some(Reverse((dist, id))) = frontier.pop() {
                if board.len() >= ef && board.peek().unwrap().0 < dist { break; }
                //Time to get all the neighbors of this particular id;
                let neighbors: &Vec<usize> = &self.nodes[id].as_ref().unwrap().neighbors[height as usize];

                for neighbor in neighbors {
                    if !visited.insert(*neighbor) { continue; }
                    let neighbor_curr_dist: OrdF32 = OrdF32(self.metric.dist(&self.nodes[*neighbor].as_ref().unwrap().data.v, input_node_data));
                    if board.len() >= ef && board.peek().unwrap().0 < neighbor_curr_dist { continue; }
                    frontier.push(Reverse((neighbor_curr_dist, *neighbor)));
                    board.push((neighbor_curr_dist, *neighbor));
                    if board.len() > ef {
                        board.pop();
                    }
                }
            }
        }
        let mut result: Vec<usize> = Vec::new();
        loop {
            if board.is_empty() { break; }
            if let Some((_, id)) = board.pop() {
                result.push(id);
            }
        }
        result.reverse();
        result
    }
    pub fn search(&self, input_node_data: &[f32], ef_search: usize, k: usize) -> Vec<usize> {
        if self.start_point.is_none() { return Vec::new(); }
        let mut start: usize = self.start_point.unwrap();
        let mut candidate: Vec<usize> = Vec::new();
        for i in (0..=self.max_height).rev() {
            if i == 0 {
                candidate = self.search_layer(input_node_data, i, start, ef_search);
            }else{
                candidate = self.search_layer(input_node_data, i, start, 1);
                if candidate.is_empty() { continue; }
                start = candidate[0];
            }
        }
        candidate.truncate(k);
        candidate
    }
}

fn main() {

}
