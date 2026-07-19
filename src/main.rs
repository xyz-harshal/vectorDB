//Mutability isn't a property the variable carries around. It's a property of the reference.
//Option and Result are an enums btw
//A ref to a struct let's us access the fields, but it doesn't gives us the references to them.
//Reaching a field just names us places - we don't own it.
//&path if mentioned explicitly will give us reference to the data till the path ends.
//the & given by the .as_ref() func will only give ref to it's target not it's childrens!
//Each core in the CPU has it's own SIMD units.
//-floating-point formats: sign bit + exponent bits + fraction bits
//-Rayon crate in rust handles the multi threading so that we can access multiple cores at the same time
//.zip() is a method that is only applied on iterator not on an data type
//In a loop if each step requires answer from the previous step then it can't be parallelized
//For a loop to be parallelized ie vectorized with SIMD then the loop should only use values in present state
//The chunks_exact(n) function gives an iterator which points to the og data in the heap but it gives the iterator in such a way so that chunks are created!

//=== DONE: SIMD dist ✓ (all metrics + normalize vectorized; gap 3.5x -> ~2.5x) ===

//=== BASELINE (pre-refactor, locked) — pass/fail criterion for memory-layout work ===
// Config: dim=32, m=16, ef_c=100, ef_search=64, k=10, Euclidean, single-threaded
// N=20k:  build ~6.0s | query ~200us | recall 0.94-0.95 | orphans 0 | avg deg L0 ~22.7
// Sweep @20k: ef16=0.64 | ef32=0.83 | ef64=0.95 | ef128=0.99 | ef256=1.00
// vs hnswlib (same-session ratios): build ~2.4x slower | query ~2.5-3x slower | recall PARITY
// Cosine (normalize-once): ~133us @ ef64, recall ~0.945
// PASS after refactor = recall/sweep identical (±noise), query faster, zero orphans
// FAIL = any recall drop -> bisect the refactor

//=== THEN: memory layout (the remaining ~2-2.5x, big refactor) ===
//TODO: Flat storage: one Vec<f32> for ALL vectors, node i's data at [i*dim .. (i+1)*dim]
//TODO: Neighbor lists as u32 instead of usize (half edge memory, 2x per cache line)
//TODO: Pooled visited-set: reusable epoch-stamped Vec<u32> instead of fresh HashSet per search
//TODO: Kill the .clone()s in insert while restructuring (folded into this refactor, not before)
//TODO: Verify vs baseline: recall identical, queries faster — else bisect

//=== ALGORITHM GARNISHES (unchanged) ===
//TODO: keepPrunedConnections -- backfill empty slots with best rejected candidates
//TODO: alpha-pruning parameter (alpha=1.0 current; try 1.2 like DiskANN, benchmark recall)

//=== RESEARCH TOYS (one addition) ===
//TODO: Benchmark on a REAL dataset (SIFT1M standard) -- random high-dim data understates recall
//TODO: Quantization f32 -> int8: per-vector scale, calibrate range, then the recall ladder
//TODO: Re-rank pipeline: search quantized, exact re-rank top-100
//TODO: Rayon: parallelize search across queries
//TODO: Own recall-vs-QPS curve vs ann-benchmarks.com

//=== HYGIENE (one item resolved by Cosine redesign) ===
//TODO: max_height -> usize, delete every cast
//TODO: Delete unused Rng import
//TODO: Invariant checks: zero orphans at layer 0, all edges valid

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

pub trait DistanceMetric {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32;
    fn needs_normalize(&self) -> bool;
}

pub struct Euclidean;

impl DistanceMetric for Euclidean {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let mut s: [f32; 8] = [0.0f32; 8];
        // chunks variable is an iterator where each element is &[f32]
        // each element size if of 8 elements
        let chunks_v1 = v1.chunks_exact(8);
        let chunks_v2 = v2.chunks_exact(8);
        let remaining_v1: &[f32] = chunks_v1.remainder();
        let remaining_v2: &[f32] = chunks_v2.remainder();
        //here every chunk of both the vectors are paired not every element but every chunk of size 8
        //the magic of SIMD is that the inner loop will execute at once
        for (a, b) in chunks_v1.zip(chunks_v2) {
            for j in 0..8 {
                let x: f32 = a[j] - b[j];
                s[j] += x * x;
            }
        }
        let mut total: f32 = s.iter().sum();
        for (a, b) in remaining_v1.iter().zip(remaining_v2) {
            let x: f32 = *a - *b;
            total += x * x;
        }
        total
    }
    fn needs_normalize(&self) -> bool { false }
}

pub struct Cosine;

impl DistanceMetric for Cosine {
    //A zero query vector under Cosine now returns distance 1.0 to everything (old code returned 0.0
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let num: f32 = DotProduct.dist(v1, v2);
        //We are doing `+` because the DotProduct.dist function gives out the -dist as value
        1.0f32 + num
    }
    fn needs_normalize(&self) -> bool { true }
}

pub struct DotProduct;

impl DistanceMetric for DotProduct {
    fn dist(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let mut s: [f32; 8] = [0.0f32; 8];
        let chunks_v1 = v1.chunks_exact(8);
        let chunks_v2 = v2.chunks_exact(8);
        let rem_v1: &[f32] = chunks_v1.remainder();
        let rem_v2: &[f32] = chunks_v2.remainder();

        for (a, b) in chunks_v1.zip(chunks_v2) {
            //vectorized loop
            for j in 0..8 {
                s[j] += a[j] * b[j];
            }
        }
        let mut res: f32 = s.iter().sum();
        for (a, b) in rem_v1.iter().zip(rem_v2) {
            res += *a * *b;
        }
        -res
    }
    fn needs_normalize(&self) -> bool { false }
}

#[derive(Clone, Debug)]
pub struct Node {
    id: usize,
    neighbors: Vec<Vec<usize>>,
}

pub struct Index {
    //This vectors array is basically the concatenation of all the vectors.
    vectors: Vec<f32>,
    dim: usize,
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
    pub fn new(m: usize, ef_construction: usize, metric: Box<dyn DistanceMetric>, dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
            nodes: Vec::new(),
            start_point: None,
            max_height: 0,
            m,
            ef_construction,
            metric,
        }
    }
    fn normalize(&mut self, i: usize) {
        let mut s: [f32; 8] = [0.0f32; 8];
        //&self.v gives &f32 per element, and the &val pattern destructure it to a plain f32.
        //Or we can just deref it by putting * before val inside the scope.
        let chunks = self.vectors[i * self.dim..(i + 1) * self.dim].chunks_exact(8);
        let rem_chunks: &[f32] = chunks.remainder();
        for a in chunks {
            for j in 0..8 {
                s[j] += a[j] * a[j];
            }
        }
        let mut norm: f32 = s.iter().sum();
        for &a in rem_chunks { norm += a * a;}
        let norm = norm.sqrt();
        if norm == 0.0 { return; }
        //&mut self.v gives out &mut f32 so we can't destructure because if we do it we will lose
        //the mutability and it would detach a copy from the vector so we just deref it in the 
        //scope to keep the mutability property and change the value.
        for val in &mut self.vectors[i * self.dim..(i + 1) * self.dim] { *val /= norm; }
    }

    //This function will basically roll a weighted die and decide, like in which layer the vector will fall.
    pub fn random_level(&self) -> usize {
        let r: f64 = rand::random();
        if r == 0.0 || self.m == 1 || self.m == 0 { return 0; }
        let ml: f64 = 1.0 / (self.m as f64).ln();
        let lev: usize = (-r.ln() * ml).floor() as usize;
        lev
    }

    pub fn select_neighbors(&self, base_vec: usize, neighbor: &Vec<usize>, m: usize) -> Vec<usize> {
        let mut survivors: Vec<usize> = Vec::new();
        for &node in neighbor {
            if survivors.len() >= m { break; }
            let base_vec_data: &[f32] = &self.vectors[base_vec * self.dim..(base_vec + 1) * self.dim];
            let node_data: &[f32] = &self.vectors[node * self.dim..(node + 1) * self.dim];
            let dist1: f32 = self.metric.dist(node_data, base_vec_data);
            let mut b: bool = true;

            for &survivor in &survivors {
                let dist2: f32 = self.metric.dist(node_data, &self.vectors[survivor * self.dim..(survivor + 1) * self.dim]);
                if dist2 < dist1 {
                    b = false;
                    break;
                }
            }
            if b { survivors.push(node); }
        }
       survivors
    }

    pub fn insert_vec(&mut self, vec: Vec<f32>) {
        self.vectors.extend_from_slice(&vec);
        let level: usize = self.random_level();
        let id: usize = self.nodes.len();
        let neighbors = vec![vec![]; level + 1];
        let node = Node {
            id,
            neighbors,
        };

        self.nodes.push(Some(node));

        if self.metric.needs_normalize() {
            self.normalize(id);
        }
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
            let mut candidate: Vec<usize> = if i <= level as u32 { self.search_layer(&self.vectors[id * self.dim..(id + 1) * self.dim], i, start, self.ef_construction)
            }else { self.search_layer(&self.vectors[id * self.dim..(id + 1) * self.dim], i, start, 1) };
            if candidate.is_empty() { continue; }
            start = candidate[0];

            if i <= level as u32 {
                let cap: usize = if i == 0 { 2 * self.m } else { self.m };
                candidate = self.select_neighbors(id, &candidate, self.m);
                self.nodes[id].as_mut().unwrap().neighbors[i as usize] = candidate.clone();
                for &survivor in &candidate {
                    //here i am taking a mutable reference of the self.nodes object
                    let mut survivor_neighbor: Vec<usize> = self.nodes[survivor].as_mut().unwrap().neighbors[i as usize].clone();
                    survivor_neighbor.push(id);
                    if survivor_neighbor.len() > cap {
                        let mut temp: Vec<(f32, usize)> = Vec::new();
                        for &surv in &survivor_neighbor {
                            temp.push((self.metric.dist(&self.vectors[surv * self.dim..(surv + 1) * self.dim], &self.vectors[survivor * self.dim..(survivor + 1) * self.dim]), surv));
                        }
                        temp.sort_by(|a, b| a.0.total_cmp(&b.0));
                        survivor_neighbor.clear();
                        for (_, addr) in &temp {
                            survivor_neighbor.push(*addr);
                        }
                       survivor_neighbor = self.select_neighbors(survivor, &survivor_neighbor, cap);
                    }
                    self.nodes[survivor].as_mut().unwrap().neighbors[i as usize] = survivor_neighbor;
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
        let c0: OrdF32 = OrdF32(self.metric.dist(&self.vectors[current_node_index * self.dim..(current_node_index + 1) * self.dim], input_node_data));
        frontier.push(Reverse((c0, current_node_index)));
        board.push((c0, current_node_index));
        visited.insert(current_node_index);

        loop {
            if frontier.is_empty() { break; }
            if let Some(Reverse((dist, id))) = frontier.pop() {
                if board.len() >= ef && board.peek().unwrap().0 < dist { break; }
                //Time to get all the neighbors of this particular id;
                let neighbors: &Vec<usize> = &self.nodes[id].as_ref().unwrap().neighbors[height as usize];

                for &neighbor in neighbors {
                    if !visited.insert(neighbor) { continue; }
                    let neighbor_curr_dist: OrdF32 = OrdF32(self.metric.dist(&self.vectors[neighbor * self.dim..(neighbor + 1) * self.dim], input_node_data));
                    if board.len() >= ef && board.peek().unwrap().0 < neighbor_curr_dist { continue; }
                    frontier.push(Reverse((neighbor_curr_dist, neighbor)));
                    board.push((neighbor_curr_dist, neighbor));
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
        //The input_node_data.to_vec() is somewhat kinda not that expensive operation and it should
        //be solved later, but for now it increases nanoseconds against the miliseconds
        let input_vec: Vec<f32> = if self.metric.needs_normalize() {
            let mut x: Vector = Vector {
                v: input_node_data.to_vec(),
            };
            x.normalize();
            x.v
        }else{ input_node_data.to_vec() };
        let mut start: usize = self.start_point.unwrap();
        let mut candidate: Vec<usize> = Vec::new();
        for i in (0..=self.max_height).rev() {
            if i == 0 {
                candidate = self.search_layer(&input_vec, i, start, ef_search);
            }else{
                candidate = self.search_layer(&input_vec, i, start, 1);
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
