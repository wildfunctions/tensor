use std::ops::{Add, Mul};

pub struct Rank1Tensor {
    dim: i32,
    components: Vec<i64>, 
}

pub struct Rank2Tensor {
    dim: i32,
    components: Vec<Vec<i64>>
}

impl Rank1Tensor {
    pub fn new(d: i32) -> Rank1Tensor { 
        Rank1Tensor { 
            dim: d, 
            components: Vec::new(),
        } 
    }
    pub fn build(d: i32, c: Vec<i64>) -> Rank1Tensor {
        Rank1Tensor {
            dim: d,
            components: c,
        }
    }
    pub fn get(&self, i: usize) -> i64 {
       self.components[i].clone() 
    }
    pub fn dim(&self) -> i32 {
        self.dim.clone()
    }
}

impl Rank2Tensor {
    pub fn new(d: i32) -> Rank2Tensor { 
        let mut temp_vec = Vec::new();
        for i in 0..d {
            temp_vec.push(Vec::new());
        }
        Rank2Tensor { 
            dim: d, 
            components: temp_vec,
        } 
    }
    pub fn build(d: i32, c: Vec<Vec<i64>>) -> Rank2Tensor {
        Rank2Tensor {
            dim: d,
            components: c,
        }
    }
    pub fn get(&self, i: usize, j: usize) -> i64 {
        self.components[i][j].clone()
    } 
    pub fn dim(&self) -> i32 {
        self.dim.clone()
    }
    pub fn print(&self) {
        for i in 0..self.dim() {
            for j in 0..self.dim() {
                print!{ "T{}{}: {}\n", i, j, self.get(i as usize, j as usize) };
            }
        }
    }
}

impl Mul<Rank1Tensor> for Rank1Tensor {
    type Output = Rank2Tensor;
    fn mul(self, other: Rank1Tensor) -> Rank2Tensor {
        let mut vec = Vec::new();    
        for i in 0..self.dim() {
            let mut temp_vec = Vec::new();
            for j in 0..other.dim() {
                temp_vec.push(self.get(i as usize) * other.get(j as usize))
            }
            vec.push(temp_vec)
        }
        Rank2Tensor::build(self.dim(), vec)
    }
}
