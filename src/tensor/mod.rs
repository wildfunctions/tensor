mod helper;

use traits;
use std::ops::Mul;

//generalized tensor
pub struct Tensor<T: traits::TensorTrait<T>> {
   dim: i32,
   rank: i32,
   components: Vec<T>, 
}

//generalized tensor impl
impl<T: traits::TensorTrait<T>> Tensor<T> {
    pub fn build(d: i32, r: i32, v: Vec<T>) -> Self {
        Tensor {
            //dim of basis vector
            dim: d,
            rank: r,
            components: v,
        } 
    } 
    pub fn clone(&self) -> Self {
        Tensor::build(self.dim.clone(), self.rank.clone(), self.components.clone())
    }
    //improve - perf is still O(rank) 
    pub fn get(&self, indices: &[i32]) -> T {
        let mut i: i32 = 0;
        //We can build dim^(rank) ahead of time
        for x in 0..self.rank {
            i = i + helper::scalar_power(self.dim, x as i32) * indices[x as usize];
        }     
        self.components[i as usize].clone() 
    }
    pub fn set(&mut self, indices: &[i32], value: T) {
        let mut i: i32 = 0;
        //We can build dim^(rank) ahead of time
        for x in 0..self.rank {
            i = i + helper::scalar_power(self.dim, x as i32) * indices[x as usize];
        }
        self.components[i as usize] = value;
    } 
    pub fn inner_product(&self, other: &Tensor<T>) -> Self {
        //equality assumptions for now
        assert_eq!(self.rank, (*other).rank);
        assert_eq!(self.dim, (*other).dim);

        let mut args: Vec<i32> = Vec::new();
        for _ in 0..(self.rank + other.rank - 2) {
            args.push(self.dim);
        }
        let new_rank: i32 = self.rank + other.rank - 2;
        let len = helper::scalar_power(self.dim, new_rank) as usize;
        let mut new_tensor = &mut Tensor::build(self.dim, new_rank, vec![T::zero(); len]);

        helper::inner_product_loop(args, Tensor::inner_product_segment, &self, &other, new_tensor);

        //this clone is bad
        (*new_tensor).clone()
    } 
    fn inner_product_segment(&mut self, t1: &Tensor<T>, t2: &Tensor<T>, indices: &[i32]) {
        let i: usize = (t1.rank - 1) as usize;
        let j: usize = (t1.rank - 1) as usize;
        println!( "{} :: {}", i, j);
        let mut v1 = indices[0..i].to_vec();
        let mut v2 = indices[j..].to_vec(); 
        
        println!( "{:?} :: {:?}", v1, v2 );

        //sum over contracted indices 
        v1.push(0);
        v2.insert(0, 0);

        let mut total: T = T::zero(); 
        for x in 0..t1.dim {
            v1[(t1.rank - 1) as usize] = x;
            v2[0] = x;
            total = total + t1.get(&v1) * t2.get(&v2);  
        } 
        
        //we set the indices of mut self
        self.set(indices, total);
    } 
    pub fn print(&self) {
       let mut v: Vec<i32> = Vec::new();
       for _ in 0..self.rank {
           v.push(self.dim.clone());
       }
       helper::print_loop(v, Tensor::print_element, &self);
    }
    fn print_element(&self, v: Vec<i32>) {
        println!( "T{:?} : {:?}", v, self.get(&v) ); 
    }
}

pub struct Rank1Tensor<T: traits::TensorTrait<T>> {
    dim: i32,
    components: Vec<T>, 
}

pub struct Rank2Tensor<T: traits::TensorTrait<T>> {
    dim: i32,
    components: Vec<Vec<T>>,
}

pub struct Rank3Tensor<T: traits::TensorTrait<T>> {
    dim: i32,
    components: Vec<Vec<Vec<T>>>,
}

impl<T: traits::TensorTrait<T>> Rank1Tensor<T> {
    pub fn new(d: i32) -> Self { 
        Rank1Tensor { 
            dim: d, 
            components: Vec::new(),
        } 
    }
    pub fn build(d: i32, c: Vec<T>) -> Self {
        Rank1Tensor {
            dim: d,
            components: c,
        }
    }
    pub fn get(&self, i: i32) -> T {
       self.components[i as usize].clone() 
    }
    pub fn dim(&self) -> i32 {
        self.dim.clone()
    }
}

impl<T: traits::TensorTrait<T>> Rank2Tensor<T> {
    pub fn new(d: i32) -> Self { 
        let mut temp_vec = Vec::new();
        for _ in 0..d {
            temp_vec.push(Vec::new());
        }
        Rank2Tensor { 
            dim: d, 
            components: temp_vec,
        } 
    }
    pub fn build(d: i32, c: Vec<Vec<T>>) -> Self {
        Rank2Tensor {
            dim: d,
            components: c,
        }
    }
    pub fn get(&self, i: i32, j: i32) -> T {
        self.components[i as usize][j as usize].clone()
    } 
    pub fn dim(&self) -> i32 {
        self.dim.clone()
    }
    pub fn print(&self) {
        for i in 0..self.dim() {
            for j in 0..self.dim() {
                print!{ "T[{},{}]: {}\n", i, j, self.get(i, j) };
            }
        }
    }
}

impl<T: traits::TensorTrait<T>> Rank3Tensor<T> {
    pub fn new(d: i32) -> Self { 
        let mut temp_vec = Vec::new();
        for _ in 0..d {
            let mut temp_vec_2 = Vec::new();
            for _ in 0..d {
                temp_vec_2.push(Vec::new());
            }
            temp_vec.push(temp_vec_2);
        }
        Rank3Tensor { 
            dim: d, 
            components: temp_vec,
        } 
    }
    pub fn build(d: i32, c: Vec<Vec<Vec<T>>>) -> Self {
        Rank3Tensor {
            dim: d,
            components: c,
        }
    }
    pub fn get(&self, i: i32, j: i32, k: i32) -> T {
       self.components[i as usize][j as usize][k as usize].clone() 
    }
    pub fn dim(&self) -> i32 {
        self.dim.clone()
    }
}

impl<T: traits::TensorTrait<T>> Mul<Rank1Tensor<T>> for Rank1Tensor<T> {
    type Output = Rank2Tensor<T>;
    fn mul(self, other: Rank1Tensor<T>) -> Rank2Tensor<T> {
        let mut vec = Vec::new();    
        for i in 0..self.dim() {
            let mut temp_vec = Vec::new();
            for j in 0..other.dim() {
                temp_vec.push(self.get(i) * other.get(j))
            }
            vec.push(temp_vec)
        }
        Rank2Tensor::build(self.dim(), vec)
    }
}
