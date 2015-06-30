mod helper;

#[allow(dead_code)]
mod special;
pub use self::special::{Rank1Tensor, Rank2Tensor};

use traits;

mod opperator;

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
