use traits;
use std::ops::Mul;

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
            i = i + scalar_power(self.dim, x as i32) * indices[x as usize];
        }     
        self.components[i as usize].clone() 
    }
    pub fn set(&mut self, indices: &[i32], value: T) {
        let mut i: i32 = 0;
        //We can build dim^(rank) ahead of time
        for x in 0..self.rank {
            i = i + scalar_power(self.dim, x as i32) * indices[x as usize];
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
        let len = scalar_power(self.dim, new_rank) as usize;
        let mut new_tensor = &mut Tensor::build(self.dim, new_rank, vec![T::zero(); len]);

        inner_product_loop(args, Tensor::print_inner_product, &self, &other, new_tensor);

        //this clone is bad
        (*new_tensor).clone()
    } 
    fn print_inner_product(&mut self, t1: &Tensor<T>, t2: &Tensor<T>, indices: &[i32]) {
        let i: usize = (t1.rank - 1) as usize;
        let j: usize = (t1.rank - 1) as usize;
        let mut v1 = indices[0..i].to_vec();
        let mut v2 = indices[j..].to_vec(); 

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
       print_loop(v, Tensor::print_element, &self);
    }
    fn print_element(&self, v: Vec<i32>) {
        println!( "T{:?} : {:?}", v, self.get(&v) ); 
    }
}

//tmp solution for n^p 
pub fn scalar_power(n: i32, p: i32) -> i32 {
    let mut a: i32 = 1;
    for _ in 0..p { a = a*n; } 
    a
}

//entry point for inner_product_loop_many
pub fn inner_product_loop<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&mut Tensor<T>, &Tensor<T>, &Tensor<T>, &[i32]),
t1: &Tensor<T>, t2: &Tensor<T>, t3: &mut Tensor<T>) {
    inner_product_loop_many(max_indices.clone(), f, t1, t2, t3, Vec::new(), 0);
}

//variable depth inner product loop
pub fn inner_product_loop_many<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&mut Tensor<T>, &Tensor<T>, &Tensor<T>, &[i32]), 
t1: &Tensor<T>, t2: &Tensor<T>, t3: &mut Tensor<T>, pargs: Vec<i32>, index: i32) {
    if max_indices.len() == 0 {
        f(t3, t1, t2, &pargs); 
    } else {
        let mut args = pargs.clone();
        let rest: Vec<i32> = max_indices[1..].to_vec();
        for _ in 0..max_indices[0] {
            if args.len() == index as usize { args.push(0); }
            if args[index as usize] < max_indices[0] {
                inner_product_loop_many(rest.clone(), f, t1, t2, t3, args.clone(), index + 1);
                args[index as usize] = args[index as usize] + 1;
            }
        }
    }
}

//entry point for print_loop_many
pub fn print_loop<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&Tensor<T>, Vec<i32>), t: &Tensor<T>) {
    print_loop_many(max_indices.clone(), f, t, Vec::new(), 0);
}

//variable depth print loop
pub fn print_loop_many<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&Tensor<T>, Vec<i32>), 
t: &Tensor<T>, pargs: Vec<i32>, index: i32) {
    if max_indices.len() == 0 {
        f(t, pargs); 
    } else {
        let mut args = pargs.clone();
        let rest: Vec<i32> = max_indices[1..].to_vec();
        for _ in 0..max_indices[0] {
            if args.len() == index as usize { args.push(0); }
            if args[index as usize] < max_indices[0] {
                print_loop_many(rest.clone(), f, t, args.clone(), index + 1);
                args[index as usize] = args[index as usize] + 1;
            }
        }
    }
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
