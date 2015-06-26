use std::ops::Mul;

pub struct Rank1Tensor {
    dim: i32,
    components: Vec<i64>, 
}

pub struct Rank2Tensor {
    dim: i32,
    components: Vec<Vec<i64>>,
}

pub struct Rank3Tensor {
    dim: i32,
    components: Vec<Vec<Vec<i64>>>,
}

//generalized tensor
pub struct Tensor {
   dim: i32,
   rank: i32,
   components: Vec<i64>, 
}

//generalized tensor impl
impl Tensor {
    pub fn build(d: i32, r: i32, v: Vec<i64>) -> Tensor {
        Tensor {
            //dim of single basis vector
            dim: d,
            rank: r,
            components: v,
        } 
    } 
    //O(rank) 
    pub fn get(&self, indices: &[i32]) -> i64 {
        let mut i: i32 = 0;
        //We can build dim^(rank) ahead of time
        for x in 0..self.rank {
            i = i + scalar_power(self.dim, x as i32) * indices[x as usize];
        }     
        self.components[i as usize].clone() 
    }
    fn print_element(&self, v: Vec<i32>) {
        println!( "T{:?} : {:?}", v, self.get(&v) ); 
    }
    pub fn print(&self) {
       let mut v: Vec<i32> = Vec::new();
       for _ in 0..self.rank {
           v.push(self.dim.clone());
       }
       vloop(v, Tensor::print_element, &self);
    }
}

//temp solution for n^p 
pub fn scalar_power(n: i32, p: i32) -> i32 {
    let mut a: i32 = 1;
    for _ in 0..p {
        a = a*n;
    }
    a
}

//entry point for vloop_many
pub fn vloop(max_indices: Vec<i32>, f: fn(&Tensor, Vec<i32>), t: &Tensor) {
    vloop_many(max_indices.clone(), f, t, Vec::new(), 0);
}

//variable depth loop
pub fn vloop_many(max_indices: Vec<i32>, f: fn(&Tensor, Vec<i32>), t: &Tensor, pargs: Vec<i32>, index: i32) {
    if max_indices.len() == 0 {
        f(t, pargs); 
    } else {
        let mut args = pargs.clone();
        let rest: Vec<i32> = max_indices[1..].to_vec();
        for _ in 0..max_indices[0] {
            if args.len() == index as usize { args.push(0); }
            if args[index as usize] < max_indices[0] {
                vloop_many(rest.clone(), f, t, args.clone(), index + 1);
                args[index as usize] = args[index as usize] + 1;
            }
        }
    }
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
    pub fn get(&self, i: i32) -> i64 {
       self.components[i as usize].clone() 
    }
    pub fn dim(&self) -> i32 {
        self.dim.clone()
    }
}

impl Rank2Tensor {
    pub fn new(d: i32) -> Rank2Tensor { 
        let mut temp_vec = Vec::new();
        for _ in 0..d {
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
    pub fn get(&self, i: i32, j: i32) -> i64 {
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

impl Rank3Tensor {
    pub fn new(d: i32) -> Rank3Tensor { 
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
    pub fn build(d: i32, c: Vec<Vec<Vec<i64>>>) -> Rank3Tensor {
        Rank3Tensor {
            dim: d,
            components: c,
        }
    }
    pub fn get(&self, i: i32, j: i32, k: i32) -> i64 {
       self.components[i as usize][j as usize][k as usize].clone() 
    }
    pub fn dim(&self) -> i32 {
        self.dim.clone()
    }
}

impl Mul<Rank1Tensor> for Rank1Tensor {
    type Output = Rank2Tensor;
    fn mul(self, other: Rank1Tensor) -> Rank2Tensor {
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
