use traits;

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
