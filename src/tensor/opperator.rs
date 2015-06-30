use traits;
use tensor::{Rank1Tensor, Rank2Tensor};
use std::ops::Mul;

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
