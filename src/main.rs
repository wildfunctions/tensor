pub mod tensor;
pub use tensor::{Rank1Tensor, Rank2Tensor};

fn main() {
    let x1 = vec![1,0];
    let x2 = vec![0,1];

    let v1 = Rank1Tensor::build(2, x1);
    let v2 = Rank1Tensor::build(2, x2); 

    let t1 = v1 * v2;
    t1.print();
}
