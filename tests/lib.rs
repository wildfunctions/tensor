extern crate tensor;

use tensor::Tensor;

#[test]
fn inner_product() {
    let v1 = Tensor::build(3, 1, vec![
        1,2,3
    ]);

    let v2 = v1.inner_product(&v1);

    let v1_dot_v1 = Tensor::build(1, 0, vec![14]);
}
