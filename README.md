# Tensor

Currently, support is only for contravariant Tensors of equal dimension basis vectors.<br/>
Also, all Tensors are assumed to use an orthonormal basis (Ex: Euclidean standard basis).

We use the following Tensor definition:<br/>
>T<sup>'α..'β</sup> = ( ∂x<sup>'</sup><sub>α</sub> / ∂x<sub>γ</sub> )..(  ∂x<sup>'</sup><sub>β</sub> / ∂x<sub>μ</sub> ) T<sup>γ..μ</sup>

After assuming an orthonormal basis, we know that ( ∂x<sup>'</sup><sub>α</sub> / ∂x<sub>γ</sub> ) will be zero for all cases α != γ.<br/> and one for α = γ
The same is true for other components such as ( ∂x<sup>'</sup><sub>β</sub> / ∂x<sub>μ</sub> ) with regards to β != μ.

This allows a straight-forward calculation of the outer and inner products of Tensors.

The inner product between two Tensors is currently defined for all Tensor ranks, provided the ranks are equal, and the indices of both Tensors are all of equal dimension.  We will gradually ease up on these restrictions.

## Example
This code serves as a sketch.<br/>


```rust
// 3d Rank3 Tensor Example
let t3 = Tensor::build(3, 3, vec![
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    
    10, 11, 12,
    13, 14, 15,
    16, 17, 18,
    
    19, 20, 21,
    22, 23, 24,
    25, 26, 27
]);

let t = t3.inner_product(&t3);
t.print();
```

The above code will perform the inner product between t3 and itself, and then print out each component of the final Tensor, t.


A more robust Tensor library is on the way.

## Purpose

This library is meant to abstract Tensor math and allow high rank Tensor calculations (Rank > 2).  Several individuals use the term 'Tensor' despite using only Vectors and Matrices.  We do not encourage this technicality.  Though Matrices and Vectors are special cases of Tensors, we do not aim to provide the highest performance for such low ranks.  We recommend 'nalgebra', https://crates.io/crates/nalgebra, for high performance Vector and Matrix operations.

## To Do

A list of additions and improvements:

* [ ] Tensor Outer Product
* [ ] Scalar Multiplication and Addition
* [ ] Tensor Addition and Subtration
* [ ] Tensor Conversion to nalgebra types
