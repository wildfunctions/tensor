# Tensor

Currently, support is only for contravariant Tensors of rank 1 and 2.
Also, all Tensors are assumed to use the standard euclidean basis.

We use the following rank 2 Tensor definition:
T<sup>'α</sup>T<sup>'β</sup> = ( ∂x<sup>'</sup><sub>α</sub> / ∂x<sub>γ</sub> )(  ∂x<sup>'</sup><sub>β</sub> / ∂x<sub>μ</sub> )T<sup>γμ</sup>

After assuming a standard basis, we know that ( ∂x<sup>'</sup><sub>α</sub> / ∂x<sub>γ</sub> ) will be zero for all instances α != γ, and the same for ( ∂x<sup>'</sup><sub>β</sub> / ∂x<sub>μ</sub> ) with regards to β != μ.

This allows for a straight-forward calculation of the outer product of rank 1 Tensors.

This code serves as a sketch.  A more robust Tensor library is on the way.
