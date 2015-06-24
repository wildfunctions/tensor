# Tensor

Currently, support is only for contravariant Tensors of rank 1 and 2.<br/>
Also, all Tensors are assumed to use an orthonormal basis (Ex: Euclidean standard basis).

We use the following rank 2 Tensor definition:<br/>
>T<sup>'α</sup><sup>'β</sup> = ( ∂x<sup>'</sup><sub>α</sub> / ∂x<sub>γ</sub> )(  ∂x<sup>'</sup><sub>β</sub> / ∂x<sub>μ</sub> )T<sup>γμ</sup>

After assuming an orthonormal basis, we know that ( ∂x<sup>'</sup><sub>α</sub> / ∂x<sub>γ</sub> ) will be zero for all case α != γ.<br/> and one for α = γ
The same is true for ( ∂x<sup>'</sup><sub>β</sub> / ∂x<sub>μ</sub> ) with regards to β != μ.

This allows a straight-forward calculation of the outer and inner products of Tensors.

This code serves as a sketch.<br/>
A more robust Tensor library is on the way.
