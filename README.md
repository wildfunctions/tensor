# Tensor

Currently, support is only for contravariant Tensors of rank 1 and 2.
Also, all Tensors are assumed to use the standard euclidean basis.

We use the following rank 2 Tensor definition:
V<sup>'a</sup>V<sup>'b</sup> = ( dx<sup>'</sup><sub>a</sub>/dx<sub>y</sub> )(  dx<sup>'</sup><sub>b</sub>/dx<sub>u</sub> )V<sup>yu</sup>

After assuming a standard basis, we know that ( dx<sup>'</sup><sub>a</sub>/dx<sub>y</sub> ) will be zero for all instances a != y, and the same for ( dx<sup>'</sup><sub>b</sub>/dx<sub>u</sub> ) with regards to b != u.

This allows for a straight-forward calculation of the outer product of rank 1 Tensors.

This code serves as a sketch.  A more robust Tensor library is on the way.
