# PathSignatures.jl

This repo contains Julia code to compute path signatures and related functions for either Euclidean or matrix Lie group valued time series. In particular, both continuous and discrete path signatures are implemented, along with efficient algorithms for signature kernel computations.

## Usage

Download the file and import the package as usual:
`include("PathSignatures.jl")`

The primary functions are `signature()` and `dsignature()`, which computes the truncated continuous and discrete path signatures. The function signatures along with the inputs/outputs are:
`signature(P, M, dtype)`
`dsignature(P, M, dtype)`
- `P` is a multi-dimensional array which represents the path. The first dimension is always the time points, and the remaining dimensions are used to represent the point at a given time. For example, an `N` dimensional Euclidean path of length `T` is a `T x N` array, while a path of length `T` in SO(n)<sup>d</sup> is a `T x n x n x d` array.
- `M` is the truncation level of the signature
- `dtype` is the datatype. Currently, the datatypes `dtype = "R"` (for Euclidean paths) and `dtype = "SO"` (for SO(n)<sup>d</sup> paths) are implemented. 

The output `S` is an element of the M-truncated tensor algebra, which is stored as a 1D array of multi-dimensional arrays. The k<sup>th</sup> element of `S` is a k-dimensional array, where each dimension is length `N`, the dimension of the Lie group. For example, to access the signature with multi-index `(1,4,1,2)`, we use `S[4][1,4,1,2]`.

Other functions include:
- `dsignature_kernel(P1, P2, M, dtype)`, which computes the inner product of the signatures S(P1) and S(P2) using a more efficient algorithm.

- `dsignature_MMDu(BP1, BP2, M, dtype)`, which computes an unbiased estimator of the maximum mean discrepancy between two collections of paths. The variables `BP1` and `BP2` must be a 1D array of arrays, where each entry contains a path. This allows the batch sizes and path lengths to be different. 

## References and Contact
Feel free to contact me at ldarrick at math.upenn.edu with any questions or suggestions. The reference accompanying this code, which contain two experiments for paths on Lie groups is:  

Lee, Darrick and Ghrist, Robert. [Path signatures on Lie groups](https://arxiv.org/abs/2007.06633), preprint.  