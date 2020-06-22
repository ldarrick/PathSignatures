
using Combinatorics
using LinearAlgebra
using TensorOperations
using Roots

#############################################################################
# COMMON VARIABLE NAMES
# T = number of timepoints (length of time series)
# N = total dimension of space
# n = dimension of a single copy of a space
# m = parameter for Lie group (ie. SO(m))
# d = number of copies of a space


###############################################################################
# Description: IteratedIntegral() is a function that takes in an array of
# cochains (an array of functions), and returns two functions:
#       IISignature() - computes the signature for a specific term (ie. [1,2,2])
#       IISignatureL() - computes the signature up to a given level
#
# Input: cochains should be an array of functions that takes in a path and evaluates
#   the cochain on the path - if a path of length T is inputted, the output is a path
#   of length T-1
# 
#   psize() is a function that returns (n_samples, n_features)
#
#   preprocess(P) is a function that preprocesses the path before starting
#       the signature computation (if required)
#
# Note: Inputs to the cochains should be type restricted
function IteratedIntegral(cochains, preprocess)
     # N = length(cochains)

    # Inputs:
    # P - a path as an (n x d) array; (n - number of points, d - dimension)
    # term - the term (multi-index) of the signature to calculate (ie. [1,2,2])
    #
    # Outputs:
    # sig - a vector of signature terms used to calculate
    #   ie. for term = [1,2,2], sig = [S^1, S^{1,2}, S^{1,2,2}]
    # sigpath - returns the signature path S^{term}(t) (used for recursion)
    # function IISignature(P, term)
    #     n = size(P,1)
    #     level = length(term)

    #     sigpath = zeros(n-1,1)
    #     eval_cochain = cochains[term[end]](P)
    #     # dP_end = eval_cc[2:end] - eval_cc[1:end-1]

    #     if level == 1
    #         integrand = eval_cochain
    #         sig = zeros(0)
    #     else
    #         sig, last_sigpath = IISignature(P, term[1:end-1])
    #         integrand = last_sigpath.*eval_cochain
    #     end

    #     sigpath = cumsum(integrand)
    #     sig = push!(sig, sigpath[end])

    #     return sig, sigpath
    # end


    function sig_forward(sigl, lastQ, K, cur_ind, cur_level, last_level, N)

        if cur_level < last_level

            for i = 1:N-1
                cur_ind[cur_level] = i
                Q = cumsum(lastQ .* view(K,:,i))
                Q = cumsum()
                sigl[cur_level][cur_ind[1:cur_level]...] = Q[end]

                sig_forward(sigl, Q, K, cur_ind, cur_level+1, last_level, N)
            end

            # On the last run through, we no longer need the information from
            # lastQ, so just use that variable instead of allocating more memory
            cur_ind[cur_level] = N
            cumsum!(lastQ, lastQ .* view(K,:,N))
            sigl[cur_level][cur_ind[1:cur_level]...] = lastQ[end]

            sig_forward(sigl, lastQ, K, cur_ind, cur_level+1, last_level, N)
        else

            for i = 1:N
                cur_ind[cur_level] = i
                sigl[cur_level][cur_ind...] = sum(lastQ .* view(K,:,i))
            end

        end
    end


    function IISignatureL(P, level)

        # P, IT, N, T = preprocess(P)
        # EIT = enumerate(IT) # create enumerated version of iterator

        T, N = size(P)
        K = zeros(T-1, N)
        for i = 1:N
            K[:,i] = P[2:end,i] - P[1:end-1,i]
        end
        # for (i, val) in EIT
        #     K[:,i] = cochains(P, val...)
        # end



        sigl = Array{Array{Float64},1}(undef, level)
        for i = 1:level
            sigl[i] = Array{Float64, i}(undef, (ones(Int,i)*N)...)
        end

        for i = 1:N
            cur_ind = zeros(Int, level)
            cur_ind[1] = i
            Q = cumsum(view(K,:,i))
            sigl[1][i] = Q[end]

            sig_forward(sigl, Q, K, cur_ind, 2, level, N)
        end

        return sigl

    end


    # Inputs
    # P1 - path as an (n1 x d) array
    # P2 - path as an (n2 x d) array
    # level - level to compute up to
    function IIKernel(P1, P2, IT, N, level)

        T1 = size(P1)[1]
        T2 = size(P2)[1]
        EIT = enumerate(IT)

        K1 = zeros(T1-1, N)
        K2 = zeros(T2-1, N)

        for (i, val) in EIT
            K1[:,i] = cochains(P1, val...)
            K2[:,i] = cochains(P2, val...)
        end

        # Compute the tensor normalization
        coeff1 = zeros(level+1)
        coeff2 = zeros(level+1)

        # Compute normalization for P1
        K = K1*K1'
        A = deepcopy(K)
        coeff1[2] = sum(view(A,:))

        for i = 2:level   
            cumsum!(A,A,dims=1)
            cumsum!(A,A,dims=2)
            A =  @. K*(1 + A)
            coeff1[i+1] = sum(view(A,:))
        end
        norm1 = coeff1[end] + 1
        c1 = coeff1[2:end] - coeff1[1:end-1]
        f1(x) = sum(c1.*(x.^((1:level)*2))) + 1 - PHI(sqrt(norm1),4.,1.)
        # println("norm1=", norm1, ", PHI(sqrt(norm1))=", PHI(sqrt(norm1), 4., 1.), ", f1(0)=", f1(0.), ", f1(2)=", f1(2), ", f1(norm1)=", f1(norm1))
        lambda1 = find_zero(f1, (0., max(2.,norm1)))

        # Compute normalization for P2
        K = K2*K2'
        A = deepcopy(K)
        coeff2[2] = sum(view(A,:))

        for i = 2:level   
            cumsum!(A,A,dims=1)
            cumsum!(A,A,dims=2)
            A =  @. K*(1 + A)
            coeff2[i+1] = sum(view(A,:))
        end
        norm2 = coeff2[end] + 1
        c2 = coeff2[2:end] - coeff2[1:end-1]
        f2(x) = sum(c2.*(x.^((1:level)*2))) + 1 - PHI(sqrt(norm2),4.,1.)
        # println("norm2=", norm2, ", PHI(sqrt(norm2))=", PHI(sqrt(norm2), 4., 1.), ", f2(0)=", f2(0.), ", f2(2)=", f2(2), ", f2(norm2)=", f2(norm2))
        lambda2 = find_zero(f2, (0., max(2.,norm2)))

        # coefficient
        ll = lambda1*lambda2

        # Compute normalized kernel
        K = K1*K2'
        A = deepcopy(K)

        for i = 2:level   
            cumsum!(A,A,dims=1)
            cumsum!(A,A,dims=2)
            if i==2
                A = @. K*(ll^(level-1) + A*ll^level)
            else
                A = @. K*(ll^(level-i+1) + A)
            end
        end

        R = 1 + sum(view(A,:))

        return R
    end


    function IIKernel_PP(P1, P2, level)
        P1, IT, N, ~ = preprocess(P1)
        P2, IT, N, ~ = preprocess(P2)

        return IIKernel(P1, P2, IT, N, level)
    end


    # This computes the unbiased MMD
    # Input: P1 and P2 are batches of time series of the given type
    # We assume that the batch index is the first index
    # For example: size(P1) = (S, T, n1, n2, ..., nl)
    #   S - size of batch
    #   T - number of time points for each time series
    #   n1, n2, ..., nl - number of dimensions for parametrized Lie group
    #   l - number of parameters required to parametrize Lie group

    # Assumption: Both time series are of the same type
    function MMDu(P1, P2, level)

        s1 = size(P1)[1]
        s2 = size(P2)[1]
        T = size(P2)[2]

        PP1, ccp, IT, N = PreprocessBatchPath(P1)
        PP2, ~, ~, ~ = PreprocessBatchPath(P2)

        # Compute MMD
        M = 0.0

        for i = 1:s1-1
            for j = i+1:s1
                M += IIKernel(PP1[i,ccp...],PP1[j,ccp...], IT, N, level)*2/(s1*(s1-1))
            end
        end

        for i = 1:s2-1
            for j = i+1:s2
                M += IIKernel(PP2[i,ccp...],PP2[j,ccp...], IT, N, level)*2/(s2*(s2-1))
            end
        end

        for i = 1:s1
            for j = 1:s2
                M -= IIKernel(PP1[i,ccp...],PP2[j,ccp...], IT, N, level)*2/(s1*s2)
            end
        end
        return M
    end

    # Gram matrix
    function GramMatrix(X, Y, level)

        nS1 = size(X)[1]
        nS2 = size(Y)[1]
        # T = size(P)[2]
        # PP, ccp, IT, N = PreprocessBatchPath(P)
        XX, IT, N = PreprocessBatchPath(X)
        YY, IT, N = PreprocessBatchPath(Y)

        K = zeros(nS1, nS2)

        for i = 1:nS1
            for j = 1:nS2
                # K[i,j] = IIKernel(PP[i,ccp...], PP[j,ccp...], IT, N, level)
                K[i,j] = IIKernel(XX[i], YY[j], IT, N, level)

                # if i !=j
                #     K[j,i] = K[i,j]
                # end
            end
        end

        return K

    end


    # Inputs
    # P1 - path as an (n1 x d) array
    # P2 - path as an (n2 x d) array
    # level - level to compute up to
    function PreprocessPath(P)
        Pnew, IT, N, ~ = preprocess(P)
        return Pnew, IT, N
    end

    # Preprocess a batch of paths and returns the preprocessed paths and the colon indexer
    function PreprocessBatchPath(P)

        # Preprocess paths
        S = size(P)[1]

        # Number of timepoints
        # T = size(P)[2]

        # Get number of dimensions of path
        l = ndims(P) - 1 # subtract 1 because of batch parameter

        # # Create colon object to index the paths
        # cc = Array{Colon,1}(undef, l)

        # Process one path to figure out size and get iterator
        # prep, IT, N, T = preprocess(P[1,cc...])
        prep, IT, N, T = preprocess(P[1])

        D = ndims(prep)

        # # Find size of preprocessed path
        # sp = size(prep)

        # # Find number of dimensions of preprocessed path and create colon object
        # lp = ndims(prep) # don't need to subtract 1 because we only preprocessed a single path
        # ccp = Array{Colon,1}(undef, lp)

        # Initialize arrays for preprocessed paths
        # PP = zeros(S, sp...)
        PP = Array{Array{Float64, D}, 1}(undef, S)

        # Preprocess all paths
        for i = 1:S
            # PP[i,ccp...], ~, ~, ~ = preprocess(P[i,cc...])
            PP[i], ~, ~, ~ = preprocess(P[i])
        end

        # return PP, ccp, IT, N
        return PP, IT, N
    end


    # return IISignature, IISignatureL, IIKernel
    return IISignatureL, IIKernel, IIKernel_PP, PreprocessPath, MMDu, GramMatrix
end

###############################################################################
# Description: Initializes IISignature functions for R^N.
#
function initializeII_R()


    function preprocess_R(P)
        T, d = size(P)
        return P, 1:d, d, T
    end

    function cochains_R(P, i)
        return P[2:end, i] - P[1:end-1, i]
    end


    # IIsig, IIsigL, IIKernel = IteratedIntegral(cochains_R, preprocess)
    IISignatureL, IIKernel, IIKernel_PP, PreprocessPath, MMDu, GramMatrix = IteratedIntegral(cochains_R, preprocess_R)

    # return IIsig, IIsigL, IIKernel
    return IISignatureL, IIKernel, IIKernel_PP, PreprocessPath, MMDu, GramMatrix
end


###############################################################################
# Description: Initializes IISignature functions for SO(m)^d.
# Note: The input to these cochains will be of size (T, m, m, d) where
#   n - parameter of SO(n)
#   d - number of copies of SO(n)
#   T - number of time points
#
# We have an additional dimension to include the inverse of matrices to make
# computation faster. In the fourth dimension, a value of 1 means the original
# matrix, and a value of 2 means the inverse matrix
#
# The idea here is to compute the Maurer-Cartan form by A^{-1}dA, and then looking
# at the forms corresponding to a basis of the Lie algebra so(n), which we choose to
# be the individual entries on the upper triangular matrix.
#
function initializeII_SO()

    function preprocess_SO(P)
        if ndims(P) == 3
            T, m, ~ = size(P)
            d = 1
            n = div((m^2-m),2)
            N = n

            Pnew = zeros(T, m, m, 2)
            Pnew[:,:,:,1] = P

            for t = 1:T-1
                # Pnew[t,:,:,2] = inv(P[t,:,:])
                Pnew[t,:,:,2] = real.(log(transpose(P[t,:,:])*P[t+1,:,:]))
            end

            IT = Array{Tuple{Int, Int}, 1}(undef, N)
            cc = 1
            for i = 1:m-1
                for j = i+1:m
                    IT[cc] = (i, j)
                    cc += 1
                end
            end

        elseif ndims(P) == 4
            T, m, ~, d = size(P)
            n = div((m^2-m),2)
            N = n*d

            Pnew = zeros(T, m, m, d, 2)
            Pnew[:,:,:,:,1] = P

            for t = 1:T-1
                for i = 1:d
                    # Pnew[t,:,:,i,2] = inv(P[t,:,:,i])
                    Pnew[t,:,:,i,2] = real.(log(transpose(P[t,:,:,i])*P[t+1,:,:,i]))
                end
            end

            IT = Array{Tuple{Int, Int, Int}, 1}(undef, N)
            cc = 1
            for k =1:d
                for i = 1:m-1
                    for j = i+1:m
                        IT[cc] = (i, j, k)
                        cc += 1
                    end
                end
            end

        else
            error("Incorrect input")
        end

        return Pnew, IT, N, T

    end


    function cochains_SO(P, i, j, k)
        ev = P[1:end-1,i,j,k,2]
        # ev = dropdims(sum(P[1:end-1,i,:,k,2].*(P[2:end,:,j,k,1] - P[1:end-1,:,j,k,1]), dims=2), dims=2)
        return ev
    end

    function cochains_SO(P, i, j)
        ev = P[1:end-1,i,j,2]
        # ev = dropdims(sum(P[1:end-1,i,:,2].*(P[2:end,:,j,1] - P[1:end-1,:,j,1]), dims=2), dims=2)
        return ev
    end


    IISignatureL, IIKernel, IIKernel_PP, PreprocessPath, MMDu, GramMatrix = IteratedIntegral(cochains_SO,preprocess_SO)

    return IISignatureL, IIKernel, IIKernel_PP, PreprocessPath, MMDu, GramMatrix
end


###############################################################################
# Description: Multiplies two elements of the tensor algebra up to given level
#

function TAMult(S, T, N, level)

    # Check if input tensors have required length
    if length(S) < level || length(T) < level
        error("Input tensors not defined up to specified level.")
    end

    # Check if tensors are the correct size
    for i = 1:level
        if any(size(S[i]).!=N) || any(size(T[i]).!=N)
            error("Input tensors are not the correct size.")
        end
    end

    TUP1 = Tuple(collect(1:level))
    TUP2 = Tuple(collect(level+1:2level))

    # Initialize output tensor
    M = Array{Array{Float64},1}(undef, level)
    for l = 1:level
        # M[l] = zeros((ones(Int,l)*N)...)
        M[l] = S[l] + T[l]
    end

    for l1 = 1:level
        for l2 = 1:level-l1
            M[l1+l2] += tensorproduct(S[l1], TUP1[1:l1], T[l2], TUP2[1:l2])
        end
    end

    return M

end


############################################################################
# Description: Tensor normalization
function PHI(x, M::Float64, a::Float64)

    if x^2 <= M
        return x^2
    else
        return M + (M^(1+a))*(M^(-a) - x^(-2a))/a
    end
end

