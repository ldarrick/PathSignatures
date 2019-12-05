
using Combinatorics
using LinearAlgebra
# using DSP

###############################################################################
# Description: IteratedIntegral() is a function that takes in an array of
# cochains (an array of functions), and returns two functions:
#       IISignature() - computes the signature for a specific term (ie. [1,2,2])
#       IISignatureL() - computes the signature up to a given level
#
# Input: cochains should be an array of functions that takes in a path and evaluates
#   the cochain on the path - if a path of length N is inputted, the output is a path
#   of length N-1
# 
#   psize() is a function that returns (n_samples, n_features)
#
# Note: Inputs to the cochains should be type restricted
function IteratedIntegral(cochains, psize)
    N = length(cochains)

    # Inputs:
    # P - a path as an (n x d) array; (n - number of points, d - dimension)
    # term - the term (multi-index) of the signature to calculate (ie. [1,2,2])
    #
    # Outputs:
    # sig - a vector of signature terms used to calculate
    #   ie. for term = [1,2,2], sig = [S^1, S^{1,2}, S^{1,2,2}]
    # sigpath - returns the signature path S^{term}(t) (used for recursion)
    function IISignature(P, term)
        n = size(P,1)
        level = length(term)

        sigpath = zeros(n-1,1)
        eval_cochain = cochains[term[end]](P)
        # dP_end = eval_cc[2:end] - eval_cc[1:end-1]

        if level == 1
            integrand = eval_cochain
            sig = zeros(0)
        else
            sig, last_sigpath = IISignature(P, term[1:end-1])
            integrand = last_sigpath.*eval_cochain
        end

        sigpath = cumsum(integrand)
        sig = push!(sig, sigpath[end])

        return sig, sigpath
    end

    function sig_forward(sigl, lastQ, K, cur_ind, cur_level, last_level)

        if cur_level < last_level

            for i = 1:N-1
                cur_ind[cur_level] = i
                Q = cumsum(lastQ .* view(K,:,i))
                sigl[cur_level][cur_ind[1:cur_level]...] = Q[end]

                sig_forward(sigl, Q, K, cur_ind, cur_level+1, last_level)
            end

            # On the last run through, we no longer need the information from
            # lastQ, so just use that variable instead of allocating more memory
            cur_ind[cur_level] = N
            cumsum!(lastQ, lastQ .* view(K,:,N))
            sigl[cur_level][cur_ind[1:cur_level]...] = lastQ[end]

            sig_forward(sigl, lastQ, K, cur_ind, cur_level+1, last_level)
        else

            for i = 1:N
                cur_ind[cur_level] = i
                sigl[cur_level][cur_ind...] = sum(lastQ .* view(K,:,i))
            end

        end
    end


    function IISignatureL(P, level)

        n, ~ = psize(P)

        sigl = Array{Array{Float64},1}(undef, level)
        for i = 1:level
            sigl[i] = Array{Float64, i}(undef, (ones(Int,i)*N)...)
        end

        K = zeros(n-1, N)
        for i = 1:N
            K[:,i] = cochains[i](P)
        end

        # K = P[2:end,:] - P[1:end-1,:]

        for i = 1:N
            cur_ind = zeros(Int, level)
            cur_ind[1] = i
            Q = cumsum(view(K,:,i))
            sigl[1][i] = Q[end]

            sig_forward(sigl, Q, K, cur_ind, 2, level)
        end

        return sigl

    end

    # Inputs
    # P1 - path as an (n1 x d) array
    # P2 - path as an (n2 x d) array
    # level - level to compute up to
    function IIKernel(P1, P2, level)

        n1, d = psize(P1)
        n2, d = psize(P2)

        K1 = zeros(n1-1, d)
        K2 = zeros(n2-1, d)

        for i = 1:d
            K1[:,i] = cochains[i](P1)
            K2[:,i] = cochains[i](P2)
        end

        K = K1*K2'

        A = zeros(level, n1-1, n2-1)
        A[1,:,:] = K

        for i = 2:level
            Q = zeros(n1,n2)
            Q[2:end,2:end] = cumsum(cumsum(A[i-1,:,:],dims=2),dims=1)
            A[i,:,:] = K.*(1 .+ Q[2:end,2:end])
        end

        R = 1 + first(sum(sum(A[level,:,:],dims=2),dims=1))

        return R

    end

    return IISignature, IISignatureL, IIKernel
end

###############################################################################
# Description: Initializes IISignature functions for R^N.
#
function initializeII_R(N)

    cochains = Array{Function,1}(undef,N)

    for i = 1:N
        cochains[i] = P::Array{Float64,2} -> (P[2:end,i] - P[1:end-1,i])
    end

    psize = P::Array{Float64,2} -> (length(P[:,1]), length(P[1,:]))

    IIsig, IIsigL, IIKernel = IteratedIntegral(cochains, psize)

    return IIsig, IIsigL, IIKernel
end

