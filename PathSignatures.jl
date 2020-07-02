
using Combinatorics
using LinearAlgebra
using TensorOperations
using Roots

###############################################################################

# COMMON PARAMETER NAMES
# T+1 = number of timepoints (length of time series), so that derivative is length T
# N = total dimension of space
# n = dimension of a single copy of a space
# b = parameter for Lie group (ie. SO(b))
# d = number of copies of a space (is. SO(b)^d)
# M = truncation level

# COMMON VARIABLE NAMES
# P = time series (path)
# BP = batch of time series (array of arrays)
# p = discrete derivative of time series
# bp = batch of discrete derivatives

###############################################################################

# TIME SERIES FORMATTING
# Ordering:
#	1) Time parameter
#	2) Lie group parametrization
#	3) Copies of space
# R^N : (T, N) array
# SO(b)^d : (T, b, b, d) array (here, N = dim G = db(b-1)/2)

###############################################################################



###############################################################################
## HELPER FUNCTIONS ###########################################################
###############################################################################


###############################################################################
# Function: initialize_TA 
# Description: Initializes array for truncated tensor algebra element.
# Input 
# 	N: dimension of underlying vector space
#	M: truncation level
# Output 
# 	A: length M array of arrays, the j^th array is a j dimensional array
#	   of length N
###############################################################################
function initialize_TA(N, M)

	A = Array{Array{Float64},1}(undef, M)
	for i = 1:M
	    # A[i] = Array{Float64, i}(undef, (ones(Int,i)*N)...)
	    A[i] = zeros((ones(Int,i)*N)...)
	end
	return A

end


###############################################################################
# Function: multiply_TA!
# Description: Multiplies two truncated tensor algebra elements, returning
#	S \otimes T.
# Input 
# 	A: variable to hold output of multiplication
#	S, T: truncated tensor algebra elements to multiply
#	M: truncation level
###############################################################################
function multiply_TA!(A, S, T, M)

    # Check if input tensors have required length
    if length(S) < M || length(T) < M
        error("Input tensors not defined up to specified level.")
    end

    N = length(S[1])

    # Check if tensors are the correct size
    for i = 1:M
        if any(size(S[i]).!=N) || any(size(T[i]).!=N)
            error("Input tensors are not the correct size.")
        end
    end

    TUP1 = Tuple(collect(1:M))
    TUP2 = Tuple(collect(M+1:2M))

    # Initialize output tensor
    # A = initialize_TA(N,level)
    for l = 1:M
        # M[l] = zeros((ones(Int,l)*N)...)
        A[l] = S[l] + T[l]
    end

    for l1 = 1:M
        for l2 = 1:M-l1
            A[l1+l2] += tensorproduct(S[l1], TUP1[1:l1], T[l2], TUP2[1:l2])
        end
    end

end


###############################################################################
# Function: tensor_exp!
# Description: Computes truncated tensor exponential
# Input 
# 	A: variable to hold output of multiplication
#	v: vector to exponentiate
#	M: truncation level
###############################################################################
function tensor_exp!(A, v, M)

	# Check if T is the right size
	if length(A) != M
		error("A is the wrong size")
	end

	N = length(v)

	for i = 1:M
		if any(size(A[i]).!=N)
			println(i)
			error("A is the wrong size")
		end
	end

	TUP = Tuple(collect(1:M))

	A[1] = v
	for m = 2:M
        A[m] = tensorproduct(A[m-1], TUP[1:m-1], v, TUP[m:m])/(factorial(m))
	end

end


###############################################################################
# Function: PHI
# Description: Function used in tensor normalization
# Input 
# 	x: argument for function
#	M, a: parameters of function
###############################################################################
function PHI(x, M::Float64, a::Float64)

    if x^2 <= M
        return x^2
    else
        return M + (M^(1+a))*(M^(-a) - x^(-2a))/a
    end
end


###############################################################################
## MAIN FUNCTIONS #############################################################
###############################################################################

###############################################################################
# Function: discrete_derivative 
# Description: Computes the discrete derivative of time series.
# Input 
# 	P: time series
# 		(T+1, N) array if dtype = "R"
# 		(T+1, b, b) or T(b, b, d) array if dtype = "SO"
# 	dtype: type of time series, currently implemented:
# 		"R": real valued time series
# 		"SO": SO(b)^d valued time series (possibly with d = 1)
# Output 
# 	p: discrete derivative of time series (T, N) array for all dtype
###############################################################################
function discrete_derivative(P, dtype)

	# Real time series
	if dtype == "R"
		return P[2:end, :] - P[1:end-1, :]

	# SO(b)^d valued time series
	elseif dtype == "SO"
		# If d=1, then the input is a size (T, b, b) array
		if ndims(P) == 3
		    T, b, ~ = size(P)
		    n = div((b^2-b),2)
		    N = n

		    Pderiv = zeros(T-1, b, b)
		    p = zeros(T-1, N)

		    for t = 1:T-1
		        Pderiv[t,:,:] = real.(log(transpose(P[t,:,:])*P[t+1,:,:]))
		    end

		    dcount = 1
		    for i = 1:b-1
		    	for j = i+1:b
		    		p[:,dcount] = Pderiv[:,i,j]
		    		dcount += 1
		    	end
		    end
		# If d > 1, then the input is a size (T, b, b, d) array
		elseif ndims(P) == 4
		    T, b, ~, d = size(P)
		    n = div((b^2-b),2)
		    N = n*d

		    Pderiv = zeros(T-1, b, b, d)
		    p = zeros(T-1, N)

		    for t = 1:T-1
		        for i = 1:d
		            Pderiv[t,:,:,i] = real.(log(transpose(P[t,:,:,i])*P[t+1,:,:,i]))
		        end
		    end

		    dcount = 1
		    for k = 1:d
			    for i = 1:b-1
			    	for j = i+1:b
			    		p[:,dcount] = Pderiv[:,i,j,k]
			    		dcount += 1
			    	end
			    end
			end

		else
		    error("Incorrect input. For SO(b)^d valued time series, input must be size 
		    	(T, b, b) or (T, b, b, d).")
		end
	end

	return p
end


#############################################################################
# Function: batch_discrete_derivative 
# Description: Batch discrete derivative computation
# Input 
# 	BP: batch of time series (array of arrays of time series)
# 	dtype: type of time series
# Output 
# 	bp: batch discrete derivative
#############################################################################
function batch_discrete_derivative(BP, dtype)

	S = size(BP)[1] # size of batch
	CC = Array{Colon,1}(undef, ndims(BP)-1)

	# Initialize the array for the collection of disrete derivatives
	bp = Array{Array{Float64, 2}, 1}(undef, S)

	for i = 1:S
		bp[i] = discrete_derivative(BP[i,CC...], dtype)
	end

	return bp
end


###############################################################################
# Function: signature 
# Description: Computes the continuous signature of interpolated time series.
# Input 
#	P: time series
#		(T+1, N) array if dtype = "R"
#		(T+1, b, b) or T(b, b, d) array if dtype = "SO"
#	M: truncation level 
#	dtype: type of time series, currently implemented:
#		"R": real valued time series
#		"SO": SO(b)^d valued time series (possibly with d = 1)
# Output 
#	S: truncated continuous signature of time series
###############################################################################
function signature(P, M, dtype)

	# Compute discrete derivative
	p = discrete_derivative(P, dtype)

	# Get size of path signature
	T, N = size(p)

	# Initialize tensor algebra
	sig1 = initialize_TA(N, M)
	sig2 = initialize_TA(N, M)
	# lastsig = initialize_TA(N,M)
	cur_exp = initialize_TA(N, M) # variable for the current tensor exponent

	# Initialize first time segment
	tensor_exp!(sig1,p[1,:], M)

	for t = 2:T
		tensor_exp!(cur_exp, p[t,:], M)
		if mod(t, 2) == 0
			multiply_TA!(sig2, sig1, cur_exp, M)
		else
			multiply_TA!(sig1, sig2, cur_exp, M)
		end
	end

	if mod(T,2) == 0
		return sig2
	else
		return sig1
	end

end


###############################################################################
# Function: dsignature 
# Description: Computes the discrete approximation of the path signature.
# Input 
#	P: time series
#		(T+1, N) array if dtype = "R"
#		(T+1, b, b) or T(b, b, d) array if dtype = "SO"
#	M: truncation level 
#	dtype: type of time series, currently implemented:
#		"R": real valued time series
#		"SO": SO(b)^d valued time series (possibly with d = 1)
# Output 
#	S: discrete signature truncated at level m
###############################################################################
function dsignature(P, M, dtype)

	# Compute discrete derivative
	p = discrete_derivative(P, dtype)

	# Get size of path signature
	T, N = size(p)

	# Initialize the tensor algebra element as an array of arrays
	S = initialize_TA(N, M)

	for i = 1:N
	    cur_ind = zeros(Int, M)
	    cur_ind[1] = i
	    Q = cumsum(view(p,:,i))
	    S[1][i] = Q[end]

	    dsig_forward(S, Q, p, cur_ind, 2, M, N)
	end

	return S
end


###############################################################################
# Function: dsig_forward 
# Description: The forward recursion step in the discrete signature function.
# Input 
#	S: current signature 
#	lastQ: last signature path
#	p: discrete derivative
#	cur_ind: current signature index
#	cur_level: current level
#	last_level: truncation level
#	N: dimension of Lie group
###############################################################################
function dsig_forward(sigl, lastQ, p, cur_ind, cur_level, last_level, N)

    if cur_level < last_level

        for i = 1:N-1
            cur_ind[cur_level] = i
            Q = cumsum(lastQ .* view(p,:,i))
            sigl[cur_level][cur_ind[1:cur_level]...] = Q[end]

            dsig_forward(sigl, Q, p, cur_ind, cur_level+1, last_level, N)
        end

        # On the last run through, we no longer need the information from
        # lastQ, so just use that variable instead of allocating more memory
        cur_ind[cur_level] = N
        cumsum!(lastQ, lastQ .* view(p,:,N))
        sigl[cur_level][cur_ind[1:cur_level]...] = lastQ[end]

        dsig_forward(sigl, lastQ, p, cur_ind, cur_level+1, last_level, N)
    else

        for i = 1:N
            cur_ind[cur_level] = i
            sigl[cur_level][cur_ind...] = sum(lastQ .* view(p,:,i))
        end
    end
end


###############################################################################
# Function: dsignature_kernel 
# Description: Computes the normalized discrete signature kernel for two paths. 
# Input 
#	P1, P2: two time series
#	M: truncation level
#	dtype: type of time series
# Output 
#	K: kernel value
###############################################################################
function dsignature_kernel(P1, P2, M, dtype)

    p1 = discrete_derivative(P1, dtype)
    p2 = discrete_derivative(P2, dtype)

    return dsignature_kernel_preprocessed(p1, p2, M)
end


###############################################################################
# Function: dsignature_kernel_preprocessed
# Description: Computes the normalizeddiscrete signature kernel for two paths,
#	where the discrete derivative has been precomputed
# Input 
#	p1, p2: discrete derivatives for two time series
#	M: truncation level
# Output 
#	K: kernel value
###############################################################################
function dsignature_kernel_preprocessed(p1, p2, M)
    T1, N1 = size(p1)
    T2, N2 = size(p2)

    if N1 != N2
    	error("Time series dimensions are not equal.")
    else
    	N = N1
    end

    # Compute the tensor normalization
    coeff1 = zeros(M+1)
    coeff2 = zeros(M+1)

    # Compute normalization for P1
    K = p1*p1'
    A = deepcopy(K)
    coeff1[2] = sum(view(A,:))

    for i = 2:M   
        cumsum!(A,A,dims=1)
        cumsum!(A,A,dims=2)
        A =  @. K*(1 + A)
        coeff1[i+1] = sum(view(A,:))
    end
    norm1 = coeff1[end] + 1
    c1 = coeff1[2:end] - coeff1[1:end-1]
    f1(x) = sum(c1.*(x.^((1:M)*2))) + 1 - PHI(sqrt(norm1),4.,1.)
    lambda1 = find_zero(f1, (0., max(2.,norm1)))

    # Compute normalization for P2
    K = p2*p2'
    A = deepcopy(K)
    coeff2[2] = sum(view(A,:))

    for i = 2:M   
        cumsum!(A,A,dims=1)
        cumsum!(A,A,dims=2)
        A =  @. K*(1 + A)
        coeff2[i+1] = sum(view(A,:))
    end
    norm2 = coeff2[end] + 1
    c2 = coeff2[2:end] - coeff2[1:end-1]
    f2(x) = sum(c2.*(x.^((1:M)*2))) + 1 - PHI(sqrt(norm2),4.,1.)
    lambda2 = find_zero(f2, (0., max(2.,norm2)))

    # coefficient
    ll = lambda1*lambda2

    # Compute normalized kernel
    K = p1*p2'
    A = deepcopy(K)

    for i = 2:M   
        cumsum!(A,A,dims=1)
        cumsum!(A,A,dims=2)
        if i==2
            A = @. K*(ll^(M-1) + A*ll^M)
        else
            A = @. K*(ll^(M-i+1) + A)
        end
    end

    K = 1 + sum(view(A,:))

    return K
end


###############################################################################
# Function: dsignature_gram_matrix
# Description: Computes the gram matrix for two batches of time series, using
#	the normalized discrete signature kernel. 
# Input 
#	BP1, BP2: two batches of time series
#	M: truncation level
#	dtype: type of time series
# Output 
#	K: gram matrix
###############################################################################
function dsignature_gram_matrix(BP1, BP2, M, dtype)

	# Compute all discrete derivatives
    bp1 = batch_discrete_derivative(BP1, dtype)
    bp2 = batch_discrete_derivative(BP2, dtype)

    S1 = length(bp1)
    S2 = length(bp2)

    K = zeros(S1, S2)

    for i = 1:S1
    	for j = 1:S2
    		K[i,j] = dsignature_kernel_preprocessed(bp1[i], bp2[j], M)
    	end
    end

    return K
end


###############################################################################
# Function: dsignature_MMDu
# Description: Computes the unbiased maximum mean discrepancy (MMD).
# Input 
#	BP1, BP2: two batches of time series
#	M: truncation level
#	dtype: type of time series
# Output 
#	MMD_val: MMD value
###############################################################################
function dsignature_MMDu(BP1, BP2, M, dtype)

	# Compute all discrete derivatives
    bp1 = batch_discrete_derivative(BP1, dtype)
    bp2 = batch_discrete_derivative(BP2, dtype)

    S1 = length(bp1)
    S2 = length(bp2)

    # Compute MMD
    MMD_val1 = 0.0
    MMD_val2 = 0.0
    MMD_val3 = 0.0

    for i = 1:S1-1	
        for j = i+1:S1
        	MMD_val1 += dsignature_kernel_preprocessed(bp1[i], bp1[j], M)
        end
    end

    for i = 1:S2-1
        for j = i+1:S2
        	MMD_val2 += dsignature_kernel_preprocessed(bp2[i], bp2[j], M)
        end
    end

    for i = 1:S1
        for j = 1:S2
        	MMD_val3 += dsignature_kernel_preprocessed(bp1[i], bp2[j], M)
        end
    end

    MMD_val = MMD_val1*2/(S1*(S1-1)) + MMD_val2*2/(S2*(S2-1)) - MMD_val3*2/(S1*S2)

    return MMD_val
end
