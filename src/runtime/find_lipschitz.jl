# Julia implementation of https://github.com/arobey1/LipSDP/blob/master/LipSDP/matlab_engine/lipschitz_multi_layer.m

function find_lipschitz(network)
    weights = []
    net_dims = []
    L_up = 1
    for (i,layer) in enumerate(network.layers)
        L_up = L_up * norm(layer.weights)
        push!(weights, layer.weights)
        push!(net_dims, size(layer.weights, 2))
    end
    push!(net_dims, size(network.layers[end].weights, 1))

    alpha = 0
    beta = 1
    # model = Model(Mosek.Optimizer); 
    # model = Model(CPLEX.Optimizer); 
    model = Model(SCS.Optimizer); 
    set_silent(model)

    N = sum(net_dims[2:end-1])
    id = Matrix(1.0I, N, N)

    @variable(model, L_sq >= 0)

    @variable(model, D[1:N] >= 0)

    @variable(model, zeta[i=1:binomial(N,2)] >= 0)

    T = Diagonal(D)

    C = reduce(hcat, collect(combinations(1:N, 2)))

    E = id[:, C[1,:]] - id[:, C[2,:]]

    T = T + E * Matrix(Diagonal(zeta)) * E'

    # Create Q matrix, which is parameterized by T, which in turn depends
    # on the chosen LipSDP formulation 
    Q = [-2 * alpha * beta * T    (alpha + beta) * T 
            (alpha + beta) * T   -2 * T]


    weights = [sparse(w) for w in weights]
    # Create A term in Lipschitz formulation
    # first_weights = blkdiag(weights[1:end-1])
    first_weights = blockdiag(weights[1:end-1]...)

    zeros_col = zeros(size(first_weights, 1), size(weights[end], 2))
    A = hcat(first_weights, zeros_col)

    # Create B term in Lipschitz formulation
    eyes = Matrix(1.0I, size(A, 1), size(A, 1))
    init_col = zeros(size(eyes, 1), net_dims[1])
    B = hcat(init_col, eyes)

    # Stack A and B matrices
    A_on_B = vcat(A, B)

    # Create M matrix encoding Lipschitz constant
    weight_term = -1 * weights[end]' * weights[end]

    middle_zeros = sparse(zeros(sum(net_dims[2:end - 2]), sum(net_dims[2 : end - 2])))

    lower_right = blockdiag(middle_zeros, weight_term)
    upper_left = sparse(Matrix(1.0I, net_dims[1], net_dims[1]))
    u_dim = size(upper_left, 1)
    l_dim = size(upper_left, 2)
    M = blockdiag(upper_left, lower_right) * L_sq
    M[u_dim+1:end, l_dim+1:end] = lower_right
    
    @objective(model, Min, L_sq)
    
    @SDconstraint(model, A_on_B' * Q * A_on_B - M ⪯ 0)
    
    optimize!(model)
    
    L = sqrt(value(L_sq))
    
    return L
end