"""
    IntervalNet(max_iter::Int64, tree_search::Symbol)

IntervalNet combines symbolic reachability analysis with constraint refinement to
minimize over-approximation of the reachable set.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: AbstractPolytope
3. Output: LazySet

# Return
`CounterExampleResult`

# Method
Symbolic reachability analysis and iterative interval refinement (search).
- `max_iter` default `100`.
- `tree_search` default `:DFS` - depth first search.
- `optimizer` default `GLPK.Optimizer`

# Property
Sound but not complete.

# Reference
[S. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana,
"Efficient Formal Safety Analysis of Neural Networks,"
*CoRR*, vol. abs/1809.08098, 2018. arXiv: 1809.08098.](https://arxiv.org/pdf/1809.08098.pdf)
[https://github.com/tcwangshiqi-columbia/IntervalNet](https://github.com/tcwangshiqi-columbia/IntervalNet)
"""

@with_kw struct IntervalNet <: Solver
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS
    optimizer = GLPK.Optimizer
    delta::Tuple{Float64, Float64} =(0,0)
    enlarge = 0
end


function solve(solver::IntervalNet, problem::Problem)
    isbounded(problem.input) || throw(UnboundedInputError("IntervalNet can only handle bounded input sets."))

    # Because of over-approximation, a split may not bisect the input set.
    # Therefore, the gradient remains unchanged (since input didn't change).
    # And this node will be chosen to split forever.
    # To prevent this, we split each node only once if the gradient of this node hasn't changed.
    # Each element in splits is a tuple (layer_index, node_index, node_gradient).

    nnet, output = problem.network, problem.output
    reach_list = []
    domain = init_symbolic_grad(problem.input)
    splits = Set()
    for i in 1:solver.max_iter
        if i > 1
            domain, splits = select!(reach_list, solver.tree_search)
        end

        reach = forward_network(solver, nnet, domain)
        result, max_violation_con = check_inclusion(solver, nnet, last(reach).sym, output)

        if result.status === :violated
            return result
        elseif result.status === :unknown
            subdomains = constraint_refinement(nnet, reach, max_violation_con, splits)
            for domain in subdomains
                push!(reach_list, (init_symbolic_grad(domain), copy(splits)))
            end

        end
        isempty(reach_list) && return CounterExampleResult(:holds)
    end
    return CounterExampleResult(:unknown)
end

function check_inclusion(solver::IntervalNet, nnet::Network,
                         reach::SymbolicInterval, output)
    # The output constraint is in the form A*x < b
    # We try to maximize output constraint to find a violated case, or to verify the inclusion.
    # Suppose the output is [1, 0, -1] * x < 2, Then we are maximizing reach.Up[1] * 1 + reach.Low[3] * (-1)

    input_domain = domain(reach)

    model = Model(solver); set_silent(model)
    x = @variable(model, [1:dim(input_domain)])
    add_set_constraint!(model, input_domain, x)

    max_violation = 0.0
    max_violation_con = nothing
    for (i, cons) in enumerate(constraints_list(output))
        # NOTE can be taken out of the loop, but maybe there's no advantage
        # NOTE max.(M, 0) * U  + ... is a common operation, and maybe should get a name. It's also an "interval map".
        a, b = cons.a, cons.b
        c = max.(a, 0)'*reach.Up + min.(a, 0)'*reach.Low

        @objective(model, Max, c * [x; 1] - b)
        optimize!(model)

        if termination_status(model) == OPTIMAL
            if compute_output(nnet, value(x)) ∉ output
                return CounterExampleResult(:violated, value(x)), nothing
            end

            viol = objective_value(model)
            if viol > max_violation
                max_violation = viol
                max_violation_con = a
            end

        else
            # TODO can we be more descriptive?
            error("No solution, please check the problem definition.")
        end

    end

    if max_violation > 0.0
        return CounterExampleResult(:unknown), max_violation_con
    else
        return CounterExampleResult(:holds), nothing
    end
end


function forward_network(solver::IntervalNet, network::Network, input)
    forward_network(solver, network, init_symbolic_grad(input))
end

function forward_network(solver::IntervalNet, network::Network, input::SymbolicIntervalGradient)
    reachable = [input = forward_layer(solver, L, input) for L in network.layers]
    return reachable
end

function forward_layer(solver::IntervalNet, layer::Layer, input)
    return forward_act(solver, forward_linear(solver, input, layer), layer)
end

# Symbolic forward_linear
function forward_linear(solver::IntervalNet, input::SymbolicIntervalGradient, layer::Layer)
    output_Low, output_Up = interval_map(layer.weights, input.sym.Low, input.sym.Up, solver.delta[1])
    output_Up[:, end] += layer.bias .+ solver.delta[2]
    output_Low[:, end] += layer.bias .- solver.delta[2]
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ)
end

function interval_map(W::AbstractMatrix{N}, l::AbstractVecOrMat, u::AbstractVecOrMat, dW) where N
    l_new = max.(W .+ dW, zero(N)) * l + min.(W .- dW, zero(N)) * u
    u_new = max.(W .+ dW, zero(N)) * u + min.(W .- dW, zero(N)) * l
    return (l_new, u_new)
end

# Symbolic forward_act
function forward_act(solver::IntervalNet, input::SymbolicIntervalGradient, layer::Layer{ReLU})
    n_node = n_nodes(layer)
    output_Low, output_Up = copy(input.sym.Low), copy(input.sym.Up)
    LΛᵢ, UΛᵢ = zeros(n_node), ones(n_node)
    # Symbolic linear relaxation
    # This is different from ReluVal
    for j in 1:n_node
        up_low, up_up = bounds(upper(input), j)
        low_low, low_up = bounds(lower(input), j)

        up_slope = relaxed_relu_gradient(up_low, up_up)
        low_slope = relaxed_relu_gradient(low_low, low_up)

        output_Up[j, :] .*= up_slope
        output_Up[j, end] += up_slope * max(-up_low, 0)

        output_Low[j, :] .*= low_slope

        LΛᵢ[j], UΛᵢ[j] = low_slope, up_slope
    end
    sym = SymbolicInterval(output_Low, output_Up, domain(input))
    LΛ = push!(input.LΛ, LΛᵢ)
    UΛ = push!(input.UΛ, UΛᵢ)
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

function forward_act(solver::IntervalNet, input::SymbolicIntervalGradient, layer::Layer{Id})
    n_node = n_nodes(layer)
    LΛ = push!(input.LΛ, ones(n_node))
    UΛ = push!(input.UΛ, ones(n_node))
    return SymbolicIntervalGradient(input.sym, LΛ, UΛ)
end
