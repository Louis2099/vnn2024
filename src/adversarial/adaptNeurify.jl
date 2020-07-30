"""
    AdaptNeurify(max_iter::Int64, tree_search::Symbol)

AdaptNeurify combines symbolic reachability analysis with constraint refinement to minimize over-approximation of the reachable set.

# Problem requirement
1. Network: any depth, ReLU activation
2. Input: AbstractPolytope
3. Output: AbstractPolytope

# Return
`CounterExampleResult` or `ReachabilityResult`

# Method
Symbolic reachability analysis and iterative interval refinement (search).
- `max_iter` default `10`.

# Property
Sound but not complete.

# Reference
[S. Wang, K. Pei, J. Whitehouse, J. Yang, and S. Jana,
"Efficient Formal Safety Analysis of Neural Networks,"
*CoRR*, vol. abs/1809.08098, 2018. arXiv: 1809.08098.](https://arxiv.org/pdf/1809.08098.pdf)
[https://github.com/tcwangshiqi-columbia/Neurify](https://github.com/tcwangshiqi-columbia/Neurify)
"""

@with_kw struct AdaptNeurify
    max_iter::Int64     = 100
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.
end

function dfs_check(solver, problem, branches::Tree, x::Int)
    (last_reach, max_violation_con, splits) = branches.data[x]
    reach = forward_layer(solver, problem.network.layers[end], last_reach)
    result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network)
    result.status == :unknown || return result, []
    unknown_leaves = []
    size(branches.children[x])[1] == 0 && return BasicResult(:unknown), [x]

    for c in branches.children[x]
        c_result, sub_unknown_leaves  = dfs_check(solver, problem, branches, c)
        c_result.status == :violated && return c_result, []
        unknown_leaves = [unknown_leaves; sub_unknown_leaves]
    end
    unknown_leaves == [] && return BasicResult(:holds), unknown_leaves
    return BasicResult(:unknown), [unknown_leaves; x]
end

function dfs_split(solver, problem, branches::Tree, x::Int)
    (last_reach, max_violation_con, splits) = branches.data[x]
    reach = forward_layer(solver, problem.network.layers[end], last_reach)
    result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network)
    result.status == :unknown || return result

    if tree_size(branches) >= solver.max_iter
        return BasicResult(:unknown)
    end
    
    reach = forward_layer(solver, problem.network.layers[end], last_reach)
    intervals = constraint_refinement!(solver, problem.network, reach, max_violation_con, splits)
    for interval in intervals
        isempty(interval) && continue
        reach, last_reach = forward_network(solver, problem.network, interval, true)
        result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network)
        add_child!(branches, x, (last_reach, max_violation_con, copy(splits)))
    end
    for c in branches.children[x]
        result = dfs_split(solver, problem, branches, c)
        result.status == :holds || return result # if status == :unknown, means splitting number exceeds max_iter, return unkown directly.
    end
    return BasicResult(:holds)
end

function solve(solver::AdaptNeurify, problem::Problem, branches = nothing)
    problem = Problem(problem.network, convert(HPolytope, problem.input), convert(HPolytope, problem.output))
    reach_lc = problem.input.constraints
    output_lc = problem.output.constraints
    n = size(reach_lc, 1)
    m = size(reach_lc[1].a, 1)
    model = Model(GLPK.Optimizer)
    @variable(model, x[1:m], base_name="x")
    @constraint(model, [i in 1:n], reach_lc[i].a' * x <= reach_lc[i].b)

    reach, last_reach = forward_network(solver, problem.network, problem.input, true)

    result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network) # This calls the check_inclusion function in ReluVal, because the constraints are Hyperrectangle
    result.status == :unknown || return result, branches

    if branches === nothing
        branches = Tree((last_reach,max_violation_con, Vector()))
    end

    # check all existing branches, find the leaves whose status is unknown
    result, unknown_leaves = dfs_check(solver, problem, branches, 1)

    result.status == :unknown || return result, branches

    for leaf in unknown_leaves
        result = dfs_split(solver, problem, branches, leaf)
        result.status == :holds || return result, branches
    end

    return BasicResult(:holds), branches
end

function adapt_solve(solver::AdaptNeurify, problem::Problem, branches, param_prediction)
    
    problem = Problem(problem.network, convert(HPolytope, problem.input), convert(HPolytope, problem.output))
    reach_lc = problem.input.constraints
    output_lc = problem.output.constraints
    n = size(reach_lc, 1)
    m = size(reach_lc[1].a, 1)
    model = Model(GLPK.Optimizer)
    @variable(model, x[1:m], base_name="x")
    @constraint(model, [i in 1:n], reach_lc[i].a' * x <= reach_lc[i].b)

    reach, last_reach = forward_network(solver, problem.network, problem.input, true)

    result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network) # This calls the check_inclusion function in ReluVal, because the constraints are Hyperrectangle
    result.status == :unknown || return result, branches

    if branches === nothing
        branches = Tree((last_reach,max_violation_con, Vector()))
    end

    # check all existing branches, find the leaves whose status is unknown
    result, unknown_leaves = dfs_check(solver, problem, branches, 1)

    result.status == :unknown || return result, branches

    for leaf in unknown_leaves
        result = dfs_split(solver, problem, branches, leaf)
        result.status == :holds || return result, branches
    end

    branch_management!(solver, problem, branches, param_prediction)

    return BasicResult(:holds), branches
end

function branch_management!(solver::AdaptNeurify, problem::Problem, branches::Tree, param_prediction)
    nodes = Vector{Vecotr}()
    for leaf in branches.leaves
        push!(nodes, nearest_nodes_to_constraints(solver, problem))
    end
    
end

function nearest_nodes_to_constraints(solver::AdaptNeurify, problem::Problem)
    
end


function constraint_refinement!(solver::AdaptNeurify, nnet::Network, reach::SymbolicIntervalGradient, max_violation_con::AbstractVector{Float64}, splits::Vector)
    i, j, influence = get_nodewise_influence(nnet, reach, max_violation_con, splits)
    push!(splits, (i, j, influence))
    # We can generate three more constraints
    # Symbolic representation of node i j is Low[i][j,:] and Up[i][j,:]
    nnet_new = Network(nnet.layers[1:i])
    reach_new = forward_network(solver, nnet_new, reach.sym.interval)
    C, d = tosimplehrep(reach.sym.interval)
    l_sym = reach_new.sym.Low[[j], 1:end-1]
    l_off = reach_new.sym.Low[[j], end]
    u_sym = reach_new.sym.Up[[j], 1:end-1]
    u_off = reach_new.sym.Up[[j], end]
    intervals = Vector(undef, 3)
    # remove zero constraints and construct new intervals
    intervals[1] = construct_interval([C; l_sym; u_sym], [d; -l_off; -u_off])
    intervals[2] = construct_interval([C; l_sym; -u_sym], [d; -l_off; u_off])
    intervals[3] = construct_interval([C; -l_sym; -u_sym], [d; l_off; u_off])
    # intervals[4] = HPolytope([C; -l_sym; u_sym], [d; l_off; -u_off]) lower bound can not be greater than upper bound
    return intervals
end

function forward_network(solver, nnet::Network, input::AbstractPolytope, return_last::Bool)
    reach = input
    last_reach = input
    for (i, layer) in enumerate(nnet.layers)
        if i == length(nnet.layers)
            last_reach = deepcopy(reach)
        end
        reach = forward_layer(solver, layer, reach)
    end
    return_last && (return reach, last_reach)
    return reach
end

function forward_layer(solver::AdaptNeurify, layer::Layer, input)
    return forward_act(forward_linear(solver, input, layer), layer)
end

# Symbolic forward_linear for the first layer
function forward_linear(solver::AdaptNeurify, input::AbstractPolytope, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    LΛ = Vector{Vector{Int64}}(undef, 0)
    UΛ = Vector{Vector{Int64}}(undef, 0)
    r = Vector{Vector{Int64}}(undef, 0)
    return SymbolicIntervalGradient(sym, LΛ, UΛ, r)
end

# Symbolic forward_linear
function forward_linear(solver::AdaptNeurify, input::SymbolicIntervalGradient, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    output_Up = max.(W, 0) * input.sym.Up + min.(W, 0) * input.sym.Low
    output_Low = max.(W, 0) * input.sym.Low + min.(W, 0) * input.sym.Up
    output_Up[:, end] += b
    output_Low[:, end] += b
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ, input.r)
end