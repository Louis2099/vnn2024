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
    max_iter::Int64     = 10
    tree_search::Symbol = :DFS # only :DFS/:BFS allowed? If so, we should assert this.

    # Becuase of over-approximation, a split may not bisect the input set. 
    # Therefore, the gradient remains unchanged (since input didn't change).
    # And this node will be chosen to split forever.
    # To prevent this, we split each node only once if the gradient of this node doesn't change. 
    # Each element in splits is a tuple (gradient_of_the_node, layer_index, node_index).

    splits = Set() # To prevent infinity loop.

    # But in some cases (which I don't have an example, just a sense), 
    # it can happen that a node indeed needs to be split twice with the same gradient.
end

function solve(solver::AdaptNeurify, problem::Problem)
    while !isempty(solver.splits) pop!(solver.splits) end
    problem = Problem(problem.network, convert(HPolytope, problem.input), convert(HPolytope, problem.output))

    reach_lc = problem.input.constraints
    output_lc = problem.output.constraints

    n = size(reach_lc, 1)
    m = size(reach_lc[1].a, 1)
    model = Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:m], base_name="x")
    @constraint(model, [i in 1:n], reach_lc[i].a' * x <= reach_lc[i].b)
    
    reach = forward_network(solver, problem.network, problem.input)    
    result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network) # This calls the check_inclusion function in ReluVal, because the constraints are Hyperrectangle
    result.status == :unknown || return result

    reach_list = [(reach, max_violation_con)]

    for i in 2:solver.max_iter
        length(reach_list) > 0 || return BasicResult(:holds)
        reach, max_violation_con = pick_out!(reach_list, solver.tree_search)
        intervals = constraint_refinement(solver, problem.network, reach, max_violation_con)
        for interval in intervals
            reach = forward_network(solver, problem.network, interval)
            result, max_violation_con = check_inclusion(solver, reach.sym, problem.output, problem.network)
            result.status == :violated && return result
            result.status == :holds || (push!(reach_list, (reach, max_violation_con)))
        end
    end
    return BasicResult(:unknown)
end

function pick_out!(reach_list, tree_search, visited, order)
    n = length(reach_list)
    if tree_search == :BFS
        i = 1
        while i <= n && visited[i]
            i+=1
        end
    else
        i = length(reach_list)
        while i >= 1 && visited[i]
            i-=1
        end
    end
    if i < 1 || i > n
        return nothing, -1
    end
    reach = reach_list[i]
    visited[i] = true
    push!(order, i)
    return reach, i
end

function solve(solver::AdaptNeurify, problem::Problem, last_reach_list, last_children, last_order, follow_previous_tree) # assume the input range doesn't change, and only the last layer's weights change
    problem = Problem(problem.network, convert(HPolytope, problem.input), convert(HPolytope, problem.output))

    reach_lc = problem.input.constraints
    output_lc = problem.output.constraints

    n = size(reach_lc, 1)
    m = size(reach_lc[1].a, 1)
    model =Model(with_optimizer(GLPK.Optimizer))
    @variable(model, x[1:m], base_name="x")
    @constraint(model, [i in 1:n], reach_lc[i].a' * x <= reach_lc[i].b)
    # println("start forwarding")
    init_reach, init_last_reach = forward_network(solver, problem.network, problem.input, model, true)
    # println("start checking")
    result = check_inclusion(init_reach.sym, problem.output, problem.network, model) # This called the check_inclusion function in ReluVal, because the constraints are Hyperrectangle
    # println("finish checking")
    result.status == :unknown || return result, Tuple[], Dict(), [], 1 

    visited = falses(solver.max_iter*4) #if we visited n nodes, then there are at most 4*n nodes in the tree. because every node has 3 children.
    order = []
    children = Dict()

    if follow_previous_tree
        for i in 1:solver.max_iter
            if i <= length(last_order)
                idx = last_order[i]
                last_reach = last_reach_list[idx]
                visited[idx] = true
            else
                last_reach, idx = pick_out!(last_reach_list, solver.tree_search, visited, last_order)
            end
            idx == -1 && return BasicResult(:holds), last_reach_list, last_children, last_order, i
            # println("last layer size")
            # println(size(problem.network.layers))
            # println(size(problem.network.layers[end].weights))
            # println(size(last_reach.sym.Low))
            # println("---")
            reach = forward_layer(solver, problem.network.layers[end], last_reach, model)
            result = check_inclusion(reach.sym, problem.output, problem.network, model)
            result.status == :violated && return result, last_reach_list, last_children, last_order, i
            if result.status != :holds && !haskey(last_children, idx)
                last_children[idx] = []
                intervals = constraint_refinement(solver, problem.network, reach, model)
                violated_results = nothing
                for interval in intervals
                    reach, last_reach = forward_network(solver, problem.network, interval, model, true)
                    result = check_inclusion(reach.sym, problem.output, problem.network, model)
                    push!(last_reach_list, last_reach)
                    push!(last_children[idx], length(last_reach_list))
                    result.status == :violated && (violated_results = result)
                end
                violated_results == nothing || return violated_results, last_reach_list, last_children, last_order, i
            end
            # result.status == :holds || (push!(last_reach_list, reach)) # This is a bug, why would I do this??????
        end
        return BasicResult(:unknown), last_reach_list, last_children, last_order, solver.max_iter
    else
        reach_list = [(init_reach,:unknown)]
        last_reach_list = [init_last_reach]
        for i in 1:solver.max_iter
            (reach, status), idx = pick_out!(reach_list, solver.tree_search, visited, order)
            idx == -1 && return BasicResult(:holds), reach_list, children, order, i
            status == :holds && continue
            intervals = constraint_refinement(solver, problem.network, reach, model)
            violated_results = nothing
            children[idx] = []
            for interval in intervals
                reach, last_reach = forward_network(solver, problem.network, interval, model, true)
                result = check_inclusion(reach.sym, problem.output, problem.network, model)
                result.status == :violated && (violated_results = result) # return later, to make sure all the children are pushed into the queue.
                push!(reach_list, (reach, result.status))
                push!(last_reach_list, last_reach)
                push!(children[idx], length(last_reach_list))
            end
            violated_results == nothing || return violated_results, last_reach_list, children, order, i
        end
    end
    return BasicResult(:unknown), last_reach_list, children, order, solver.max_iter
end


function constraint_refinement(solver::AdaptNeurify, nnet::Network, reach::SymbolicIntervalGradient, model::JuMP.Model)
    i, j, gradient = get_nodewise_gradient(nnet, reach.LΛ, reach.UΛ)
    # We can generate three more constraints
    # Symbolic representation of node i j is Low[i][j,:] and Up[i][j,:]
    nnet_new = Network(nnet.layers[1:i])
    reach_new = forward_network(solver, nnet_new, reach.sym.interval, model)
    C, d = tosimplehrep(reach.sym.interval)
    l_sym = reach_new.sym.Low[[j], 1:end-1]
    l_off = reach_new.sym.Low[[j], end]
    u_sym = reach_new.sym.Up[[j], 1:end-1]
    u_off = reach_new.sym.Up[[j], end]
    intervals = Vector{HPolytope{Float64}}(undef, 3)
    intervals[1] = HPolytope([C; l_sym; u_sym], [d; -l_off; -u_off])
    intervals[2] = HPolytope([C; l_sym; -u_sym], [d; -l_off; u_off])
    intervals[3] = HPolytope([C; -l_sym; -u_sym], [d; l_off; u_off])
    # intervals[4] = HPolytope([C; -l_sym; u_sym], [d; l_off; -u_off]) lower bound can not be greater than upper bound
    return intervals
end

function forward_network(solver, nnet::Network, input::AbstractPolytope, model::JuMP.Model, return_last::Bool)
    reach = input
    last_reach = input
    for (i, layer) in enumerate(nnet.layers)
        if i == length(nnet.layers)
            last_reach = deepcopy(reach)
        end
        reach = forward_layer(solver, layer, reach, model)
    end
    return_last && (return reach, last_reach)
    return reach
end

function forward_layer(solver::AdaptNeurify, layer::Layer, input, model::JuMP.Model)
    return forward_act(forward_linear(solver, input, layer), layer, model)
end

# Symbolic forward_linear for the first layer
function forward_linear(solver::AdaptNeurify, input::AbstractPolytope, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    sym = SymbolicInterval(hcat(W, b), hcat(W, b), input)
    LΛ = Vector{Vector{Int64}}(undef, 0)
    UΛ = Vector{Vector{Int64}}(undef, 0)
    return SymbolicIntervalGradient(sym, LΛ, UΛ)
end

# Symbolic forward_linear
function forward_linear(solver::AdaptNeurify, input::SymbolicIntervalGradient, layer::Layer)
    (W, b) = (layer.weights, layer.bias)
    output_Up = max.(W, 0) * input.sym.Up + min.(W, 0) * input.sym.Low
    output_Low = max.(W, 0) * input.sym.Low + min.(W, 0) * input.sym.Up
    output_Up[:, end] += b
    output_Low[:, end] += b
    sym = SymbolicInterval(output_Low, output_Up, input.sym.interval)
    return SymbolicIntervalGradient(sym, input.LΛ, input.UΛ)
end