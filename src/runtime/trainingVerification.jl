
function check_inclusion(solver, nnet, reach, output)
    # The output constraint is in the form A*x < b
    # We try to maximize output constraint to find a violated case, or to verify the inclusion.
    # Suppose the output is [1, 0, -1] * x < 2, Then we are maximizing reach.Up[1] * 1 + reach.Low[3] * (-1)

    input_domain = domain(reach)

    model = Model(solver); set_silent(model)
    x = @variable(model, [1:dim(input_domain)])
    add_set_constraint!(model, input_domain, x)

    max_violation = -1e9
    max_violation_con = nothing
    for (i, cons) in enumerate(constraints_list(output))
        # NOTE can be taken out of the loop, but maybe there's no advantage
        # NOTE max.(M, 0) * U  + ... is a common operation, and maybe should get a name. It's also an "interval map".
        a, b = cons.a, cons.b
        c = max.(a, 0)'*reach.Up + min.(a, 0)'*reach.Low

        @objective(model, Max, c * [x; 1] - b)
        optimize!(model)

        if termination_status(model) == OPTIMAL

            viol = objective_value(model)

            if viol > max_violation
                max_violation = viol
                max_violation_con = a
            end

            if compute_output(nnet, value(x)) ∉ output
                return CounterExampleResult(:violated, value(x)), nothing, max_violation
            end

        else
            # TODO can we be more descriptive?
            error("No solution, please check the problem definition.")
        end

    end

    if max_violation > 0.0
        return CounterExampleResult(:unknown), max_violation_con, max_violation
    else
        return CounterExampleResult(:holds), nothing, max_violation
    end
end

function ordinal_split!(solver, problem, branches::Tree, x::Int, max_size::Int, splits_order::Array{Tuple{Int64,Int64},1})
    
    (domain, max_violation, splits) = branches.data[x]
    nnet, output = problem.network, problem.output
    
    reach = forward_network(solver, nnet, domain)
    result, max_violation_con, max_violation = check_inclusion(solver, nnet, reach.sym, output)
    branches.data[x][2] = max_violation

    result.status == :unknown || return result

    if tree_size(branches) >= solver.max_iter
        return BasicResult(:unknown)
    end
    
    k = length(splits)
    subdomains = constraint_refinement(nnet, reach, max_violation_con, splits, splits_order[k+1])
    for subdomain in subdomains
        add_child!(branches, x, (subdomain, 0, copy(splits)))
    end

    for c in branches.children[x]
        result = ordinal_split!(solver, problem, branches, c, max_size, splits_order)
        result.status == :holds || return result # if status == :unknown, means splitting number exceeds max_iter, return unkown directly.
    end
    return BasicResult(:holds)
end


function init_split(solver, problem, max_branch)
    # split sequantially
    splits_order = Array{Tuple{Int64,Int64},1}(undef, max_branch)
    k = 0
    for (i,l) in enumerate(problem.network.layers)
        for j in 1:n_nodes(l)
            k += 1
            k > max_branch && break
            splits_order[k] = (i,j)
        end
        k > max_branch && break
    end
    branches = Tree((problem.input, 0, Vector()))
    result = ordinal_split!(solver, problem, branches, 1, max_branch, splits_order)
    return result, branches
end

function solve(problems::TrainingProblem, max_branches=50, fix_branch=true, branch_management=false, perturbation_tolerence=false, incremental_computation=false)
    
    solver = Neurify(max_iter = 1) # max_iter=1 because we are doing branch management outside.
    
    problems = TrainingProblem(problems.networks, convert(HPolytope, problems.input), convert(HPolytope, problems.output))

    cnt = 0
    total_time = 0

    sat_idx = []
    vio_idx = []
    tim_idx = []
    err_idx = []
    tim_rec = []
    sat_rec = []

    n = length(problems.networks)

    net = problems.networks[1]
    problem = Problem(net, problems.input, problems.output)

    result, branches = init_split(solver, problem, max_branches)
    println(result)
    println(branches)
    println("========")
    result, unknown_leaves = dfs_check(solver, problem, branches, 1)

    for leaf in unknown_leaves
        result = dfs_split(solver, problem, branches, leaf, solver.max_iter)
        result.status == :holds || return result, branches
    end

    return BasicResult(:holds), branches

    for i = 1:n

        problem = Problem(problems.networks[i], problems.input, problems.output)

        timed_result = @timed solve(solver, problem)
        result, iter = timed_result[1]

        total_time += timed_result[2]
        append!(tim_rec, timed_result[2])
        println("Output: ")
        println(result)
        println("")
        
        if result.status == :violated 
            noisy = NeuralVerification.compute_output(net, result.counter_example)
            append!(vio_idx, i)
            # println("======== found counter example ========")
            # println("index: " * string(i))
            # println("Time: " * string(timed_result[2]) * " s")
            # println("counter_pred   ", noisy[:,:]')
            # println("=======================================")
        elseif result.status == :unknown
            append!(tim_idx, i)
            # println("Timed out")
        else
            append!(sat_idx, i)
            # println("Holds")
        end
    end

    return total_time, sat_idx, vio_idx, tim_idx

end


function solve(solver, problem::Problem, branches = nothing)
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
        result = dfs_split(solver, problem, branches, leaf, solver.max_iter)
        result.status == :holds || return result, branches
    end

    return BasicResult(:holds), branches
end



function dfs_check(solver, problem, branches::Tree, x::Int)
    (input_set, max_violation_con, splits) = branches.data[x]

    reach = forward_network(solver, problem.network, problem.input)
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

function dfs_split!(solver, problem, branches::Tree, x::Int, max_size::Int)
    (domain, max_violation_con, splits) = branches.data[x]

    if tree_size(branches) >= solver.max_iter
        return BasicResult(:unknown)
    end

    nnet, output = problem.network, problem.output
    reach_list = []
    domain = init_symbolic_grad(problem.input)
    
    max_max_vio = -1e9
    final_result = CounterExampleResult(:holds)

    reach = forward_network(solver, nnet, domain)
    result, max_violation_con, max_violation = check_inclusion(solver, nnet, last(reach).sym, output)

    if result.status === :violated
        final_result = result
    end
    
    k = length(splits)
    if k < length(splits_order)
        subdomains = constraint_refinement(nnet, reach, max_violation_con, splits, splits_order[k+1])
        for domain in subdomains
            push!(reach_list, (init_symbolic_grad(domain), copy(splits)))
        end
    else
        println(splits, " ", max_violation)
        max_max_vio = max(max_max_vio, max_violation)
        if final_result.status === :holds
            final_result = result
        end
    end
    isempty(reach_list) && break
end
