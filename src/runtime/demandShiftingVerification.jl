function solve(problems::DemandShiftingProblem, split_method=:split_by_node_heuristic, max_branches=50, branch_management=false, incremental_computation=false, samples=nothing)
    
    # solver = perturbation_tolerence > 0 ? IntervalNet(max_iter = 1, delta = (perturbation_tolerence, perturbation_tolerence))
    #                                     : Neurify(max_iter = 1) # max_iter=1 because we are doing branch management outside.

    solver = IntervalNet(max_iter = 1, delta = (0, 0))

    problems = DemandShiftingProblem(problems.network, convert(HPolytope, problems.input), [convert(HPolytope, x) for x in problems.outputs])

    cnt = 0
    total_time = 0

    sat_idx = []
    vio_idx = []
    tim_idx = []
    err_idx = []
    tim_rec = []
    sat_rec = []
    cnt_rec = []
    cov_rec = []
    
    splits_order = generate_ordinal_splits_order(problems.network, max_branches)
    last_unk_cnt = nothing
    branches = nothing
    samples_branch = nothing
    cnts = nothing
    coverage = nothing
    out_reach = nothing
    
    @showprogress 1 "Verifying output constraints change..." for (i, output) in enumerate(problems.outputs)
        
        problem = Problem(problems.network, problems.input, output)

        if i == 1 || !branch_management
            result, branches, samples_branch = init_split(solver, problem, max_branches, split_method, splits_order, samples)
        end

        timed_result = @timed check_all_leaves_demand_shifting(solver, problem, branches, incremental_computation, out_reach)
        result, result_dict, cnts, out_reach = timed_result.value

        if i > 1 && branch_management && cnts[2] > last_unk_cnt[2]
            # println("recompute, cnts: ",cnts)
            result, branches, samples_branch = init_split(solver, problem, max_branches, split_method, splits_order, samples)
            timed_result = @timed check_all_leaves_demand_shifting(solver, problem, branches, incremental_computation, out_reach)
            result, result_dict, cnts, out_reach = timed_result.value
        end
        
        last_unk_cnt = cnts
        total_time += timed_result.time
        coverage = compute_coverage(branches, result_dict, size(problems.network.layers[1].weights, 2), samples_branch)

        append!(cov_rec, coverage)
        append!(cnt_rec, cnts)
        append!(tim_rec, timed_result.time)

        # println("idx, cnts, coverage: ", i, " ",cnts, " ", coverage)
        
        # if branch_management
        #     merge_holds_nodes!(solver, problem, branches, result_dict) # try to merge holds nodes to save memory resources.
        # end
        # unknown_leaves = [k for (k,v) in result_dict if v.status==:unknown] # because branches.leaves may change in the split process
        # for leaf in unknown_leaves
        #     result_dict[leaf].status == :unknown && ordinal_split!(solver, problem, branches, leaf, max_branches, splits_order) # split unknown nodes
        # end
        
        # println(branches.size)
        if result.status == :violated 
            noisy = NeuralVerification.compute_output(nnet, result.counter_example)
            append!(vio_idx, i)
            println("======== found counter example ========")
            println("index: " * string(i))
            println("Time: " * string(timed_result[2]) * " s")
            println("Input:   ", result.counter_example)
            println("Counter pred:   ", noisy[:,:]')
            println("=======================================")
        elseif result.status == :unknown
            append!(tim_idx, i)
            # println("Timed out")
        else
            append!(sat_idx, i)
            # println("Holds")
        end
    end

    return tim_rec, cnt_rec, cov_rec

end
