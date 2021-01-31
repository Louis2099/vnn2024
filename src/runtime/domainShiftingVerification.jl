function solve(problems::DomainShiftingProblem, max_branches=50, branch_management=false, reachable_set_relaxation=0, samples=nothing)
    
    # solver = perturbation_tolerence > 0 ? IntervalNet(max_iter = 1, delta = (perturbation_tolerence, perturbation_tolerence))
    #                                     : Neurify(max_iter = 1) # max_iter=1 because we are doing branch management outside.

    solver = IntervalNet(max_iter = 1, delta = (0, 0))

    problems = DomainShiftingProblem(problems.network, [convert(HPolytope, x) for x in problems.inputs], convert(HPolytope, problems.output))

    cnt = 0
    total_time = 0

    sat_idx = []
    vio_idx = []
    tim_idx = []
    err_idx = []
    tim_rec = []
    sat_rec = []
    cnt_rec = []

    problem = Problem(problems.network, problems.inputs[1], problems.output)
    splits_order = generate_ordinal_splits_order(problems.network, max_branches)
    result, branches, samples_branch = init_split_given_order(solver, problem, max_branches, splits_order)
    # println(solve(solver, problem))

    timed_result = @timed check_all_leaves_domain_shifting(solver, problem, branches, reachable_set_relaxation)
    result, result_dict, cnts, enlarged_inputs = timed_result.value
    
    println("size(branches.leaves)")
    println(length(branches.leaves))
    last_unk_cnt = cnts

    # print_tree(branches, 1)
    
    for (i, input) in enumerate(problems.inputs)
        
        problem = Problem(problems.network, input, problems.output)

        if !branch_management
            result, branches, samples_branch = init_split_given_order(solver, problem, max_branches, splits_order)
        end

        timed_result = @timed check_all_leaves_domain_shifting(solver, problem, branches, reachable_set_relaxation, enlarged_inputs, result_dict)
        result, result_dict, cnts, enlarged_inputs = timed_result.value

        if branch_management && cnts[2] > last_unk_cnt[2]
            println("recompute, cnts: ",cnts)
            result, branches, samples_branch = init_split_given_order(solver, problem, max_branches, splits_order)
            timed_result = @timed check_all_leaves_domain_shifting(solver, problem, branches, reachable_set_relaxation, enlarged_inputs, result_dict)
            result, result_dict, cnts, enlarged_inputs = timed_result.value
            last_unk_cnt = cnts
        end
        println("cnts: ",cnts)
        total_time += timed_result.time
        append!(cnt_rec, cnts)

        # calc_area(branches, result_dict)
        append!(tim_rec, timed_result.time)
        # println("Output: ")
        # println(result)
        # println("")
        # println(branches.size)
        if branch_management
            merge_holds_nodes!(solver, problem, branches, result_dict) # try to merge holds nodes to save memory resources.
        end
        # merge_holds_nodes_general!(solver, problem, branches, result_dict) # try to merge holds nodes to save memory resources.
        # break
        # println(branches.size)
        # println(result_dict)
        unknown_leaves = [k for (k,v) in result_dict if v.status==:unknown] # because branches.leaves may change in the split process
        # println(unknown_leaves)
        for leaf in unknown_leaves
            # println(leaf, ' ', result_dict[leaf].status)
            result_dict[leaf].status == :unknown && ordinal_split!(solver, problem, branches, leaf, max_branches, splits_order) # split unknown nodes
        end

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

    return tim_rec, cnt_rec

end
