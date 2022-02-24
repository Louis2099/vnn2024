function solve(problems::AdaptingProblem, split_method=:split_by_node_heuristic, max_branches=50; branch_management=false, interval_range=nothing, incremental_computation=false, samples=nothing)
    
    # solver = interval_range > 0 ? IntervalNet(max_iter = 1, delta = (interval_range, interval_range))
    #                                     : Neurify(max_iter = 1) # max_iter=1 because we are doing branch management outside.

    isnothing(interval_range) && (interval_range = [(0.,0.) for i in 1:length(problems.networks[1].layers)])

    solver = IntervalNet(max_iter = 1, deltas = interval_range)

    problems = AdaptingProblem(problems.networks, convert(HPolytope, problems.input), convert(HPolytope, problems.output))

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

    splits_order = generate_ordinal_splits_order(problems.networks[1], max_branches)
    last_net_id = 1
    last_unk_cnt = nothing
    branches = nothing
    samples_branch = nothing
    cnts = nothing
    coverage = nothing
    forward_result_dict = nothing
    # print_tree(branches, 1)
    
    @showprogress 1 "Verifying last layer change..."  for (i, nnet) in enumerate(problems.networks)
        
        # println("====")
        diffs = net_diffs(nnet, problems.networks[last_net_id]);
        in_INN = all([l1[1] <= l2[1] && l1[2] <= l2[2] for (l1, l2) in zip(diffs, interval_range)])
        if i > 1 && in_INN
            append!(cnt_rec, cnts)
            append!(cov_rec, coverage)
            append!(tim_rec, 0)
            continue
        end
        last_net_id = i

        problem = Problem(nnet, problems.input, problems.output)

        if i == 1 || !branch_management
            result, branches, samples_branch = init_split(solver, problem, max_branches, split_method, splits_order, samples)
            forward_result_dict = nothing
        end

        timed_result = @timed check_all_leaves(solver, problem, branches, incremental_computation, forward_result_dict)
        result, result_dict, cnts, forward_result_dict = timed_result.value

        if i > 1 && branch_management && cnts[2] > last_unk_cnt[2]
            # println("recompute, cnts: ",cnts)
            result, branches, samples_branch = init_split(solver, problem, max_branches, split_method, splits_order, samples)
            timed_result = @timed check_all_leaves(solver, problem, branches, incremental_computation)
            result, result_dict, cnts, forward_result_dict = timed_result.value
        end
        
        last_unk_cnt = cnts
        total_time += timed_result.time
        coverage = compute_coverage(branches, result_dict, size(problems.networks[1].layers[1].weights, 2), samples_branch)

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
