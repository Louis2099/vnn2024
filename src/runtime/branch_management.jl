function split_by_node(nnet::Network,
    reach::Vector{<:SymbolicIntervalGradient},
    max_violation_con,
    splits,
    splits_order)
    if isnothing(splits_order)
        i, j, influence = get_max_nodewise_influence(nnet, reach, max_violation_con, splits, false)
    else
        i,j = splits_order
    end
    # We can generate three more constraints
    # Symbolic representation of node i j is Low[i][j,:] and Up[i][j,:]
    aL, bL = reach[i].sym.Low[j, 1:end-1], reach[i].sym.Low[j, end]
    aU, bU = reach[i].sym.Up[j, 1:end-1], reach[i].sym.Up[j, end]
    
    # custom intersection function that doesn't do constraint pruning
    ∩ = (set, lc) -> HPolytope([constraints_list(set); lc])
    
    subsets = Union{Nothing, HPolytope}[domain(reach[1])] # reach is the list of reachable set of each layer of the network.
    
    # If either of the normal vectors is the 0-vector, we must skip it.
    # It cannot be used to create a halfspace constraint.
    # NOTE: how can this come about, and does it mean anything?
    if !iszero(aL)
        subsets = subsets .∩ [HalfSpace(aL, -bL), HalfSpace(aL, -bL), HalfSpace(-aL, bL)]
    end
    if !iszero(aU)
        subsets = subsets .∩ [HalfSpace(aU, -bU), HalfSpace(-aU, bU), HalfSpace(-aU, bU)]
    end
    
    # empty_idx = filter(x->isempty(subsets[x]), eachindex(subsets))
    return subsets, (i, j)
end

function split_by_dimension(reach::Vector{<:SymbolicIntervalGradient})
    
    original_domain = domain(reach[1])
    B = box_approximation(original_domain)
    dim_sizes = [(high(B, i) - low(B, i)) for i in 1:dim(B)]
    (max_len, max_dim) = findmax(dim_sizes)
    
    a = zeros(length(dim_sizes))
    a[max_dim] = 1.0
    b = (high(B, max_dim) + low(B,max_dim)) / 2.0
    
    # custom intersection function that doesn't do constraint pruning
    ∩ = (set, lc) -> HPolytope([constraints_list(set); lc])
    
    subsets = Union{Nothing, HPolytope}[original_domain] # reach is the list of reachable set of each layer of the network.
    subsets = subsets .∩ [HalfSpace(a, b), HalfSpace(-a, -b)]
    
    return subsets, (0, 0)
end

function ordinal_split!(solver, problem, branches::Tree, x::Int, max_size::Int, split_method=:split_by_node_heuristic, splits_order = nothing)
    (domain, splits) = branches.data[x]
    nnet, output = problem.network, problem.output
    reach = forward_network(solver, nnet, domain)
    result, max_violation_con = check_inclusion(solver, nnet, last(reach).sym, output)
    # branches.data[x] = (domain, max_violation_con, splits) # because max_violation_con is not calculated before (set as 0)
    
    # result.status == :unknown || (return result)
    result.status == :holds && (return result)
    
    if tree_size(branches) >= max_size
        return BasicResult(:unknown)
    end
    
    if split_method == :split_by_dim
        subdomains, split_node = split_by_dimension(reach)
    else
        if isnothing(splits_order)
            subdomains, split_node = split_by_node(nnet, reach, max_violation_con, splits, nothing)
        else
            k = length(splits)
            subdomains, split_node = split_by_node(nnet, reach, max_violation_con, splits, splits_order[k+1])
        end
    end
    
    for (idx, subdomain) in enumerate(subdomains)
        if isempty(subdomain)
            continue
        end
        new_splits = copy(splits)
        push!(new_splits, (split_node, idx))
        add_child!(branches, x, (subdomain, new_splits))
    end
    
    for c in branches.children[x]
        result = ordinal_split!(solver, problem, branches, c, max_size, split_method, splits_order)
        result.status == :holds || return result # status == :unknown means splitting number exceeds max_iter, return unkown directly.
    end
    return BasicResult(:holds)
end

function generate_ordinal_splits_order(nnet, max_branches)
    splits_order = Array{Tuple{Int64,Int64},1}(undef, max_branches)
    k = 0
    for (i,l) in enumerate(nnet.layers)
        for j in 1:n_nodes(l)
            k += 1
            k > max_branches && break
            splits_order[k] = (i,j)
        end
        k > max_branches && break
    end
    return splits_order
end

function init_split(solver, problem, max_branches, split_method = :split_by_node_heuristic, splits_order=nothing, samples=nothing)
    # split sequantially
    branches = Tree((problem.input, Vector()))
    result = ordinal_split!(solver, problem, branches, 1, max_branches, split_method, splits_order) # split by dimension
    samples_branch = nothing
    # @show length(branches.leaves)
    if !isnothing(samples)
        samples_branch = []
        for (i,sample) in enumerate(samples)
            for leaf in branches.leaves
                (domain, splits) = branches.data[leaf]
                if in(sample, domain)
                    push!(samples_branch, leaf)
                    break
                end
            end
        end
    end
    return result, branches, samples_branch
end

function check_node(solver, problem, domain, last_layer_reach = nothing)
    if isnothing(last_layer_reach)
        reach = forward_network(solver, problem.network, domain)
    else
        reach = [last_layer_reach, forward_layer(solver, problem.network.layers[end], last_layer_reach)]
    end
    result, max_violation_con = check_inclusion(solver, problem.network, last(reach).sym, problem.output)
    return result, reach
end

function check_all_leaves(solver, problem, branches, incremental_computation=false, forward_result_dict=nothing)
    result_dict = Dict()
    if incremental_computation && !isnothing(forward_result_dict)
        for leaf in branches.leaves
            result_dict[leaf], reach = check_node(solver, problem, branches.data[leaf][1], forward_result_dict[leaf])
            forward_result_dict[leaf] = reach[end-1]
        end
    else
        forward_result_dict = Dict()
        for leaf in branches.leaves
            result_dict[leaf], reach = check_node(solver, problem, branches.data[leaf][1])
            forward_result_dict[leaf] = reach[end-1]
        end
    end
    
    hold_cnt = count(x->x[2].status==:holds,result_dict)
    unkn_cnt = count(x->x[2].status==:unknown,result_dict)
    viol_cnt = count(x->x[2].status==:violated,result_dict)
    
    violated_idx = [k for (k,v) in result_dict if v.status==:violated]
    
    length(violated_idx) > 0 && return result_dict[violated_idx[1]], result_dict, (hold_cnt, unkn_cnt, viol_cnt), forward_result_dict
    count(x->x[2].status==:unknown,result_dict) > 0 && return BasicResult(:unknown), result_dict, (hold_cnt, unkn_cnt, viol_cnt), forward_result_dict
    return BasicResult(:holds), result_dict, (hold_cnt, unkn_cnt, viol_cnt), forward_result_dict
end


function check_all_leaves_demand_shifting(solver, problem, branches, incremental_computation=false, prev_out_reach=nothing)
    result_dict = Dict()
    if incremental_computation && !isnothing(prev_out_reach)
        for leaf in branches.leaves
            result_dict[leaf], max_violation_con = check_inclusion(solver, problem.network, prev_out_reach[leaf].sym, problem.output)
        end
    else
        prev_out_reach = Dict()
        for leaf in branches.leaves
            result_dict[leaf], reach = check_node(solver, problem, branches.data[leaf][1])
            prev_out_reach[leaf] = reach[end]
        end
    end
    
    hold_cnt = count(x->x[2].status==:holds,result_dict)
    unkn_cnt = count(x->x[2].status==:unknown,result_dict)
    viol_cnt = count(x->x[2].status==:violated,result_dict)
    
    violated_idx = [k for (k,v) in result_dict if v.status==:violated]
    
    length(violated_idx) > 0 && return result_dict[violated_idx[1]], result_dict, (hold_cnt, unkn_cnt, viol_cnt), prev_out_reach
    count(x->x[2].status==:unknown,result_dict) > 0 && return BasicResult(:unknown), result_dict, (hold_cnt, unkn_cnt, viol_cnt), prev_out_reach
    return BasicResult(:holds), result_dict, (hold_cnt, unkn_cnt, viol_cnt), prev_out_reach
end

function update_all_leaves!(solver, problem, branches, prev_input)
    input = problem.input
    diff_idx = []
    # println("==========================")
    # @show length(input.constraints), length(prev_input.constraints)
    for i in 1:length(input.constraints)
        # spurious, may not be correct. but works well for now.
        input.constraints[i] != prev_input.constraints[i] && push!(diff_idx, i)
    end
    # @show diff_idx
    # @show input.constraints[diff_idx]
    for leaf in branches.leaves
        branches.data[leaf][1].constraints[diff_idx] = input.constraints[diff_idx]
        # @show branches.data[leaf][1].constraints[diff_idx]
    end
end

function enlarge_domain(domain, reachable_set_relaxation)
    enlarge_domain = copy(domain)
    reachable_set_relaxation < 0 && return enlarge_domain
    for (i,con) in enumerate(domain.constraints)
        enlarge_domain.constraints[i] = HalfSpace(con.a, con.b + reachable_set_relaxation)
    end
    return enlarge_domain
end

function set_distance(A, B)
    # find max distance between 2 sets.
    # when max_dis < 0, means A is included in B

    model = Model(GLPK.Optimizer); set_silent(model)
    x = @variable(model, [1:dim(A)])
    add_set_constraint!(model, A, x)
    
    max_dis = -Inf
    for (i, cons) in enumerate(constraints_list(B))
        # NOTE can be taken out of the loop, but maybe there's no advantage
        # NOTE max.(M, 0) * U  + ... is a common operation, and maybe should get a name. It's also an "interval map".
        a, b = cons.a, cons.b

        @objective(model, Max, a' * x - b)
        optimize!(model)

        if termination_status(model) == OPTIMAL
            viol = objective_value(model)
            max_dis = max(max_dis, viol / norm(a))
        else
            error("No solution, please check the problem definition.")
        end
    end
    return max_dis
    # return max(max_dis, 0)
end


function set_violation(A, B)
    # find max violation between 2 sets.
    # when max_vio < 0, means A is included in B

    model = Model(GLPK.Optimizer); set_silent(model)
    x = @variable(model, [1:dim(A)])
    add_set_constraint!(model, A, x)
    
    max_vio = -Inf
    for (i, cons) in enumerate(constraints_list(B))
        # NOTE can be taken out of the loop, but maybe there's no advantage
        # NOTE max.(M, 0) * U  + ... is a common operation, and maybe should get a name. It's also an "interval map".
        a, b = cons.a, cons.b

        @objective(model, Max, a' * x - b)
        optimize!(model)

        if termination_status(model) == OPTIMAL
            viol = objective_value(model)
            max_vio = max(max_vio, viol)
        else
            error("No solution, please check the problem definition.")
        end
    end
    return max_vio
    # return max(max_vio, 0)
end

function check_all_leaves_domain_shifting(solver, problem, branches, reachable_set_relaxation=-1, enlarged_inputs=Dict(), prev_results=Dict(), lipschitz=nothing)
    result_dict = Dict()
    isnothing(enlarged_inputs) && (enlarged_inputs = Dict())
    rsr_saved_cnt = 0
    for leaf in branches.leaves
        (domain, splits) =  branches.data[leaf]
        if haskey(enlarged_inputs, leaf) && haskey(prev_results, leaf)
            if reachable_set_relaxation >= 0 && (isempty(domain) || issubset(domain, enlarged_inputs[leaf])) # rsr=0 -> branch_management
                result_dict[leaf] = prev_results[leaf]
                rsr_saved_cnt += 1
                continue
            end
            if !isnothing(lipschitz)
                input_dis = set_distance(domain, enlarged_inputs[leaf])
                if input_dis < prev_results[leaf].min_dis / lipschitz
                    result_dict[leaf] = RuntimeResult(prev_results[leaf].status, prev_results[leaf].min_dis - input_dis * lipschitz)
                    continue
                end
            end
            # if reachable_set_relaxation > 0 || !isnothing(lipschitz)
            #     input_dis = set_distance(domain, enlarged_inputs[leaf])
            #     if reachable_set_relaxation > 0 && input_dis <= 0
            #         result_dict[leaf] = prev_results[leaf]
            #         continue
            #     elseif !isnothing(lipschitz) && max(input_dis,0) < prev_results[leaf].min_dis / lipschitz
            #         result_dict[leaf] = RuntimeResult(prev_results[leaf].status, prev_results[leaf].min_dis - max(input_dis,0) * lipschitz)
            #         continue
            #     end
            # end
        end
        enlarged_inputs[leaf] = enlarge_domain(domain, reachable_set_relaxation)
        result_dict[leaf], reach = check_node(solver, problem, enlarged_inputs[leaf])
    end
    # println("RSR saved:", rsr_saved_cnt, "/", length(branches.leaves))
    
    hold_cnt = count(x->x[2].status==:holds,result_dict)
    unkn_cnt = count(x->x[2].status==:unknown,result_dict)
    viol_cnt = count(x->x[2].status==:violated,result_dict)
    
    violated_idx = [k for (k,v) in result_dict if v.status==:violated]
    
    length(violated_idx) > 0 && return result_dict[violated_idx[1]], result_dict, (hold_cnt, unkn_cnt, viol_cnt), enlarged_inputs
    count(x->x[2].status==:unknown,result_dict) > 0 && return BasicResult(:unknown), result_dict, (hold_cnt, unkn_cnt, viol_cnt), enlarged_inputs
    return BasicResult(:holds), result_dict, (hold_cnt, unkn_cnt, viol_cnt), enlarged_inputs
end

function split_given_path(solver, problem, split_path)
    # this can be slow because we have to split from the beginning.
    # for example, if we want to merge two paths:
    # 1+2+4+  and  1+2-4+.   We can not directly remove different constraints.
    # Because the activation condition of node 4 is different for  after we split 2+ and 2-.
    # And 1+4+ may not exist, because 4+ can be empty
    nnet, domain, output = problem.network, problem.input, problem.output
    splits = Vector()
    
    for choice in split_path
        (node, sgn) = choice
        reach = forward_network(solver, nnet, domain)
        result, max_violation_con = check_inclusion(solver, nnet, last(reach).sym, output)
        push!(splits, (node, 0)) # set idx=0, because this split_path may not be part of any tree path
        subdomains, split_node = split_by_node(nnet, reach, max_violation_con, splits, node)
        domain = subdomains[sgn]
        if isnothing(domain)
            return BasicResult(:holds, )
        end 
    end
    
    reach = forward_network(solver, nnet, domain)
    result, max_violation_con = check_inclusion(solver, nnet, last(reach).sym, output)
    
    return result, (domain, splits)
end

# This function is deprecated, because two paths only differ in a middle node are not mergeable.
# function merge_holds_nodes_general!(solver, problem, branches, result_dict)
#     """
#     The split path is in the form:
#     ((i1,j1), sgn1), ((i2,j2), sgn2) ...

#     (i,j) is the position of the split ReLU node in the network. 
#     sgn is the index of the subdomain we choose. 
#     In neurify, we split the domain into three subdomains. Therefore, the sign can be 1-3. 

#     We denote
#     ((i,j), sgn): choice
#     (i,j):        node
#     sgn:          sign

#     To merge paths, we must have 3 holding paths that are only different in one sign.
#     To find such paths. We define a dictionary pool.

#     pool:   key:   split_path with a choice removed.
#     value: [(removed_choice1, path_idx1), (removed_choice2, path_idx2), ... ],
#     """
#     pool = Dict()
#     new_leaves = []
#     merged_path_idx = []
#     try_cnt = 0
#     suc_cnt = 0
#     leaves = copy(branches.leaves) # in case the branches.leaves changes in the loop
#     for (i, leaf_idx) in enumerate(leaves)
#         leaf = branches.data[leaf_idx]
#         result_dict[leaf_idx].status == :holds || continue
#         (domain, split_path) = leaf
#         # println(split_path)
#         for j in 1:length(split_path)
#             pruned_path = [split_path[1:j-1]; split_path[j+1:end]]  # remove a choice from the split path, then use the pruned path as the key.
#             if haskey(pool, pruned_path)  # check all paths that has the same pruned path
#                 choice_idx = findall(x -> x[1][1]==split_path[j][1], pool[pruned_path])  # find choices that have the same node
#                 if length(choice_idx) == 2 # find two nodes, that is, all sign of this node hold, possible to merge
#                     # println("find identical")
#                     # println("split_path")
#                     # println(split_path)
#                     # println("split_path[j]")
#                     # println(split_path[j])
#                     idx = [[x[2] for x in pool[pruned_path][choice_idx]]; leaf_idx]  # get all mergable leaf idx
#                     idx = filter(x->!(x in merged_path_idx), idx) # remove paths that are already merged
#                     length(idx) == 3 || continue 
#                     # println("idx (if consecutive, we are actually replacing three leaves with their parent)")
#                     println(idx)
#                     result, merged_node = split_given_path(solver, problem, pruned_path)
#                     # println("merge result")
#                     # println(result)
#                     try_cnt += 1
#                     result.status == :holds || continue
#                     suc_cnt += 1
#                     merged_path_idx = [merged_path_idx; idx]
#                     # println("merged_node")
#                     # println(merged_node)
#                     id = add_child!(branches, idx[1], merged_node)# remove idx[1] from leaves, set parent of the merged_node as idx[1]
#                     connect!(branches, idx[2], id)  # to remove idx[2] from leaves
#                     connect!(branches, idx[3], id)  # to remove idx[3] from leaves
#                 end
#             else
#                 pool[pruned_path] = []
#             end
#             push!(pool[pruned_path], (split_path[j], leaf_idx))  # store (removed_choice, leaf_idx) as the value.
#         end
#     end
#     println("try merge: ", try_cnt)
#     println("success:   ", suc_cnt)
#     return [filter(x->!(x in merged_path_idx), branches.leaves); new_leaves]
# end

function merge_holds_nodes!(solver, problem, branches, result_dict)
    """
    If all the siblings of a leaf node and itself hold, try to replace them with their parent.
    """
    holds_cnt = zeros(branches.size)
    branches.size == 1 && return
    leaves = copy(branches.leaves)
    try_cnt = 0
    suc_cnt = 0
    while !isempty(leaves)
        leaf = pop!(leaves)
        result_dict[leaf].status == :holds || continue
        # println(leaf, ' ', branches.parent[leaf])
        holds_cnt[branches.parent[leaf]] += 1
        if holds_cnt[branches.parent[leaf]] == length(branches.children[branches.parent[leaf]])
            # println("try to merge")
            try_cnt += 1
            result, last_reach = check_node(solver, problem, branches.data[branches.parent[leaf]][1])
            # result = BasicResult(:holds) # to test split
            (result.status == :holds) || continue
            # println("merge success")
            suc_cnt += 1
            # println("branches.parent[leaf]")
            # println(branches.parent[leaf])
            # println("before")
            # println(branches.leaves)
            result_dict[branches.parent[leaf]] = result
            push!(leaves, branches.parent[leaf])
            delete_all_children!(branches, branches.parent[leaf])
            # println("after")
            # println(branches.leaves)
        end
    end
    # println("try merge: ", try_cnt)
    # println("success:   ", suc_cnt)
end



function calc_length(branches, result_dict)
    (inputset, splits) = branches.data[1]
    # p = plot(inputset, color="blue")
    hold_area = 0
    for leaf in branches.leaves
        # println(leaf)
        if result_dict[leaf].status == :holds
            (domain, splits) = branches.data[leaf]
            isempty(domain) && continue
            println("domain")
            println(domain)
            println(isempty(domain))
            println(LazySets.diameter(domain))
            hold_area += LazySets.diameter(domain)
            if LazySets.diameter(domain) < 1e-6
                continue
            end
            # plot!(p, domain, color="green", alpha=0.8)
        end
        
        if result_dict[leaf].status == :unknown
            (domain, splits) = branches.data[leaf]
            if LazySets.diameter(domain) < 1e-6
                continue
            end
            # plot!(p, domain, color="orange", alpha=0.8)
        end
    end 
    return hold_area / area(inputset)
end


function calc_area(branches, result_dict)
    (inputset, splits) = branches.data[1]
    # p = plot(inputset, color="blue")
    hold_area = 0
    for leaf in branches.leaves
        # println(leaf)
        if result_dict[leaf].status == :holds
            (domain, splits) = branches.data[leaf]
            # println("domain")
            # println(domain)
            hold_area += area(domain)
            if area(domain) < 1e-6
                continue
            end
            # plot!(p, domain, color="green", alpha=0.8)
        end
        
        if result_dict[leaf].status == :unknown
            (domain, splits) = branches.data[leaf]
            if area(domain) < 1e-6
                continue
            end
            # plot!(p, domain, color="orange", alpha=0.8)
        end
    end 
    # display(p)
    # println("hold area ratio:", )
    return hold_area / area(inputset)
end

function calc_sampling_coverage(branches, result_dict, samples_branch)
    # println("samples_branch")
    # println(samples_branch)
    # println(result_dict)
    hold_cnt = sum([result_dict[x].status == :holds for x in samples_branch])
    return hold_cnt*1.0/length(samples_branch)
end

function compute_coverage(branches, result_dict, input_dim, samples_branch)
    isnothing(samples_branch) && return 0
    
    if input_dim == 1
        return calc_length(branches, result_dict)
    elseif input_dim == 2
        return calc_area(branches, result_dict)
    else
        return calc_sampling_coverage(branches, result_dict, samples_branch)
    end
end