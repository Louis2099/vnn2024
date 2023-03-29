"""
    Ai2{T}

`Ai2` performs over-approximated reachability analysis to compute the over-approximated
output reachable set for a network. `T` can be `Hyperrectangle`, `Zonotope`, or
`HPolytope`, and determines the amount of over-approximation (and hence also performance
tradeoff). The original implementation (from [1]) uses Zonotopes, so we consider this
the "benchmark" case. The `HPolytope` case is more precise, but slower, and the opposite
is true of the `Hyperrectangle` case.

Note that initializing `Ai2()` defaults to `Ai2{Zonotope}`.
The following aliases also exist for convenience:

```julia
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle}
```

# Problem requirement
1. Network: any depth, ReLU activation (more activations to be supported in the future)
2. Input: AbstractPolytope
3. Output: AbstractPolytope

# Return
`ReachabilityResult`

# Method
Reachability analysis using split and join.

# Property
Sound but not complete.

# Reference
[1] T. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev,
"Ai2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,"
in *2018 IEEE Symposium on Security and Privacy (SP)*, 2018.

## Note
Efficient over-approximation of intersections and unions involving zonotopes relies on Theorem 3.1 of

[2] Singh, G., Gehr, T., Mirman, M., Püschel, M., & Vechev, M. (2018). Fast
and effective robustness certification. In Advances in Neural Information
Processing Systems (pp. 10802-10813).
"""
struct Ai2{T<:Union{Hyperrectangle, Zonotope, HPolytope}} <: Solver end

Ai2() = Ai2{Zonotope}()
const Ai2h = Ai2{HPolytope}
const Ai2z = Ai2{Zonotope}
const Box = Ai2{Hyperrectangle}

function solve(solver::Ai2, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

function solve(solver::Ai2, problem::Problem; max_iter=100, sampling_size=1)
    sampling_size = max(sampling_size, 1)
    nnet, output = problem.network, problem.output
    reach_list = []
    domain = problem.input
    for i in 1:max_iter
        if i > 1
            domain = select!(reach_list, :DFS)
        end

        reach = forward_network(solver, nnet, domain)
        # println(reach)
        # println(length(reach))
        result = check_inclusion(reach, problem.output)

        if result.status === :violated
            counter_examples = sample_counter_examples(domain, output, nnet, sampling_size)
            return CounterExamplesResult(:violated, counter_examples), i
        elseif result.status === :unknown
            subdomains = split_by_dimension(domain)
            for domain in subdomains
                push!(reach_list, domain)
            end
        end
        isempty(reach_list) && return CounterExamplesResult(:holds), i
    end
    
    return CounterExamplesResult(:unknown), max_iter
end

function split_by_dimension(domain)
    
    B = box_approximation(domain)
    dim_sizes = [(high(B, i) - low(B, i)) for i in 1:dim(B)]
    (max_len, max_dim) = findmax(dim_sizes)
    
    a = zeros(length(dim_sizes))
    a[max_dim] = 1.0
    b = (high(B, max_dim) + low(B,max_dim)) / 2.0
    
    # custom intersection function that doesn't do constraint pruning
    ∩ = (set, lc) -> HPolytope([constraints_list(set); lc])
    
    subsets = Union{Nothing, HPolytope}[domain] # reach is the list of reachable set of each layer of the network.
    subsets = subsets .∩ [HalfSpace(a, b), HalfSpace(-a, -b)]
    
    return subsets
end

# Ai2h and Ai2z use affine_map
# Box uses approximate_affine_map for the linear region if it is propagating a zonotope
forward_linear(solver::Ai2h, L::Layer{ReLU}, input::AbstractPolytope) = affine_map(L, input)
forward_linear(solver::Ai2z, L::Layer{ReLU}, input::AbstractZonotope) = affine_map(L, input)
forward_linear(solver::Box, L::Layer{ReLU}, input::AbstractZonotope) = approximate_affine_map(L, input)
# method for Zonotope and Hyperrectangle, if the input set isn't a Zonotope overapproximate
forward_linear(solver::Union{Ai2z, Box}, L::Layer{ReLU}, input::AbstractPolytope) = forward_linear(solver, L, overapproximate(input, Hyperrectangle))

# Forward_act is different for Ai2h, Ai2z and Box
forward_act(solver::Ai2h, L::Layer{ReLU}, Ẑ::AbstractPolytope) = convex_hull(UnionSetArray(forward_partition(L.activation, Ẑ)))
forward_act(solver::Ai2z, L::Layer{ReLU}, Ẑ::AbstractZonotope) = overapproximate(Rectification(Ẑ), Zonotope)
forward_act(slver::Box, L::Layer{ReLU}, Ẑ::AbstractPolytope) = rectify(Ẑ)

# For ID activation do an affine map for all methods
forward_linear(solver::Ai2, L::Layer{Id}, input::AbstractPolytope) = affine_map(L, input)
forward_act(solver::Ai2, L::Layer{Id}, input::AbstractPolytope) = input

function convex_hull(U::UnionSetArray{<:Any, <:HPolytope})
    tohrep(VPolytope(LazySets.convex_hull(U)))
end
