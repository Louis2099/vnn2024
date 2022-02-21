@with_kw struct NNDynTrack <: Solver
    optimizer = CPLEX.Optimizer
end

function solve(solver::NNDynTrack, problem::TrackingProblem, start_values=nothing; u_ref = nothing, xu_init = nothing)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-8)
    set_optimizer_attribute(model, "CPX_PARAM_EPAGAP", 1e-6)
    
    z = init_vars(model, problem.network, :z, with_input=true)

    δ = init_vars(model, problem.network, :δ, binary=true)
    # get the pre-activation bounds:
    model[:bounds] = get_bounds(problem.network, problem.input, false)
    model[:before_act] = true

    add_set_constraint!(model, problem.input, first(z))
    add_set_constraint!(model, problem.output, last(z))
    encode_network!(model, problem.network, BoundedMixedIntegerLP())

    
    # o = symbolic_abs((last(z) - problem.output_ref).*problem.output_cost)
    d = (last(z) - problem.output_ref)
    o = isnothing(u_ref) ? dot(d, d.*problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    
    @objective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)

    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)

    if termination_status(model) == OPTIMAL
        return TrackingResult(:holds, value(first(z)), objective_value(model)), value.(all_variables(model))
    else
        # @show termination_status(model)
        return TrackingResult(:violated), start_values
    end
end
