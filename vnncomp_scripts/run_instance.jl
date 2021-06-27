using NeuralVerification, LazySets
include("vnnlib_parser.jl")

function verify_an_instance(onnx_file, spec_file)
    use_gz = split(onnx_file, ".")[end] == "gz"
    nnet_file = use_gz ? onnx_file[1:end-7] * "nnet" : onnx_file[1:end-4] * "nnet"
    net = read_nnet(nnet_file)
    n_in = size(net.layers[1].weights)[2]
    n_out = length(net.layers[end].bias)
    
    specs = read_vnnlib_simple(spec_file, n_in, n_out)
    for spec in specs
        X_range, Y_cons = spec
        lb = [bd[1] for bd in X_range]
        ub = [bd[2] for bd in X_range]
        X = Hyperrectangle(low = lb, high = ub)
        res = nothing
        A = []
        b = []
        for Y_con in Y_cons
            A = hcat(Y_con[1]...)'
            b = Y_con[2]
            if length(b) > 1
                Y_adv = HPolytope(A, b)
                Y = Complement(Y_adv)
                solver = MIPVerify()
                prob = Problem(net, X, Y)
                res = solve(solver, prob)
            else
                Y = HPolytope(-A, -b)
                solver = ReluVal(max_iter=100)
                prob = Problem(net, X, Y)
                res = solve(solver, prob)[1]
            end
            
            res.status == :violated && (return "violated")
            res.status == :unknown && (return "unknown")
        end
    end
    return "holds"
end

function main(args)
    result = @timed verify_an_instance(args[1], args[2])
    open(args[3], "w") do io
       write(io, result.value)
    end
end
main(ARGS)