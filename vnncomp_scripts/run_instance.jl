using NeuralVerification, LazySets
function load_vnnlib_spec(spec_file)
    specs = []
    lines = readlines(spec_file)
    for line in lines
        pat = r" [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
        numbers = map(eachmatch(pat, line)) do m
            parse(Float64, m.match)
        end
        if length(numbers) < 3
            continue
        end
        append!(specs, [numbers])
    end
    print("[DONE] spec loaded")
    return specs
end
function verify_an_instance(onnx_file, spec_file)
    use_gz = split(onnx_file, ".")[end] == "gz"
    nnet_file = use_gz ? onnx_file[1:end-7] * "nnet" : onnx_file[1:end-4] * "nnet"
    net = read_nnet(nnet_file)
    specs = load_vnnlib_spec(spec_file)
    @show specs
    for i in 1:length(specs)
        leq = (length(specs) == 2 && i == 1) ? true : false

        X = Hyperrectangle(low = [specs[i][1]], high = [specs[i][2]])
        Y = leq ? HalfSpace([-1.], -specs[i][3]) : HalfSpace([1.], specs[i][3])

        solver = ReluVal(max_iter=100)
        prob = Problem(net, X, Y)
        res = solve(solver, prob)[1]
        
        @show res
        if res.status == :violated
            return "violated"
        end
        if res.status == :unknown
            return "unknown"
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