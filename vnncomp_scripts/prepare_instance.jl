using PyCall
function onnx_to_nnet(onnx_file)
    pushfirst!(PyVector(pyimport("sys")."path"), "./")
    nnet = pyimport("NNet")
    use_gz = split(onnx_file, ".")[end] == "gz"
    if use_gz
        onnx_file = onnx_file[1:end-3]
    end
    nnet_file = onnx_file[1:end-4] * "nnet"
    isfile(nnet_file) && return
    nnet.onnx2nnet(onnx_file, nnetFile=nnet_file)
end

function main(args)
    onnx_to_nnet(args[1])
end
main(ARGS)