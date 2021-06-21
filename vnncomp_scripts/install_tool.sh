sudo apt update -y
sudo apt wget -y
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz
tar -xvzf julia-1.6.1-linux-x86_64.tar.gz
sudo cp -r julia-1.6.1 /opt/
sudo ln -s /opt/julia-1.6.0/bin/julia /usr/local/bin/julia
sudo apt install build-essential -y
sudo apt install git -y
sudo apt install python3 -y
sudo apt install pip -y


https://github.com/intelligent-control-lab/ComposableNeuralVerification.jl.git
cd ComposableNeuralVerification.jl
git checkout vnncomp
echo '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.add("PyCall")
ENV["PYTHON"]="/usr/bin/python3"
Pkg.build("PyCall")
' | julia

chmod +x vnncomp_scripts/*.sh
pip install -r vnncomp_scripts/NNet/test_requirements.txt
