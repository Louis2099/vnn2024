apt-get update -y
apt-get install wget -y
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz
tar -xvzf julia-1.6.1-linux-x86_64.tar.gz
cp -r julia-1.6.1 /opt/
ln -s /opt/julia-1.6.0/bin/julia /usr/local/bin/julia
export PATH="$PATH:/opt/julia-1.6.0/bin/julia"
apt-get install build-essential -y
apt-get install git -y
apt-get install python3 -y
apt-get install pip -y

cd NeuralVerification.jl
echo '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.add("PyCall")
' | julia

chmod +x vnncomp_scripts/*.sh
pip install -r vnncomp_scripts/NNet/test_requirements.txt