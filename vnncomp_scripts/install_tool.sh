apt-get update -y
apt-get -y install sudo
sudo apt-get install wget -y
sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz
sudo tar -xvzf julia-1.6.1-linux-x86_64.tar.gz
sudo cp -r julia-1.6.1 /opt/
sudo ln -s /opt/julia-1.6.1/bin/julia /usr/local/bin/julia
sudo apt-get install build-essential -y
sudo apt-get install git -y
sudo apt-get install python3 -y
sudo apt-get install pip -y
sudo apt-get install psmisc

cd NeuralVerification.jl
echo '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.add("PyCall")
' | julia

chmod +x vnncomp_scripts/*.sh
pip install -r vnncomp_scripts/NNet/test_requirements.txt