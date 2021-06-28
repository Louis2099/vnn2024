apt-get update -y
apt-get install sudo -y
sudo apt-get install software-properties-common -y
sudo apt-add-repository universe -y
sudo apt-get update -y
sudo apt-get install wget -y
sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz
sudo tar -xvzf julia-1.6.1-linux-x86_64.tar.gz
sudo cp -r julia-1.6.1 /opt/
sudo ln -s /opt/julia-1.6.1/bin/julia /usr/local/bin/julia
sudo apt-get install build-essential -y
sudo apt-get install git -y
sudo apt-get install python3 -y
sudo apt-get install python3-pip -y
sudo apt-get install psmisc
source ~/.bashrc

script_name=$0
script_path=$(dirname "$0")
project_path=$(dirname "$script_path")

cd $project_path
echo '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.add("PyCall")
' | julia

script_name=$0
script_path=$(dirname "$0")
chmod +x ${script_path}/*.sh
pip3 install -r "${script_path}/NNet/test_requirements.txt"