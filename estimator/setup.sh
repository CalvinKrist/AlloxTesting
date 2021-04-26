set -x

apt remove python3
apt update
apt install python3 python3-pip software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt install python3.7
python3.7 -m pip install --upgrade pip
python3.7 -m pip install numpy
python3.7 -m pip install scipy
python3.7 -m pip install tensorflow==1.15
python3.7 -m pip install scikit-image
python3.7 -m pip install scikit-learn


mkdir data
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --output data/cifar.tar.gz
tar -xvf data/cifar.tar.gz -C data/.
rm data/cifar.tar.gz