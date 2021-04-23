### Setting up the Environment

* Install the datasets without using sudo:
```
mkdir data
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --output data/cifar.tar.gz
tar -xvf data/cifar.tar.gz -C data/.
rm data/cifar.tar.gz
```
* Set up python environment using `virtualenv`
```bash
module load python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy
pip install scipy
pip install tensorflow==1.15
pip install tensorflow-gpu==1.15
pip install scikit-image
pip install pandas==0.24
```

* Modify scrpts
	* Some scripts like `jobs/googleNet/run.sh` might specify `python3.7`. However, within the virtual environment, this type of version specification will attempt to use the system python installations instead of the virtual environment ones, and you will likely get errors. It must be modifed to use `python` instead of `python3` or `python3.7`.n

* To get GPU info run `srun --partition=gpu lshw -C display`
* Verify that Tensorflow can detect GPUs with `srun --partition=gpu ./gpuTest.sh`. You should see output that lists various GPUs
* Do a GPU test with:
```bash
cd jobs/googleNet
srun --partition=gpu ./run.sh 0.05 10 0.9 1 --gpu --numGPUs 1
```