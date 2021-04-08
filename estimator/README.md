### Setup

#### Requirements:

* Python 3.3+ < Python 3.8
* TensorFlow 1.9+, but NOT tensorflow 2
* Numpy
* Scipy
* SK-Image

Then run `./setup.sh` which will download all the datasets required and attempt to install all python packages.

### How it Works

From the paper:
```
We run a small sampling job for each real job to obtain the parameters for both CPU and GPU configurations, as we discussed in Section 4.1. The total overhead of sampling jobs is 3% of the real jobs.
```

The estimator consists of two parts:
* `estimator.py`, which is the actual linear estimator. 
* `main.py`, which uses the estimator to run experiments.