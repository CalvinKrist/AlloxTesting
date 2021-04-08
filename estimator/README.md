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

### Estimator

`estimator.py` defines two types of classes: *Jobs* and the *Estimator*. There is a job superclass called *Job*, and then specific job types (like GoogleNet) are children of that class. The *Job* superclass contains very basic information that is shared across all jobs, and knows how to run itself and how to retrieve the output and error. Subclasses define how to copy themselves (needed by the *Estimator*), any additional parameters they need, and the actual arguments needed to run the job.

The estimator currently runs a specific number of samples of the job at a specific number of epochs. It then finds the average time and uses simple fractions to determine how long the job woul take to run given more iterations. 

#### Adding a new Job

1. Get the model and put it in a new directory.
2. Create a `run.sh` script that can train the model based on given parameters. See `jobs/googleNet/run.sh` for an example.
3. Create a new subclass of `Job` and implement the `copy` and `get_args` functions.