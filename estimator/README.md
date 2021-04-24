### Setup

#### Requirements:

* Python 3.7
* TensorFlow 1.9+, but NOT tensorflow 2
* Numpy
* Scipy
* SK-Image

Then run `./setup.sh` which will download all the datasets required and attempt to install all python packages.

### How it Works

From the paper:
```
The completion time on each resource is linearly estimated based the two small samples of the job. Totally, there are four samples for each job on CPU and GPU. In our experiment, the length of the sample jobs is 3% of the real jobs.... The mean absolute error is 8% and the standard deviation is 11%.
```

The estimator consists of two parts:
* `estimator.py`, which is the actual linear estimator. 
* `main.py`, which uses the estimator to run experiments.

Additionally, there is a utility class called `TimeWritter` which is used to help measure how long each phase of a machine learning model takes to run. See [Adding a new Job](#adding-a-new-job) for information on how to integrate it to a job, and see [Time Writter](#time-writter) for informaiton on how it works and the design.

### Estimator

`estimator.py` defines two types of classes: *Jobs* and the *Estimator*. There is a job superclass called *Job*, and then specific job types (like GoogleNet) are children of that class. The *Job* superclass contains very basic information that is shared across all jobs, and knows how to run itself and how to retrieve the output and error. Subclasses define how to copy themselves (needed by the *Estimator*), any additional parameters they need, and the actual arguments needed to run the job.

The estimator currently runs a specific number of samples of the job at a specific number of epochs. It then finds the average time and uses simple fractions to determine how long the job woul take to run given more iterations. 

#### Adding a new Job

1. Get the model and put it in a new directory.
2. Create a `run.sh` script that can train the model based on given parameters. See `jobs/googleNet/run.sh` for an example.
3. Create a new subclass of `Job` and implement the `copy` and `get_args` functions.
4. Integrate the `time_writter` into the job. See [jobs/googleNet/examples/inception_cifar.py](jobs/googleNet/examples/inception_cifar.py) for an example. Note the following in particular:
	* `time_writter.Start()` is called right at the start of the `train` function
	* `time_writter.LogUpdate()` is called at the start of each epoch
	* `time_writter.LogUpdate()` is called after the last epoch
	* `time_writter.LogUpdate()` and `time_writter.PrintResults()` are called at the very end of the `train` function
	* Because the `time_writter` is in a different module, it must be imported in a special manner. This is done around line 20 of [jobs/googleNet/examples/inception_cifar.py](jobs/googleNet/examples/inception_cifar.py).

### Time Writter

The `TimeWritter` is used to track timing within jobs. It uses a singleton design pattern, where the module has a single instance of the `TimeWritter` class and exposes functions to use it. The `TimeWritter` class can be started, and then every time `LogUpdate()` is called it marks how long it has been since the writter was started or `LogUpdate` was last called. It can output the results of this logging, and additionally the module has a function to parse this output.

Thus, by using the `time_writter` module as described in [Adding a new Job](#adding-a-new-job), when the job is done there will be output like such: 

`<TIME_WRITTER_OUTPUT>[13.973396399989724, 8.200004231184721e-06, 2.094237900004373]</TIME_WRITTER_OUTPUT>`

Assuming a batch / epoch-based job, this shows that initial construction of the model and loading of TensorFlow took about 14 seconds, then a single epoch took 8.2e-06 seconds, and then saving the model to a file took about 2 seconds. Passing the output of the job (including the above string) to `time_writter.parse_output(output)` will return the float list of `[13.973396399989724, 8.200004231184721e-06, 2.094237900004373]`, which can then be summed to find the total job time or analyzed using knowledge of the job structure for a more accurate analysis. For example, summing the first and last elements of the list gives a y intercept and the remaining elements will give the slope. 

### Job Hardware Support

The `run.sh` of a job can be used to control the hardware it is run on. You can run it on the gpu with `--gpu --numGPUs NUM_GPUS` at the end of the normmal `run.sh` command, or you can run it on the cpu with `--cpu --numThreads NUM_THREADS` in the same way. 

Different from AlloX, we do not support configuring CPU or GPU memory constriants. This is because Tensorfloe 1.15, the version needed for all the machine learning jobs, does not support limiting the memory for the CPU, although it does allow this for the GPU. Additionally, we do not support running on multiple CPUs because to do so in any meaningful way requires a parameter server.