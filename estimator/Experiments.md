### Baselines

Each job is run in full (500 epochs for GoogleNet and AlexNet, 30000 for LeNet) 10 times on GPU and CPU each to get estimates for how long the jobs should take. The results are stored in `results/baselines/JOBTYPE_HARDWARE_X`, such as `results/baseline/googleNet_GPU_4`.

### Experiments

All experiment results are stored in the following filepath format: `results/ESTIMATION_TYPE/JOB_TYPE/config_i/HARDWAREy`, for example, `results/linearRegression/googleNet/config_2/GPU80`. The config types are as follows:

#### linearRegression

* (0.6, 0.9): `results/linearRegression/JOB/config_0`
* (0.4, 1.1): `results/linearRegression/JOB/config_1`
* (0.2, 1.3): `results/linearRegression/JOB/config_2`

#### timeWritter

* 0.002: `results/timeWritter/JOB/config_0`
* 0.004: `results/timeWritter/JOB/config_1`
* 0.008: `results/timeWritter/JOB/config_2`
* 0.014: `results/timeWritter/JOB/config_2`

The files are the output of the job being run. Because this includes the output of the `time_writter`, the file can be loaded in and parsed with the `time_writter` libraries to do all analysis. The jobs were run only as long as needed by that specific estimation strategy and configuration. For example, time writter config 2 (0.008) for LeNet would be run for 240 epochs.