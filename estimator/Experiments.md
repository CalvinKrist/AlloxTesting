## Baselines

Each job is un in full (500 epochs for GoogleNet and AlexNet, 30000 for LeNet) 10 times on GPU and CPU each to get estimates for how long the jobs should take. The results are stored in `results/baselines.json` in the following format: 

```
{
	"googleNet": {
		"cpu" : [t1, t2, ..., t10],
		"gpu" : [t1, t2, ..., t10]
	}, 
	"alexNet": {
		"cpu" : [t1, t2, ..., t10],
		"gpu" : [t1, t2, ..., t10]
	}, 
	"leNet": {
		"cpu" : [t1, t2, ..., t10],
		"gpu" : [t1, t2, ..., t10]
	}, 
}
``` 

## Linear Regression

All of the linear regression results are stored in `results/linearRegression`, with further subdirectories for each job type, ie `results/linearRegression/googleNet`. Within each job type, there are again subdirectories for each configuration:

* (0.6, 0.9): `results/linearRegression/JOB/config_0`
* (0.4, 1.1): `results/linearRegression/JOB/config_1`
* (0.2, 1.3): `results/linearRegression/JOB/config_2`

For each configuration for each job, each job type was estimated 100 times with for CPU and GPU esimations. These estimations are saved as `estimation_i.json`. The format of these files is JSON, and were created by writting to file `json.dumps(job.__dict__)`, which stores a dictionary representation of the estimated job. Thus, it can be loaded in to a dictionary in Python and the meaningful properties -- `cpu_compl_time` and `gpu_compl_time` -- can be easily accessed.

## Time Writter

All of the linear regression results are stored in `results/timeWritter`, with further subdirectories for each job type, ie `results/timeWritter/googleNet`. Within each job type, there are again subdirectories for each configuration:

* 0.002: `results/timeWritter/JOB/config_0`
* 0.004: `results/timeWritter/JOB/config_1`
* 0.008: `results/timeWritter/JOB/config_2`
* 0.014: `results/timeWritter/JOB/config_2`

For each configuration for each job, each job type was estimated 100 times with for CPU and GPU esimations. These estimations are saved as `estimation_i.json`. The format of these files is JSON, and were created by writting to file `json.dumps(job.__dict__)`, which stores a dictionary representation of the estimated job. Thus, it can be loaded in to a dictionary in Python and the meaningful properties -- `cpu_compl_time` and `gpu_compl_time` -- can be easily accessed.