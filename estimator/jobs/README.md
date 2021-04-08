### What jobs

From the paper:
```
Each user has 10 popular Tensorflow jobs, e.g., Googlenet, Lenet, and Alexnet. The job configurations such as batch sizes and batch numbers are different, resulting in the speedup rates of using one GPU versus one CPU ranging from 1.8 to 10. For jobs on CPU, the number of threads is set at 19 to best utilize the virtual cores while leaving one core for other services on each node.
```

Jobs added:
* Goognelet on the CIFAR dataset

### Running Jobs

* Within each job directory, there is a script called `run.sh`. This can then be called to run the job, or provide a template for how a job should be run.