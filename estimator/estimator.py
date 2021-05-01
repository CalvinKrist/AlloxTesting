import subprocess
import time_writter
import os
import numpy as np
import re

class Job:
	JOB_ID = 0
	def __init__(self, epochs):
		self.epochs = epochs
		self.args = []

		self.id = Job.JOB_ID
		Job.JOB_ID += 1

		# Configuration
		self.user = 1

		self.mem = 15
		self.thread_count = 8
		self.cpu_compl_time = float('inf')
		self.cpu_err = 1.0

		self.gpu_count = 1
		self.gpu_mem = 2.0
		self.gpu_compl_time = float('inf')
		self.gpu_err = 1.0

	# This no longer runs anything, but not creates a slurm script that can be run later
	def run(self, hardware, job_name, output):
		if not os.path.exists('slurm_scripts'):
			os.makedirs('slurm_scripts')

		f = '''
		#!/bin/bash
		set -ex
		git pull
		'''
		f += ' '.join(self.get_args(hardware))

		with open('slurm_scripts/' + job_name + ".sh", "w") as slurm:
			slurm.write(f)

		process = subprocess.run(["/bin/bash", 'slurm_scripts/' + job_name + ".sh"])

	def run_cpu(self, job_name, output):
		self.run("cpu", job_name, output)

	def run_gpu(self, job_name, output):
		self.run("gpu", job_name, output)

	def get_args(self, hardware):
		raise Exception("Unsupported function 'get_args' for class JOB")


	def __str__(self):
		s = "# " + str(self.id) + "\n"
		s += "1 " + str(self.id) + " 1 ARRIVAL_TIME queue" + str(self.user) + "\n"
		s += "stage -1.0 " + str(self.thread_count) + " " + str(self.mem) + " " + str(self.cpu_compl_time) + " " + str(self.gpu_count) + " " + \
			str(self.gpu_mem) + " " + str(self.gpu_compl_time) + " 1 " + str(self.cpu_err) + " " + str(self.gpu_err) + "\n"
		s += "0 \n"
		return s

class GoogleNetJob(Job):
	def __init__(self, epochs=500, batch_size=128, learn_rate=0.001, keep_prob=0.4):
		Job.__init__(self, epochs)
		self.batch_size = batch_size
		self.learn_rate = learn_rate
		self.keep_prob = keep_prob

	def get_args(self, hardware):
		if hardware == "cpu":
			return ["jobs/googleNet/run.sh", str(self.learn_rate), str(self.batch_size), str(self.keep_prob), str(self.epochs),
			       "--cpu", "--numThreads", str(self.thread_count)]
		return ["jobs/googleNet/run.sh", str(self.learn_rate), str(self.batch_size), str(self.keep_prob), 
		        str(self.epochs), "--gpu", "--numGPUs", str(self.gpu_count)]

class AlexNetJob(Job):
	def get_args(self, hardware):
		if hardware == "cpu":
			return ["jobs/AlexNet/run.sh", str(self.epochs), "--cpu", "--numThreads", str(self.thread_count)]
		return ["jobs/AlexNet/run.sh", str(self.epochs), "--gpu", "--numGPUs", str(self.gpu_count)]

class LeNetJob(Job):
	def get_args(self, hardware):
		if hardware == "cpu":
			return ["jobs/LeNet/run.sh", str(self.epochs), "--cpu", "--numThreads", str(self.thread_count)]
		return ["jobs/LeNet/run.sh", str(self.epochs), "--gpu", "--numGPUs", str(self.gpu_count)]


def estimate_job_time(tw_cpu_output, model_name):
    '''
	Parses output from experiments and returns estimated cpu time

    :param tw_cpu_output: single log file containing time writer output
    :param model_name: str, alexnet or lenet models
	:return cpu_estimated: float, estimated cpu time
    '''
    times = []
    if model_name=="lenet":
        times = parse_output_lenet(tw_cpu_output)
    if model_name=="alexnet":
        times = parse_output_alexnet(tw_cpu_output)
    # Find arguments
    if "<ARGS>[" not in tw_cpu_output or "]</ARGS>" not in tw_cpu_output:
        raise Exception("Arguments not included: ", tw_cpu_output)
    args = tw_cpu_output.split("<ARGS>[")[-1]
    args = args.split("]</ARGS>")[0]
    args = args.replace("'", '')
    args = args.split(", ")
    
    # process args to call either linear regression or timewritter
    linearReg = False
    timeWritter = False
    config = []
    for i in args:
        if "linearRegression" in i:
            linearReg = True
        if "timeWritter" in i:
            timeWritter = True
        if "config" in i:
            config.append(i)
    if not config:
        raise Exception("No config value specified in args")
    if not linearReg and not timeWritter:
        raise Exception("Linear Regression Estimation or timewritter estimation not specified in args")
    config_num = int(config[0].split("=")[1].split("'")[0])
    
    cpu_estimated = float('inf')
    # Call linear reg estimation or time writter estimation depending on config param
    if linearReg:
        if config_num==0:
            cpu_estimated = estimate_job_time_linreg(times, 0.006, 0.009, model_name)
            aggregate_results(cpu_estimated, "linReg", config_num, model_name)
        elif config_num==1:
            cpu_estimated = estimate_job_time_linreg(times, 0.004, 0.011, model_name)
            aggregate_results(cpu_estimated, "linReg", config_num, model_name)
        elif config_num==2:
            cpu_estimated = estimate_job_time_linreg(times, 0.002, 0.013, model_name)
            aggregate_results(cpu_estimated, "linReg", config_num, model_name)
        else:
            raise Exception("Invalid config: ", config)
    if timeWritter:
        if config_num==0:
            cpu_estimated = estimate_job_time_time_writter(times, 0.002, model_name)
            aggregate_results(cpu_estimated, "tw", config_num, model_name)
        elif config_num==2:
            cpu_estimated = estimate_job_time_time_writter(times, 0.008, model_name)
            aggregate_results(cpu_estimated, "tw", config_num, model_name)
        elif config_num==3:
            cpu_estimated = estimate_job_time_time_writter(times, 0.014, model_name)
            aggregate_results(cpu_estimated, "tw", config_num, model_name)
        else:
            raise Exception("Invalid config: ", config)
    return cpu_estimated

def estimate_job_time_linreg(times, a, b, model_name):
	
	###################################
	#  Linear Regression Estimation   # 
	###################################
    total_iterations = 0
    if model_name == "lenet":
        total_iterations = 30000
    else:
        total_iterations = 500

	# Using a and b proportions, find corresponding epoch number
    a_epochs = round(total_iterations * a)
    times_iters = times[1:-1]
    b_epochs = len(times_iters)

    a_epochs_times = 0
    b_epochs_times = 0
    for i in times_iters[:a_epochs]:
        a_epochs_times += i
    for j in times_iters:
        b_epochs_times += j
    a_epochs_times += times[0]
    
    b_epochs_times += times[-1]
    b_epochs_times += times[0]
    
    cpu_estimated = float('inf')
    
    slope = (b_epochs_times - a_epochs_times) / (b_epochs - a_epochs)
    b = a_epochs_times - (slope * a_epochs)
    
    cpu_estimated = (slope*total_iterations)+b
    return cpu_estimated

def estimate_job_time_time_writter(times, proportion, model_name):
	###################################
	#####    Detailed Estimator   #####
	###################################
    total_iterations = 0
    if model_name == "lenet":
        total_iterations = 30000
    else:
        total_iterations = 500
    cpu_estimated = float('inf')
    
    sum_times = 0
    for i in times[1:-1]:
        sum_times += i
    slope = sum_times/len(times[1:-1])
    b = times[0]+times[-1]
    
    cpu_estimated = (slope*total_iterations)+b
    return cpu_estimated