import subprocess
import time_writter
from sklearn.linear_model import LinearRegression
import csv 
import numpy as np
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
		self.thread_count = 16
		self.cpu_compl_time = float('inf')
		self.cpu_err = 1.0

		self.gpu_count = 1
		self.gpu_mem = 2.0
		self.gpu_compl_time = float('inf')
		self.gpu_err = 1.0

	def run_cpu(self):
		process = subprocess.Popen(self.get_args("cpu"), stdout=subprocess.PIPE)
		output, error = process.communicate()
		return output, error

	def run_gpu(self):
		process = subprocess.Popen(self.get_args("gpu"), stdout=subprocess.PIPE)
		output, error = process.communicate()
		return output, error

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
		if hardware is "cpu":
			return ["jobs/googleNet/run.sh", str(self.learn_rate), str(self.batch_size), str(self.keep_prob), str(self.epochs),
			       "--cpu", "--numThreads", str(self.thread_count)]
		return ["jobs/googleNet/run.sh", str(self.learn_rate), str(self.batch_size), str(self.keep_prob), 
		        str(self.epochs), "--gpu", "--numGPUs", str(self.gpu_count)]


def estimate_job_time(tw_cpu_output):
	with open(tw_cpu_output) as csv_file:
		csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONE)
		output = []
		args = []
		num_epochs = []
		for row in csv_reader:
			output.append(row)
			for field in row:
				if field.startswith('"<ARGS>'):
					args.append(row)
				if field.startswith("MAX_ITER"):
					num_epochs.append(row)
	# find number of epochs
	punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
	epochs_proc = ""
	for char in num_epochs[0][0]:
		if char not in punctuations:
			epochs_proc = epochs_proc + char
	epochs = [int(s) for s in epochs_proc.split() if s.isdigit()][0]
	# process args to call either linear regression or timewritter
	linearReg = False
	timeWritter = False
	config = []
	for i in args[0]:
		if "linearRegression" in i:
			linearReg = True
		if "timeWritter" in i:
			timeWritter = True
		if "config" in i:
			config.append(i)
	config_num = int(config[0].split("=")[1].split("'")[0])
	cpu_estimated = float('inf')
	# Call linear reg estimation or time writter estimation depending on config param
	if linearReg:
		if config_num==0:
			cpu_estimated = estimate_job_time_linreg(tw_cpu_output, 0.6, 0.9, epochs)
		if config_num==1:
			cpu_estimated = estimate_job_time_linreg(tw_cpu_output, 0.4, 1.1, epochs)
		if config_num==2:
			cpu_estimated = estimate_job_time_linreg(tw_cpu_output, 0.2, 1.3, epochs)
	if timeWritter:
		if config_num==0:
			cpu_estimated = estimate_job_time_time_writter(tw_cpu_output, 0.002, epochs)
		if config_num==1:
			cpu_estimated = estimate_job_time_time_writter(tw_cpu_output, 0.004, epochs)
		if config_num==2:
			cpu_estimated = estimate_job_time_time_writter(tw_cpu_output, 0.008, epochs)
		if config_num==3:
			cpu_estimated = estimate_job_time_time_writter(tw_cpu_output, 0.014, epochs)
	return cpu_estimated
		

# job: the job whose CPU and GPU time should be estimated
# a: the lower percent to use for the estimaton
# b: the upper percent to use for the estimation
def estimate_job_time_linreg(tw_cpu_output, a, b, num_epochs):
	###################################
	#####    ESTIMATE JOB TIME    #####
	###################################

	# We want to use the list of times to build a linear regression 
	#   model to predict the entire job completion time. i.e. if we 
	#   train for job.epochs amount. The independent variable is the 
	#   number of epochs, and the dependent variable is the job 
	#   completion time at job.epochs amount. 

	a_epochs = round(num_epochs * a)
	b_epochs = round(num_epochs * b)

	with open(tw_cpu_output, "r") as f:
		lines = f.read()
		# omit the first and last loading/writing values
		epoch_times = time_writter.parse_output(lines)[1:-1]

	a_time = epoch_times[a_epochs]
	b_time = epoch_times[b_epochs]

	cpu_estimated = float('inf')

	X_cpu = np.array([a_epochs, b_epochs]).reshape(2,1)
	y_cpu = [a_time, b_time]
	reg = LinearRegression().fit(X_cpu, y_cpu)
	cpu_estimated = reg.predict([[original_epochs]])[0]

	return cpu_estimated

# job: the job whose CPU and GPU time should be estimated
# num_epochs: the number of epochs to run the estimation job for
def estimate_job_time_time_writter(tw_cpu_output, proportion, num_epochs):

	###################################
	#####    ESTIMATE JOB TIME    #####
	###################################

	cpu_estimated = float('inf')
	# gpu_estimated = float('inf')

	proportion_epoch = round(num_epochs * proportion)

	with open(tw_cpu_output, "r") as f:
		lines = f.read()
		# omit the first and last loading/writing values
		epoch_times = time_writter.parse_output(lines)[1:-1]
	first_epoch_time = epoch_times[0]
	proportion_epoch_time = epoch_times[proportion_epoch]

	X_cpu = np.array([1, proportion_epochs]).reshape(2,1)
	y_cpu = [first_epoch_time, proportion_epoch_time]

	reg = LinearRegression().fit(X_cpu, y_cpu)
	cpu_estimated = reg.predict([[last_epoch_time]])[0]

	return cpu_estimated