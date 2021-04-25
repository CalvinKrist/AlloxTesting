import subprocess
import time_writter
import os

CPU_TEMPLATE = ""
with open("cpu_slurm_template.sh") as f:
	CPU_TEMPLATE = f.read()
GPU_TEMPLATE = ""
with open("gpu_slurm_template.sh") as f:
	GPU_TEMPLATE = f.read()

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
		self.thread_count = 10
		self.cpu_compl_time = float('inf')
		self.cpu_err = 1.0

		self.gpu_count = 1
		self.gpu_mem = 2.0
		self.gpu_compl_time = float('inf')
		self.gpu_err = 1.0

	# This no longer runs anything, but not creates a slurm script that can be run later
	def run(self, template, hardware, job_name, output):
		if not os.path.exists('slurm_scripts'):
			os.makedirs('slurm_scripts')

		f = template + "\nsrun ./" + ' '.join(self.get_args(hardware))
		f = f.replace("<NAME>", job_name)
		f = f.replace("<OUTPUT>", output)

		with open('slurm_scripts/' + job_name, "w") as slurm:
			slurm.write(f)

	def run_cpu(self, job_name, output):
		self.run(CPU_TEMPLATE, "cpu", job_name, output)

	def run_gpu(self, job_name, output):
		self.run(GPU_TEMPLATE, "gpu", job_name, output)

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

# output: the output of the job whose CPU and GPU time should be estimated
# a: the lower percent to use for the estimaton
# b: the upper percent to use for the estimation
def estimate_job_time_linreg(output, a, b):
	measurements = time_writter.parse_output(output)
	
	estimated_cpu_time = 0
	estimated_gpu_time = 0

	return (estimated_cpu_time, estimated_gpu_time)

# job: the output of the job whose CPU and GPU time should be estimated
# prop: the proportion of epochs to run for
def estimate_job_time_writter(output, prop):
	measurements = time_writter.parse_output(output)
	
	estimated_cpu_time = 0
	estimated_gpu_time = 0

	return (estimated_cpu_time, estimated_gpu_time)