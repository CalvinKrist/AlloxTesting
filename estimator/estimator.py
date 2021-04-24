import subprocess
import time_writter

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

# job: the job whose CPU and GPU time should be estimated
# a: the lower percent to use for the estimaton
# b: the upper percent to use for the estimation
def estimate_job_time_linreg(job, a, b):
	original_epochs = job.epochs
	job.epochs = round(job.epochs * a)

	###################################
	#####    ESTIMATE JOB TIME    #####
	###################################
	cpu_estimated = float('inf')
	gpu_estimated = float('inf')

	job.epochs = original_epochs
	job.cpu_compl_time = cpu_estimated
	job.gpu_compl_time = gpu_estimated
	job.cpu_err = 0.1
	job.gpu_err = 0.1

# job: the job whose CPU and GPU time should be estimated
# num_epochs: the number of epochs to run the estimation job for
def estimate_job_time_time_writter(job, num_epochs):
	original_epochs = job.epochs
	job.epochs = num_epochs

	times = []
	for _ in range(self.num_samples):
		output, error = jopCopy.run_cpu()
		measurements = time_writter.parse_output(output)
		times.append(sum(measurements))
		
	avgTime = sum(times) / len(times)
	cpu_estimated = avgTime / a * original_epochs

	times = []
	for _ in range(self.num_samples):
		output, error = jopCopy.run_gpu()
		measurements = time_writter.parse_output(output)
		times.append(sum(measurements))
		
	avgTime = sum(times) / len(times)
	gpu_estimated = avgTime / a * original_epochs

	job.epochs = original_epochs
	job.cpu_compl_time = cpu_estimated
	job.gpu_compl_time = gpu_estimated
	job.cpu_err = 0.1
	job.gpu_err = 0.1