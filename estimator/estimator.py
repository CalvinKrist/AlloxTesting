import subprocess
import time_writter

class Job:
	JOB_ID = 0
	def __init__(self, name, epochs):
		self.name = name
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

	def run(self):
		process = subprocess.Popen(self.get_args(), stdout=subprocess.PIPE)
		print("Running job " + self.name)
		output, error = process.communicate()
		print(self.name + " executed: \n\toutput: " + str(output) + "\n\terror: " + str(error))
		return output, error

	def copy(self):
		raise Exception("Unsupported function 'copy' for class JOB")

	def get_args(self):
		raise Exception("Unsupported function 'get_args' for class JOB")


	def __str__(self):
		s = "# " + str(self.id) + "\n"
		s += "1 " + str(self.id) + " 1 ARRIVAL_TIME queue" + str(self.user) + "\n"
		s += "stage -1.0 " + str(self.thread_count) + " " + str(self.mem) + " " + str(self.cpu_compl_time) + " " + str(self.gpu_count) + " " + \
			str(self.gpu_mem) + " " + str(self.gpu_compl_time) + " 1 " + str(self.cpu_err) + " " + str(self.gpu_err) + "\n"
		s += "0 \n"
		return s

class GoogleNetJob(Job):
	def __init__(self, name, epochs=10, batch_size=100, learn_rate=0.1, keep_prob=0.5):
		Job.__init__(self, name, epochs)
		self.batch_size = batch_size
		self.learn_rate = learn_rate
		self.keep_prob = keep_prob

	def get_args(self):
		return ["jobs/googleNet/run.sh", str(self.learn_rate), str(self.batch_size), str(self.keep_prob), str(self.epochs)]

	def copy(self):
		return GoogleNetJob(self.name + " copy", self.epochs, self.batch_size, self.learn_rate, self.keep_prob)

class Estimator:
	def __init__(self, num_samples, num_epochs):
		self.num_samples = num_samples
		self.num_epochs = num_epochs

	def estimate_job_time(self, job):
		# Copy the job and change its parameters to estimation parameters
		jopCopy = job.copy()
		jopCopy.epochs = self.num_epochs

		times = []

		for _ in range(self.num_samples):
			output, error = jopCopy.run()
			measurements = time_writter.parse_output(output)
			times.append(sum(measurements))
			
		avgTime = sum(times) / len(times)
		print("Average time : " + str(avgTime) + " over " + str(self.num_epochs) + " epochs.")
		estimated = avgTime / self.num_epochs * job.epochs
		print("Estimated time : " + str(estimated) + " over " + str(job.epochs) + " epochs.")

		job.cpu_compl_time = estimated
		job.cpu_err = 0.1