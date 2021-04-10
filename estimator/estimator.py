import subprocess
import time_writter

class Job:
	def __init__(self, name, epochs):
		self.name = name
		self.epochs = epochs
		self.args = []

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

		return estimated