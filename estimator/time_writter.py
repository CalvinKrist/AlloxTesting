import time

class TimeWritter:
	def __init__(self):
		self.times = []
		self.t1 = 0

	def start(self):
		self.t1 = time.perf_counter()

	def log_update(self):
		t2 = time.perf_counter()	
		self.times.append(t2 - self.t1)
		self.t1 = t2

	def print_results(self):
		print("<TIME_WRITTER_OUTPUT>" + str(self.times) + "</TIME_WRITTER_OUTPUT>")

writter = TimeWritter()

def Start():
	writter.start()

def LogUpdate():
	writter.log_update()

def PrintResults():
	writter.print_results()

def Reset():
	writter.t1 = 0
	writter.times = []

def parse_output(output):
	if "<TIME_WRITTER_OUTPUT>[" not in output or "]</TIME_WRITTER_OUTPUT>" not in output:
		raise Exception("Invalid time writter output: " + output)
	
	s = output.split("<TIME_WRITTER_OUTPUT>[")[1]
	s = s.split("]</TIME_WRITTER_OUTPUT>")[0]
	s = s.split(", ")
	s = list(float(i) for i in s)

	return s