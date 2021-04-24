from estimator import *
import os
import json
import sys

def calc_baselines():
	# Run each job in full 10 times, record the runtime
	for i in range(10):
		googleNet = GoogleNetJob(epochs=500)
		alexNet = AlexNetJob(epochs=500)
		leNet = LeNetJob(epochs=30000)

		if "--cpu" in sys.argv:
			googleNet.run_cpu("googleNet_CPU_" + str(i), "results/baselines/googleNet_CPUbaseline_" + str(i))
			alexNet.run_cpu("alexNet_GPU_" + str(i), "results/baselines/alexNet_GPUbaseline_" + str(i))
			leNet.run_cpu("leNet_CPU_" + str(i), "results/baselines/leNet_CPUbaseline_" + str(i))

		if "--gpu" in sys.argv:
			alexNet.run_gpu("alexNet_CPU_" + str(i), "results/baselines/alexNet_CPUbaseline_" + str(i))
			googleNet.run_gpu("googleNet_GPU_" + str(i), "results/baselines/googleNet_GPUbaseline_" + str(i))
			leNet.run_gpu("leNet_GPU_" + str(i), "results/baselines/leNet_GPUbaseline_" + str(i))

if __name__ == '__main__':
	##############################################
	####    Get baselines and save to JSON    ####
	##############################################

	if not os.path.exists('results'):
		os.makedirs('results')
	if not os.path.exists('results/baselines'):
		os.makedirs('results/baselines')

	if "--baselines" in sys.argv:
		calc_baselines()

	###############################
	####    Run experiments    ####
	###############################
	estimation_methods = ["linearRegression", "timeWritter"]
	if "linearRegression" not in sys.argv:
		estimation_methods.remove("linearRegression")
	if "timeWritter" not in sys.argv:
		estimation_methods.remove("timeWritter")

	for estimation_name in estimation_methods:
		if not os.path.exists('results/' + estimation_name):
			os.makedirs('results/' + estimation_name)

		job_types = {"googleNet" : GoogleNetJob, "alexNet" : AlexNetJob, "leNet" : LeNetJob}
		if "googleNet" not in sys.argv:
			del job_types["googleNet"]
		if "alexNet" not in sys.argv:
			del job_types["alexNet"]
		if "leNet" not in sys.argv:
			del job_types["leNet"]

		configurations = {"linearRegression" : [(0.6, 0.9), (0.4, 1.1), (0.2, 1.3)], "timeWritter" : [0.002, 0.004, 0.008, 0.014]}

		for job_type, job_class in job_types.items():
			if not os.path.exists('results/' + estimation_name + '/' + job_type):
				os.makedirs('results/' + estimation_name + '/' + job_type)

			for i, config in enumerate(configurations[estimation_name]):
				if not os.path.exists('results/' + estimation_name + '/' + job_type + "/config_" + str(i)):
					os.makedirs('results/' + estimation_name +'/' + job_type + "/config_" + str(i))
				file_path = 'results/' + estimation_name + '/' + job_type + "/config_" + str(i)

				# Estimate each job type at each configuration at each estimation method 100 times
				for itr in range(100):

					if job_type == "leNet":
						if estimation_name == "linearRegression":
							job = job_class(epochs=round(30000*config[1]))
						else:
							job = job_class(epochs=round(30000*config))
					else:
						if estimation_name == "linearRegression":
							job = job_class(epochs=round(500*config[1]))
						else:
							job = job_class(epochs=round(500*config))
					
					if "--cpu" in sys.argv:
						job.run_cpu(job_type + "_" + estimation_name + "_" + str(i) + "_" + str(itr), file_path + "/cpu" + str(itr))
					if "--gpu" in sys.argv:
						job.run_gpu(job_type + "_" + estimation_name + "_" + str(i) + "_" + str(itr), file_path + "/gpu" + str(itr))