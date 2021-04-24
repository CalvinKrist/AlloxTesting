from estimator import *
import os
import time
import json

def get_baselines():
	print("Calculating baselines")
	# Run each job in full 10 times, record the runtime
	baselines = {"googleNet" : {"cpu" : [], "gpu" : []}, "alexNet" : {"cpu" : [], "gpu" : []}, "leNet" : {"cpu" : [], "gpu" : []}}
	for i in range(10):
		googleNet = GoogleNetJob(epochs=500)
		alexNet = AlexNetJob(epochs=500)
		leNet = LeNetJob(epochs=30000)

		print("Calculating googleNet baseline " + str(i))
		start = time.perf_counter()
		googleNet.run_cpu()
		end = time.perf_counter()
		baselines["googleNet"]["cpu"].append(end - start)

		start = time.perf_counter()
		googleNet.run_gpu()
		end = time.perf_counter()
		baselines["googleNet"]["gpu"].append(end - start)

		print("Calculating alexNet baseline " + str(i))
		start = time.perf_counter()
		alexNet.run_cpu()
		end = time.perf_counter()
		baselines["alexNet"]["cpu"].append(end - start)

		start = time.perf_counter()
		alexNet.run_gpu()
		end = time.perf_counter()
		baselines["alexNet"]["gpu"].append(end - start)

		print("Calculating leNet baseline " + str(i))
		start = time.perf_counter()
		leNet.run_cpu()
		end = time.perf_counter()
		baselines["leNet"]["cpu"].append(end - start)

		start = time.perf_counter()
		leNet.run_gpu()
		end = time.perf_counter()
		baselines["leNet"]["gpu"].append(end - start)

	return baselines

if __name__ == '__main__':
	##############################################
	####    Get baselines and save to JSON    ####
	##############################################

	if not os.path.exists('results'):
		os.makedirs('results')

	baselines = get_baselines()
	with open('results/baselines.json', 'w') as outfile:
		json.dump(baselines, outfile)

	###############################
	####    Run experiments    ####
	###############################
	estimation_methods = ["linearRegression", "timeWritter"]
	for estimation_name in estimation_methods:
		if not os.path.exists('results/' + estimation_name):
			os.makedirs('results/' + estimation_name)

		job_types = {"googleNet" : GoogleNetJob, "alexNet" : AlexNetJob, "leNet" : LeNetJob}
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
					print("Estimating " + file_path)

					if job_type == "leNet":
						job = job_class(epochs=30000)
					else:
						job = job_class(epochs=500)
					
					if estimation_name == "linearRegression":
						estimate_job_time_linreg(job, config[0], config[1])
					elif estimation_name == "timeWritter":
						estimate_job_time_writter(job, config)
		
					jsonStr = json.dumps(job.__dict__)
					file_name = file_path + "/estimation_" + str(itr) + ".json"
					with open(file_name, "w") as f:
						f.write(jsonStr)