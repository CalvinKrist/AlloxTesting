import subprocess
import json

################################
####    Submit baselines    ####
################################
count = 0
jobs = ["googleNet", "alexNet"]
for job in jobs:
	for i in range(10):
		count += 1
		job_name = job + "Baseline" + str(i)
		command = {"command" : ["/base.sh", "--cpu", job, "--id=" + str(i), "--baseline"]}
		proc = ["aws", "batch", "submit-job", "--job-name", job_name, "--job-queue", "cpu-queue", "--job-definition", "CPUBase", "--container-overrides", json.dumps(command)]
		process = subprocess.Popen(proc, stdout=subprocess.PIPE)
		output, error = process.communicate()

methods = ["linearRegression", "timeWritter"]
configCount = {"timeWritter" : 4, "linearRegression" : 3}
for job in jobs:
	for method in methods:
		for config in range(configCount[method]):
			for i in range(100):
				#count += 1
				'''job_name = job + "_" + method + "_" + str(config) + "_" + str(i)
				command = {"command" : ["/base.sh", "--cpu", job, "--id=" + str(i), method, "--config=" + str(config)]}
				proc = ["aws", "batch", "submit-job", "--job-name", job_name, "--job-queue", "cpu-queue", "--job-definition", "CPUBase", "--container-overrides", json.dumps(command)]
				process = subprocess.Popen(proc, stdout=subprocess.PIPE)
				output, error = process.communicate()'''
print(count)
#aws batch submit-job --job-name myJob --job-queue cpu-queue --job-definition CPUBase --container-overrides '{"command":["/base.sh","--cpu","leNet"]}'