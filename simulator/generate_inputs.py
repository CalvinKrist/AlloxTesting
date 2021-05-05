from numpy.random import default_rng
import copy

def createInputFiles(seed,mu,sigma,scale,out_file,real_mean=0,real_std=0):
    number_of_queues = 10
    number_of_jobs = 1000

    # Arrival times
    with open("poissrnd.csv") as f:
        arrivals = f.readlines()
        arrivals = arrivals[0][:-1].split(",")
    #print((arrivals))
    
    arrival_idx = 0
    jobs_arrive_at_a_time = 5
    wait_jobs = 0
    arrival_time = 0
    
    queues = []
    jobs = genJobs(seed,number_of_jobs*number_of_queues,real_mean,real_std)
    err_rng = default_rng()
    cpu_errs = err_rng.normal(mu,sigma,size=len(jobs))
    gpu_errs = err_rng.normal(mu,sigma,size=len(jobs))

    for i,job in enumerate(jobs):
        cpu_err = cpu_errs[i]
        gpu_err = gpu_errs[i]
        job['cpu_err'] = cpu_err + scale
        job['gpu_err'] = gpu_err + scale

    for i in range(number_of_queues):
        queues.append(f"queue{i}")
    job_id = 0
    doc_string = ""
    for _ in range(number_of_jobs):
        for que in queues: 
            job = jobs[job_id]     
            doc_string += f"# {job_id}\n"
            doc_string += f"1 {job_id} 1 {arrival_time} {que}\n"
            doc_string += f"stage -1 {job['cpu']} {job['mem']} {job['cpu_compl']} {job['gpu']} {job['gpu_mem']} {job['gpu_compl']} 1 {job['cpu_err']} {job['gpu_err']}\n"
            doc_string += f"0\n"
            job_id+=1
            wait_jobs+=1
            if(wait_jobs == jobs_arrive_at_a_time):
                wait_jobs = 0
                arrival_idx = (arrival_idx+1) % len(arrivals)
                arrival_time+=int(arrivals[arrival_idx])
            
    with open(f"./input_files/{out_file}",'w') as f:
        f.write(doc_string)
    
"""
# 1
1 1 1 0 queue1
stage -1.0 32.0 1.0 90.0 1.0 2.0 26.0 1 0.1413438554619842 0.0707886234068464
0
# 2
1 2 1 0 queue2
stage -1.0 32.0 15.0 138.0 1.0 2.0 44.0 1 0.19966375411931717 0.10460691720908298
0
# 3
1 3 1 0 queue3
stage -1.0 32.0 16.0 160.0 1.0 2.0 43.0 1 0.08433035439793143 -0.04753371668553091
0
"""
def genJobs(seed,num,real_mean=0,real_std=0):
    
    jobs_possible = [{'cpu' : 32.0, 'mem' : 15.0, 'cpu_compl' : 50.0, 'gpu' : 1.0, 'gpu_mem': 2.0, 'gpu_compl' : 25.0},
                     {'cpu' : 32.0, 'mem' : 20.0, 'cpu_compl' : 150.0, 'gpu' : 1.0, 'gpu_mem': 2.0, 'gpu_compl' : 75.0},
                     {'cpu' : 32.0, 'mem' : 20.0, 'cpu_compl' : 250.0, 'gpu' : 1.0, 'gpu_mem': 2.0, 'gpu_compl' : 125.0}]
    
    if(real_mean != 0):
        jobs_possible = [{'cpu' : 32.0, 'mem' : 15.0, 'cpu_compl' : real_mean, 'gpu' : 1.0, 'gpu_mem': 2.0, 'gpu_compl' : real_mean/2}]
    rng = default_rng(seed=seed)
    jobs = []
    
    for _ in range(num):
        job = (copy.copy(jobs_possible[rng.integers(len(jobs_possible))-1]))
        if(real_std != 0):
            job['cpu_compl'] = rng.normal(real_mean,real_std)
            job['gpu_compl'] = job['cpu_compl']/2
        jobs.append(job)
    return jobs



if __name__ == '__main__':
    baseline_mean=178.7087746978
    baseline_std=1.4014685177968365
    est_mean=271.0005316963945
    est_std=18.425086031121456
    scale = (est_mean-baseline_mean)/baseline_mean
    sigma = est_std/est_mean
    print(f"sigma: {sigma} mu: {scale}")
    createInputFiles(seed=12345,mu=0,sigma=sigma,scale=scale,out_file="jobs_input_10_Google_debug.txt",real_mean=baseline_mean,real_std=baseline_std)
    """
    true mean
    
    est std
    est mean
    
    mu = (mean_e - mean_t)/(mean_t)
    sig = std
    """