from estimator import *

if __name__ == '__main__':
    job = GoogleNetJob("myJob", epochs=100, batch_size=10)
    estimator = Estimator(2, 1)

    #estimator.estimate_job_time(job)
    print(job)