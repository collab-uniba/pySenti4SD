import os

class CoreUtils():

    @staticmethod
    def check_jobs_number(jobs_number):
        max_jobs = os.cpu_count()
        if jobs_number > max_jobs:
            jobs_number = max_jobs
        elif jobs_number < 0:
            if jobs_number == -1:
                jobs_number = max_jobs
            elif jobs_number > -(max_jobs - 1):
                jobs_number = max_jobs + jobs_number
            else:
                jobs_number = 1
        return jobs_number
