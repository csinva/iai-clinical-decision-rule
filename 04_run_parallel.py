from slurmpy import Slurm

partition = 'low'

# sweep different ways to initialize weights
params_to_vary = range(300)

# run
s = Slurm("iai classfy", {"partition": partition, "time": "1-0"})

# iterate
for i in params_to_vary:
    param_str = f'module load python; python3 04_classify_all.py {i}'
    s.run(param_str)
