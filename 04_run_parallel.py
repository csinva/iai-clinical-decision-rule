from slurmpy import Slurm

if __name__ == '__main__':
    partition = 'low'

    # sweep different ways to initialize weights
    params_to_vary = range(400)

    # run
    s = Slurm("iai classfy", {"partition": partition, "time": "3-0"})

    # iterate
    for i in params_to_vary:
        param_str = f'module load python; python3 04_classify_all.py {i}'
        s.run(param_str)
