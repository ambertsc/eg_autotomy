import os

import argparse

def exp_0x000(num_workers):
    
    population_size = 64
    generations = 100
    performance_threshold = 36
    algo = "GeneticPopulation"
    policy = "MLPBodyPolicy"
    seeds = [1,2,3,4,5]


    for seed in seeds:
        for use_autotomy in [0,1]:

            exp_tag = f"{algo}_{policy}_p{population_size}_g{generations}_s{seed}_w{num_workers}_u{use_autotomy}"

            exp_cmd = f"python -m bevodevo.train -n BackAndForthEnv-v0 "\
                    f" -p {population_size} -a {algo} -pi {policy} "\
                    f" -g {generations} -x {exp_tag} -s {seed} -u {use_autotomy}"\
                    f" -w {num_workers}"

            print("begin experiment")
            print(exp_cmd)
            os.system(exp_cmd)

if __name__ == "__main__": #pragma: no cover

    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--num_workers", type=int, default=0, \
            help="number of workers to use for training")

    args = parser.parse_args()
    exp_0x000(args.num_workers)
