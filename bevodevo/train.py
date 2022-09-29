import os
import sys
import argparse
import subprocess

import torch
import numpy as np
import time

import gym
import pybullet
import pybullet_envs


from mpi4py import MPI
comm = MPI.COMM_WORLD

from bevodevo.policies.rnns import GatedRNNPolicy
from bevodevo.policies.mlps import MLPPolicy,\
        HebbianMLP, ABCHebbianMLP
from bevodevo.policies.body_mlps import MLPBodyPolicy,\
        HebbianMLPBody, ABCHebbianMLPBody

from bevodevo.algos.es import ESPopulation
from bevodevo.algos.cmaes import CMAESPopulation
from bevodevo.algos.pges import PGESPopulation
from bevodevo.algos.nes import NESPopulation
from bevodevo.algos.ga import GeneticPopulation
from bevodevo.algos.random_search import RandomSearch

import eg_envs

# TODO: reminder to implement RL baselines 
#from bevodevo.algos.vpg import VanillaPolicyGradient
#from bevodevo.algos.dqn import DQN


def train(**kwargs):
    
    if "gatedrnn" in kwargs["policy"].lower():
        policy_fn = GatedRNNPolicy
        kwargs["policy"] = "GatedRNNPolicy" 
    elif "impala" in kwargs["policy"].lower():
        policy_fn = ImpalaCNNPolicy
        kwargs["policy"] = "ImpalaCNNPolicy"
    elif "cppnmlp" in kwargs["policy"].lower():
        policy_fn = CPPNMLPPolicy
        kwargs["policy"] = "CPPNMLPPolicy"
    elif "abchebbianmlpbody" in kwargs["policy"].lower():
        policy_fn = ABCHebbianMLPBody
        kwargs["policy"] = "ABCHebbianMLPBody"
    elif "abchebbianmlp" in kwargs["policy"].lower():
        policy_fn = ABCHebbianMLP
        kwargs["policy"] = "ABCHebbianMLP"
    elif "cppnhebbianmlp" in kwargs["policy"].lower():
        policy_fn = CPPNHebbianMLP
        kwargs["policy"] = "CPPNHebbianMLP"
    elif "hebbiancamlp2" in kwargs["policy"].lower():
        policy_fn = HebbianCAMLP2
    elif "hebbiancamlp" in kwargs["policy"].lower():
        policy_fn = HebbianCAMLP
    elif "hebbianmlpbody" in kwargs["policy"].lower():
        policy_fn = HebbianMLPBody
        kwargs["policy"] = "HebbianMLPBody"
    elif "hebbianmlp" in kwargs["policy"].lower():
        policy_fn = HebbianMLP
        kwargs["policy"] = "HebbianMLP"
    elif "mlpbodypolicy" in kwargs["policy"].lower():
        policy_fn = MLPBodyPolicy
        kwargs["policy"] = "MLPBodyPolicy"
    elif "mlppolicy" in kwargs["policy"].lower():
        policy_fn = MLPPolicy
        kwargs["policy"] = "MLPPolicy"
    else:
        assert False, "policy not found, check spelling?"

    if "ESPopulation" == kwargs["algorithm"]:
        population_fn = ESPopulation
    elif "CMAESPopulation" == kwargs["algorithm"]:
        population_fn = CMAESPopulation
    elif "Genetic" in kwargs["algorithm"]:
        population_fn = GeneticPopulation
    elif "PGES" in kwargs["algorithm"]:
        population_fn = PGESPopulation
    elif "NES" in kwargs["algorithm"]:
        population_fn = NESPopulation
    elif "dqn" in kwargs["algorithm"]:
        population_fn = DQN
    elif "vpg" in kwargs["algorithm"].lower():
        population_fn = VanillaPolicyGradient
    elif "andom" in kwargs["algorithm"]:
        population_fn = RandomSearch
    else:
        assert False, "population algo not found, check spelling?"

    num_workers = kwargs["num_workers"]

    if "use_autotomy" in kwargs.keys():
        kwargs["allow_autotomy"] = kwargs["use_autotomy"]
    else:
        kwargs["allow_autotomy"] = 0

    population = population_fn(policy_fn, **kwargs)
    
    population.train(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment parameters")

    parser.add_argument("-a", "--algorithm", type=str,\
            help="name of es learning algo", default="ESPopulation")
    parser.add_argument("-b", "--body_dim", type=int,\
            help="body dim", \
            default=8)
    parser.add_argument("-g", "--generations", type=int,\
            help="number of generations", default=50)
    parser.add_argument("-m", "--mode", default=0,\
            help="mode (0,1,2, or 3) for body co-evolution")
    parser.add_argument("-n", "--env_name", type=str, \
            help="name of environemt", default="InvertedPendulumBulletEnv-v0")
    parser.add_argument("-o", "--goal", type=int, nargs="+", default=[48, 16],\
            help="displacement objectives: forward (g[0]) and reverse (g[1])")
    parser.add_argument("-p", "--population_size", type=int,\
            help="number of individuals in population", default=64)
    parser.add_argument("-pi", "--policy", type=str,\
            help="name of policy architecture", default="MLPPolicy")
    parser.add_argument("-s", "--seeds", type=int, nargs="+", default=42,\
            help="seed for initializing pseudo-random number generator")
    parser.add_argument("-w", "--num_workers", type=int,\
            help="number of cpu thread workers", default=0)
    parser.add_argument("-t", "--performance_threshold", type=float,\
            help="performance threshold to use for early stopping", default=float("Inf"))
    parser.add_argument("-u", "--use_autotomy", type=int, default=1,\
            help="allow autotomy in training (for envs that support it)")
    parser.add_argument("-x", "--exp_name", type=str, \
            help="name of experiment", default="temp_exp")


    args = parser.parse_args()

    if "-v" not in args.env_name:
        args.env_name += "-v0"

    if type(args.seeds) is not list:
        args.seeds = [args.seeds]

    kwargs = dict(args._get_kwargs())

    # use subprocess to get the current git hash, store
    hash_command = ["git", "rev-parse", "--verify", "HEAD"]
    git_hash = subprocess.check_output(hash_command)

    # store the command-line call for this experiment
    entry_point = []
    entry_point.append(os.path.split(sys.argv[0])[1])
    args_list = sys.argv[1:]

    sorted_args = []
    for aa in range(0, len(args_list)):

        if "-" in args_list[aa]:
            sorted_args.append([args_list[aa]])
        else: 
            sorted_args[-1].append(args_list[aa])

    sorted_args.sort()
    entry_point = "python -m symr.benchmark "

    for elem in sorted_args:
        entry_point += " " + " ".join(elem)

    kwargs["entry_point"] = entry_point 
    kwargs["git_hash"] = git_hash.decode("utf8")[:-1]

    train(**kwargs)
