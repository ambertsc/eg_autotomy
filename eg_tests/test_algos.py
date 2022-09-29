import os
import argparse
import subprocess
import sys

import unittest

import numpy as np

import eg_auto
import eg_auto.helpers
from eg_auto.helpers import check_connected

import gym
import eg_envs

from bevodevo.algos.cmaes import CMAESPopulation
from bevodevo.algos.es import ESPopulation
from bevodevo.algos.ga import GeneticPopulation
from bevodevo.algos.nes import NESPopulation
from bevodevo.algos.pges import PGESPopulation
from bevodevo.algos.random_search import RandomSearch

from bevodevo.policies.mlps import MLPPolicy

class TestESPopulation(unittest.TestCase):

    def setUp(self):

        policy_fn = MLPPolicy

        self.population = ESPopulation(policy_fn=policy_fn)

    def test_mantle(self):

        parser = argparse.ArgumentParser("Experiment parameters")
        parser.add_argument("-n", "--env_name", type=str, \
                help="name of environemt", default="InvertedPendulumBulletEnv-v0")
        parser.add_argument("-p", "--population_size", type=int,\
                help="number of individuals in population", default=6)
        parser.add_argument("-w", "--num_workers", type=int,\
                help="number of cpu thread workers", default=0)
        parser.add_argument("-a", "--algorithm", type=str,\
                help="name of es learning algo", default="ESPopulation")
        parser.add_argument("-pi", "--policy", type=str,\
                help="name of policy architecture", default="MLPPolicy")
        parser.add_argument("-g", "--generations", type=int,\
                help="number of generations", default=2)
        parser.add_argument("-t", "--performance_threshold", type=float,\
                help="performance threshold to use for early stopping", default=float("Inf"))
        parser.add_argument("-x", "--exp_name", type=str, \
                help="name of experiment", default="temp_exp")
        parser.add_argument("-s", "--seeds", type=int, nargs="+", default=42,\
                help="seed for initializing pseudo-random number generator")

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

        self.population.mantle(**kwargs)


class TestPGESPopulation(TestESPopulation):

    def setUp(self):

        policy_fn = MLPPolicy

        self.population = PGESPopulation(policy_fn=policy_fn)

class TestNESPopulation(TestESPopulation):

    def setUp(self):

        policy_fn = MLPPolicy

        self.population = NESPopulation(policy_fn=policy_fn)

class TestCMAESPopulation(TestESPopulation):

    def setUp(self):

        policy_fn = MLPPolicy

        self.population = CMAESPopulation(policy_fn=policy_fn)

class TestGeneticPopulation(TestESPopulation):

    def setUp(self):

        policy_fn = MLPPolicy

        self.population = GeneticPopulation(policy_fn=policy_fn)

class TestRandomSearch(TestESPopulation):

    def setUp(self):

        policy_fn = MLPPolicy

        self.population = RandomSearch(policy_fn=policy_fn)


if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)
