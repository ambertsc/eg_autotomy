import os
import argparse

import unittest

import numpy as np

from eg_auto.helpers import check_connected

import gym
import eg_envs

from evogym import EvoWorld, EvoSim, \
        EvoViewer, sample_robot

from bevodevo.enjoy import enjoy

class TestEnjoy(unittest.TestCase):

    def setUp(self):
        pass

    def test_no_render(self):
        parser = argparse.ArgumentParser("Experiment parameters")

        parser.add_argument("-n", "--env_name", type=str, \
                help="name of environemt", default="BackAndForthEnv-v0")
        parser.add_argument("-pi", "--policy", type=str,\
                help="name of policy architecture", default="MLPPolicy")
        parser.add_argument("-e", "--episodes", type=int,\
                help="number of episodes", default=1)
        parser.add_argument("-s", "--save_frames", type=bool, \
                help="save frames or not", default=False)
        parser.add_argument("-nr", "--no_render", type=bool,\
                help="don't render", default=True)
        parser.add_argument("-ms", "--max_steps", type=int,\
                help="maximum number of steps per episode", default=10)
        parser.add_argument("-f", "--file_path", type=str,\
                help="file path to model parameters", \
                default="./results/test_exp/")
        parser.add_argument("-a", "--num_agents", type=int,\
                help="how many agents to evaluate", \
                default=1)

        args = parser.parse_args()

        exit_code = enjoy(args)

        self.assertEqual(0, exit_code)

    def test_no_render_body(self):
        parser = argparse.ArgumentParser("Experiment parameters")

        parser.add_argument("-n", "--env_name", type=str, \
                help="name of environemt", default="BackAndForthEnv-v0")
        parser.add_argument("-pi", "--policy", type=str,\
                help="name of policy architecture", default="MLPBodyPolicy")
        parser.add_argument("-e", "--episodes", type=int,\
                help="number of episodes", default=1)
        parser.add_argument("-s", "--save_frames", type=bool, \
                help="save frames or not", default=False)
        parser.add_argument("-nr", "--no_render", type=bool,\
                help="don't render", default=True)
        parser.add_argument("-ms", "--max_steps", type=int,\
                help="maximum number of steps per episode", default=10)
        parser.add_argument("-f", "--file_path", type=str,\
                help="file path to model parameters", \
                default="./results/test_exp/")
        parser.add_argument("-a", "--num_agents", type=int,\
                help="how many agents to evaluate", \
                default=1)
        parser.add_argument("-b", "--body_dim", type=int,\
                help="body dim", \
                default=8)
        parser.add_argument("-u", "--use_autotomy", type=int, default=1,\
                help="allow autotomy in training (for envs that support it)")

        args = parser.parse_args()

        exit_code = enjoy(args)

        self.assertEqual(0, exit_code)

    
if __name__ == "__main__": #pragma: no cover
    unittest.main(verbosity=2)
