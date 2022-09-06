
import os

import unittest

import numpy as np

from eg_auto.helpers import check_connected

import gym
import eg_envs

from evogym import EvoWorld, EvoSim, \
        EvoViewer, sample_robot

from bevodevo.policies.rnns import GatedRNNPolicy


class TestGatedRNNPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = GatedRNNPolicy(params=None)

    def test_mlp_forward(self):

        x = np.random.randn(1, self.policy.input_dim)
        
        output = self.policy(x)

        self.assertEqual(output.shape[-1], self.policy.action_dim)

if __name__ == "__main__":

    unittest.main(verbosity=2)
