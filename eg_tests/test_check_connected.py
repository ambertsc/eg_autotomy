import os

import unittest

import numpy as np

from eg_auto.helpers import check_connected

import envs
import gym
from envs.back_and_forth_env import BackAndForthEnvClass

class TestCheckConnected(unittest.TestCase):

    def setUp(self):
        pass

    def test_check_connected_simple(self):

        bad_body = np.array([[1,0,0,0], [0,0,0,0], [0,0,0,1]])
        good_body = np.array([[1,0,0,0], [1,1,1,1], [0,0,0,1]])

        good_result = check_connected(good_body)
        bad_result = check_connected(bad_body)

        self.assertTrue(good_result)
        self.assertFalse(bad_result)

    def test_check_connected_random(self):
        
        for step in range(100):

            body = np.random.randint(0,6, size=(6,6)) 

            result = check_connected(body)

            if result:
                env = gym.make("BackAndForthEnv-v0", body=body)


        self.assertTrue(True)

if __name__ == "__name__":

    unittest.main(verbosity=2)
