import os

import unittest

import numpy as np

from eg_auto.helpers import check_connected

import envs
import gym
from envs.back_and_forth_env import BackAndForthEnvClass

from evogym import EvoWorld, EvoSim, \
        EvoViewer, sample_robot



class TestAdaptiveWalkEnv(unittest.TestCase):

    def setUp(self):
        pass

    def test_env_init(self):
        body, connections = sample_robot((4,4))
        env = gym.make("AdaptiveWalkEnv-v0", body=body)

        self.assertTrue(True)

    def test_steps(self):

        body, connections = sample_robot((4,4))
        env = gym.make("AdaptiveWalkEnv-v0", body=body)

        _ = env.reset()
        total_reward = 0.

        for step in range(10):

            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            total_reward += reward

        self.assertTrue(True)

class TestBackAndForthEnv(unittest.TestCase):

    def setUp(self):
        pass

    def test_env_init(self):
        body, connections = sample_robot((4,4))
        env = gym.make("BackAndForthEnv-v0", body=body)

        self.assertTrue(True)

    def test_steps(self):

        body, connections = sample_robot((4,4))
        env = gym.make("BackAndForthEnv-v0", body=body)

        _ = env.reset()
        total_reward = 0.
        pruning = np.random.randint(0,2, size=(env.robot_body_elements,))


        for step in range(10):
            action = env.action_space.sample()

            action[-env.robot_body_elements:] = pruning

            obs, reward, done, info = env.step(action)
            total_reward += reward

        self.assertTrue(True)
