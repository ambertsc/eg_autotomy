import os

import unittest

import numpy as np

from eg_auto.helpers import check_connected

import gym
import eg_envs
from eg_envs.back_and_forth_env import BackAndForthEnvClass

from evogym import EvoWorld, EvoSim, \
        EvoViewer, sample_robot



class TestAdaptiveWalkEnv(unittest.TestCase):

    def setUp(self):
        pass

    def test_env_init(self):
        body, connections = sample_robot((4,4))
        env = gym.make("AdaptiveWalkEnv-v0", body=body)

        self.assertTrue(True)

    def test_remove_robot(self):

        body, connections = sample_robot((4,4))
        env = gym.make("AdaptiveWalkEnv-v0", body=body)
        env.remove_robot("robot")

        self.assertFalse("robot" in env.world.objects.keys())

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

    def test_env_init_default_body(self):
        env = gym.make("BackAndForthEnv-v0")

        _ = env.reset()
        action = env.action_space.sample()

        o, r, d, i = env.step(action)

        self.assertEqual(dict, type(i))

    def test_remove_robot(self):

        body, connections = sample_robot((4,4))
        env = gym.make("BackAndForthEnv-v0", body=body)
        env.remove_robot("robot")

        self.assertFalse("robot" in env.world.objects.keys())


    def test_steps(self):

        body = np.array([[1,0.,0,1],[1,3,3,1],[1,3,3,4],[1,0,0,4],[0,0,0,4]])
        connections = None

        env = gym.make("BackAndForthEnv-v0", body=body)

        _ = env.reset()
        total_reward = 0.
        pruning = np.zeros((env.robot_body_elements,))

        self.assertFalse(env.mode)

        action = env.action_space.sample()

        action[-env.robot_body_elements:] = pruning

        obs, reward, done, info = env.step(action)
        total_reward += reward

        env = gym.make("BackAndForthEnv-v0", body=body, goal=[2,48])
        _ = env.reset()

        pruning = np.ones((env.robot_body_elements,))

        #pruning[1] = 0
        pruning[8] = 0
        #pruning[9] = 0
        #pruning[13] = 0

        for step in range(10):
            action = env.action_space.sample()

            action[-env.robot_body_elements:] = pruning

            obs, reward, done, info = env.step(action)
            total_reward += reward

        env.reverse_direction(action)
        self.assertTrue(env.mode)

        self.assertTrue(True)
