import os

import unittest

import numpy as np

from eg_auto.helpers import check_connected

import gym
import eg_envs

from evogym import EvoWorld, EvoSim, \
        EvoViewer, sample_robot

from bevodevo.policies.mlps import MLPPolicy, HebbianMLP,  ABCHebbianMLP
from bevodevo.policies.body_mlps import MLPBodyPolicy, HebbianMLPBody,  ABCHebbianMLPBody
from bevodevo.policies.body_mlps2 import MLPBodyPolicy2

class TestMLPPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = MLPPolicy(params=None)

    def test_mlp_forward(self):

        x = np.random.randn(1, self.policy.input_dim)
        
        output = self.policy(x)

        self.assertEqual(output.shape[-1], self.policy.action_dim)

    def test_set_params(self):
        
        my_params = self.policy.get_params()

        new_params = np.random.randn(*my_params.shape)

        self.policy.set_params(new_params)
        recovered_params = self.policy.get_params()

        # some precision is lost going back and forth
        # between numpy and torch here, therefore 1e-6
        
        self.assertGreater(1e-6, np.abs(new_params - recovered_params).max())

        self.policy.set_params(my_params)
        recovered_params = self.policy.get_params()

        self.assertNotIn(False, my_params == recovered_params)


class TestHebbianMLP(TestMLPPolicy):

    def setUp(self):
        self.policy = HebbianMLP(params=None)

class TestABCHebbianMLP(TestMLPPolicy):

    def setUp(self):
        self.policy = ABCHebbianMLP(params=None)

    def test_mlp_forward(self):

        x = np.random.randn(1, self.policy.input_dim)
        
        output = self.policy(x)

        self.assertEqual(output.shape[-1], self.policy.action_dim)


class TestMLPBodyPolicy(TestMLPPolicy):

    def setUp(self):
        self.policy = MLPBodyPolicy(params=None)

    def test_set_params(self):
        
        my_params = self.policy.get_params()

        new_params = np.random.randint(0, 4, my_params.shape)

        my_body = self.policy.get_body().ravel()
        my_autotomy = self.policy.get_autotomy().ravel()

        new_params[-2*self.policy.body_elements:-self.policy.body_elements] = my_body
        new_params[-self.policy.body_elements:] = my_autotomy

        self.policy.set_params(new_params)
        recovered_params = self.policy.get_params()

        # some precision is lost going back and forth
        # between numpy and torch here, therefore 1e-6
        
        self.assertGreater(1e-6, np.abs(new_params - recovered_params).max())

        self.policy.set_params(my_params)
        recovered_params = self.policy.get_params()

        self.assertNotIn(False, my_params == recovered_params)

    def test_autotomy(self):

        my_params = self.policy.get_params()
        self.policy.reset()

        temp_env = gym.make("BackAndForthEnv-v0", body=self.policy.body)
        obs = temp_env.reset()

        action = self.policy.get_action(obs)

        my_autotomy = self.policy.get_autotomy()

        action_autotomy = action[:, -self.policy.body_elements:]

        self.assertEqual(2, np.unique(action_autotomy).shape[0])
        self.assertEqual(2, np.unique(my_autotomy).shape[0])
        self.assertEqual(0, (my_autotomy.ravel() - action_autotomy.squeeze()).sum())

    def test_body(self):

        my_params = self.policy.get_params()
        self.policy.reset()

        temp_env = gym.make("BackAndForthEnv-v0", body=self.policy.body)
        my_body = self.policy.get_body()
        my_params_body = my_params[\
                -2*self.policy.body_elements:-self.policy.body_elements]

        self.assertLessEqual(5, np.unique(my_body).shape[0])
        self.assertLessEqual(5, np.unique(my_params_body).shape[0])
        self.assertEqual(0, (my_body.ravel() - my_params_body.squeeze()).sum())

class TestMLPBodyPolicy2(TestMLPBodyPolicy):

    def setUp(self):
        self.policy = MLPBodyPolicy2(params=None)

    def test_set_params(self):
        
        my_params = self.policy.get_params()

        new_params = np.random.randint(0, 4, my_params.shape)

        my_body = self.policy.body.ravel()
        my_autotomy = np.clip(new_params[-self.policy.body_elements:],0,1)

        new_params[-3*self.policy.body_elements:-2*self.policy.body_elements] = my_body
        new_params[-self.policy.body_elements:] = my_autotomy

        self.policy.set_params(new_params)
        recovered_params = self.policy.get_params()

        # some precision is lost going back and forth
        # between numpy and torch here, therefore 1e-6
        
        self.assertGreater(1e-6, np.abs(new_params - recovered_params).max())

        self.policy.set_params(my_params)
        recovered_params = self.policy.get_params()

        self.assertNotIn(False, my_params == recovered_params)

    def test_autotomy(self):

        my_params = self.policy.get_params()
        self.policy.reset()

        temp_env = gym.make("BackAndForthEnv-v0", body=self.policy.body)
        obs = temp_env.reset()

        action = self.policy.get_action(obs)

        my_autotomy = self.policy.get_autotomy()

        action_autotomy = action[:, -self.policy.body_elements:]

        self.assertEqual(2, np.unique(action_autotomy).shape[0])
        self.assertEqual(2, np.unique(my_autotomy).shape[0])

    def test_body(self):

        my_params = self.policy.get_params()
        self.policy.reset()

        temp_env = gym.make("BackAndForthEnv-v0", body=self.policy.body)

        self.policy.body_multiplier = 100. * self.policy.body_dim**2
        
        my_body = self.policy.get_body()
        my_params_body = my_params[\
                -3*self.policy.body_elements:-2*self.policy.body_elements]

        
        self.assertEqual(2, len((my_body).shape))

        self.assertEqual(0, (my_body.ravel() - my_params_body.squeeze()).sum())


class TestGivenBody(unittest.TestCase):

    def setUp(self, mode=0):
        self.policy = MLPBodyPolicy(params=None, mode=mode)
        self.mode = mode

    def test_given_body(self):
        self.policy.reset()
        
        my_body = self.policy.get_body()
        my_body_should_be = self.policy.given_body(mode=self.mode)

        self.assertEqual(0, (my_body.ravel() - my_body_should_be.ravel()).sum())
        self.assertEqual(my_body.shape, my_body_should_be.shape)
        
class TestGivenBody1(TestGivenBody):

    def setUp(self, mode=1):
        self.policy = MLPBodyPolicy(params=None, mode=mode)
        self.mode = mode

class TestGivenBody2(TestGivenBody):

    def setUp(self, mode=2):
        self.policy = MLPBodyPolicy(params=None, mode=mode)
        self.mode = mode

class TestGivenBody3(TestGivenBody):

    def setUp(self, mode=3):
        self.policy = MLPBodyPolicy(params=None, mode=mode)
        self.mode = mode


class TestHebbianMLPBodyPolicy(TestMLPBodyPolicy):

    def setUp(self):
        self.policy = HebbianMLPBody(params=None)

class TestABCHebbianMLPBodyPolicy(TestMLPBodyPolicy):

    def setUp(self):
        self.policy = ABCHebbianMLPBody(params=None)

if __name__ == "__main__": #pragma: no cover
    unittest.main(verbosity=2)
        
