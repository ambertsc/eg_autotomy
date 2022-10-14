from collections import OrderedDict
from functools import reduce

import numpy as np
import scipy
from scipy.ndimage import label

import torch
import torch.nn as nn
import torch.nn.functional as F

from evogym import EvoWorld, EvoSim, \
        EvoViewer, sample_robot

from bevodevo.policies.mlps import MLPPolicy, HebbianMLP, ABCHebbianMLP
from bevodevo.policies.body_mlps import MLPBodyPolicy

import gym
import eg_envs
from eg_auto.helpers import check_connected

class MLPBodyPolicy2(MLPBodyPolicy):
    
    def __init__(self, **kwargs):

        if "autotomy_multiplier" in kwargs.keys():
            self.autotomy_multiplier = kwargs["autotomy_multiplier"]
        else:
            self.autotomy_multiplier = 3
        if "body_multiplier" in kwargs.keys():
            self.body_multiplier = kwargs["body_multiplier"]
        else:
            self.body_multiplier = 48

        super().__init__(**kwargs)

    def max_body(self): 

        my_shape = self.body.shape
        my_body_prob = np.random.rand(*my_shape)
        my_body_prob = torch.tensor(my_body_prob).reshape(-1,1).float()
        my_body_prob = torch.softmax(my_body_prob, dim=0).reshape(*my_shape)

        my_body_prob *= self.body_multiplier
        my_body_prob = 1.0 * (my_body_prob > torch.rand_like(my_body_prob)).numpy()

        self.body_prob = my_body_prob 
        self.body = self.body_prob  * np.random.randint(1,4, self.body.shape)

    def init_body(self):

        if self.mode == 0:
            self.body, self.connections = sample_robot((self.body_dim, self.body_dim)) 
            self.max_body()

            while self.body.max() < 3 or not(check_connected(self.body)):
                # avoid a bot with no actuators
                self.max_body()
        else:
            self.body, self.connections = self.given_body(mode=self.mode), None

        temp_env = gym.make("BackAndForthEnv-v0", body=self.body)
        self.active_action_dim = temp_env.action_space.sample().ravel().shape[0]

        my_autotomy = np.random.randint(0,2, size=self.body.shape)
    
        self.set_autotomy(my_autotomy)

        self.body_elements = self.autotomy.shape[0] * self.autotomy.shape[1]


    def set_body_prob(self, body_prob):
        
        self.body_prob = body_prob

    def get_body(self):

        return self.body

    def get_autotomy(self):
        
        my_shape = self.autotomy.shape
        
        # only consider autotomy where there is a body
        
        my_autotomy = (self.autotomy).reshape(self.body.shape)

        my_autotomy[self.body ==0] = 0.
        
        my_autotomy = torch.tensor(my_autotomy).reshape(-1,1).float()
        my_autotomy = torch.softmax(my_autotomy, dim=0).reshape(*my_shape)

        my_autotomy *= self.autotomy_multiplier

        my_autotomy = 1.0 * (my_autotomy > torch.rand_like(my_autotomy)).numpy()

        return my_autotomy

    def set_autotomy(self, autotomy):

        self.autotomy = autotomy

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.named_parameters():


            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

            param_start = param_stop

        # set the body plan
        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.body.shape)

        if self.mode == 0:
            updated_body = my_params[param_start:param_stop]
        else:
            updated_body = self.given_body(mode=self.mode)
        self.set_body(updated_body)

        param_start = param_stop
        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.autotomy.shape)
        if self.mode == 0:
            updated_body_prob = my_params[param_start:param_stop]
        else:
            updated_body_prob = np.ones(self.body.shape)
        self.set_body_prob(updated_body_prob)

        # set the body autotomy plan
        param_start = param_stop
        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.autotomy.shape)
        temp = my_params[param_start:param_stop]
        self.set_autotomy(temp)


    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            self.body_dim = 5
            params = np.append(params, param[1].detach().numpy().ravel())

        params = np.append(params, self.body.ravel())
        params = np.append(params, self.body_prob.ravel())
        params = np.append(params, self.autotomy.ravel())

        return params
