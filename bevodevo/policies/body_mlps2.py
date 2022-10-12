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
        self.autotomy_multiplier = 5
        super().__init__(**kwargs)


    def get_autotomy(self):
        
        my_shape = self.autotomy.shape

        my_autotomy = torch.tensor(self.autotomy).reshape(-1,1).float()
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


        params = np.append(params, self.get_body().ravel())
        params = np.append(params, self.autotomy.ravel())

        return params
