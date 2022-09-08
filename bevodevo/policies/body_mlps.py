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

import gym
import eg_envs
from eg_auto.helpers import check_connected

class MLPBodyPolicy(MLPPolicy):

    def __init__(self, **kwargs):

        if "body_dim" in kwargs.keys():
            self.body_dim = min([kwargs["body_dim"], 5])
        else:
            self.body_dim = 5

        self.init_body()

        super().__init__(**kwargs)

        self.action_dim = 2 * reduce(lambda x,y: x*y, self.body.shape)
        self.max_observation_dim = 96
        self.input_dim = self.max_observation_dim
        self.init_params()
        
    def get_action(self, x):

        x_holder = np.zeros((1,self.max_observation_dim))
        x_holder[:, :x.shape[-1]] = x
        y = self.forward(x_holder)

        if self.discrete:
            act = torch.argmax(y, dim=-1)
        else:
            act = y

        act = act[:,:self.number_actuators]

        return act.detach().cpu().numpy()
    
    def init_body(self):

        self.body, self.connections = sample_robot((self.body_dim, self.body_dim)) 

        while self.body.max() < 3:
            # avoid a bot with no actuators
            self.body, self.connections = sample_robot((self.body_dim, self.body_dim)) 

        temp_env = gym.make("BackAndForthEnv-v0", body=self.body)
        self.number_actuators = temp_env.action_space.sample().ravel().shape[0]

    def get_body(self):

        return self.body
        #, self.connections

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())


        params = np.append(params, self.body.ravel())

        return params

    def set_body(self, new_body):
        
        new_body = np.reshape(new_body, self.body.shape)
        new_body = np.uint8(np.clip(new_body, 0,4))
        mask = label((new_body > 0))

        most = 0
        keep_index = None
        for check in range(1, np.max(mask[0])):

            temp = (mask[0] == check).sum()

            if temp > most:
                most = temp
                keep_index = check

        if keep_index is not None:
            new_body *= np.array((mask[0] == keep_index), dtype=np.uint8)

        if not check_connected((new_body > 0) * 1.0):
            pass
        elif new_body.max() < 3:
            pass
        elif (new_body > 0.0).sum() < 2:
            pass
        else:
            self.body = np.clip(np.uint8(new_body), 0,4).reshape(self.body.shape)

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

            param_start = param_stop

        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.body.shape)

        temp = my_params[param_start:param_stop]

        self.set_body(temp)


class HebbianMLPBody(HebbianMLP, MLPBodyPolicy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        if self.lr_layers is not None and self.plastic:
            for param in self.lr_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())

        params = np.append(params, self.body.ravel())

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.layers.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

            param_start = param_stop

        if self.plastic:
            for name, param in self.lr_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

                param_start = param_stop

        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.body.shape)

        temp = my_params[param_start:param_stop]
        self.set_body(temp)


class ABCHebbianMLPBody(ABCHebbianMLP, MLPBodyPolicy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        if self.lr_layers is not None and self.plastic:
            for param in self.lr_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())
            for param in self.a_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())
            for param in self.b_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())
            for param in self.c_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())

        params = np.append(params, self.body.ravel())

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.layers.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

            param_start = param_stop

        if self.plastic:
            for name, param in self.lr_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

                param_start = param_stop

            for name, param in self.a_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

                param_start = param_stop

            for name, param in self.b_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

                param_start = param_stop
                
            for name, param in self.c_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)


                param_start = param_stop

        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.body.shape)

        temp = my_params[param_start:param_stop]
        self.set_body(temp)

