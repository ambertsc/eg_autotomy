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
            self.body_dim = min([kwargs["body_dim"], 8])
        else:
            self.body_dim = 8

        if "mode" in kwargs.keys():
            self.mode = kwargs["mode"]
            self.body_dim = 5
        else:
            # coevo with body
            self.mode = 0
            self.body_dim = 5
            
        self.init_body()

        super().__init__(**kwargs)

        self.action_dim = 2 * reduce(lambda x,y: x*y, self.body.shape)
        self.max_observation_dim = 200
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

        act = act[:,:self.active_action_dim]
        act[:,-self.body_elements:] = torch.tensor(\
                self.get_autotomy().ravel()[None,:], dtype=act.dtype)


        return act.detach().cpu().numpy()
    

    def given_body(self, mode=1):
        """
        return a predertimined robot body plan
        mode 1 - square
        mode 2 - table
        mode 3 - comb
        """

        if mode == 3:
            # comb
            new_body = np.array([[0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0],\
            [1, 1, 1, 1, 1],\
            [4, 3, 4, 3, 4],\
            [2, 0, 4, 0, 2.]])
        elif mode == 2:
            # table
            new_body = np.array([[1, 1, 1, 1, 1.],\
            [1, 3, 3, 3, 1],\
            [4, 3, 3, 3, 4],\
            [4, 0, 0, 0, 4],\
            [2, 0, 0, 0, 2]])
        elif self.mode == 0:
            new_body = self.body
        else:
            # square
            new_body = np.array([[4., 4, 4, 4, 4],\
            [4, 3, 3, 3, 1],\
            [4, 3, 3, 3, 4],\
            [4, 3, 3, 3, 4],\
            [4, 4, 4, 4, 4]])

        return new_body

    def get_body(self):

        return self.body
        #, self.connections

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

    def get_autotomy(self):

        return self.autotomy

    def set_autotomy(self, autotomy):

         autotomy_map = np.round(autotomy)
         autotomy_clipped = np.clip(autotomy_map, 0,1.)

         self.autotomy = autotomy_clipped

    def init_body(self):

        if self.mode == 0:
            self.body, self.connections = sample_robot((self.body_dim, self.body_dim)) 
            while self.body.max() < 3:
                # avoid a bot with no actuators
                self.body, self.connections = sample_robot((self.body_dim, self.body_dim)) 
        else:
            self.body, self.connections = self.given_body(mode=self.mode), None


        temp_env = gym.make("BackAndForthEnv-v0", body=self.body)
        self.active_action_dim = temp_env.action_space.sample().ravel().shape[0]

        my_autotomy = np.random.randint(0,2, size=self.body.shape)
        self.set_autotomy(my_autotomy)

        self.body_elements = self.autotomy.shape[0] * self.autotomy.shape[1]

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())


        params = np.append(params, self.get_body().ravel())
        params = np.append(params, self.get_autotomy().ravel())

        return params


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
        params = np.append(params, self.get_autotomy().ravel())

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

        # set the body autotomy plan
        param_start = param_stop
        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.autotomy.shape)
        temp = my_params[param_start:param_stop]
        self.set_autotomy(temp)


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
        params = np.append(params, self.get_autotomy().ravel())

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

        # set the body autotomy plan
        param_start = param_stop
        param_stop = param_start \
                + reduce(lambda x,y: x*y, self.autotomy.shape)
        temp = my_params[param_start:param_stop]
        self.set_autotomy(temp)

