from collections import OrderedDict
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevodevo.policies.base import Policy

class MLPPolicy(nn.Module):

    def __init__(self, **kwargs):
        super(MLPPolicy, self).__init__()

        self.use_grad = kwargs["use_grad"] if "use_grad" in kwargs.keys() else False

        # architecture params
        self.input_dim = kwargs["dim_x"] if "dim_x" in kwargs.keys() else 5
        self.action_dim = kwargs["dim_y"] if "dim_y" in kwargs.keys() else 1
        self.hid_dims = kwargs["dim_h"] if "dim_h" in kwargs.keys() else 16
        self.hid_dims = [self.hid_dims] if type(self.hid_dims) is not list else self.hid_dims
        self.activations = kwargs["activations"] \
                if "activations" in kwargs.keys() else nn.ReLU
        self.discrete = kwargs["discrete"] if "discrete" in kwargs.keys() else False
        self.use_bias = False
        self.var = 1.e-2

        if type(self.activations) == list:
            if len(self.activations) <= (len(self.hid_dims)+1):
                # use no activation after list of act fns are used up
                for ii in range(len(self.activations), len(self.hid_dims)+1):
                    # identity function for layers missing activations
                    self.activations.append(lambda x: x)
            elif len(self.activations) >= (len(self.hid_dims)+1):
                print("warning: activation list has {} functions but MLP has only {} layers"\
                        .format(len(self.activations), len(self.hid_dims)+1))
                print("... truncating action function list")

                self.activations = self.activations[:len(self.hid_dims)]
        else:
            self.activations = [self.activations] * len(self.hid_dims)

        if self.discrete:
            pass
            # TODO: test/implement discrete action spaces
            #self.activations.append(lambda x: x)
        else:
            self.activations.append(nn.Tanh)

        self.init_params()

        if kwargs["params"] is not None: 
            self.set_params(kwargs["params"])

    def init_params(self):

        self.layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        for jj in range(1, len(self.hid_dims)):
            self.layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.layers.add_module("activation{}".format(jj), self.activations[jj]())

        self.layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))
        if self.discrete:
            pass
        else:
            self.layers.add_module("output_activation",\
                    self.activations[-1]())

        for param in self.layers.parameters():
            param.requires_grad = self.use_grad

        self.num_params = self.get_params().shape[0]

    def forward(self, x):

        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        x = x.to(torch.float32)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.layers(x)
        return x

    def get_action(self, x):

        y = self.forward(x)

        if self.discrete:
            act = torch.argmax(y, dim=-1)
        else:
            act = y

        return act.detach().cpu().numpy()

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

    def reset(self):
        pass


class HebbianMLP(MLPPolicy):

    def __init__(self, **kwargs):
        self.plastic = kwargs["plastic"] if "plastic" in kwargs.keys() else True
        self.lr_layers = None
        self.e_min = -1.
        self.e_max = 1.
        
        super(HebbianMLP, self).__init__(**kwargs)
        if self.plastic:
            self.init_traces()
        else:
            self.clear_nodes()

    def init_traces(self):

        self.dim_list = [self.input_dim]
        self.dim_list.extend(self.hid_dims)
        self.dim_list.append(self.action_dim)
        

        # clear node activations, start at 0 everywhere
        self.clear_nodes()

        # initialize learning rate values
        self.lr_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        self.eligibility_layers = [torch.zeros(self.input_dim, self.hid_dims[0])]

        for jj in range(1, len(self.hid_dims)-1):
            self.lr_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.lr_layers.add_module("activation{}".format(jj), self.activations[jj]())

            self.eligibility_layers.append(torch.zeros(self.hid_dims[jj], self.hid_dims[jj+1]))

        self.lr_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))

        self.eligibility_layers.append(torch.zeros(self.hid_dims[-1], self.action_dim))

        if self.discrete:
            pass
        else:
            self.lr_layers.add_module("output_activation",\
                    self.activations[-1]())

        for param in self.lr_layers.parameters():
            param.requires_grad = self.use_grad

        self.num_params = self.get_params().shape[0]


    def clear_nodes(self):

        self.nodes = [torch.zeros(1,elem) for elem in self.dim_list]

    def clear_traces(self):

        if self.plastic:
            self.eligibility_layers = [0.0 * elem for elem in self.eligibility_layers]

    def forward(self, x):


        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        x = x.to(torch.float32)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        trace_count = 0
        self.nodes[trace_count] = x.clone()

        for name, module in self.layers.named_modules():

            if "layer" in name:

                trace_count += 1
                x = module(x)
                self.nodes[trace_count] = x.clone()
            elif "activation" in name:
                x = module(x)

        if self.plastic:
            self.update()

        return x

    def update(self):

        num_layers = len(list(self.layers.named_parameters()))
        layer_count = 0

        for lr_param, param in zip(list(self.lr_layers.named_parameters()), \
                list(self.layers.named_parameters())):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]

            self.eligibility_layers[layer_count] += torch.matmul(self.nodes[layer_count].T, \
                    self.nodes[layer_count+1]) 

            self.eligibility_layers[layer_count] = torch.clamp(self.eligibility_layers[layer_count], min=self.e_min, max=self.e_max)

            for ii in range(layer_dim_x):
                for jj in range(layer_dim_y):

                    param[1][jj,ii] = param[1][jj,ii] + lr_param[1][jj,ii] * self.eligibility_layers[layer_count][ii,jj] 

            layer_count += 1

    def get_params(self):
        params = np.array([])

        for param in self.layers.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        if self.lr_layers is not None and self.plastic:
            for param in self.lr_layers.named_parameters():
                params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.layers.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

        if self.plastic:
            for name, param in self.lr_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

    def reset(self):

        self.clear_nodes()
        self.clear_traces()

class ABCHebbianMLP(HebbianMLP):

    def __init__(self, **kwargs):

        super(ABCHebbianMLP, self).__init__(**kwargs)

    def init_traces(self):

        self.dim_list = [self.input_dim]
        self.dim_list.extend(self.hid_dims)
        self.dim_list.append(self.action_dim)
        # clear node activations, start at 0 everywhere
        self.clear_nodes()

        # learning rules are encoded by lr, A, B, C. 
        # \delta W_{ij} = lr * (A * o_i*o_j + B * o_i + C * o_j)
        # initialize learning rate values
        self.lr_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))


        # Hebbian coefficient A
        self.a_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        # pre-synaptic coefficient B
        self.b_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        # post-synaptic coefficient C
        self.c_layers = nn.Sequential(OrderedDict([\
                ("layer0", nn.Linear(self.input_dim, self.hid_dims[0], bias=self.use_bias)),\
                ("activation_0", self.activations[0]())\
                ]))

        self.eligibility_layers = [torch.zeros(self.input_dim, self.hid_dims[0])]

        for jj in range(1, len(self.hid_dims)-1):
            self.lr_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))

            self.a_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.b_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))
            self.c_layers.add_module("layer{}".format(jj),\
                    nn.Linear(self.hid_dims[jj], self.hid_dims[jj+1], bias=self.use_bias))

            self.eligibility_layers.append(torch.zeros(self.hid_dims[jj], self.hid_dims[jj+1]))

        self.lr_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))

        self.a_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))
        self.b_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))
        self.c_layers.add_module("output_layer",\
                nn.Linear(self.hid_dims[-1], self.action_dim, bias=self.use_bias))

        self.eligibility_layers.append(torch.zeros(self.hid_dims[-1], self.action_dim))


        for params in [self.lr_layers.parameters(), self.a_layers.parameters(), \
                self.b_layers.parameters(), self.c_layers.parameters()]:
            for param in params: 
                param.requires_grad = self.use_grad

        self.num_params = self.get_params().shape[0]
        

    def update(self):

        num_layers = len(list(self.layers.named_parameters()))
        layer_count = 0

        for lr_param, param, A, B, C in zip(\
                list(self.lr_layers.named_parameters()),\
                list(self.layers.named_parameters()),\
                list(self.a_layers.named_parameters()),\
                list(self.b_layers.named_parameters()),\
                list(self.c_layers.named_parameters())):

            layer_dim_x, layer_dim_y = param[1].shape[1], param[1].shape[0]

            self.eligibility_layers[layer_count] += torch.matmul(self.nodes[layer_count].T, self.nodes[layer_count+1]) 

            self.eligibility_layers[layer_count] = torch.clamp(self.eligibility_layers[layer_count], min=self.e_min, max=self.e_max)

            for ii in range(layer_dim_x):
                for jj in range(layer_dim_y):

                    param[1][jj,ii] = torch.clamp(param[1][jj,ii] \
                    + lr_param[1][jj,ii] \
                    * (\
                    A[1][jj,ii] * self.eligibility_layers[layer_count][ii,jj] \
                    +B[1][jj,ii] * self.nodes[layer_count][:,ii] \
                    +C[1][jj,ii] * self.nodes[layer_count+1][:,jj] \
                        ), min=-10, max=10)
                            

            layer_count += 1

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

        return params

    def set_params(self, my_params):

        param_start = 0
        for name, param in self.layers.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                    requires_grad=self.use_grad)

        if self.plastic:
            for name, param in self.lr_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)

            for name, param in self.a_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)
            for name, param in self.b_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)
            for name, param in self.c_layers.named_parameters():

                param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

                param[:] = torch.nn.Parameter(torch.tensor(\
                        my_params[param_start:param_stop].reshape(param.shape), requires_grad=self.use_grad), \
                        requires_grad=self.use_grad)




if __name__ == "__main__":

    # run tests

    args = {}
    args["dim_x"] = 6
    args["dim_y"] = 1
    args["dim_h"] = 16
    args["params"] = None

    temp = MLPPolicy(args)
    temp = HebbianMLP(args)
    temp = ABCHebbianMLP(args)
    temp = HebbianCAMLP(args)
    temp = CPPNHebbianMLP(args)
    temp = CPPNMLPPolicy(args)

    print("OK")
