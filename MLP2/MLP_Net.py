import torch
from torch import nn
from torch.nn import Module,Linear,Sequential,Tanh

import json

class Reshape2d(Module):
    def __init__(self,in_shape:tuple,out_shape:tuple):
        super().__init__()

        self.in_shape = in_shape 
        self.out_shape = out_shape

    def forward(self,x):
        suffix_n = len(self.in_shape)
        assert x.shape[-suffix_n:] == self.in_shape
        prefix_shape = x.shape[:-suffix_n]

        suffix_n2 = len(self.out_shape)
        out_shape = prefix_shape + self.out_shape

        y = x.reshape(out_shape)
        return y


class MLP(Module):
    """
    
    """
    struct_str =""  
    def __init__(self):
        super().__init__()


class MLP0(MLP):
    """
    For an MLP with 2 hidden layers, each containing 511 hidden units, input patches of size 13x13 pixels and output patches of size 13x13 pixels
    """
    struct_str = json.dumps([13,511,511,13])
    noise_advice = json.dumps(['AWG',0.1])
    def __init__(self):
        super().__init__()
        self.Activation = Tanh

        self.input_layer = Sequential( 
            Reshape2d((13,13),(13*13,)),
            Linear(13*13,511,True)
        )
        self.median_layer = Sequential(
            Linear(511,511,True),
            self.Activation()
        )
        self.output_layer = Sequential( 
            Linear(511,13*13,True), 
            Reshape2d((13*13,),(13,13))
        )
    def forward(self,x):
        x_in = self.input_layer(x)
        x1 = self.median_layer(x_in)
        x_out = self.output_layer(x1)
        return x_out


class MLP1(MLP):
    """
    For an MLP with 3 hidden layers, each containing 511 hidden units, input patches of size 13x13 pixels and output patches of size 13x13 pixels
    """
    struct_str = json.dumps([13,511,511,511,13])
    noise_advice = json.dumps(['AWG',0.1])
    def __init__(self):
        super().__init__()
        self.Activation = Tanh

        self.input_layer = Sequential( 
            Reshape2d((13,13),(13*13,)),
            Linear(13*13,511,True)
        )
        self.median_layer = Sequential(
            Linear(511,511,True),
            self.Activation(),
            Linear(511,511,True),
            self.Activation(),
        )
        self.output_layer = Sequential( 
            Linear(511,13*13,True), 
            Reshape2d((13*13,),(13,13))
        )
    def forward(self,x):
        x_in = self.input_layer(x)
        x1 = self.median_layer(x_in)
        x_out = self.output_layer(x1)
        return x_out

class MLP2(MLP):
    """
    For an MLP with 4 hidden layers, each containing 511 hidden units, input patches of size 13x13 pixels and output patches of size 13x13 pixels
    """
    struct_str = json.dumps([13,511,511,511,511,13])
    noise_advice = json.dumps(['AWG',0.1])
    def __init__(self):
        super().__init__()
        self.Activation = Tanh

        self.input_layer = Sequential( 
            Reshape2d((13,13),(13*13,)),
            Linear(13*13,511,True)
        )
        self.median_layer = Sequential(
            Linear(511,511,True),
            self.Activation(),
            Linear(511,511,True),
            self.Activation(),
            Linear(511,511,True),
            self.Activation(),
        )
        self.output_layer = Sequential( 
            Linear(511,13*13,True), 
            Reshape2d((13*13,),(13,13))
        )
    def forward(self,x):
        x_in = self.input_layer(x)
        x1 = self.median_layer(x_in)
        x_out = self.output_layer(x1)
        return x_out

