# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:27:43 2022

@author: Miguel
"""

import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn import GraphConv, SumPooling, AvgPooling
from dgl.nn.pytorch import Sequential


class LinearDens(nn.Module):
    
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    
    LinearDens is introduced as a way of use nn.Linear together with GraphConv since Sequential() 
    requires the layers to be joined have the same input. In this case, a dgl.graph and a features tensor.
    """
    def __init__(self, in_feats, out_feats):
    
        super(LinearDens, self).__init__()
    
        self.linear = nn.Linear(in_feats, out_feats)
    
    def forward(self, graph, feat, weight=None, edge_weight=None):
        
        return self.linear(feat)



def graph_potential(n_gc_layers: int = 1,
                    n_fc_layers: int = 1,
                    activation_function: str = 'Tanh',
                    pooling: str = 'add',
                    n_node_features: int = 55,
                    n_edge_features: int = 1,
                    n_neurons: int = None ) -> torch.nn.Module: 

    r""" This is a function that returns a model with the desired number of Graph-Convolutional (n_gc_layers) layers and 
    Fully-Connected (n_fc_layers) linear layers, using the specified non-linear activation layers interspaced
    between them. 

    Args:
        
        n_gc_layers (int): number of Graph-Convolutional layers (GraphConv from dgl). Default: 1 
        n_fc_layers (int): number of densely-connected -Fully Connected- linear layers (LinearDens). Default: 1
        activation_function (char): nonlinear activation layer; can be any activation available in torch. Default: Tanh
        pooling (char): the type of pooling of node results. If set to 'add' performs a sum over all nodes (e.g. return total energy).
                        In other case, performs the 'mean' to return energy per atom. Default: Add
        n_node_features (int): length of node feature vectors. Default: 55
        
        
        n_edge_features (int): length of edge feature vectors; currently 1, the distance.
        ATENCION PQ CON LA FUNCION QUE ESTAMOS USANDO NO ES NECESARIO SABER LOS EDGES
        
        
        n_neurons (int): number of neurons in deep layers (if they exist) in the densely-connected network.  
                       If set to None, it is internally set to n_node_features. Default: None

    """


    # First, we define the activation layer, the pooling layer and the number of neurons:
    
    activation_layer = eval("torch.nn." + activation_function + "()")
    
    if pooling == 'add':
        poolingLayer = SumPooling()
   
    else:
        poolingLayer = AvgPooling()

    if n_neurons == None:
        n_neurons = n_node_features

    
    # Now, the Graph-convolutional layers: 

    conv = GraphConv(in_feats = n_node_features, out_feats = n_node_features, activation = activation_layer)
    
    model = conv
      
    for n in range(1, n_gc_layers-1):
        
        model = Sequential(model, conv)
    
    
    
    # Then the fully-connected layers: 

    linear1 = LinearDens(n_node_features, 1)           # only 1 fully connected layer; last conv layer --> final result)
    linear2 = LinearDens(n_node_features, n_neurons)   # conv layer --> fully connected layer
    linear3 = LinearDens(n_neurons, n_neurons)         # fully connected layer --> fully connected layer
    linear4 = LinearDens(n_neurons, 1)                 # fully connected layer --> final result
    
    
    if n_fc_layers == 1:
        
        model = Sequential(model, linear1)
    
    else:
        
        model = Sequential(model, linear2)
        
        model = Sequential(model, activation_layer)
        
        for n in range(1, n_fc_layers-1):
            
            model = Sequential(model, linear3)
            
            model= Sequential(model, activation_layer)
            
        # and finally the exit layer

        model = Sequential(model, linear4)
    
    # last but not least, the pooling layer

    model = Sequential(model, poolingLayer)
    
    return model
