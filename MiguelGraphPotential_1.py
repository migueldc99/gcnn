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


def graph_potential(n_gc_layers: int = 1,
                    n_fc_layers: int = 1,
                    activation_function: str = 'Tanh',
                    pooling: str = 'add',
                    n_node_features: int = 55,
                    n_edge_features: int = 1,
                    n_neurons: int = None ) -> torch.nn.Module: 

    r""" 
    A function that returns a model with the desired number of Graph-Convolutional (n_gc_layers) layers and 
    Fully-Connected (n_fc_layers) linear layers, using the specified non-linear activation layers interspaced
    between them. 

    Args:
        
        n_gc_layers (int): number of Graph-Convolution layers (GraphConv from dgl); default: 1 
        n_fc_layers (int): number of densely-connected -Fully Connected- linear layers (DenseGraphConv); default: 1
        activation_function (char): nonlinear activation layer; can be any activation available in torch; default: Tanh
        pooling (char): the type of pooling of node results; can be 'add', i.e. summing over all nodes (e.g. to 
                return total energy; default) or 'mean', to return energy per atom.
        n_node_features (int): length of node feature vectors; default: 55
        
        
        n_edge_features (int): length of edge feature vectors; currently 1, the distance.
        ATENCION PQ CON LA FUNCION QUE ESTAMOS USANDO NO ES NECESARIO SABER LOS EDGES
        
        n_neurons (int): number of neurons in deep layers (if they exist) in the densely-connected network.  
                       If set to None, it is internally set to n_node_features; default: None

    """


    activation_layer = eval("torch.nn." + activation_function + "()")
    
    if pooling == 'add':
        
        poolingLayer = SumPooling()
   
    else:
    
        poolingLayer = AvgPooling()

    if n_neurons == None:
        
        n_neurons = n_node_features

    
    # first the Graph-convolutional layers: 
    
    r"""     
        GraphConv(in_feats, out_feats, norm='both', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
    """
    
    conv = GraphConv(in_feats = n_node_features, out_feats = n_edge_features, activation = activation_layer)
    # Convolution and activation at the same time
    
    model = conv
    
    for n in range(0, n_gc_layers):
    
        model = Sequential(model, conv)
    
    
    # then the fully-connected layers: 
    
    # Applies a linear transformation to the incoming data: y = xA^T + b
    linear1 = nn.Linear(n_node_features, 1)           # only 1 fully connected layer 
                                                      # (has n_node_features as it is connected to the last conv layer)
    linear2 = nn.Linear(n_node_features, n_neurons)   # conv layer --> fully connected layer
    linear3 = nn.Linear(n_neurons, n_neurons)         # fully connected layer --> fully connected layer
    linear4 = nn.Linear(n_neurons, 1)                 # fully connected layer --> final result
    
    
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