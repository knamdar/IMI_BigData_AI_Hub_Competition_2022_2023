#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Feb 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition
@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF and cuML from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""

import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class NGNN_GCNConv(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NGNN_GCNConv, self).__init__()
        self.conv = GraphConv(input_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, output_channels)

    def forward(self, g, x, edge_weight=None):
        x = self.conv(g, x, edge_weight)
        x = F.relu(x)
        x = self.fc(x)
        return x

class NGNN_GCN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NGNN_GCN, self).__init__()
        self.conv1 = NGNN_GCNConv(input_channels, hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, output_channels)

    def forward(self, g, input_channels):
        h = self.conv1(g, input_channels)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# define GCN model
class BuiltinGCN(nn.Module):
    """Graph convolutional network using DGL supported graph convolution modules
    
    Parameters
    ----------
    in_feats (int): input feature size
    h_feats (int): hidden feature size
    num_classes (int): number of classes
    """
    def __init__(self, in_feat, h_feat, num_classes):
        super(BuiltinGCN, self).__init__()
        self.layer1=GraphConv(in_feat, h_feat, norm='both')
        self.layer2=GraphConv(h_feat, h_feat, norm='both')
        self.layer3=GraphConv(h_feat, num_classes, norm='both')

    def forward(self, g, h):
        """Forward computation
        
        Parameters
        ----------
        g (DGLGraph): the input graph
        features (Tensor): the input node features
        """
        h=self.layer1(g, h)
        h=F.relu(h)
        h=self.layer2(g, h)
        h=F.relu(h)
        h=self.layer3(g, h)
        return h


# define SimpleGraphNet
class SimpleGraphNet(nn.Module):
    """Simple graph neural network
    
    Parameters
    ----------
    in_feats (int): input feature size
    h_feats (int): hidden feature size
    num_classes (int): number of classes
    """
    def __init__(self, in_feats, h_feats, num_classes):
        # for inheritance we use super() to refer to the base class
        super(SimpleGraphNet, self).__init__()
        
        # two linear layers where each one will have its own weights, W
        # first layer computes the hidden layer
        self.layer1 = nn.Linear(in_feats, h_feats)
        # use num_classes units for the second layer to compute the classification of each node
        self.layer2 = nn.Linear(h_feats, num_classes)

    def forward(self, g, h, adj): 
        """Forward computation
        
        Parameters
        ----------
        g (DGLGraph): the input graph
        h (Tensor): the input node features
        adj (Tensor): the graph adjacency matrix
        """
        # apply first linear layer's transform weights 
        x=self.layer1(h)
        
        # perform matrix multiplication with the adjacency matrix and node features to 
        # aggregate/recombine across neighborhoods
        x=torch.mm(adj, x)
        
        # apply a relu activation function
        x=F.relu(x)
        
        # apply second linear layer's transform weights
        x=self.layer2(x)
        return x

# ref: https://github.com/senadkurtisi/pytorch-GCN/blob/main/src/model.py
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias

        return torch.sparse.mm(adj, x)


class GCN(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes, dropout, use_bias=True):
        super(GCN, self).__init__()
        self.gcn_1 = GCNLayer(node_features, hidden_dim, use_bias)
        self.gcn_2 = GCNLayer(hidden_dim, num_classes, use_bias)
        self.dropout = nn.Dropout(p=dropout)

    def initialize_weights(self):
        self.gcn_1.initialize_weights()
        self.gcn_2.initialize_weights()

    def forward(self, x, adj):
        x = F.relu(self.gcn_1(x, adj))
        x = self.dropout(x)
        x = self.gcn_2(x, adj)
        return x