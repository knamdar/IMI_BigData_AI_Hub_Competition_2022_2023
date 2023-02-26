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
import time
import torch.nn.functional as F

# define evaluate
def evaluate(model, g, labels, mask):
    """Model evaluation for particular set

    Parameters
    ----------
    model (nn.Module): the model
    g (DGLGraph): the input graph
    labels (Tensor): the ground truth labels
    mask (Tensor): the mask for a specific subset
    """
    # assign features
    features=g.ndata['feat']

    # set to evaluation mode
    model.eval()

    with torch.no_grad():
        # put features through model to obtain logits 
        logits=model(g, features)

        # get logits and labels for particular set
        logits=logits[mask]
        labels=labels[mask]

        # get most likely class and count the number of corrects
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)

        # return accuracy
        return correct.item() * 1.0 / len(labels), indices.tolist(), labels.tolist()

# define train
def train(model, optimizer, num_epochs, g, labels, train_mask, valid_mask, test_mask):
    """Model training

    Parameters
    ----------
    model (nn.Module): the model
    features (Tensor): the feature tensor
    labels (Tensor): the ground truth labels
    """
    # assign features
    features=g.ndata['feat']
    
    # use a standard optimization pipeline using the adam optimizer
    #optimizer=torch.optim.Adam(model.parameters(), lr=0.0002)


    # standard training pipeline with early stopping
    best_acc=0.0
    for epoch in range(num_epochs): 
        start=time.time()

        # set to training mode
        model.train()
        
        # forward step
        # calculate logits and loss
        # print("features", features)
        logits=model(g, features)
        # print("logits", logits.size())
        # print("labels[train_mask]", labels[train_mask])
        # calculate loss using log_softmax and negative log likelihood
        logp=F.log_softmax(logits, dim=-1)
        # print("logp", logp)
        
        #loss=F.nll_loss(torch.nan_to_num(logp[train_mask], nan=0.0), labels[train_mask])
        if torch.cuda.is_available():
            cls_w = torch.tensor([1/10000,1/4]).cuda()
        else:
            cls_w = torch.tensor([1/10000,1/4])
        loss=F.nll_loss(logp[train_mask], labels[train_mask], cls_w)
        # print("loss==torch.nan", loss==torch.nan)
        # print("logp[train_mask]", torch.isnan(logp[train_mask]).any())
        # print("labels[train_mask]", torch.isnan(labels[train_mask]).any())
        # print("loss",loss)

        # backward step
        # zero out gradients before accumulating the gradients on backward pass
        optimizer.zero_grad()
        loss.backward()

        # apply the optimizer to the gradients
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # evaluate on validation and test sets
        val_acc, _,_=evaluate(model, g, labels, valid_mask)
        test_acc, indices, lbls =evaluate(model, g, labels, test_mask)

        # compare validation accuracy with best accuracy at 10 epoch intervals, which will update if exceeded
        if (epoch%10==0) & (val_acc>best_acc):
            best_acc=val_acc
        print("Epoch {:03d} | Loss {:.4f} | Validation Acc {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), val_acc, test_acc, time.time()-start))
    return indices, lbls
