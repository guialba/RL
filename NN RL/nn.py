import torch
from torch.autograd import Variable
from torch.distributions import normal
import torch.nn as nn
import torch.nn.functional as f

import itertools
import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Rectangle

class BetaModel(nn.Module):
    def __init__(self):
        super(BetaModel, self).__init__()
        n_features = 2
        n_models = 2
        n_hidden = 10

        self.model = nn.Sequential(
            nn.Linear(n_features, n_hidden).double(),
            nn.ReLU(),
            # nn.Linear(n_hidden, n_hidden).double(),
            # nn.ReLU(),
            nn.Linear(n_hidden, n_models).double(),
            nn.ReLU(),
            nn.Softmax(dim=1)
            # nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def plot(self):
        subset_color = [
            "gold",
            "darkorange",
            "darkorange",
            "red",
        ]

        subset_sizes = tuple(np.array([[i.in_features, i.out_features] for i in self.model if type(i) == torch.nn.modules.linear.Linear]).flatten().tolist())
        extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
        layers = [range(start, end) for start, end in extents]
        G = nx.Graph()
        for i, layer in enumerate(layers):
            G.add_nodes_from(layer, layer=i)
        for layer1, layer2 in nx.utils.pairwise(layers):
            G.add_edges_from(itertools.product(layer1, layer2))

        # color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
        pos = nx.multipartite_layout(G, subset_key="layer")
        nx.draw(G, pos, with_labels=False)


class Model:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.params = torch.nn.Parameter(torch.rand(2).type(torch.DoubleTensor))

    def loss(self, beta, sigma, s, a, s_):
        ## Get Probability Values
        mi = (
            torch.exp(
                normal.Normal(
                    ( s[:,0] + a[:,0]*self.env.force).reshape((-1,1)), torch.exp(sigma)
                ).log_prob( s_[:,0].reshape((-1,1)) )
            ) 
            * 
            torch.exp(
                normal.Normal(
                    ( s[:,1] + a[:,1]*self.env.force).reshape((-1,1)), torch.exp(sigma)
                ).log_prob( s_[:,1].reshape((-1,1)) )
            )
        )
        ##

        p_theta = torch.sum((mi * beta), 1) 
        return -torch.sum(torch.log(p_theta))
   
    # def decompose_data(self, data):
    #     ## Factorate Action
    #     acts = np.stack((
    #         np.take(self.env.actions[:,0], data.a.iloc[:-1]), 
    #         np.take(self.env.actions[:,1], data.a.iloc[:-1])
    #     ), axis=-1)
    #     ##

    #     s = torch.from_numpy(data['s'].iloc[:-1].apply(pd.Series).to_numpy()).type(torch.DoubleTensor)
    #     a = torch.from_numpy(acts).type(torch.DoubleTensor)
    #     s_ = torch.from_numpy(data['s'].iloc[1:].apply(pd.Series).to_numpy()).type(torch.DoubleTensor)

    #     return s, a, s_

    def decompose_data(self, data):
        ## Factorate Action
        acts = np.stack((
            np.take(self.env.actions[:,0], data.a.iloc[:]), 
            np.take(self.env.actions[:,1], data.a.iloc[:])
        ), axis=-1)
        ##

        s = torch.from_numpy(data['s'].iloc[:].apply(pd.Series).to_numpy()).type(torch.DoubleTensor)
        a = torch.from_numpy(acts).type(torch.DoubleTensor)
        s_ = torch.from_numpy(data['s_'].iloc[:].apply(pd.Series).to_numpy()).type(torch.DoubleTensor)

        return s, a, s_

    def batch_train(self, historic_data, epochs=100, lr=1e-4, momentum=.9, log=False):
        s, a, s_ = self.decompose_data(historic_data)
        optim = torch.optim.SGD(list(self.model.parameters()) + [self.params], lr=lr, momentum=momentum)
        register = []
        
        self.model.train(True)
        for epoch in range(epochs):
            optim.zero_grad()
            outputs = self.model(s)

            ll = self.loss(outputs, self.params, s, a, s_)
            if log:
                print(epoch, ll.item(), torch.exp(self.params))
            ll.backward()
            optim.step() 
            register.append(ll.item())
        self.model.train(False)

        return register
    
    def plot(self, ax=None, t=None, n=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        
        size = 10
        res = 50
        lin = np.linspace(-size, size, res).reshape(-1,1)
        X,Y = np.meshgrid(lin, lin)

        d = torch.from_numpy( np.stack((X, Y), axis=-1).reshape(-1, 2) ).type(torch.DoubleTensor)
        with torch.no_grad():
            corr = self.model(d)[:,1].reshape(int(X.size**(1/2)), int(X.size**(1/2)))
            p = ax.imshow(corr, extent=(int(min(lin))-1, int(max(lin))+1, int(max(lin))+1, int(min(lin))-1), vmin = 0, vmax = 1)
            plt.colorbar(p)
        ax.invert_yaxis()
        
        return ax




