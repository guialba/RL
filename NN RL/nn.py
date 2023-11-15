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

class ParModel(nn.Module):
    def __init__(self):
        super(ParModel, self).__init__()
        n_features = 2
        n_params = 2
        n_hidden = 10

        self.model = nn.Sequential(
            nn.Linear(n_features, n_hidden).double(),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden).double(),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden).double(),
            nn.ReLU(),
            nn.Linear(n_hidden, n_params).double(),
        )
        
    def forward(self, x):
        return self.model(x)
    

class BetaModel(nn.Module):
    def __init__(self):
        super(BetaModel, self).__init__()
        n_features = 2
        n_models = 2
        n_hidden = 10

        self.model = nn.Sequential(
            nn.Linear(n_features, n_hidden).double(),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden).double(),
            nn.ReLU(),
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
    def __init__(self, env, model=None, sigmas=None, taus=None, lr=1e-4, momentum=.9):
        self.env = env
        self.model = model or BetaModel()

        self.lr = lr
        self.momentum = momentum

        self.params = []
        if sigmas is None:
            self.sigmas = torch.nn.Parameter(torch.rand(2).type(torch.DoubleTensor))
            self.params.append(self.sigmas)
        else:
            self.sigmas = torch.log(torch.tensor([*sigmas])).type(torch.DoubleTensor)

        if taus is None:
            self.taus = torch.nn.Parameter(torch.rand(2).type(torch.DoubleTensor))
            self.params.append(self.taus)
        else:
            self.taus = torch.tensor([*taus]).type(torch.DoubleTensor)


    def loss(self, beta, sigma, tau, s, a, s_):
        ## Get Probability Values
        mi = (
            torch.exp(
                normal.Normal(
                    # ( s[:,0] + a[:,0]*force).reshape((-1,1)), torch.exp(sigma)
                    (s[:,0] + torch.outer(a[:,0], tau).T).T, torch.exp(sigma)
                ).log_prob( s_[:,0].reshape((-1,1)) )
            ) 
            * 
            torch.exp(
                normal.Normal(
                    # ( s[:,1] + a[:,1]*force).reshape((-1,1)), torch.exp(sigma)
                    (s[:,1] + torch.outer(a[:,1], tau).T).T, torch.exp(sigma)
                ).log_prob( s_[:,1].reshape((-1,1)) )
            )
        )
        ##

        p_theta = torch.sum((mi * beta), 1) 
        return -torch.sum(torch.log(p_theta))
   
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


    def batch_train(self, historic_data, epochs=100, log=False):
        s, a, s_ = self.decompose_data(historic_data)       
        optim = torch.optim.SGD(list(self.model.parameters()) + self.params, lr=self.lr, momentum=self.momentum)
        register = []
        
        self.model.train(True)
        for epoch in range(epochs):
            optim.zero_grad()
            outputs = self.model(s)

            ll = self.loss(outputs, self.sigmas, self.taus, s, a, s_)
            if log:
                print(epoch, ll.item(), torch.exp(self.sigmas), self.taus)
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
        ax.text(-size,size+1.5, f'sigma: {torch.round(torch.exp(self.sigmas), decimals=3).tolist()}, tau: {torch.round(self.taus, decimals=3).tolist()}')

        return ax




class GeneralModel:
    def __init__(self, env, model=None, lr=1e-4, momentum=.9):
        self.env = env
        self.model = model or ParModel()

        self.lr = lr
        self.momentum = momentum
        self.optim = torch.optim.SGD(list(self.model.parameters()), lr=self.lr, momentum=self.momentum)

    def loss(self, params, s, a, s_):
        p1, p2 = torch.exp(params[:, 0]), params[:, 1]
        # p2 = torch.take(torch.tensor(self.env.sigma), 1*(torch.sum(s**2, 1)**(1/2) < self.env.psi))
        distribution = normal.Normal(0, p1)
        value = s_- (s + a*p2.reshape((-1,1)))
        return -torch.sum( distribution.log_prob(value[:,0]) + distribution.log_prob(value[:,1]) )

   
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


    def batch_train(self, historic_data, epochs=100, log=False):
        s, a, s_ = self.decompose_data(historic_data)       
        register = []
        
        self.model.train(True)
        for epoch in range(epochs):
            self.optim.zero_grad()
            outputs = self.model(s)
            # if log:
            #     # print(epoch, torch.any(outputs[:,0] == 0.0), torch.any(outputs[:,1] == 0.0))
                # print(epoch, torch.any(outputs == 0.0), torch.any(outputs == np.inf), torch.any(outputs == np.nan))

            ll = self.loss(outputs, s, a, s_)
            if log:
                print(epoch, ll.item())

            ll.backward()
            self.optim.step() 
            register.append(ll.item())
        self.model.train(False)

        return register
    
    def plot(self, ax=None, param=0):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        
        size = 10
        res = 50
        lin = np.linspace(-size, size, res).reshape(-1,1)
        X,Y = np.meshgrid(lin, lin)

        d = torch.from_numpy( np.stack((X, Y), axis=-1).reshape(-1, 2) ).type(torch.DoubleTensor)
        with torch.no_grad():
            corr = self.model(d)[:,param].reshape(int(X.size**(1/2)), int(X.size**(1/2)))
            bounds = {'vmin': min(self.env.theta[1:][param]), 'vmax': max(self.env.theta[1:][param])}
            # bounds = {}
            p = ax.imshow(  corr if param>0 else torch.exp(corr), 
                            extent=(
                                int(min(lin))-1, int(max(lin))+1, 
                                int(max(lin))+1, int(min(lin))-1
                            ), 
                            **bounds
                        )
            plt.colorbar(p)
        ax.set_title(f'theta_{param}')
        ax.invert_yaxis()
        # ax.text(-size,size+1.5, f'sigma: {torch.round(torch.exp(self.sigmas), decimals=3).tolist()}, tau: {torch.round(self.taus, decimals=3).tolist()}')

        return ax
    