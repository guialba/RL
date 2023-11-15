from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment import Trajectory
from nn import Model

class Agent:
    def __init__(self, env, td_model_steps=5, memory_size=0, state_space_size=50, model=Model, **model_params):
        self.env = env
        self.td_model_steps = td_model_steps
        self.memory_size = memory_size
        self.model_params = model_params
        self.model =  model(self.env, **model_params)

        self.state_space_size = state_space_size
        self.trajectory = None
        
    def discretize_state(self, s):
        bounds = [self.env.observation_space.low.tolist(), self.env.observation_space.high.tolist()]
        discretizer = KBinsDiscretizer(n_bins=self.state_space_size, encode='ordinal', strategy='uniform')
        discretizer.fit(bounds)
        discrete_state = discretizer.transform([s])[0]
        return tuple(map(int, discrete_state))

        # return (np.array(s)*100).astype(int)  

    def plan(self, s):
        calc_dist = lambda s_: min([((goal.low[0]-s_[0])**2 + (goal.low[1]-s_[1])**2)**(1/2) for goal in self.env.goals])
        sigma, tau = self.model.infer(s)
        dists = np.array([calc_dist(self.env.transition(s,a,tau,sigma)) for a in range(4)])

        return np.argmin(dists)
 
    def episode(self, size_limit=None, log=False):
        if self.trajectory is None:
            step, terminated,  = 0, False,
            trajectory = Trajectory()

            S = self.env.reset()
        else:
            step, terminated,  = 0, self.trajectory.run.iloc[-1].end,
            trajectory = self.trajectory

            S = self.trajectory.run.iloc[-1].s

        if log:
            print(S)

        while not terminated:
            # take action
            A = self.plan(S)
            S_, R, terminated = self.env.step(A)
            trajectory.step(step, S, A, R, S_, terminated)
            
            if log:
                print(step, S,A,R, S_)

            if not(step % self.td_model_steps):
                self.train(trajectory)

            S=S_
            step += 1   
            if size_limit is not None:
                terminated = terminated or (step >= size_limit)

        self.trajectory = trajectory  
        return trajectory

    def train(self, trajectory, epochs=1000, log=False, seed=0):
        np.random.seed(seed)

        ll = self.model.batch_train(trajectory.run[-self.memory_size:], epochs)
        # ll = self.model.batch_train(trajectory.run, epochs)
        if log:
            print(ll[0], ll[-1])
            # self.model.plot(trajectory.plot(self.env.plot(background=False))).get_figure().savefig(f'logs\epi-{epi}.png')

    def plot(self, kind='values'):
        def plot_background(n):
            for k in range(n):
                ax[k] = self.env.plot(ax[k], background=False)
                ax[k] = self.trajectory.plot(ax[k])

        if  str.lower(kind)=='values':
            _, ax = plt.subplots(ncols=self.model.n_params, figsize=(self.model.n_params*5, 5))
            plot_background(self.model.n_params)
            self.model.plot_values(ax)

        elif  str.lower(kind)=='probs':
            _, ax = plt.subplots(ncols=self.model.k, figsize=(self.model.k*5, 5))
            plot_background(self.model.k)
            self.model.plot_probs(ax)

        
    
    