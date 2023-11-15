from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment import Trajectory
from nn import Model

class Agent:
    def __init__(self, env, td_model_steps=5, state_space_size=50, model=None, **model_params):
        self.env = env
        self.td_model_steps = td_model_steps
        self.model =  model or Model(self.env, **model_params)

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
        calc_dist = lambda s_: ((self.env.goal.low[0]-s_[0])**2 + (self.env.goal.low[1]-s_[1])**2)**(1/2)
        sigma, tau = self.model.infer(s)
        dists = np.array([calc_dist(self.env.transition(s,a,tau,sigma)) for a in range(4)])

        return np.argmin(dists)

    
    def episode(self, size_limit=None, log=False):
        step, terminated,  = 0, False,
        trajectory = Trajectory()

        S = self.env.reset()
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
                terminated = (step >= size_limit)

        self.trajectory = trajectory  
        return trajectory

    def train(self, trajectory, epochs=1000, log=False, seed=0):
        np.random.seed(seed)

        ll = self.model.batch_train(trajectory.run, epochs)
        if log:
            print(ll[0], ll[-1])
            # self.model.plot(trajectory.plot(self.env.plot(background=False))).get_figure().savefig(f'logs\epi-{epi}.png')


    def plot(self, params=False):
        if params:
            self.model.plot(self.trajectory.plot(self.env.plot(background=False)), param=0)
            self.model.plot(self.trajectory.plot(self.env.plot(background=False)), param=1)
        else: 
           self.model.plot(self.trajectory.plot(self.env.plot(background=False)))
    
    