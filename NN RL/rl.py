from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment import Trajectory
from nn import Model

def epsilon_greedy(S, Q, e=.2, z=4):
    explore = np.random.choice([0,1], p=[e, 1-e])
    return np.random.randint(0,z) if explore else np.argmax(Q[S])

class Agent:
    def __init__(self, env, model=None, alpha=.1, gamma=.9, epsilon=.3, state_space_size=50, init_seed=None, **model_params):
        self.env = env
        self.model =  model or Model(self.env, **model_params)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_space_size = state_space_size

        if init_seed is None:
            self.Q = np.zeros(shape=[self.state_space_size, self.state_space_size, 4])
        else:
            np.random.seed(init_seed)
            self.Q = np.random.rand(self.state_space_size, self.state_space_size, 4)
        
        self.trajectories = []
        self.lls = []

    def discretize_state(self, s):
        bounds = [self.env.observation_space.low.tolist(), self.env.observation_space.high.tolist()]
        discretizer = KBinsDiscretizer(n_bins=self.state_space_size, encode='ordinal', strategy='uniform')
        discretizer.fit(bounds)
        discrete_state = discretizer.transform([s])[0]
        return tuple(map(int, discrete_state))

        # return (np.array(s)*100).astype(int)  

    def plan(self):
        pass
    
    def episode(self, log=False):
        step, terminated = 0, False,
        trajectory = Trajectory()

        observation = self.env.reset()
        S = self.discretize_state(observation)
        if log:
            print(observation, S)

        while not terminated:
            # take action
            A = epsilon_greedy(S, self.Q, self.epsilon)
            observation_, R, terminated = self.env.step(A)
            trajectory.step(step, observation, A, R, observation_, terminated)
            
            S_ = self.discretize_state(observation_) 
            if log:
                print(observation_, S_)
                print(step, observation, S,A,R, observation_,S_)

            # Learning Step
            self.Q[(*S, A)] += self.alpha*(R + self.gamma * np.max(self.Q[S_]) - self.Q[(*S, A)])
            S, observation = S_, observation_
            step += 1     
        return trajectory

    def train(self, episodes=100, epochs=100, log=False, save=True, model_learn=True, seed=0):
        np.random.seed(seed)

        for epi in range(episodes):
            trajectory = self.episode(log) # Controle
            ll = self.model.batch_train(trajectory.run, epochs) if model_learn else [None]
            self.plan() # Predição
            if log:
                print(ll[0], ll[-1])
                self.model.plot(trajectory.plot(self.env.plot(background=False))).get_figure().savefig(f'logs\epi-{epi}.png')
            if save:
                self.trajectories.append(trajectory)
                self.lls.append(ll)

    def plot(self, ax=None, type='total_reward'):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 4))

        if type == "total_reward":
            ax.plot([t.run.sum().r for t in self.trajectories])
        if type == "cumulative_reward":
            ax.plot(np.cumsum([t.run.sum().r for t in self.trajectories]))
        if type == "moving_avg":
            ax.plot(np.convolve([t.run.sum().r for t in self.trajectories], np.ones(100)/100, mode='valid'))
        if type == "cumulative_avg":
            ax.plot(np.cumsum([t.run.sum().r for t in self.trajectories]) / len(self.trajectories))
        return ax
    
    