from typing import Optional

import numpy as np

import gym
from gym import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class NormalMoveEnv():
    def __init__(self, punishment=-1e3, psi=5, sigma=(.005,.3), tau=(.5, 1.), wolrd_size=10., goal=[[5.,5.],[9.,9.]]):
        self.wolrd_size = wolrd_size
        self.goal = goal
        self.punishment = punishment
        self.actions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        self.psi, self.sigma, self.tau = psi, sigma, tau
        self.theta = [psi, sigma, tau]

        self.beta = lambda s: 1*(np.sum(s**2)**(1/2) < self.psi)
        # self.mu = lambda s,a: (s + self.actions[a]*self.force + np.random.normal(0, self.sigma[self.beta(s)], 2)).astype(np.float32)
        self.mu = lambda s,a: (s + self.actions[a]*self.tau[self.beta(s)] + np.random.normal(0, self.sigma[self.beta(s)], 2)).astype(np.float32)

        # self.goal = spaces.Box(low=5., high=9., shape=(2,), dtype=np.float32)
        self.goal = spaces.Box(low=np.array(goal[0], dtype=np.float32), high=np.array(goal[1], dtype=np.float32), dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-wolrd_size, high=wolrd_size, shape=(2,), dtype=np.float32)

        self.state = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = np.array([0.,0.], dtype=np.float32)
        return self.state
        
    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        # m = self.beta(self.state)
        new_state = self.mu(self.state, action)

        out_bound = (not self.observation_space.contains(new_state))
        if not out_bound:
            self.state = new_state

        on_goal = self.goal.contains(self.state)
        # terminated = on_goal or out_bound
        reward = on_goal + (self.punishment if out_bound else -1)

        return self.state, reward, on_goal
    
    def plot(self, ax=None, background=True):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        # ax.invert_yaxis()

        if background:
            size = 10
            res = 50
            lin = np.linspace(-size, size, res).reshape(-1,1)
            X,Y = np.meshgrid(lin,lin)
            t = [self.beta(s) for row in np.dstack((X,Y)) for s in row]

            p = ax.imshow(np.array(t).reshape(res, res), extent=(int(min(lin))-1, int(max(lin))+1, int(max(lin))+1, int(min(lin))-1), vmin = 0, vmax = 1)
            plt.colorbar(p)
            ax.invert_yaxis()

        diff = lambda a,b: b-a
        # ax.add_patch(Circle((0, 0), self.psi+self.force, edgecolor='red', fill=False))
        ax.add_patch(Circle((0, 0), self.psi+(sum(self.tau)/len(self.tau)), edgecolor='red', fill=False))

        length = diff(*list(zip(self.goal.low, self.goal.high))[0])
        ax.add_patch(Rectangle(self.goal.low, length, length, edgecolor='green', fill=False))
        return ax
    

    