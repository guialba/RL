from typing import Optional

import numpy as np

import gym
from gym import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class NormalMoveEnv():
    def __init__(self):
        self.force = .5
        self.punishment = 1e3
        self.actions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        self.psi, self.sigma = 5, (.005,.3)
        self.theta = [self.psi, self.sigma]

        self.beta = lambda s: 1*(np.sum(s**2)**(1/2) < self.psi)
        self.mu = lambda s,a: (s + self.actions[a]*self.force + np.random.normal(0, self.sigma[self.beta(s)], 2)).astype(np.float32)

        self.goal = spaces.Box(low=5., high=9., shape=(2,), dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10., high=10., shape=(2,), dtype=np.float32)

        self.state = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = np.array([0.,0.], dtype=np.float32)
        return self.state
        
    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        # m = self.beta(self.state)
        self.state = self.mu(self.state, action)

        out_bound = (not self.observation_space.contains(self.state))
        on_goal = self.goal.contains(self.state)

        terminated = on_goal or out_bound
        reward = on_goal - (self.punishment if out_bound else 1)

        return self.state, reward, terminated
    
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
        ax.add_patch(Circle((0, 0), self.psi+self.force, edgecolor='red', fill=False))

        length = diff(*list(zip(self.goal.low, self.goal.high))[0])
        ax.add_patch(Rectangle(self.goal.low, length, length, edgecolor='green', fill=False))
        return ax
    

    