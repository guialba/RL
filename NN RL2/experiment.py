
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class Trajectory:
    def __init__(self, env=None, size=None, policy=None):
        self.policy = (lambda _: np.random.choice(4, 1)[0]) if policy is None else policy
        self.env = env
        self.size = size or np.inf

        self.run = pd.DataFrame({'step':[], 's':[], 'a':[], 'r':[], 's_':[], 'end':[]})
        
        if env is not None:
            self.generate()

    def step(self, step, s, a, r, s_, end):
        self.run = pd.concat([self.run, pd.DataFrame({'step':[int(step)], 's':[s], 'a':[int(a)], 'r':[int(r)], 's_':[s_], 'end':[int(end)]})])

    def generate(self):   
        i, end = 0, False
        s = self.env.reset()
        while not end:
            a = self.policy(s)
            s_, r, end = self.env.step(a)
            i += 1
            self.step(i, s, a, r, s_, end)
            # self.run = pd.concat([self.run, pd.DataFrame({'step':[int(i)], 's':[s], 'a':[int(a)], 'r':[int(r)], 's_':[s_], 'end':[int(end)]})])
            # self.run.append((i, s,a,r,s_ end))
            end = end or (i>=self.size)
            s = s_
        return self.run
    
    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        df_s = pd.DataFrame(self.run.s.to_list() + self.run.s_.to_list()[-1:], columns=['x','y'])
        ax.plot(df_s.x, df_s.y, color='red')

        return ax