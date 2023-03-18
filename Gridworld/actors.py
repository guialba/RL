from IPython.display import clear_output
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def e_greedy(values, e):
    should_explore = np.random.rand() < e

    explore = random.choice(list(values.keys()))
    exploit = max(values, key=values.get)
    # exploit = list(values.keys())[np.argmax(list(values.values()))]
    # return max(values, key=values.get)

    return explore if should_explore else exploit


class Human:
    def __init__(self, grid):
        self.g = grid
        self.resetCount()

    def resetCount(self):
        self.totalReward = 0
        self.steps = 0

    def run(self):
        actions = ['up', 'down', 'right', 'left', 'w', 's', 'd', 'a']
        pos = (0,0)
        while True:
            clear_output(wait=False)
            print('Steps:', self.steps) 
            print('Rewards:', self.totalReward) 
            self.g.print(display=pos) 
            print(pos) 
            time.sleep(.2)

            a = input()
            end = False
            if a in actions:
                print(self.g.ACTIONS[actions.index(a.lower())%4])
                pos, r, end = self.g.transition(pos, 
                                            self.g.ACTIONS[actions.index(a.lower())%4]
                                        )
                self.totalReward += r
                self.steps += 1
            
            if 'exit' in a or end:
                clear_output(wait=False)
                print('Steps:', self.steps) 
                print('Rewards:', self.totalReward) 
                self.g.print(display=pos) 
                print(pos) 
                break

class AI(Human):
    def __init__(self, grid):
        super().__init__(grid)
        self.S = [(x, y) for y in range(self.g.size[0]) for x in range(self.g.size[1])]
        self.Q = {s: {a:0 for a in self.g.ACTIONS} for s in self.S}

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        
        getValue = lambda x: max(self.Q[x], key=self.Q[x].get)
        v_arr = {s: (getValue(s), self.Q[s][getValue(s)]) for s in self.Q}

        v = np.array([item[1] for item in v_arr.values()])
        v = v.reshape([*self.g.size])
        ax.imshow(v)

        xr = range(v.shape[1])
        yr = range(v.shape[0])

        arrows = np.array([item[0] for item in v_arr.values()])
        arrows = arrows.reshape([*self.g.size, 2])

        arr = ax.quiver(xr, yr, arrows[:, :, 0].T, arrows[:, :, 1].T, pivot='mid')

        for iy, ix in np.ndindex(v.shape):
            angle = math.atan2(arrows[ix, iy, 1], arrows[ix, iy, 0]) * 180 / math.pi 
            ax.quiverkey(arr, ix, iy, 1, label=f'{round(v[iy, ix], 2)}', angle=angle, labelpos='N', coordinates='data')


        # Start Marker
        plt.scatter(0, 0, marker='o', s=1000, c='w', alpha=0.5)
        # Goals Markers
        for pos, eff in self.g.effects.items():
            if 'terminal' in eff: 
                plt.scatter(*pos, marker='*', s=1000, c='w', alpha=0.5)

        return ax , arrows, arr

class QLearning(AI):
    def run(self, alpha=0.5, epsilon=0.1, gamma=0.9):
        current_state = (0,0)
        while True:
            S = current_state
            # A = random.choice(self.g.ACTIONS) if random.random() < epsilon else max(self.Q[S], key=self.Q[S].get)
            A = e_greedy(self.Q[S], epsilon)
            S_, R, end = self.g.transition(S, A)
            a = max(self.Q[S_], key=self.Q[S_].get)
            self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])
            
            current_state = S_
            self.totalReward += R
            self.steps += 1

            if end:
                break

class DynaQ(AI):
    def __init__(self, grid):
        super().__init__(grid)
        self.MODEL = {s: {a: (0,0) for a in self.g.ACTIONS} for s in self.S}        
        self.observations = {s: {a:0 for a in self.g.ACTIONS} for s in self.S}

    def run(self, n=5, alpha=0.1, epsilon=0.1, gamma=0.9):
        current_state = (0,0)
        while True:
            S = current_state
            A = e_greedy(self.Q[S], epsilon)
            S_, R, end = self.g.transition(S, A)
            a = max(self.Q[S_], key=self.Q[S_].get)
            self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])
            self.MODEL[S][A] = (S_, R)

            current_state = S_
            self.totalReward += R
            self.steps += 1
            self.observations[S][A] += 1 
            
            # Planing
            for _ in range(n):
                observed_s = {s:self.observations[s] for s in self.observations if sum(self.observations[s].values())>0}
                S = random.choice(list(observed_s.keys()))
                
                observed_a = {a:observed_s[S][a] for a in observed_s[S] if observed_s[S][a] > 0}
                A = random.choice(list(observed_a.keys()))
                
                S_, R = self.MODEL[S][A]
                a = max(self.Q[S_], key=self.Q[S_].get)
                self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])
            
            if end:
                break

    def plot(self):
        ax, arrows, arr = super().plot()
        getValue = lambda x: max(self.Q[x], key=self.Q[x].get)
        
        v_vis = {s: (getValue(s), self.observations[s][getValue(s)]) for s in self.observations}
        vis = np.array([item[1] for item in v_vis.values()])
        vis = vis.reshape([*self.g.size])
        for iy, ix in np.ndindex(vis.shape):
            angle = math.atan2(arrows[ix, iy, 1], arrows[ix, iy, 0]) * 180 / math.pi 
            ax.quiverkey(arr, ix, iy, 1, label=f'{int(vis[iy, ix])}', angle=angle, labelpos='S', coordinates='data')



