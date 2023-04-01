from IPython.display import clear_output
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from queue import PriorityQueue
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

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
        self.trajectory = []
        self.totalReward = 0
        self.steps = 0
        self.current_state = (0,0)

    def takeAction(self, S, A):
        self.trajectory.append({'s': S, 'a': A})
        S_, R, end = self.g.transition(S, A)

        self.current_state = S_
        self.totalReward += R
        self.steps += 1

        return S_, R, end


    def run(self):
        actions = ['up', 'down', 'right', 'left', 'w', 's', 'd', 'a']
        while True:
            S = self.current_state
            clear_output(wait=False)
            print('Steps:', self.steps) 
            print('Rewards:', self.totalReward) 
            self.g.print(display=S) 
            print(S) 
            time.sleep(.2)

            a = input()
            end = False
            if a in actions:
                A = self.g.ACTIONS[actions.index(a.lower())%4]
                print(A)
                _, _, end = self.takeAction(S, A)
            
            if 'exit' in a or end:
                clear_output(wait=False)
                print('Steps:', self.steps) 
                print('Rewards:', self.totalReward) 
                self.g.print(display=S) 
                print(S) 
                break

class AI(Human):
    def __init__(self, grid):
        super().__init__(grid)
        self.S = [(x, y) for y in range(self.g.size[0]) for x in range(self.g.size[1])]
        self.Q = {s: {a:0 for a in self.g.ACTIONS} for s in self.S}

        self.observations = {s: {a:[] for a in self.g.ACTIONS} for s in self.S}

    def takeAction(self, S, A):
        S_, R, end = super().takeAction(S, A)
        self.observations[S][A].append((S_, R)) 
        return S_, R, end

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        
        # Value
        getValue = lambda x: max(self.Q[x], key=self.Q[x].get)

        v_arr = {s: (getValue(s), self.Q[s][getValue(s)]) for s in self.Q}
        v = np.array([item[1] for item in v_arr.values()])
        v = v.reshape([*self.g.size])

        v_vis = {s: (getValue(s), len(self.observations[s][getValue(s)])) for s in self.observations}
        vis = np.array([item[1] for item in v_vis.values()])
        vis = vis.reshape([*self.g.size])
        
        ax.imshow(v)

        # Start Marker
        plt.scatter(0, 0, marker='o', s=1000, c='w', alpha=0.5)
        # Goals Markers
        for pos, eff in self.g.effects.items():
            if 'terminal' in eff: 
                plt.scatter(*pos, marker='*', s=1000, c='w', alpha=0.5)

        #Policy
        xr = range(v.shape[1])
        yr = range(v.shape[0])

        # arrows = np.array([(item[0][0], -item[0][1]) for item in v_arr.values()])
        arrows = np.array([(item[0][0], -item[0][1]) for item in v_arr.values()])
        arrows = arrows.reshape([*self.g.size, 2])

        arr = ax.quiver(xr, yr, arrows[:, :, 0], arrows[:, :, 1], pivot='mid', color=(0,0,0,1))

        for iy, ix in np.ndindex(v.shape):
            ax.quiverkey(arr, ix, iy, 1, label=f'{round(v[iy, ix], 2)}', labelpos='N', coordinates='data', color=(0,0,0,0))
            ax.quiverkey(arr, ix, iy, 1, label=f'{int(vis[iy, ix])}', labelpos='S', coordinates='data', color=(0,0,0,0))

        # return ax , arrows, arr

class QLearning(AI):
    def run(self, alpha=0.5, epsilon=0.5, gamma=0.9):
        while True:
            S = self.current_state
            A = e_greedy(self.Q[S], epsilon)
            S_, R, end = self.takeAction(S, A)

            a = max(self.Q[S_], key=self.Q[S_].get)
            self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])

            if end:
                break

class DynaQ(AI):
    def __init__(self, grid):
        super().__init__(grid)
        self.MODEL = {s: {a: (0,0) for a in self.g.ACTIONS} for s in self.S}     

    def run(self, n=5, alpha=0.1, epsilon=0.1, gamma=0.9):
        while True:
            S = self.current_state
            A = e_greedy(self.Q[S], epsilon)
            S_, R, end = self.takeAction(S, A)
            a = max(self.Q[S_], key=self.Q[S_].get)
            self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])
            self.MODEL[S][A] = (S_, R)

            # Planing
            for _ in range(n):
                observed_s = [s for s in self.observations for a in self.observations[s] if len(self.observations[s][a])>0]
                S = random.choice(observed_s)
                
                observed_a = [a for a in self.observations[S] if len(self.observations[S][a])>0]
                A = random.choice(observed_a)
                
                S_, R = self.MODEL[S][A]
                a = max(self.Q[S_], key=self.Q[S_].get)
                self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])
            
            if end:
                break

class PiorSweep(AI):
    def __init__(self, grid):
        super().__init__(grid)
        self.MODEL = {s: {a: (0,0) for a in self.g.ACTIONS} for s in self.S}    

    def run(self, n=5, alpha=0.1, epsilon=0.1, gamma=0.9, theta=0.1):
        PQueue = PriorityQueue()

        while True:
            S = self.current_state
            A = e_greedy(self.Q[S], epsilon)
            S_, R, end = self.takeAction(S, A)
            self.MODEL[S][A] = (S_, R)
            a = max(self.Q[S_], key=self.Q[S_].get)
            P = abs(R + gamma*self.Q[S_][a] - self.Q[S][A])
            if P > theta: PQueue.put((P, S, A))
            
            # Planing
            step = 0
            while step < n and not PQueue.empty():
                    _, S, A = PQueue.get()
                    S_, R = self.MODEL[S][A]
                    a = max(self.Q[S_], key=self.Q[S_].get)
                    self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])

                    exp_s_a = [(s, a, self.observations[s][a]) for s in self.observations for a in self.observations[s] if len(self.observations[s][a]) > 0]
                    predicted = [(s, a, r) for s, a, exp in exp_s_a for s_, r in exp if s_== S]
                    for S__, A__, R__ in predicted:
                        a = max(self.Q[S], key=self.Q[S].get)
                        P = abs(R__ + gamma*self.Q[S_][a] - self.Q[S__][A__])
                        if P > theta: PQueue.put((P, S__, A__))
                    step += 1

            if end: break


class BehavioralCloning(AI):
    def __init__(self, grid):
        super().__init__(grid)
        
        self.Pi = {s: 0 for s in self.S}

    def estimate_Pi(self, x):
        n = np.array([sum(all(x[n][0] == s) and all(x[n][1] == a) for n in range(len(x))) for a in self.g.ACTIONS for s in self.g.S]).reshape(len(self.g.ACTIONS), len(self.g.S)).T
        p = lambda s,a: n[s,a] / sum(n[s]) 
        return np.array([[p(s,a) for a in range(len(self.g.ACTIONS))] for s in range(len(self.g.S))])

    def estimate_P(self, x):
        n = np.array([sum(all(x[n][0] == s) and all(x[n][1] == a) and all(x[n+1][0] == s_) for n in range(len(x)-1)) for s in self.g.S for a in self.g.ACTIONS for s_ in self.g.S]).reshape(len(self.g.S), len(self.g.ACTIONS), len(self.g.S)).T
        p = lambda s,a,s_: n[s,a,s_] / sum(n[s,a]) 
        return np.array([[[p(s,a, s_) for s_ in range(len(self.g.S))] for a in range(len(self.g.ACTIONS))] for s in range(len(self.g.S))])

    def estimatePiModel(self,x):
        X, y = list(zip(*x))
        regX = LogisticRegression().fit(X, np.array(y)[:,0])
        regY = LogisticRegression().fit(X, np.array(y)[:,1])
        self.pi = lambda p: (*regX.predict(np.array([p])).round(), *regY.predict(np.array([p])).round())
    

    def imitate(self, trajectory):
        xi = np.array([(step['s'], step['a']) for step in trajectory])
        # pi = self.estimate_Pi(xi)
        # p = self.estimate_P(xi)
        self.estimatePiModel(xi)
        # return pi, p
        

        

    def run(self, alpha=0.5, gamma=0.9):
        while True:
            S = self.current_state
            A = self.Pi(self.Q[S])
            S_, R, end = self.takeAction(S, A)

            a = max(self.Q[S_], key=self.Q[S_].get)
            self.Q[S][A] = self.Q[S][A] + alpha*(R + gamma* self.Q[S_][a] - self.Q[S][A])

            if end:
                break

