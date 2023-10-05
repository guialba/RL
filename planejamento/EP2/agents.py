import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation  as animation 

from discretizers import fixed_discretized_state


def epsilon_greedy(S, Q, e):
    explore = np.random.choice([0,1], p=[e, 1-e])
    return (np.random.randint(0,2), np.argmax(Q[S]))[explore]


class RLAgent:
    def __init__(self, env, alpha=.5, gamma=.9, epsilon=.1, state_shape=(3, 3, 10, 10), discretizer=fixed_discretized_state, init_seed=None):
        self.dicretizer = discretizer
        self.state_shape = state_shape
        
        self.env = env
       
        self.alpha = alpha
        self.gamma = gamma 
        self.epsilon = epsilon

        if init_seed is None:
            self.Q = np.zeros(shape=[*state_shape, 2])
        else:
            np.random.seed(init_seed)
            self.seeds = np.random.rand(*(state_shape, 2))
        self.data = pd.DataFrame(data={'episode':[], 'step':[], 'obs':[], 'S':[], 'A':[], 'R':[]})


    def step(self, episodes=1000, seed=0):
        np.random.seed(seed)
        self.seeds = np.random.randint(low=0, high=1e6, size=episodes)
        pass

    def train(self, episodes=1000, log=False, seed=0):
        for episode  in self.step(episodes, log, seed):
            self.data = pd.concat([self.data, pd.DataFrame(data=episode)], ignore_index=True)

    def play(self):
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams['figure.dpi'] = 150  
        # plt.ioff()

        fig, _ = plt.subplots()
        frames = []
        truncated, terminated = False, False
        observation,_ = self.env.reset(seed=0)
        S = self.dicretizer(observation, self.state_shape)
        while not (truncated or terminated):
            frames.append(self.env.render())
            observation, _, terminated, truncated, _ = self.env.step(np.argmax(self.Q[S]))
            S = self.dicretizer(observation, self.state_shape) 

        def animate(t):
            plt.imshow(frames[t], animated=True)
        return animation.FuncAnimation(fig, animate, frames=len(frames))

    def plot(self, axs=None):
        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(10, 4))
        axs.plot(self.data[['episode', 'R']].groupby(['episode']).sum())
        return axs


class SARSA(RLAgent):
    def step(self, episodes=1000, log=False, save=False, seed=0):
        super().step(episodes, seed)

        for episode in range(episodes):
            terminated, truncated = False, False
            step = 0

            observation,_ = self.env.reset(seed=int(self.seeds[episode]))
            if log:
                print(observation)

            S = self.dicretizer(observation, self.state_shape)
            A = epsilon_greedy(S, self.Q, self.epsilon)

            data = {'episode': [episode], 'step':[np.nan], 'obs':[observation], 'S':[S], 'A':[A], 'R':[0]}
            while not (truncated or terminated):
                observation, R, terminated, truncated, _ = self.env.step(A)
                if log:
                    print(observation)
                S_ = self.dicretizer(observation, self.state_shape) 
                A_ = epsilon_greedy(S_, self.Q, self.epsilon)
    
                step += 1

                if log:
                    print(episode, step, S,A,R,S_,A_)
                
                self.Q[(*S, A)] += self.alpha*(R + self.gamma*self.Q[(*S_, A_)] - self.Q[(*S, A)])
                S, A = S_, A_

                data['episode'].append(episode)
                data['step'].append(step)
                data['obs'].append(observation)
                data['S'].append(S)
                data['A'].append(int(A))
                data['R'].append(int(R))

            if save:
                self.data = pd.concat([self.data, pd.DataFrame(data=data)], ignore_index=True)

            yield data


class QLearning(RLAgent):  
    def step(self, episodes=1000, log=False, save=False, seed=0):
        super().step(episodes, seed)

        for episode in range(episodes):
            terminated, truncated = False, False
            step = 0

            observation,_ = self.env.reset(seed=int(self.seeds[episode]))
            if log:
                print(observation)

            S = self.dicretizer(observation, self.state_shape)

            data = {'episode': [episode], 'step':[np.nan], 'obs':[observation], 'S':[S], 'A':[np.nan], 'R':[0]}
            while not (truncated or terminated):
                A = epsilon_greedy(S, self.Q, self.epsilon)
                observation, R, terminated, truncated, _ = self.env.step(A)
                if log:
                    print(observation)

                S_ = self.dicretizer(observation, self.state_shape) 

                step += 1

                if log:
                    print(episode, step, S,A,R,S_)
                
                self.Q[(*S, A)] += self.alpha*(R + self.gamma * np.max(self.Q[S_]) - self.Q[(*S, A)])
                S = S_

                data['episode'].append(episode)
                data['step'].append(step)
                data['obs'].append(observation)
                data['S'].append(S)
                data['A'].append(int(A))
                data['R'].append(int(R))

            if save:
                self.data = pd.concat([self.data, pd.DataFrame(data=data)], ignore_index=True)

            yield data
