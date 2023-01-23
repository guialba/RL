import random
import pandas as pd
import numpy as np

class Bandit:
    def __init__(self, k=10, e=0.1, alpha=None, nonStationaryParams=(0,0)):
        self.k = k
        self.e = e
        self.alpha = alpha
        self.nonStationaryParams = nonStationaryParams
        self.fixed_alpha = (alpha is not None)
        self.age = 0

        self.config = {
            'k': self.k,
            'e': self.e,
            'alpha': self.alpha,
            'nonStationaryParams': self.nonStationaryParams
        }

        self.reset()

    def reset(self):
        self.q_star = np.random.normal(0, 1, self.k)
        self.best_action = np.argmax(self.q_star)
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)

    def increaseNonStationeryFactor(self):
        nonStationaryFactor =  np.random.normal(*self.nonStationaryParams, self.k)
        self.q_star += nonStationaryFactor 
        self.best_action = np.argmax(self.q_star)

    def chooseAction(self):
        should_explore = np.random.rand() < self.e
        explore = np.random.choice(self.k)
        exploit = np.argmax(self.Q)

        return explore if should_explore else exploit

    def getReward(self, action):
        return np.random.normal(self.q_star[action], 1)
    
    
    def run(self, steps=1000):
        for step in range(steps):
            self.increaseNonStationeryFactor()

            action = self.chooseAction()
            reward = self.getReward(action)

            self.age += 1
            self.N[action] += 1

            alpha = self.alpha if self.fixed_alpha else (1/self.N[action])
            self.Q[action] += alpha * (reward - self.Q[action])

            yield step, action, reward


class BanditExperiment:
    def __init__(self, steps=1000, bandits_config=None, runs=2000, bandits=None):
        self.steps = steps
        self.runs = runs
        self.bandits_config = bandits_config if type(bandits_config) == type([]) else [bandits_config]
        
        self.results = pd.DataFrame({"bandit": [],"run": [], "steps": [], "actions": [], "rewards": [], "optimal_actions": [], "Q": [], "q_star": []})
        
        if bandits is None:
            self.bandits = {cfg_id: [Bandit(**cfg) for _ in range(self.runs)] for cfg_id, cfg in enumerate(self.bandits_config)}
        else:
            self.bandits = bandits


    def getResults(self):
        for bandit_id, bandits in self.bandits.items():
            for run_id, bandit in enumerate(bandits):
                for step, action, reward in bandit.run(self.steps):
                    temp = pd.DataFrame({"bandit": [bandit_id], "run": [run_id], "steps": [step], "actions": [action], "rewards": [reward], "optimal_actions": [bandit.best_action], "Q": [bandit.Q], "q_star": [bandit.q_star]})
                    self.results = pd.concat([self.results, temp])
        return self.results, self.bandits

