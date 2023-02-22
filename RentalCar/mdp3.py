import numpy as np
from scipy.stats import poisson

class MDP:
    ACTIONS = []
    STATES = []
    REWARDS = [] 

    Pi = []
    V = []
    # Q = []

    def __init__(self):
        self.V = np.zeros(len(self.STATES))
        self.Pi = np.zeros(len(self.STATES), dtype=int)

    def transition(self, s, a):
        s_ = s
        return s_

    def p(self, s_=None, r=None, s=None, a=None):
        return 1

    def action_value_function(self, s, a, gamma=0.9):
        return sum(self.p(s_,r, s,a)*(r + gamma*self.V[s_]) for s_, _ in enumerate(self.STATES) for r, _ in enumerate(self.REWARDS))

    def state_value_optimality_function(self, s, gamma=0.9):
        return np.argmax([self.action_value_function(s, a, gamma) for a, _ in enumerate(self.ACTIONS)])

    def policy_evaluation(self, theta=1e-4, inplace=True):
        newV = self.V if inplace else np.zeros_like(self.V)
        i = 0
        while True:
            delta=0
            for s,_ in enumerate(self.STATES):
                v = newV[s]
                newV[s] = self.action_value_function(s, self.Pi[s])
                delta = max(delta, np.abs(v-newV[s]))
            i += 1
            if delta < theta: 
                return np.round(newV, decimals=2, out=newV), i

    def policy_iteration(self):
        policy_stable = False
        while not policy_stable:
            _, i = self.policy_evaluation()
            for s,_ in enumerate(self.STATES):
                old_action = self.Pi[s]               
                self.Pi[s] = self.state_value_optimality_function(s)
                if old_action == self.Pi[s] or i == 1:
                    policy_stable = True
        return self.V, self.Pi

