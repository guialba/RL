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

    def p(self, s=None, a=None):
        return 1

    def bellman_expectation(self, s_, r, gamma=0.9):
        result = 0
        for s, _ in enumerate(self.STATES):
            # for a, _ in enumerate(self.ACTIONS):
                # for s2, _ in enumerate(self.STATES):
            if (s, s_) in self.ps: 
                result += self.ps[s, s_]*(r + gamma*self.V[s_])
        return result

    def policy_evaluation(self, theta=1e-4, inplace=True):
        newV = self.V if inplace else np.zeros_like(self.V)
        i = 0
        while True:
            delta=0
            for s,_ in enumerate(self.STATES):
                v = newV[s]
                newV[s] = self.bellman_expectation(*self.transition(s, self.Pi[s]))
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
                acts = [self.bellman_expectation(*self.transition(s, a)) for a,_ in enumerate(self.ACTIONS)]
                self.Pi[s] = np.argmax(acts)
                if old_action == self.Pi[s] or i == 1:
                    policy_stable = True
        return self.V, self.Pi

class Grid(MDP):
    north = (-1, 0)
    south = (1, 0)
    east = (0, 1)
    west = (0, -1)
    
    def __init__(self, 
                size=(5,5), 
                special_transitions={1:16, 3:14}, 
                rewards={1:10, 3:5}
            ):
        self.size = size
        self.special_transitions = special_transitions

        self.ACTIONS = [np.array(self.north), np.array(self.east), np.array(self.south), np.array(self.west)]
        self.STATES = [np.array((x,y)) for x in range(size[0]) for y in range(size[1])]
        self.REWARDS = [0 if s not in rewards else rewards[s] for s, _ in enumerate(self.STATES)]

        super().__init__()

    def transition(self, s, a):
        reward = self.REWARDS[s]
        if s in self.special_transitions:
            newState = self.special_transitions[s] 
        else:
            newState = tuple(self.STATES[s] + self.ACTIONS[a])

            if not ((0 <= newState[0] < self.size[0]) and (0 <= newState[1] < self.size[1])):
                reward = -1
                newState = s
            else:
                newState = np.ravel_multi_index(newState, self.size)
        
        return newState, reward

    def print_Pi(self):
        maps = {self.west:"\u2190", self.north:"\u2191", self.east:"\u2192", self.south:"\u2193"}
        row = [[maps[self.ACTIONS[i]] for i in line] for line in self.Pi]
        render = [' '.join(line) for line in row]
        print('\n'.join(render))

    def print(self, display='index'):
        maps = {self.west:"\u2190", self.north:"\u2191", self.east:"\u2192", self.south:"\u2193"}
        plot = ''
        for x in range(self.size[0]):
            c = 0
            l = 0
            contents = '| '
            for y in range(self.size[1]):
                s = np.ravel_multi_index((x,y), self.size)
                if display == 'index':
                    icon = f'{x},{y}'
                if display == 'policy':
                    icon = f'{maps[tuple(self.ACTIONS[self.Pi[s]])]}'
                elif type(display) in [type([]), type(np.zeros((1,1)))]:
                    icon = display[x][y]

                c += 1
                l += len(str(icon))
                contents += str(icon)+' | '
            plot += '-'+'-'*(l+c*3) + '\n' + contents + '\n'

        plot += '-'+'-'*(l+c*3)

        print(plot)

class CarRental(MDP):
    MAX_CARS_PER_LOCATION = 20
    MAX_CARS_MOVED = 5
    COST_PER_CAR_MOVED = 2
    RENTAL_PRICE = 10
    
    def __init__(self, max_car=20):
        self.MAX_CARS_PER_LOCATION = max_car
        self.ACTIONS = [i for i in range(-self.MAX_CARS_MOVED, self.MAX_CARS_MOVED+1)]
        self.STATES = [(l1, l2) for l1 in range(self.MAX_CARS_PER_LOCATION) for l2 in range(self.MAX_CARS_PER_LOCATION)]
        self.REWARDS = []
        self.ps = {}

        super().__init__()
        self.Pi.fill(5)

    def reward(self, n_requests, moves):
        return self.RENTAL_PRICE*n_requests - np.abs(moves)*self.COST_PER_CAR_MOVED

    def p(self, s, a):
        rented = [n if self.requests[location][0]>=n else self.requests[location][0] for location, n in enumerate(self.STATES[s])]
        for s_, newS in enumerate(self.STATES):
            if ((self.STATES[s][0] + a[0] - rented[0] == newS[0]) 
            and (self.STATES[s][1] + a[1] - rented[1] == newS[1])): 
                self.ps[(s, s_)] = sum(req[0]*ret[0] for req, ret in zip(self.requests, self.returns)) 
            else: self.ps[(s, s_)] = 0

    def transition(self, s, a, lambda_requests=[3,4], lambda_returns=[3,2]):
        requests = [np.random.poisson(lambda_request) for lambda_request in lambda_requests]
        returns  = [np.random.poisson(lambda_return) for lambda_return in lambda_returns]
        self.requests = [(v, poisson.pmf(v, lambda_requests[i])) for i, v in enumerate(requests)]
        self.returns = [(v, poisson.pmf(v, lambda_requests[i])) for i, v in enumerate(returns)]

        a_effect = [-self.ACTIONS[a], self.ACTIONS[a]]

        s_ = list(self.STATES[s])
        r = list(self.STATES[s])
        for location, n in enumerate(self.STATES[s]):
            rented = n if self.requests[location][0]>=n else self.requests[location][0]
            s_[location] = n - rented + a_effect[location]
            r[location] = self.reward(rented, self.ACTIONS[a])

            s_[location] = s_[location] + self.returns[location][0] if (s_[location] + self.returns[location][0]) <= self.MAX_CARS_PER_LOCATION-1 else self.MAX_CARS_PER_LOCATION-1

        new_s = np.ravel_multi_index(s_, [self.MAX_CARS_PER_LOCATION]*len(self.STATES[s]))

        self.p(s, a_effect)
        
        return new_s, np.sum(r)

    def print(self, display):
        plot = ''
        for x in range(self.MAX_CARS_PER_LOCATION):
            c = 0
            l = 0
            contents = '| '
            for y in range(self.MAX_CARS_PER_LOCATION):
                s = np.ravel_multi_index((x,y), (self.MAX_CARS_PER_LOCATION, self.MAX_CARS_PER_LOCATION))
                if display == 'index':
                    icon = f'{x},{y}'
                if display == 'policy':
                    icon = f'{self.ACTIONS[self.Pi[s]]}'
                elif type(display) in [type([]), type(np.zeros((1,1)))]:
                    icon = display[x][y]
                c += 1
                l += len(str(icon))
                contents += str(icon)+' | '
            plot += '-'+'-'*(l+c*3) + '\n' + contents + '\n'

        plot += '-'+'-'*(l+c*3)

        print(plot)

