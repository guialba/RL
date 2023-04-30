import numpy as np

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

    def bellman_expectation(self, s_, r, p_s=1, gamma=0.9):
        return p_s*(r + gamma*self.V[s_])

    def policy_evaluation(self, theta=1e-4, inplace=True):
        newV = self.V if inplace else np.zeros_like(self.V)
        while True:
            delta=0
            for s,_ in enumerate(self.STATES):
                v = newV[s]
                newV[s] = self.bellman_expectation(*self.transition(s, self.Pi[s]))
                delta = max(delta, np.abs(v-newV[s]))
            if delta < theta: 
                return np.round(newV, decimals=2, out=newV)

    def policy_iteration(self):
        policy_stable = False
        while not policy_stable:
            self.policy_evaluation()
            for s,_ in enumerate(self.STATES):
                old_action = self.Pi[s]
                acts = [self.bellman_expectation(*self.transition(s, a)) for a,_ in enumerate(self.ACTIONS)]
                self.Pi[s] = np.argmax(acts)
                if old_action == self.Pi[s]:
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
