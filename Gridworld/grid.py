import numpy as np

class RLagent:
    pass


class Grid:
    north = (-1, 0)
    south = (1, 0)
    east = (0, 1)
    west = (0, -1)
    def __init__(self, 
                size=(5,5), 
                effects={(0,1):(4,1), (0,3):(2,3)}, 
                rewards={(0,1):10, (0,3):5}
            ):
        self.size = size
        self.effects = effects
        self.rewards = rewards
        self.grid = np.zeros(size)

    def transition(self, state, action):
        if tuple(state) in self.effects:
            reward = self.rewards[tuple(state)]
            newState = self.effects[tuple(state)] 
        else:
            newState = tuple(np.array(state)+np.array(action))
            reward = 0

        if not ((0 <= newState[0] < self.size[0]) and (0 <= newState[1] < self.size[1])):
            reward = -1
            newState = state
        
        return newState, reward

    def print(self, display='index'):
        plot = ''
        c = 0
        l = 0
        for x, row in enumerate(self.grid):
            c = 0
            l = 0
            lines = '-'
            contents = '| '
            for y, cell in enumerate(row):
                if display == 'index':
                    icon = f'{x},{y}'
                elif type(display) in [type([]), type(np.zeros((1,1)))]:
                    icon = display[x][y]
                else:
                    icon = cell

                c += 1
                l += len(str(icon))
                contents += str(icon)+' | '
            plot += '-'+'-'*(l+c*3) + '\n' + contents + '\n'

        plot += '-'+'-'*(l+c*3)

        print(plot)




