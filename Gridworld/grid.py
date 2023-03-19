import numpy as np



class Grid:
    up = (0, -1)
    down = (0, 1)
    right = (1, 0)
    left = (-1, 0)

    ACTIONS = [up, down, right, left]
    
    def __init__(self, 
                size=(5,5), 
            ):
        self.size = size
        self.effects = {
            (3,3):{'terminal': True, 'reward': 10}
            # Rotate Actions Zones
            # Upper left - UP-UP, DOWN-DOWN, RIGHT-RIGHT, LEFT-LEFT 
            # (0,1):{'move': (4,1), 'reward': 10},
            # (0,3):{'move': (2,3), 'reward': 5}
        }

    def transition(self, state, action):
        newState = tuple(np.array(state)+np.array(action))
        reward = 0
        terminal = False

        if tuple(state) in self.effects:
            if 'reward' in self.effects[tuple(state)]:
                reward = self.effects[tuple(state)]['reward']
            if 'move' in self.effects[tuple(state)]:
                newState = self.effects[tuple(state)]['move']
        
        if not ((0 <= newState[0] < self.size[0]) and (0 <= newState[1] < self.size[1])):
            reward = -1
            newState = state

        if tuple(newState) in self.effects:
            if 'reward' in self.effects[tuple(newState)]:
                reward = self.effects[tuple(newState)]['reward']
            if 'terminal' in self.effects[tuple(newState)]:
                terminal = self.effects[tuple(newState)]['terminal']
        
        return newState, reward, terminal

    def print(self, display='index'):
        plot = ''
        c = 0
        l = 0
        for y in range(self.size[0]):
            c = 0
            l = 0
            # lines = '-'
            contents = '| '
            for x in range(self.size[1]):
                if display == 'index':
                    icon = f'{x},{y}'
                elif type(display) in [type([]), type(np.zeros((1,1)))]:
                    icon = display[x][y]
                elif type(display) == type(()):
                    if x == display[0] and y == display[1]:
                        icon = 'o' 
                    else:
                        icon = ' '
                else:
                    icon = ' '

                c += 1
                l += len(str(icon))
                contents += str(icon)+' | '
            plot += '-'+'-'*(l+c*3) + '\n' + contents + '\n'

        plot += '-'+'-'*(l+c*3)

        print(plot)



