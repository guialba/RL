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
        self.S = [(i,j) for j in range(size[1]) for i in range(size[0])]
        self.size = size
        self.effects = {
            (4,2):{'terminal': True, 'reward': 10},

            # Blocked
            (2,1):{'blocked': True},
            (2,2):{'blocked': True},
            (2,3):{'blocked': True},
            (1,2):{'blocked': True},
            (3,2):{'blocked': True},
           
            # Rotate Actions Zones
            #   * Upper Left - UP-UP, DOWN-DOWN, RIGHT-RIGHT, LEFT-LEFT 
            #   * Upper Right - UP-RIGHT, DOWN-LEFT, RIGHT-DOWN, LEFT-UP 
            (3,0):{'noise': lambda a: [-a[1], a[0]]},
            (4,0):{'noise': lambda a: [-a[1], a[0]]},
            (3,1):{'noise': lambda a: [-a[1], a[0]]},
            (4,1):{'noise': lambda a: [-a[1], a[0]]},
            #   * Lower Right - UP-DOWN, DOWN-UP, RIGHT-LEFT, LEFT-RIGHT 
            (3,3):{'noise': lambda a: [-a[0], -a[1]]},
            (4,3):{'noise': lambda a: [-a[0], -a[1]]},
            (3,4):{'noise': lambda a: [-a[0], -a[1]]},
            (4,4):{'noise': lambda a: [-a[0], -a[1]]},
            #   * Lower Left - UP-LEFT, DOWN-RIGHT, RIGHT-UP, LEFT-DOWN 
            (0,3):{'noise': lambda a: [a[1], -a[0]]},
            (1,3):{'noise': lambda a: [a[1], -a[0]]},
            (0,4):{'noise': lambda a: [a[1], -a[0]]},
            (1,4):{'noise': lambda a: [a[1], -a[0]]},
        }

    def getBlockeds(self):
        return [s for s, eff in self.effects.items() if eff.get('blocked', False)]

    def transition(self, state, action):
        newState = tuple(np.array(state)+np.array(action))
        reward = 0
        terminal = False

        if tuple(state) in self.effects:
            if 'noise' in self.effects[tuple(state)]:
                newState = tuple(np.array(state)+np.array(self.effects[tuple(state)]['noise'](action))) 
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
            if 'blocked' in self.effects[tuple(newState)]:
                if self.effects[tuple(newState)]['blocked']:
                    reward = -1
                    newState = state
                    terminal = False
        
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



