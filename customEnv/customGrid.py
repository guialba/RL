import numpy as np

class CustomGrid:
    UP = (-1, 0)
    DOWN = (1, 0)
    RIGHT = (0, 1)
    LEFT = (0, -1)

    ACTIONS = [UP, DOWN, RIGHT, LEFT]

    def __init__(self, 
                size=(4,4), 
                effects={(0,1):(4,1), (0,3):(2,3)}, 
                rewards={(0,1):10, (0,3):5}
                zones=None,
            ):
        self.size = size
        self.effects = effects
        self.rewards = rewards
        self.grid = np.zeros(size)