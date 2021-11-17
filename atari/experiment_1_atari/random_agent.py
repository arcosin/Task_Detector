
import random


class RandomAgent:
    def __init__(self, actSize):
        super().__init__()
        self.actSize = actSize

    def act(self, state):
        return random.randint(0, self.actSize - 1)





#===============================================================================
