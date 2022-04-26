from base import BaseAgent
from tools import *


class Agent(BaseAgent):

    def do_turn(self) -> Action:
        set_requirements(self.grid, self.grid_height, self.grid_width, self.turn_count,self.max_turn_count, self.agent_scores[0])
        return what_to_do()


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
