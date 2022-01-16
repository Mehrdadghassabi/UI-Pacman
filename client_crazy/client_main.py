
from base import BaseAgent, Action
import exhaustive_searches
import tools


class Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.solution = []
        self.diamonds = None
        self.graph_dict = None
        self.state = 0
        self.eaten = [0, 0, 0, 0]

    def do_turn(self) -> Action:
        # in no solution situations : returns NOOP
        agent = tools.find_agent(self.grid, self.character)

        if self.state == 0:
            self.diamonds = tools.find_gems(self.grid)
            graph = tools.Graph(self.grid_height, self.grid_width, self.grid)
            graph.make_graph()
            self.graph_dict = graph.graph_dict

            if len(self.diamonds) == 0:
                self.state = 2
                return Action.NOOP
            elif len(self.diamonds) == 1:
                self.state = 2

            best = exhaustive_searches.best_diamond_bfs(agent, self.graph_dict,
                                                        self.grid, self.agent_scores[self.id-1],
                                                        self.eaten)
            self.solution = best[1]

            action = self.solution.pop(0)
            self.eaten = tools.eat_it_if_you_can(self.eaten, agent, action, self.grid)
            return tools.move_action(action)

        elif self.state == 1:
            if len(self.solution) == 1:
                # to find new path
                self.state = 0
            action = self.solution.pop(0)
            self.eaten = tools.eat_it_if_you_can(self.eaten, agent, action, self.grid)
            return tools.move_action(action)

        elif self.state == 2:
            return Action.NOOP

        elif self.state == 3:
            if len(self.solution) == 0:
                # to find new path
                self.state = 0

                return Action.TELEPORT

            action = self.solution.pop(0)
            self.eaten = tools.eat_it_if_you_can(self.eaten, agent, action, self.grid)
            return tools.move_action(action)


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
