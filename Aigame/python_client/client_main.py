import random
from base import BaseAgent, Action
from queue import PriorityQueue

i = 0
aclis = []
available_goals = []
unavailable_goals = []


def manhattan(start, goal):
    # print("man: " + str(abs(start[0] - goal[0]) + abs(start[1] - goal[1])))
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


class Agent(BaseAgent):

    def neighbors(self, node):
        width = self.grid_width
        height = self.grid_height
        x = node[0]
        j = node[1]

        neis = []
        if x == 0 and j == 0:
            nei1 = (1, 0)
            nei2 = (0, 1)
            neis.append(nei1)
            neis.append(nei2)
        elif x == 0 and j == width - 1:
            nei1 = (1, width - 1)
            nei2 = (0, width - 2)
            neis.append(nei1)
            neis.append(nei2)
        elif x == height - 1 and j == 0:
            nei1 = (height - 2, 0)
            nei2 = (height - 1, 1)
            neis.append(nei1)
            neis.append(nei2)
        elif x == height - 1 and j == width - 1:
            nei1 = (height - 1, width - 2)
            nei2 = (height - 2, width - 1)
            neis.append(nei1)
            neis.append(nei2)
        elif x == 0 and j > 0:
            nei1 = (1, j)
            nei2 = (0, j - 1)
            nei3 = (0, j + 1)
            neis.append(nei1)
            neis.append(nei2)
            neis.append(nei3)
        elif x > 0 and j == 0:
            nei1 = (x, 1)
            nei2 = (x - 1, 0)
            nei3 = (x + 1, 0)
            neis.append(nei1)
            neis.append(nei2)
            neis.append(nei3)
        elif x == height - 1 and j > 0:
            nei1 = (height - 2, j)
            nei2 = (height - 1, j - 1)
            nei3 = (height - 1, j + 1)
            neis.append(nei1)
            neis.append(nei2)
            neis.append(nei3)
        elif x > 0 and j == width - 1:
            nei1 = (x, width - 2)
            nei2 = (x - 1, width - 1)
            nei3 = (x + 1, width - 1)
            neis.append(nei1)
            neis.append(nei2)
            neis.append(nei3)
        else:
            nei1 = (x, j + 1)
            nei2 = (x, j - 1)
            nei3 = (x + 1, j)
            nei4 = (x - 1, j)
            neis.append(nei1)
            neis.append(nei2)
            neis.append(nei3)
            neis.append(nei4)

        # print(neis)
        return neis

    def cost(self, start, goal):
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = dict()
        cost_so_far = dict()
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()
            print(str(current[1][0]) + str(current[1][1]))
            if current[1][0] == goal[0] and current[1][1] == goal[1]:
                print("Goal found")
                break

            for nnext in self.neighbors(current[1]):
                new_cost = cost_so_far[current[1]] + 1
                if nnext not in cost_so_far or new_cost < cost_so_far[nnext]:
                    cost_so_far[nnext] = new_cost
                    priority = new_cost + manhattan(nnext, goal)
                    # print("ummm re" + str(self.grid[nnext[0]][nnext[1]]))
                    if "E" in self.grid[nnext[0]][nnext[1]] or "2" in self.grid[nnext[0]][nnext[1]] \
                            or "3" in self.grid[nnext[0]][nnext[1]] or "4" in self.grid[nnext[0]][nnext[1]]:
                        frontier.put((priority, nnext))
                        # print("nnext: " + str(nnext[0]) + str(nnext[1]) + " priority: " + str(priority))
                    came_from[nnext] = current[1]
        return cost_so_far, came_from

    def goals_list(self, current):
        lavailable_goals = []
        lunavailable_goals = []
        for x in range(self.grid_height):
            for j in range(self.grid_width):
                # print(self.grid[i][j])
                if self.grid[x][j] == "1":
                    tup1 = (x, j)
                    lavailable_goals.append(tup1)
                if self.grid[x][j] == "2":
                    tup1 = (x, j)
                    if self.agent_scores[0] >= 15 + manhattan(current, tup1):
                        # print("Scpo2 : " + str(15 + manhattan(current, tup1)))
                        lavailable_goals.append(tup1)
                    else:
                        lunavailable_goals.append(tup1)
                if self.grid[x][j] == "3":
                    tup1 = (x, j)
                    if self.agent_scores[0] >= 50 + manhattan(current, tup1):
                        # print("Scpo3 : " + str(50 + manhattan(current, tup1)))
                        lavailable_goals.append(tup1)
                    else:
                        lunavailable_goals.append(tup1)
                if self.grid[x][j] == "4":
                    tup1 = (x, j)
                    if self.agent_scores[0] >= 140 + manhattan(current, tup1):
                        # print("Scpo4 : " + str(140 + manhattan(current, tup1)))
                        lavailable_goals.append(tup1)
                    else:
                        lunavailable_goals.append(tup1)
        return lavailable_goals, lunavailable_goals

    def find_state(self, st):
        for x in range(self.grid_height):
            for j in range(self.grid_width):
                if st in self.grid[x][j]:
                    tup1 = (x, j)
                    return tup1

    def eat_the_goal(self, start, goal, came_from):
        temp_goal = goal
        list_of_action = []
        list_of_action_reversed = []
        # print("cccc: " + str(came_from[temp_goal]))
        while temp_goal not in came_from:
            available_goals.remove(temp_goal)
            unavailable_goals.append(temp_goal)
            temp_goal = available_goals[0]
        while temp_goal != start:
            goal_nei = came_from[temp_goal]
            print("hey im in the loop temp goal is: " + str(temp_goal))
            print("hey im in the loop  goalnei is: " + str(goal_nei))
            x = temp_goal[0] - goal_nei[0]
            y = temp_goal[1] - goal_nei[1]
            print("x = " + str(x))
            print("y = " + str(y))
            print("---------------------------------------------------")
            if x == 1 and y == 0:
                list_of_action.append(Action.DOWN)
            if x == -1 and y == 0:
                list_of_action.append(Action.UP)
            if x == 0 and y == 1:
                list_of_action.append(Action.RIGHT)
            if x == 0 and y == -1:
                list_of_action.append(Action.LEFT)
            tup1 = (goal_nei[0], goal_nei[1])
            temp_goal = tup1
        while bool(list_of_action):
            item = list_of_action.pop()
            # print("item is: "+str(item))
            list_of_action_reversed.append(item)
        return list_of_action_reversed

    def sort_available_goals(self, size, current):
        print(size)
        for x in range(size):
            small = x
            for j in range(x, size):
                # print("hi: " + str(self.goal_score_function(available_goals[j], current)))
                # print("bye: " + str(self.goal_score_function(available_goals[small], current)))
                if self.goal_score_function(available_goals[j], current) < self.goal_score_function(
                        available_goals[small], current):
                    small = j
            temp = available_goals[small]
            available_goals[small] = available_goals[x]
            available_goals[x] = temp

        return available_goals

    def goal_score_function(self, goal, current):
        x = goal[0]
        y = goal[1]
        distance = manhattan(current, goal)
        if self.grid[x][y] == "1":
            return 10 / distance ** 2
        elif self.grid[x][y] == "2" and self.agent_scores[0] > 15:
            return 25 / distance ** 2
        elif self.grid[x][y] == "3" and self.agent_scores[0] > 50:
            return 35 / distance ** 2
        elif self.grid[x][y] == "4" and self.agent_scores[0] > 140:
            return 75 / distance ** 2
        else:
            return 0

    def do_turn(self) -> Action:
        global i
        global aclis
        global available_goals
        global unavailable_goals
        print(
            "-----------------------------------------------------------------------------------------------------------------------------")
        print("my score is:: " + str(self.agent_scores[0]))
        print("turn: " + str(i))
        start = self.find_state("A")
        available_goals, unavailable_goals = self.goals_list(start)
        goal = available_goals[0]
        cost_so_far, came_from = self.cost(start, goal)
        print("av: " + str(available_goals))
        print("un: " + str(unavailable_goals))
        # print("step: " + str(i) + "score: " + str(self.agent_scores[0]))
        print("my current position: " + str(start))
        # print("distance to goal: " + str(cost_so_far[goal]))
        if not bool(aclis):
            aclis = self.eat_the_goal(start, goal, came_from)
            available_goals = self.sort_available_goals(len(available_goals), start)
            # aclis = self.sort_available_goals(available_goals, len(available_goals), start)
            print("the path is=== " + str(aclis))
        # print("goal came_from: " + str(came_from[goal]))
        # print(self.agent_scores)
        i = i + 1
        # while bool(aclis):
        if bool(aclis):
            ki = aclis[0]
            aclis.remove(ki)
            print(ki)
            return ki
        else:
            return random.choice(
                [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.TELEPORT, Action.NOOP, Action.TRAP])


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
