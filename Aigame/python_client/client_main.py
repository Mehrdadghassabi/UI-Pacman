from base import BaseAgent, Action
from queue import PriorityQueue

# step number
i = 0
# action list waiting to be taken
aclis = []
# list of available_goals
available_goals = []
# list of unavailable_goals
unavailable_goals = []
# number of yellow Diamond eaten
yellow_diamond_eaten = 0
# number of green Diamond eaten
green_diamond_eaten = 0
# number of red Diamond eaten
red_diamond_eaten = 0
# number of blue Diamond eaten
blue_diamond_eaten = 0


# manhattan distance of 2 node
# regardless  of existence a path
def manhattan(start, goal):
    # print("man: " + str(abs(start[0] - goal[0]) + abs(start[1] - goal[1])))
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


class Agent(BaseAgent):

    # return von Neumann neighbours of a node
    # note that :
    # the frontier nodes have 3 von Neumann neighbour
    # the corner nodes have 3 von Neumann neighbour
    # the others have 4 von Neumann neighbour
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

    #  Find paths from start to goal
    #  We’re not only trying to find the shortest distance
    #  We also want to take into account travel time
    #  There is 3 approach for doing this
    #  1: Breadth First Search explores equally in all directions
    #  This is an incredibly useful algorithm, not only for regular path finding,
    #  but also for procedural map generation, flow field pathfinding,
    #  distance maps, and other types of map analysis.
    #  2: Dijkstra’s Algorithm (also called Uniform Cost Search)
    #  lets us prioritize which paths to explore.
    #  Instead of exploring all possible paths equally,
    #  it favors lower cost paths.
    #  We can assign lower costs to encourage moving on roads,
    #  higher costs to avoid forests, higher costs to discourage going near enemies, and more.
    #  When movement costs vary, we use this instead of Breadth First Search.
    # A* is a modification of Dijkstra’s Algorithm that is optimized for a single destination.
    # Dijkstra’s Algorithm can find paths to all locations;
    # A* finds paths to one location, or the closest of several locations.
    # It prioritizes paths that seem to be leading closer to a goal.
    # 3: A* Dijkstra’s Algorithm works well to find the shortest path,
    # but it wastes time exploring in directions that aren’t promising.
    # Greedy Best First Search explores in promising directions but it may not find the shortest path.
    # The A* algorithm uses both the actual distance from the start and the estimated distance to the goal.
    def cost(self, start, goal):
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = dict()
        cost_so_far = dict()
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            current = frontier.get()
            # print(str(current[1][0]) + str(current[1][1]))
            if current[1][0] == goal[0] and current[1][1] == goal[1]:
                # print("Goal found")
                break

            for nnext in self.neighbors(current[1]):
                new_cost = cost_so_far[current[1]] + 1
                if nnext not in cost_so_far or new_cost < cost_so_far[nnext]:
                    cost_so_far[nnext] = new_cost
                    priority = new_cost + manhattan(nnext, goal)
                    # print("ummm re" + str(self.grid[nnext[0]][nnext[1]]))
                    if "E" in self.grid[nnext[0]][nnext[1]] or \
                            "1" in self.grid[nnext[0]][nnext[1]] or \
                            "2" in self.grid[nnext[0]][nnext[1]] or \
                            "3" in self.grid[nnext[0]][nnext[1]] or "4" in self.grid[nnext[0]][nnext[1]] \
                            or "T" in self.grid[nnext[0]][nnext[1]]:
                        frontier.put((priority, nnext))
                        # print("nnext: " + str(nnext[0]) + str(nnext[1]) + " priority: " + str(priority))
                    came_from[nnext] = current[1]
        return cost_so_far, came_from

    # finding available_goals due to score by goals_list (regardless of the path)
    # considering manhattan distance from the current state
    def goals_list(self, current):
        lavailable_goals = []
        lunavailable_goals = []
        for x in range(self.grid_height):
            for j in range(self.grid_width):
                # print(self.grid[i][j])
                if self.grid[x][j] == "1":
                    tup1 = (x, j)
                    if self.agent_scores[0] >= 15 + manhattan(current, tup1) and yellow_diamond_eaten <= 15:
                        # print("Scpo2 : " + str(15 + manhattan(current, tup1)))
                        lavailable_goals.append(tup1)
                    else:
                        lunavailable_goals.append(tup1)
                if self.grid[x][j] == "2":
                    tup1 = (x, j)
                    if self.agent_scores[0] >= 15 + manhattan(current, tup1) and green_diamond_eaten <= 8:
                        # print("Scpo2 : " + str(15 + manhattan(current, tup1)))
                        lavailable_goals.append(tup1)
                    else:
                        lunavailable_goals.append(tup1)
                if self.grid[x][j] == "3":
                    tup1 = (x, j)
                    if self.agent_scores[0] >= 50 + manhattan(current, tup1) and red_diamond_eaten <= 5:
                        # print("Scpo3 : " + str(50 + manhattan(current, tup1)))
                        lavailable_goals.append(tup1)
                    else:
                        lunavailable_goals.append(tup1)
                if self.grid[x][j] == "4":
                    tup1 = (x, j)
                    if self.agent_scores[0] >= 140 + manhattan(current, tup1) and blue_diamond_eaten <= 4:
                        # print("Scpo4 : " + str(140 + manhattan(current, tup1)))
                        lavailable_goals.append(tup1)
                    else:
                        lunavailable_goals.append(tup1)
        return lavailable_goals, lunavailable_goals

    # find the Agent st=A location
    # by searching all of the grid
    def find_state(self, st):
        for x in range(self.grid_height):
            for j in range(self.grid_width):
                if st in self.grid[x][j]:
                    tup1 = (x, j)
                    return tup1

    # find the nearest_teleport
    # cost 500 means no teleport found
    def find_nearest_teleport(self, start):
        tup1 = (0, 0)
        cost = 500
        for x in range(self.grid_height):
            for j in range(self.grid_width):
                if "T" in self.grid[x][j]:
                    temptup = (x, j)
                    # print(self.grid[x][j])
                    # print("x: " + str(x) + " j: " + str(j))
                    cost_so_far, came_from = self.cost(start, temptup)
                    # print("i: " + str(cost_so_far) + "j: " + str(cost_so_far))
                    # print(cost_so_far[temptup])
                    if temptup in cost_so_far:
                        if cost_so_far[temptup] < cost:
                            tup1 = temptup
                            cost = cost_so_far[temptup]
        # print("i: " + str(tup1[0]) + " j: " + str(tup1[1]))
        if "T" in self.grid[tup1[0]][tup1[1]]:
            # print("i: " + str(tup1[0]) + " j: " + str(tup1[1]))
            return tup1
        else:
            pass

    def count_number_of_eaten_goals(self, goal):
        global yellow_diamond_eaten
        global green_diamond_eaten
        global red_diamond_eaten
        global blue_diamond_eaten

        if self.grid[goal[0]][goal[1]] == "1":
            # print("yellow_diamond_eaten: " + str(yellow_diamond_eaten))
            yellow_diamond_eaten = yellow_diamond_eaten + 1
        elif self.grid[goal[0]][goal[1]] == "2":
            # print("green_diamond_eaten: " + str(green_diamond_eaten))
            green_diamond_eaten = green_diamond_eaten + 1
        elif self.grid[goal[0]][goal[1]] == "3":
            # print("red_diamond_eaten: " + str(red_diamond_eaten))
            red_diamond_eaten = red_diamond_eaten + 1
        elif self.grid[goal[0]][goal[1]] == "4":
            # print("blue_diamond_eaten: " + str(blue_diamond_eaten))
            blue_diamond_eaten = blue_diamond_eaten + 1

    # Just the Goal
    # How?
    # by having came_from array
    # you just need to return list of actions
    # follow the cost function
    # to understand what does came_from array is
    def eat_the_goal(self, start, goal, came_from):
        self.count_number_of_eaten_goals(goal)
        temp_goal = goal
        list_of_action = []
        list_of_action_reversed = []
        while temp_goal != start:
            goal_nei = came_from[temp_goal]
            # print("hey im in the loop temp goal is: " + str(temp_goal))
            # print("hey im in the loop  goalnei is: " + str(goal_nei))
            x = temp_goal[0] - goal_nei[0]
            y = temp_goal[1] - goal_nei[1]
            # print("x = " + str(x))
            # print("y = " + str(y))
            # print("---------------------------------------------------")
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

    # sorting the available_goals
    # having more goal_score_function means
    # the goal has a higher priority
    def sort_available_goals(self, size, current):
        # print(size)
        for x in range(size):
            small = x
            for j in range(x, size):
                # print("hi: " + str(self.goal_score_function(available_goals[j], current)))
                # print("bye: " + str(self.goal_score_function(available_goals[small], current)))
                if self.goal_score_function(available_goals[j], current) > self.goal_score_function(
                        available_goals[small], current):
                    small = j
            temp = available_goals[small]
            available_goals[small] = available_goals[x]
            available_goals[x] = temp
        # print("currrren" + str(current[0]) + str(current[1]))
        # print(available_goals)
        return available_goals

    # by having current & goal
    # this is score function priority
    # having more score means having more priority
    def goal_score_function(self, goal, current):
        x = goal[0]
        y = goal[1]
        distance = manhattan(current, goal)
        # print(" Agent score " + str(self.agent_scores[0]))

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

    # just for debugging
    # print the ----- line
    # print the score at step i
    # print turn number
    def print_score_turn(self, score):
        global i
        print(
            "---------------------------------------------------------------"
            "--------------------------------------------------------------")
        print("my score is:: " + str(score))
        print("turn: " + str(i))

    # just for debugging
    # print the available_goals & unavailable_goals
    # print the current position
    def print_availability(self, start):
        print("av: " + str(available_goals))
        print("un: " + str(unavailable_goals))
        # print("step: " + str(i) + "score: " + str(self.agent_scores[0]))
        print("my current position: " + str(start))
        # print("distance to goal: " + str(cost_so_far[goal]))

    # some goals are available due to their points but there is no direct path to them
    # just make them unavailable!!
    # and if there is no available goal target the nearest teleport
    def make_wall_goal_unavailable(self, start, goal, came_from):
        while goal not in came_from:
            available_goals.remove(goal)
            unavailable_goals.append(goal)
            if not bool(available_goals):
                # print("empty here")
                available_goals.append(self.find_nearest_teleport(start))
            goal = available_goals[0]
        return goal

    # Ok we are at step number i
    # find the current state of agent A
    # then we find available_goals due to score by goals_list (regardless of the path)
    # if all available_goals are eaten return noop
    # else
    # take the first available_goal as the target goal
    # search for a path between start and target goal
    # the goals that there is no path to them should be unavailable by make_wall_goal_unavailable
    # There you go!! its the final decision three decision are available
    # 1: finding the best path that should be traversed
    # 2: traversing that path
    # 3: find a teleport if there is no available goal
    def do_turn(self) -> Action:
        global i
        global aclis
        global available_goals
        global unavailable_goals
        i = i + 1
        # self.print_score_turn(self.agent_scores[0])

        start = self.find_state("A")
        available_goals, unavailable_goals = self.goals_list(start)

        if not bool(available_goals):
            return Action.NOOP

        goal = available_goals[0]
        cost_so_far, came_from = self.cost(start, goal)

        goal = self.make_wall_goal_unavailable(start, goal, came_from)
        # self.print_availability(start)

        if not bool(aclis):
            # print("im in part one step is: " + str(i))
            available_goals = self.sort_available_goals(len(available_goals), start)
            # temp
            goal = available_goals[0]
            cost_so_far, came_from = self.cost(start, goal)
            aclis = self.eat_the_goal(start, goal, came_from)
            # print("the path is=== " + str(aclis))
        if bool(aclis):
            # print("im in part two step is: " + str(i))
            first_ac = aclis[0]
            aclis.remove(first_ac)
            return first_ac
        else:
            # print("im in part three step is: " + str(i))
            available_goals.remove(goal)
            return Action.TELEPORT


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
