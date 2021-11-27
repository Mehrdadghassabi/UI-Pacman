from base import BaseAgent, Action
from queue import PriorityQueue

# increasing :) (:
# it was phase 1 final day and im so exhausted
# total days spent = 9
# total hours spent = 80


i = 0
# step number
aclis = []
# action list waiting to be taken
available_goals = []
# list of available_goals
unavailable_goals = []
# list of unavailable_goals
yellow_diamond_eaten = 0
# number of yellow Diamond eaten
green_diamond_eaten = 0
# number of green Diamond eaten
red_diamond_eaten = 0
# number of red Diamond eaten
blue_diamond_eaten = 0
# number of blue Diamond eaten
tbts = False
# tele better than straight
Inf = 1000
# Inf
prefer = 3
# how much do we prefer straight than tele
this_level = []
# current depth of me-enemy tree for trapping
mmeenemy_tree = {}
# tree for choose to trap
minus_Inf = -1000


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
    # cost 500 and (-1,-1) means no teleport found
    def find_nearest_teleport(self, start):
        global Inf
        tup1 = (0, 0)
        cost = Inf
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
            # print("nothing found")
            tup2 = (-1, -1)
            return tup2

    # just a method to count the eaten diamond
    # due to games rule number of eating diamond are limited
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
        if self.grid[goal[0]][goal[1]] == "T":
            list_of_action.append(Action.TELEPORT)
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
    def sort_available_goals(self, size, current, enemycurrent, scared_from_enemy):
        # print(size)
        for x in range(size):
            small = x
            for j in range(x, size):
                # print("hi: " + str(self.goal_score_function(available_goals[j], current)))
                # print("bye: " + str(self.goal_score_function(available_goals[small], current)))
                if self.goal_score_function(available_goals[j], current,
                                            enemycurrent, scared_from_enemy) > self.goal_score_function(
                        available_goals[small], current, enemycurrent, scared_from_enemy):
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
    # the function doesnt give us the best answer
    # so lets use Q learning
    # updating...
    def goal_score_function(self, goal, current, enemycurrent, scared_from_enemy):
        x = goal[0]
        y = goal[1]
        distance = manhattan(current, goal)
        enemydis = manhattan(enemycurrent, goal)

        # current is myposition
        # distance is manhattan distance of me and the goal
        # enemydis is manhattan distance of enemy and the goal
        # enemycurrent is enemy position
        # print(" Agent score " + str(self.agent_scores[0]))

        if self.grid[x][y] == "1":
            if scared_from_enemy:
                return (10 / distance ** 2) + enemydis
            if not scared_from_enemy:
                return 10 / distance ** 2
        elif self.grid[x][y] == "2" and self.agent_scores[0] > 15:
            if scared_from_enemy:
                return (25 / distance ** 2) + enemydis
            if not scared_from_enemy:
                return 25 / distance ** 2
        elif self.grid[x][y] == "3" and self.agent_scores[0] > 50:
            if scared_from_enemy:
                return (35 / distance ** 2) + enemydis
            if not scared_from_enemy:
                return 35 / distance ** 2
        elif self.grid[x][y] == "4" and self.agent_scores[0] > 140:
            if scared_from_enemy:
                return (75 / distance ** 2) + enemydis
            if not scared_from_enemy:
                return 75 / distance ** 2
        elif self.grid[x][y] == "T" and tbts:
            return 1000
        else:
            return 0

    # finding gravity center of teleports
    # since coming out of teleports are random
    # we need to compute an Expected value
    # somehow help us made the decision
    def center_gravity_teleports(self, start):
        global Inf
        xsum = 0
        ysum = 0
        teleports_number = 0
        nearest_teleport = self.find_nearest_teleport(start)
        for x in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[x][j] == "T" and x != nearest_teleport[0] and j != nearest_teleport[1]:
                    xsum = xsum + x
                    ysum = ysum + j
                    teleports_number = teleports_number + 1
        if teleports_number != 0:
            xave = xsum / teleports_number
            yave = ysum / teleports_number
            tup1 = (xave, yave)
        else:
            # print("there is no tele so there is no tgc")
            tup1 = (-Inf, -Inf)
        # print("man gravity is : " + str(tup1))
        return tup1, nearest_teleport

    # this function would tell us is it right to target teleport
    # in order to find better accessibility
    # it would change when q learner added
    # updating..
    def tele_better_than_straight(self, start, goal):
        global tbts
        global Inf
        global prefer
        tele_gc, nearest_teleport = self.center_gravity_teleports(start)
        # if there is no telegc return False
        if tele_gc[0] == -Inf and tele_gc[1] == -Inf:
            tbts = False
            return False
        rounded_tele_gcx = round(tele_gc[0])
        rounded_tele_gcy = round(tele_gc[1])
        rounded_tele_gc = (rounded_tele_gcx, rounded_tele_gcy)
        cost_so_far, came_from = self.cost(rounded_tele_gc, goal)
        if goal in cost_so_far:
            tele_gc_goal = cost_so_far[goal]
            # print(cost_so_far[goal])
        else:
            tele_gc_goal = Inf
            # print("Inf")
        pogo = manhattan(start, goal)
        tego = manhattan(start, nearest_teleport) + tele_gc_goal + prefer
        tbts = (tego < pogo)
        # self.print_tele_better_than_straight(pogo, tego, nearest_teleport, tbts, tele_gc_goal, rounded_tele_gc)
        return tbts

    # just for debugging
    # print the nearest teleport
    # print cost of straight
    # print cost of tele
    def print_tele_better_than_straight(self, pogo, tego, nearest_teleport, tbts, tele_gc_goal, rounded_tele_gc):
        print("nearest tele= " + str(nearest_teleport))
        print("pogo= " + str(pogo))
        print("togo= " + str(tego))
        print("tele_gc_goal: " + str(tele_gc_goal))
        print("rounded_tele_gc" + str(rounded_tele_gc))
        print(tbts)

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

    # just for debugging
    # printing meenmy tree at given depth
    # this tree help us to understand is this proper to trap now?
    def print_meenmy_tree_at_given_dep(self, maxdep, dep):
        # print(meenemy_tree)
        revdep = abs(maxdep + 1 - dep)
        print(mmeenemy_tree[revdep])
        print("--------------------------------------------------")

    # just for debugging
    # get meenmy tree at given depth
    # this tree help us to understand is this proper to trap now?
    def get_meenmy_tree_at_given_dep(self, maxdep, dep):
        revdep = abs(maxdep + 1 - dep)
        return mmeenemy_tree[revdep]

    # some goals are available due to their points but there is no direct path to them
    # just make them unavailable!!
    # and if there is no available goal target the nearest teleport
    # if no teleport found in this case it means everything done
    # just break the while loop
    def make_wall_goal_unavailable(self, start, goal, came_from):
        while goal not in came_from:
            available_goals.remove(goal)
            unavailable_goals.append(goal)
            if not bool(available_goals):
                # print("empty here")
                tup1 = self.find_nearest_teleport(start)
                if not tup1[0] == -1 and not tup1[1] == -1:
                    available_goals.append(self.find_nearest_teleport(start))
                else:
                    break
            goal = available_goals[0]
        return goal

    # bfs search for the me-enemy tree
    # following minimax algorithm
    def breadth_first_search_trap(self, meturn, remaindep):
        global this_level
        global meenemy_tree
        global mmeenemy_tree
        # tredep = {}
        if remaindep == 0:
            return
        nex_level = []
        for nodtup in this_level:
            # print("remain depth: " + str(remaindep))
            if meturn:
                nodxtrn = nodtup[0]
                nodytrn = nodtup[1]
                nodxNtrn = nodtup[2]
                nodyNtrn = nodtup[3]
            else:
                nodxtrn = nodtup[2]
                nodytrn = nodtup[3]
                nodxNtrn = nodtup[0]
                nodyNtrn = nodtup[1]
            nodtrn = (nodxtrn, nodytrn)
            nodNtrn = (nodxNtrn, nodyNtrn)
            neis = self.neighbors(nodtrn)
            for neibor in neis:
                if meturn:
                    nodchldx = neibor[0]
                    nodchldy = neibor[1]
                    child = (nodchldx, nodchldy, nodxNtrn, nodyNtrn)
                else:
                    nodchldx = neibor[0]
                    nodchldy = neibor[1]
                    child = (nodxNtrn, nodyNtrn, nodchldx, nodchldy)
                nex_level.append(child)
            # this_level = nex_level
            # counter = counter + 1
            # print(counter)
        # print(this_level)
        # print("______________________________________________________________")
        # tredep = (remaindep, this_level)
        # meenemy_tree.append(tredep)
        mmeenemy_tree[remaindep] = this_level
        this_level = nex_level
        self.breadth_first_search_trap(not meturn, remaindep - 1)

    # following minimax algorithm
    # this method tell us
    # that is this propare to trap now?
    def is_thisـproper_to_trap(self, start, enemypos, scared_from_enemy):
        global this_level
        startx = start[0]
        starty = start[1]
        enemyposx = enemypos[0]
        enemyposy = enemypos[1]
        isprop = False
        if self.grid_height > self.grid_width:
            maxdep = int(self.grid_width / 2)
        else:
            maxdep = int(self.grid_height / 2)

        r = [startx, starty, enemyposx, enemyposy]
        this_level = [r]
        meturn = True
        # root = Node(r)
        # print("max dep: " + str(maxdep))
        # nex_level = []
        # counter = 0

        if manhattan(start, enemypos) == 1 and scared_from_enemy:
            return True

        self.breadth_first_search_trap(meturn, maxdep)
        # self.print_meenmy_tree_at_given_dep(maxdep, 1)
        for x in range(maxdep - 1):
            # print(x+1)
            # print("#######################################")
            menemytreeatdepx = self.get_meenmy_tree_at_given_dep(maxdep, x + 1)
            for nod in menemytreeatdepx:
                bo = (nod[2] == startx) and (nod[3] == starty)
                isprop = isprop or bo
                # print(bo)
                # print(nod)
            # self.print_meenmy_tree_at_given_dep(maxdep, x + 1)
        # self.print_meenmy_tree()
        # print(meenemy_tree.index(0))
        # print("--------------------------------------------------")
        return isprop

    # agent one is you and agent two is enemy
    # return a boolean to determine that
    # is agent1 scared from agent2 ?
    # if agent2 score is more than first one then agent1 is scared
    # this method is useful for attack&flee
    # if agent1 doesnt scare from agent2 it can follow agent2 and hit it
    def agent_one_scaring_from_agent_two(self, scorone, scortwo):
        return scorone < scortwo

    # by having your position and enemy position
    # if youre not scared from the enemy
    # you can chased the enemy and hit it
    # and if youre scared from enemy
    # you should flee and avoid from hiting
    # the method do this by following the chase tree
    def attack_or_flee(self, start, enemypos, scared_from_enemy):
        width = self.grid_width
        height = self.grid_height
        qi = start[0]
        qj = start[1]
        qm = enemypos[0]
        qn = enemypos[1]
        qx = int(height / 2)
        qy = int(width / 2)
        if scared_from_enemy:
            # print("im scareeed")
            if qi - qm > 0:
                if qj - qn > 0:
                    if qx - qi > 0:
                        return Action.DOWN
                    elif qy - qj > 0:
                        return Action.RIGHT
                    else:
                        if qj == width - 1:
                            return Action.LEFT
                        else:
                            return Action.RIGHT
                elif qj - qn == 0:
                    if qx - qi > 0:
                        return Action.DOWN
                    elif qy - qj > 0:
                        return Action.RIGHT
                    elif qj - qy > 0:
                        return Action.LEFT
                    else:
                        if qj == width - 1:
                            return Action.LEFT
                        else:
                            return Action.RIGHT
                elif qj - qn < 0:
                    if qx - qi > 0:
                        return Action.DOWN
                    elif qj - qy > 0:
                        return Action.LEFT
                    else:
                        if qj == 0:
                            return Action.RIGHT
                        else:
                            return Action.LEFT
            if qi - qm == 0:
                if qj - qn > 0:
                    if qy - qj > 0:
                        return Action.RIGHT
                    elif qx - qi > 0:
                        return Action.DOWN
                    elif qi - qx > 0:
                        return Action.UP
                    else:
                        if qi == 0:
                            return Action.DOWN
                        else:
                            return Action.UP
                if qj - qn < 0:
                    if qj - qy > 0:
                        return Action.LEFT
                    elif qx - qi > 0:
                        return Action.DOWN
                    elif qi - qx > 0:
                        return Action.UP
                    else:
                        if qi == 0:
                            return Action.DOWN
                        else:
                            return Action.UP
            if qi - qm < 0:
                if qj - qn > 0:
                    if qi - qx > 0:
                        return Action.UP
                    elif qy - qj > 0:
                        return Action.RIGHT
                    else:
                        if qj == width - 1:
                            return Action.LEFT
                        else:
                            return Action.RIGHT
                elif qj - qn == 0:
                    if qi - qx > 0:
                        return Action.UP
                    elif qy - qj > 0:
                        return Action.RIGHT
                    elif qj - qy > 0:
                        return Action.LEFT
                    else:
                        if qj == width - 1:
                            return Action.LEFT
                        else:
                            return Action.RIGHT
                elif qj - qn < 0:
                    if qi - qx > 0:
                        return Action.UP
                    elif qj - qy > 0:
                        return Action.LEFT
                    else:
                        if qj == 0:
                            return Action.RIGHT
                        else:
                            return Action.LEFT
        else:
            # print("im brave")
            # print(manhattan(start, enemypos))
            if qi - qm > 0:
                if qj - qn > 0:
                    if qx - qi > 0:
                        return Action.UP
                    elif qy - qj > 0:
                        return Action.LEFT
                    else:
                        return Action.LEFT
                elif qj - qn == 0:
                    return Action.DOWN
                elif qj - qn < 0:
                    if qx - qi > 0:
                        return Action.UP
                    elif qj - qy > 0:
                        return Action.RIGHT
                    else:
                        return Action.RIGHT
            if qi - qm == 0:
                if qj - qn > 0:
                    return Action.LEFT
                if qj - qn < 0:
                    return Action.RIGHT
            if qi - qm < 0:
                if qj - qn > 0:
                    if qi - qx > 0:
                        return Action.DOWN
                    elif qy - qj > 0:
                        return Action.LEFT
                    else:
                        return Action.LEFT
                elif qj - qn == 0:
                    return Action.DOWN
                elif qj - qn < 0:
                    if qi - qx > 0:
                        return Action.DOWN
                    elif qj - qy > 0:
                        return Action.RIGHT
                    else:
                        return Action.RIGHT

    # Ok we are at step number i
    # find the current state of agent A
    # then we find available_goals due to score by goals_list (regardless of the path)
    # if all available_goals are eaten return noop
    # if tele_better_than_straight mark the nearest teleport as the target goal
    # else
    # take the first available_goal as the target goal
    # search for a path between start and target goal
    # the goals that there is no path to them should be unavailable by make_wall_goal_unavailable
    # There you go!! its the final decision three decision are available
    # 1: finding the best path that should be traversed
    # 2: traversing that path
    # 3: find a teleport if there is no available goal
    def find_appropriate_turn(self):
        global i
        global aclis
        global available_goals
        global unavailable_goals
        i = i + 1
        # self.print_score_turn(self.agent_scores[0])
        start = self.find_state("A")
        enemypos = self.find_state("B")
        myscore = self.agent_scores[0]
        enemyscore = self.agent_scores[1]
        scared_from_enemy = self.agent_one_scaring_from_agent_two(myscore, enemyscore)

        available_goals, unavailable_goals = self.goals_list(start)

        propare = self.is_thisـproper_to_trap(start, enemypos, scared_from_enemy)
        if propare:
            return Action.TRAP
        # print(root)

        if not bool(available_goals):
            return self.attack_or_flee(start, enemypos, scared_from_enemy)
            # return Action.NOOP

        goal = available_goals[0]
        cost_so_far, came_from = self.cost(start, goal)

        goal = self.make_wall_goal_unavailable(start, goal, came_from)
        if self.tele_better_than_straight(start, goal):
            nearest_teleports = self.find_nearest_teleport(start)
            available_goals.append(nearest_teleports)
        # self.print_availability(start)
        if not bool(aclis):
            # print("im in part one step is: " + str(i))
            available_goals = self.sort_available_goals(len(available_goals), start, enemypos, scared_from_enemy)
            if not bool(available_goals):
                return self.attack_or_flee(start, enemypos, scared_from_enemy)
                # return Action.NOOP
            goal = available_goals[0]
            cost_so_far, came_from = self.cost(start, goal)
            goal = self.make_wall_goal_unavailable(start, goal, came_from)
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

    # Do the turn
    # updating...
    def do_turn(self) -> Action:
        return self.find_appropriate_turn()


if __name__ == '__main__':
    data = Agent().play()
    print("FINISH : ", data)
