from base import BaseAgent, Action


def move_action(action):
    if action == "up":
        return Action.UP
    if action == "down":
        return Action.DOWN
    if action == "right":
        return Action.RIGHT
    if action == "left":
        return Action.LEFT
    if action == "noop":
        return Action.NOOP
    if action == "teleport":
        return Action.TELEPORT
    if action == "trap":
        return Action.TRAP


def find_agent(my_grid: list, character):
    for r in range(len(my_grid)):
        for c in range(len(my_grid[0])):
            if len(my_grid[r][c]) >= 2:
                if my_grid[r][c][1] == character:
                    return r, c
            if len(my_grid[r][c]) > 2:
                if my_grid[r][c][2] == character:
                    return r, c


def is_gem(my_grid: list, r, c):
    if my_grid[r][c] == '1' or my_grid[r][c] == '2' or \
            my_grid[r][c] == '3' or my_grid[r][c] == '4':
        return True
    return False


def gem_value(gem):
    gem_scores = [10, 25, 35, 75]
    return gem_scores[int(gem) - 1]


def achievable_gem(gem, score, eaten):
    max_each = [15, 8, 5, 4]
    min_req = [-100000, 15, 50, 140]
    if score >= min_req[int(gem)-1] and eaten[int(gem)-1] < max_each[int(gem)-1]:
        return True
    return False


def eat_it_if_you_can(eaten, agent, action, my_grid: list):
    if action == "right":
        if is_gem(my_grid, agent[0], agent[1]+1):
            eaten[int(my_grid[agent[0]][agent[1]+1])-1] += 1
    elif action == "left":
        if is_gem(my_grid, agent[0], agent[1]-1):
            eaten[int(my_grid[agent[0]][agent[1]-1])-1] += 1
    elif action == "up":
        if is_gem(my_grid, agent[0]-1, agent[1]):
            eaten[int(my_grid[agent[0]-1][agent[1]])-1] += 1
    elif action == "down":
        if is_gem(my_grid, agent[0]+1, agent[1]):
            eaten[int(my_grid[agent[0]+1][agent[1]])-1] += 1
    return eaten


def is_teleport(my_grid: list, r, c):
    if my_grid[r][c][0] == 'T':
        return True
    return False


def find_teleports(my_grid: list):
    teleport_list = []
    for r in range(len(my_grid)):
        for c in range(len(my_grid[0])):
            if is_teleport(my_grid, r, c):
                teleport_list.append((r, c))
    return teleport_list


def find_gems(my_grid: list):
    gems_list = []
    for r in range(len(my_grid)):
        for c in range(len(my_grid[0])):
            if is_gem(my_grid, r, c):
                gems_list.append((int(my_grid[r][c]), (r, c)))
    gems_list.sort()
    return gems_list


def taken_action(parent: tuple, child: tuple):
    if parent[0]-1 == child[0]:
        return "up"
    if parent[0]+1 == child[0]:
        return "down"
    if parent[1]+1 == child[1]:
        return "right"
    if parent[1]-1 == child[1]:
        return "left"


def get_parent(child: tuple, parent_action):
    if parent_action == "up":
        return child[0]+1, child[1]
    if parent_action == "down":
        return child[0]-1, child[1]
    if parent_action == "right":
        return child[0], child[1]-1
    if parent_action == "left":
        return child[0], child[1]+1


class Node:
    def __init__(self, content, children: list, coordinate: tuple):
        self.coordinate = coordinate
        self.content = content
        self.actionSrc = None
        self.parentSrc = -1
        self.children = children


class Graph:
    def __init__(self, height, width, my_grid):
        self.graph_dict = {}
        self.my_grid = my_grid
        self.height = height
        self.width = width

    def make_graph(self):
        for r in range(self.height):
            for c in range(self.width):
                children = []
                if r + 1 < self.height:
                    children.append((r+1, c))
                if r - 1 > -1:
                    children.append((r-1, c))
                if c + 1 < self.width:
                    children.append((r, c+1))
                if c - 1 > -1:
                    children.append((r, c-1))

                new_node = Node(self.my_grid[r][c], children, (r, c))
                self.graph_dict[(r, c)] = new_node





