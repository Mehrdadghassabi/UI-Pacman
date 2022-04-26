from base import Action
from genetic_motor import define_requirment


class Node:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value
        self.visibility = True
        self.point = 0
        self.kind = None
        self.gem_value = None


class Map:
    def __init__(self, height, width):
        self.grid = [[None for j in range(width)] for i in range(height)]
        self.primary_grid = [[None for j in range(width)] for i in range(height)]
        self.gems_collected = [0 for i in range(4)]
        self.minimums = [0, 15, 50, 140]
        self.gem_limits = [15, 8, 5, 4]
        self.height = height
        self.width = width
        self.holes = []
        self.gems = []
        self.path_to_go = []
        self.sequence = []
        self.agent_x = int
        self.agent_y = int
        self.teleporting = False

    def set_map(self, grid: list):
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = Node(i, j, grid[i][j])
                if 'A' in grid[i][j]:
                    self.agent_x = i
                    self.agent_y = j

    # def set_point(self):
    #     for i in self.gems:
    #         if not i.visibility:
    #             continue
    #         point = 0
    #         distance = abs(i.x - self.agent_x) + abs(i.y - self.agent_y)
    #         point -= distance * 0.2
    #         point += i.gem_value
    #         if i.x != 0:
    #             if self.grid[i.x - 1][i.y].kind == 'gem':
    #                 point += self.grid[i.x - 1][i.y].gem_value * 1.6
    #         if i.y != 0:
    #             if self.grid[i.x][i.y - 1].kind == 'gem':
    #                 point += self.grid[i.x][i.y - 1].gem_value * 1.6
    #         if i.x != self.height - 1:
    #             if self.grid[i.x + 1][i.y].kind == 'gem':
    #                 point += self.grid[i.x + 1][i.y].gem_value * 1.6
    #         if i.y != self.width - 1:
    #             if self.grid[i.x][i.y + 1].kind == 'gem':
    #                 point += self.grid[i.x][i.y + 1].gem_value * 1.6
    #         if i.x != 0 and i.y != 0:
    #             if self.grid[i.x - 1][i.y - 1].kind == 'gem':
    #                 point += self.grid[i.x - 1][i.y - 1].gem_value * 1.6
    #         if i.x != 0 and i.y != self.width - 1:
    #             if self.grid[i.x - 1][i.y + 1].kind == 'gem':
    #                 point += self.grid[i.x - 1][i.y + 1].gem_value * 1.6
    #         if i.x != self.height - 1 and i.y != 0:
    #             if self.grid[i.x + 1][i.y - 1].kind == 'gem':
    #                 point += self.grid[i.x + 1][i.y - 1].gem_value * 1.6
    #         if i.x != self.height - 1 and i.y != self.width - 1:
    #             if self.grid[i.x + 1][i.y + 1].kind == 'gem':
    #                 point += self.grid[i.x + 1][i.y + 1].gem_value * 1.6
    #         i.point = point

    def set_kind(self, turn):
        for i in range(self.height):
            for j in range(self.width):
                temp = (self.grid[i][j]).value
                if temp == 'EA':
                    (self.grid[i][j]).kind = 'empty'
                elif temp == 'E':
                    (self.grid[i][j]).kind = 'empty'
                elif temp == 'T':
                    (self.grid[i][j]).kind = 'teleport'
                    if turn == 1:
                        self.holes.append(self.grid[i][j])
                elif temp == '1':
                    (self.grid[i][j]).kind = 'gem'
                    (self.grid[i][j]).gem_value = 10
                    if turn == 1:
                        self.gems.append(self.grid[i][j])
                elif temp == '2':
                    (self.grid[i][j]).kind = 'gem'
                    (self.grid[i][j]).gem_value = 25
                    if turn == 1:
                        self.gems.append(self.grid[i][j])
                elif temp == '3':
                    (self.grid[i][j]).kind = 'gem'
                    (self.grid[i][j]).gem_value = 35
                    if turn == 1:
                        self.gems.append(self.grid[i][j])
                elif temp == '4':
                    (self.grid[i][j]).kind = 'gem'
                    (self.grid[i][j]).gem_value = 75
                    if turn == 1:
                        self.gems.append(self.grid[i][j])
                else:
                    (self.grid[i][j]).kind = 'wall'

    def set_visibility(self, score):
        for j in self.sequence:
            i = self.grid[j[0]][j[1]]
            if i.value == '2':
                if score < self.minimums[1]:
                    (self.grid[i.x][i.y]).visibility = False
                    i.visibility = False
                else:
                    (self.grid[i.x][i.y]).visibility = True
                    i.visibility = True
            if i.value == '3':
                if score < self.minimums[2]:
                    (self.grid[i.x][i.y]).visibility = False
                    i.visibility = False
                else:
                    (self.grid[i.x][i.y]).visibility = True
                    i.visibility = True
            if i.value == '4':
                if score < self.minimums[3]:
                    (self.grid[i.x][i.y]).visibility = False
                    i.visibility = False
                else:
                    (self.grid[i.x][i.y]).visibility = True
                    i.visibility = True

    def bfs(self, x_goal, y_goal):
        queue = [(self.grid[self.agent_x][self.agent_y], [])]
        visited = {}

        while len(queue) > 0:
            node, path = queue.pop(0)
            path.append(node)
            visited[node] = 1
            adj_nodes = []

            if node.x == x_goal and node.y == y_goal:
                self.path_to_go = path
                return 1

            if node.x != 0:
                if (self.grid[node.x - 1][node.y]).kind != 'wall':
                    adj_nodes.append(self.grid[node.x - 1][node.y])
            if node.y != 0:
                if (self.grid[node.x][node.y - 1]).kind != 'wall':
                    adj_nodes.append(self.grid[node.x][node.y - 1])
            if node.x != self.height - 1:
                if (self.grid[node.x + 1][node.y]).kind != 'wall':
                    adj_nodes.append(self.grid[node.x + 1][node.y])
            if node.y != self.width - 1:
                if (self.grid[node.x][node.y + 1]).kind != 'wall':
                    adj_nodes.append(self.grid[node.x][node.y + 1])

            for item in adj_nodes:
                if item not in visited:
                    queue.append((item, path[:]))

        return -1


grid: Map


def set_requirements(main_grid, grid_height, grid_width, turn, max_turn, score):
    global grid
    if turn == 1:
        grid = Map(grid_height, grid_width)

    if len(grid.path_to_go) == 0 and not grid.teleporting:
        grid.sequence = define_requirment(main_grid, grid_height, grid_width, score, turn)
        # print(grid.sequence, 'seq')
        grid.set_map(main_grid)
        grid.set_kind(turn)
        if turn == 1:
            for i in range(grid.height):
                for j in range(grid.width):
                    grid.primary_grid[i][j] = grid.grid[i][j]
        grid.set_visibility(score)
        # grid.set_point()

        if not gems_visible() or not grid.gems or not grid.sequence:
            exit('No gem remains')

        x_goal, y_goal = grid.sequence.pop(0)
        while not grid.sequence:
            if 'E' in grid.grid[x_goal][y_goal].value:
                x_goal, y_goal = grid.sequence.pop(0)
        # print(x_goal, y_goal, ' goal 1')

        # for i in grid.gems:
        #     if i.point > maximum:
        #         if collectible(i.x, i.y, score):
        #             maximum = i.point
        #             x_goal = i.x
        #             y_goal = i.y

        if grid.bfs(x_goal, y_goal) == -1:
            minimum = 1000
            res = None
            for i in grid.holes:
                if abs(grid.agent_x - i.x) + abs(grid.agent_y - i.y) < minimum:
                    if grid.bfs(i.x, i.y) != -1:
                        minimum = abs(grid.agent_x - i.x) + abs(grid.agent_y - i.y)
                        res = i

            grid.sequence.insert(0, (x_goal, y_goal))
            x_goal, y_goal = res.x, res.y
            grid.bfs(x_goal, y_goal)
            if len(grid.path_to_go) == 1:
                grid.path_to_go.clear()
            grid.teleporting = True
        # print(x_goal, y_goal, ' goal 2')

        # alternative = probe(score)
        # if alternative != -1:
        #     x_goal, y_goal = alternative
        #     grid.bfs(x_goal, y_goal)
        #     grid.teleporting = False

        if len(grid.path_to_go) > max_turn - turn:
            exit("Next gem is not accessible because of max_turn limitation")

        for i in grid.gems:
            if i.x == x_goal and i.y == y_goal:
                grid.gems.remove(i)
                return


# def collectible(x_goal, y_goal, score):
#     distance = abs(x_goal - grid.agent_x) + abs(y_goal - grid.agent_y)
#     current_gem = grid.primary_grid[grid.agent_x][grid.agent_y].value
#     if current_gem == '1':
#         score += 10
#     if current_gem == '2':
#         score += 25
#     if current_gem == '3':
#         score += 35
#     if current_gem == '4':
#         score += 75
#
#     temp = 0
#     if (grid.grid[x_goal][y_goal]).value == '2':
#         temp = grid.minimums[1]
#     if (grid.grid[x_goal][y_goal]).value == '3':
#         temp = grid.minimums[2]
#     if (grid.grid[x_goal][y_goal]).value == '4':
#         temp = grid.minimums[3]
#     return score - distance >= temp


def what_to_do():
    global grid
    if not grid.path_to_go:
        if grid.teleporting:
            grid.teleporting = False
            return Action.TELEPORT
        exit('No path found')

    a = grid.path_to_go[0]
    b = grid.path_to_go[1]
    grid.path_to_go.pop(0)
    if len(grid.path_to_go) == 1:
        grid.path_to_go.pop(0)

    check(a.x, a.y)
    # print((a.x, a.y, b.x, b.y))
    # print(grid.teleporting)
    if a.x != b.x:
        if a.x > b.x:
            return Action.UP
        else:
            return Action.DOWN
    else:
        if a.y > b.y:
            return Action.LEFT
        else:
            return Action.RIGHT


# def probe(score):
#     for i in grid.path_to_go:
#         if i == grid.path_to_go[-1]:
#             continue
#         if i.x != 0:
#             if grid.grid[i.x - 1][i.y].value in ['1', '2', '3', '4'] and \
#                     grid.grid[i.x - 1][i.y].visibility and collectible(i.x - 1, i.y, score):
#                 return i.x - 1, i.y
#
#         if i.y != 0:
#             if grid.grid[i.x][i.y - 1].value in ['1', '2', '3', '4'] and \
#                     grid.grid[i.x][i.y - 1].visibility and collectible(i.x, i.y - 1, score):
#                 return i.x, i.y - 1
#
#         if i.x != grid.height - 1:
#             if grid.grid[i.x + 1][i.y].value in ['1', '2', '3', '4'] and \
#                     grid.grid[i.x + 1][i.y].visibility and collectible(i.x + 1, i.y, score):
#                 return i.x + 1, i.y
#
#         if i.y != grid.width - 1:
#             if grid.grid[i.x][i.y + 1].value in ['1', '2', '3', '4'] and \
#                     grid.grid[i.x][i.y + 1].visibility and collectible(i.x, i.y + 1, score):
#                 return i.x, i.y + 1
#
#     return -1

def gems_visible():
    for i in grid.gems:
        if i.visibility:
            return True

    return False


def check(x, y):
    if (grid.grid[x][y]).value in ['1', '2', '3', '4'] and grid.grid[x][y].visibility:
        if (x, y) in grid.sequence:
            grid.sequence.remove((x, y))
