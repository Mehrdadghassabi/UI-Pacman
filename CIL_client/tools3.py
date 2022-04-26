def find_elements(map, x, y):
    def find_agent():
        for i in range(len(map)):
            for j in range(len(map[i])):
                if 'A' in map[i][j]:
                    return [i, j]

        # if map[0][0] == 'EA':
        #     return [0, 0]
        # if map[0][y - 1] == 'EA':
        #     return [0, y - 1]
        # if map[x - 1][0] == 'EA':
        #     return [x - 1, 0]
        # if map[x - 1][y - 1] == 'EA':
        #     return [x - 1, y - 1]

    def find_gem_sequence():
        sequence = list()
        gem_id = 0
        for i in range(x):
            for j in range(y):
                if map[i][j] == '1':
                    sequence.append([gem_id, 1, i, j])
                    gem_id += 1
                if map[i][j] == '2':
                    sequence.append([gem_id, 2, i, j])
                    gem_id += 1
                if map[i][j] == '3':
                    sequence.append([gem_id, 3, i, j])
                    gem_id += 1
                if map[i][j] == '4':
                    sequence.append([gem_id, 4, i, j])
                    gem_id += 1
        return gem_id, sequence

    def find_wall():
        wall_sequence = list()
        for i in range(x):
            for j in range(y):
                if map[i][j] == 'W':
                    wall_sequence.append([i, j])
        return wall_sequence

    def find_teleport():
        teleport_sequence = list()
        for i in range(x):
            for j in range(y):
                if map[i][j] == 'T':
                    teleport_sequence.append([i, j])
        return teleport_sequence

    return find_agent(), find_gem_sequence(), find_wall(), find_teleport()
