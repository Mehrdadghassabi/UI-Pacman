import numpy as np
def is_terminal(observation):
    agentx = observation[0][0]
    agenty = observation[0][1]
    turn = observation[1]
    diamond = observation[2]
    dic_color_number = observation[3]
    if turn == 0 or (len(diamond)==0):
        return True
    if dic_color_number['y'] == 15 and dic_color_number['g'] == 8 and dic_color_number['r'] == 5 and dic_color_number['b'] == 4:
        return True

    return False



def getNextAction(observation, gridmap, height, width, hole, score_agent,character, epsilon, q_values,submask_copy):
    agentx = observation[0][0]
    agenty = observation[0][1]
    turn = observation[1]
    diamond = observation[2]
    dic_color_number = observation[3]

    if np.random.random() > epsilon:
        #q_values[agentx][agenty]?
        # print("summask",bin(submask_copy))
        # print("action ",q_values[agentx, agenty,submask_copy])
        #action = np.argmax(q_values[agentx, agenty,submask_copy])
        action = np.argmax(np.array(q_values[(agentx, agenty,submask_copy)]))
        if action == 2:
            if agentx - 1 >= 0 and gridmap[agentx - 1][agenty] != 'W':
                return action

        if action == 3:
            if agentx + 1 < height and gridmap[agentx + 1][agenty] != 'W':
                return action

        if action == 0:
            if agenty - 1 >= 0 and gridmap[agentx][agenty - 1] != 'W':
                return action

        if action == 1:
            if agenty + 1 < width and gridmap[agentx][agenty + 1] != 'W':
                return action
        if action == 4:
            return action


    flag =False
    while not flag:

        if gridmap[agentx][agenty] == 'T' + character or gridmap[agentx][agenty] == 'T':
            action = np.random.randint(5)
        else:
            action = np.random.randint(4)

        # 0=>left,1=>right,2=>up,3=>down,4=>teleport
        if action == 2:
            if agentx-1 >= 0 and gridmap[agentx-1][agenty] != 'W':
                flag=True

        if action == 3:
            if agentx + 1 < height and gridmap[agentx + 1][agenty] != 'W':
                flag=True

        if action == 0:
            if agenty - 1 >= 0 and gridmap[agentx][agenty-1] != 'W':
                flag = True

        if action == 1:
            if agenty + 1 < width and gridmap[agentx][agenty+1] != 'W':
                flag = True
        if action == 4:
            flag=True

    return action


def getlocation(action_agent, location_agent_old, holes):
    list_location = []
    # 0=>left,1=>right,2=>up,3=>down,4=>teleport
    if action_agent == 0:
        return (location_agent_old[0], location_agent_old[1]-1)
    if action_agent == 1:
        return (location_agent_old[0], location_agent_old[1] + 1)
    if action_agent == 2:
        return (location_agent_old[0]-1, location_agent_old[1])
    if action_agent == 3:
        return (location_agent_old[0]+1, location_agent_old[1])
    if action_agent == 4:
        for hole in holes:
            if hole != (location_agent_old[0], location_agent_old[1], 0):
                list_location.append((hole[0], hole[1]))

        num_random = np.random.randint(len(list_location))

        return list_location[num_random]


def getreward(rewards_copy, location_agent_new, score_agent,gridmap,diccolornumber_agent,character):

    if gridmap[location_agent_new[0]][location_agent_new[1]] == 'E' or gridmap[location_agent_new[0]][location_agent_new[1]] == 'E'+character or \
            gridmap[location_agent_new[0]][location_agent_new[1]] == 'T' or gridmap[location_agent_new[0]][location_agent_new[1]]=='T' + character:
        return rewards_copy[location_agent_new[0]][location_agent_new[1]]
    if gridmap[location_agent_new[0]][location_agent_new[1]] == '1':
        if diccolornumber_agent['y'] < 15:
           return rewards_copy[location_agent_new[0]][location_agent_new[1]]
        else:
            return -1

    if gridmap[location_agent_new[0]][location_agent_new[1]] == '2':
        if diccolornumber_agent['g'] < 8 and score_agent >= 15:
           return rewards_copy[location_agent_new[0]][location_agent_new[1]]
        else:
            return -1

    if gridmap[location_agent_new[0]][location_agent_new[1]] == '3':
        if diccolornumber_agent['r'] < 5 and score_agent >= 50:
            return rewards_copy[location_agent_new[0]][location_agent_new[1]]
        else:
            return -1

    if gridmap[location_agent_new[0]][location_agent_new[1]] == '4':
        if diccolornumber_agent['b'] < 4 and score_agent >= 140:
            return rewards_copy[location_agent_new[0]][location_agent_new[1]]
        else:
            return -1