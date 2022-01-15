from dfs_depth_n import dfs_depth_n
from math import log, floor
from dijkstra_level2 import dij_show_way,dij_show_action
from queue import LifoQueue
from minmax_alpha_beta import minmax
from minimax_alpha_beta1 import minmax1
from TrapOrNot import trapornot
from hitortrap import hitortrap
from HoleorNot import holeornot
import numpy as np
from matplotlib import pyplot as plt
import qlearning

#change diamondlist to dictionary, should other functions
diamond = {}
hole = {}
start_agent = tuple()
score = 0
size_action = 0
diccolornumber_agent = {'y': 0, 'g': 0, 'r': 0, 'b': 0}
diccolornumber_enemy = {'y': 0, 'g': 0, 'r': 0, 'b': 0}
actions = LifoQueue()
sizedh = 0
sizedh_minmax=0
prev_action = ''
walls = 0
score_agent = 0
score_enemy = 0
start_enemy = tuple()
agent_id = 0
enemy_id = 0
enemy_trap = []
agent_trap = []
transposition = {}
transposition_size = 0
max_depth = 0
trapcountinfo = 0
holecounter = 0
befor_score_agent=0
count_of_hits=0
before_start_agent=0
before_start_enemy=0
diamond_list = []
score_old = 0
# submask = 0
q_values = None

def getinfo(gridmap, height, width, character,scoreinitial):
    global diamond
    global hole
    global start_agent
    global score
    global sizedh
    global walls
    score = scoreinitial

    for i in range(0, height):
        for j in range(0, width):
            if gridmap[i][j]=='W':
                walls += 1
            if gridmap[i][j] == 'T':
                hole.append((i, j, 0))
            if gridmap[i][j] == '1':
                diamond.append((i, j, 10))
            if gridmap[i][j] == '2':
                diamond.append((i, j, 25))
            if gridmap[i][j] == '3':
                diamond.append((i, j, 35))
            if gridmap[i][j] == '4':
                diamond.append((i, j, 75))
            if gridmap[i][j] == ('E'+character):
                start_agent = (i, j)
            if (gridmap[i][j] == 'W'):
                walls += 1
    sizedh = len(diamond) + len(hole)
    # print(diamond, "diamond")
    # print(hole,"hole")


def startturn(gridmap, height, width, turn, maxturn, timelimit,character):
    global diamond
    global hole
    global diccolornumber
    global score
    global start_agent #should change in other turns
    global sizedh
    global walls
    way = LifoQueue()


    depth = floor(log((10 ** 4) * timelimit, max(sizedh, 2)))

    if (walls // (height + width)) * 100 < 5 and len(hole) == 0:
        depth -= 1

    # print(start,"start in startturn")

    # print(depth,"depth")
    # print(score,"befordfs")
    #we change maxturn-turn to maxturn-turn+1 because the first turn is 1
    next_move = dfs_depth_n(gridmap, height, width, maxturn - turn + 1, depth, start_agent[0], start_agent[1], diamond, hole, score, diccolornumber)
    # print(score,"afterdfsscore")
    # print(next_move, "next_move")
    if next_move == ():
        for i in range(maxturn-turn+1):
           way.put('n')
    else:
        if gridmap[next_move[0]][next_move[1]] == '1':
            diccolornumber['y'] += 1
            score += 10
            diamond.remove((next_move[0], next_move[1], 10))
        if gridmap[next_move[0]][next_move[1]] == '2':
            diccolornumber['g'] += 1
            score += 25
            diamond.remove((next_move[0], next_move[1], 25))
        if gridmap[next_move[0]][next_move[1]] == '3':
            diccolornumber['r'] += 1
            score += 35
            diamond.remove((next_move[0], next_move[1], 35))
        if gridmap[next_move[0]][next_move[1]] == '4':
            diccolornumber['b'] += 1
            score += 75
            diamond.remove((next_move[0], next_move[1], 75))

        # print(start," start")
        # print(next_move, "next_move")
        way = dij_show_way(start_agent[0], start_agent[1], next_move[0], next_move[1], gridmap, height, width, score,character)
    start_agent = next_move
    # print(score,"scoreafter")
    # while not way.empty():
    #     print(way.get())
    return way


def action_state_func(gridmap, height, width, turn, maxturn, character, maxscore, time):
    global actions
    global size_action
    global score
    global prev_action
    global hole
    global start_agent
    if turn == 1 and actions.empty():
        getinfo(gridmap, height, width, character, maxscore)
        actions = startturn(gridmap, height, width, turn, maxturn, time,character)
        size_action = actions.qsize()
    if actions.empty():
        score -= size_action
        if prev_action == 't':
            for item_hole in hole:
                if gridmap[item_hole[0]][item_hole[1]] == 'T'+character:
                    start_agent = (item_hole[0], item_hole[1])
                    # print(start,"start hole")
        actions = startturn(gridmap, height, width, turn, maxturn, time,character)
        # print(actions)
        size_action = actions.qsize()

    # while not actions.empty():
    #     print(actions.get())
    # print("weeeeee innnnn")
    prev_action = actions.get()
    # print(prev_action,"current_action")
    return prev_action

def getinfophase2(gridmap, height, width, turn, maxturn, character,scoreinitial, scores, trapcount,timelimit):
    global diamond
    global hole
    global start_agent
    global score_agent
    global score_enemy
    global sizedh_minmax
    global walls
    global start_enemy
    global enemy_list
    global transposition
    global transposition_size
    global max_depth
    score_agent = scoreinitial
    score_enemy = scoreinitial

    if character == 'A':
        character_enemy = 'B'
        agent_id = 0
        enemy_id = 1
        score_agent = scores[0]
        score_enemy = scores[1]
    else:
        character_enemy = 'A'
        agent_id = 1
        enemy_id = 0
        score_enemy = scores[0]
        score_agent = scores[1]

    if turn == 1:
        for i in range(0, height):
            for j in range(0, width):
                s = str(gridmap[i][j])
                if gridmap[i][j] == 'W':
                    walls += 1
                if gridmap[i][j] == 'T':
                    hole[(i, j, 0)] = True
                if gridmap[i][j] == '1':
                    diamond[(i, j, 10)] = True
                if gridmap[i][j] == '2':
                    diamond[(i, j, 25)] = True
                if gridmap[i][j] == '3':
                    diamond[(i, j, 35)] = True
                if gridmap[i][j] == '4':
                    diamond[(i, j, 75)] = True
                if s.find(character) != -1:
                    start_agent = (i, j)
                if s.find(character_enemy) != -1:
                    start_enemy = (i, j)


        transposition_size = height * width - (walls+len(diamond)+len(hole))

        sizedh_minmax = len(diamond) + len(hole)
    if turn != 1:
        previous_enemy_place = start_enemy
        prevous_enemy_score = score_enemy
        for i in range(0, height):
            for j in range(0, width):
                s = str(gridmap[i][j])
                if s.find(character) != -1:
                    start_agent = (i, j)
                if s.find(character_enemy) != -1:
                    start_enemy = (i, j)


        if start_enemy == previous_enemy_place and (start_enemy not in enemy_trap):
            enemy_trap.append(start_enemy)


        # score_enemy -= 1
        # score_agent -= 1
        # if start_agent == previous_enemy_place:
        #     if prevous_enemy_score > score_agent:
        #         score_agent -= 19
        #     elif prevous_enemy_score < score_agent:
        #         score_enemy -= 19
        #     elif score_agent == prevous_enemy_score:
        #         score_enemy -= 19
        # if start_enemy == start_agent:
        #     if score_enemy > score_agent:
        #         score_agent -= 19
        #     elif score_agent > score_enemy:
        #         score_enemy -= 19
        #     elif score_agent == score_agent:
        #         score_agent -= 19
        # fortrap = str(gridmap[start_enemy[i]][start_enemy[j]])
        # if fortrap.find(character.lower()):
        #     score_enemy -= 39
        if (start_enemy[0],start_enemy[1],10) in diamond:
            if diccolornumber_enemy['y'] < 15:
                diccolornumber_enemy['y'] += 1
                del diamond[ (start_enemy[0],start_enemy[1],10)]

        if (start_enemy[0],start_enemy[1],25) in diamond:
            if diccolornumber_enemy['g'] < 8 and score_enemy-25 > 14:
                diccolornumber_enemy['g'] += 1
                del diamond [(start_enemy[0],start_enemy[1],25)]

        if (start_enemy[0],start_enemy[1],35) in diamond:
            if diccolornumber_enemy['r'] < 5 and score_enemy - 35 > 49:
                diccolornumber_enemy['r'] += 1
                del diamond[(start_enemy[0],start_enemy[1],35)]
        if (start_enemy[0], start_enemy[1], 75) in diamond:
            if diccolornumber_enemy['b'] < 4 and score_enemy - 75 > 139:
                diccolornumber_enemy['b'] += 1
                del diamond[(start_enemy[0], start_enemy[1], 75)]

        if (start_agent[0], start_agent[1], 10) in diamond:
            if diccolornumber_agent['y'] < 15:
                diccolornumber_agent['y'] += 1
                del diamond[(start_agent[0], start_agent[1], 10)]

        if (start_agent[0], start_agent[1], 25) in diamond:
            if diccolornumber_agent['g'] < 8 and score_agent - 25 > 14:
                diccolornumber_agent['g'] += 1
                del diamond[(start_agent[0], start_agent[1], 25)]

        if (start_agent[0], start_agent[1], 35) in diamond:
            if diccolornumber_agent['r'] < 5 and score_agent - 35 > 49:
                diccolornumber_agent['r'] += 1
                del diamond[(start_agent[0], start_agent[1], 35)]

        if (start_agent[0], start_agent[1], 75) in diamond:
            if diccolornumber_agent['b'] < 4 and score_agent - 75 > 139:
               diccolornumber_agent['b'] += 1
               del diamond[(start_agent[0], start_agent[1], 75)]



    agent_trap=[]

    # print(character_enemy,"character enemy")
    # print(character,"character")
    depth_minmax = floor(log((10 ** 4) * timelimit, max(sizedh_minmax, 2)))

    if (walls // (height + width)) * 100 < 5 and len(hole) == 0:

        depth_minmax -= 1

    depth_minmax = max(max_depth, depth_minmax)
    if depth_minmax % 2:
        depth_minmax = max(depth_minmax-1, 2)

    # print("-"*15)
    # print(maxturn-turn+1)
    # print("-" * 15)
    next_move,max_depth = minmax(gridmap, height, width, maxturn-turn+1, maxturn-turn+1, diamond, hole, start_agent[0], start_agent[1], start_enemy[0], start_enemy[1], trapcount, depth_minmax, score_agent, score_enemy, diccolornumber_agent,diccolornumber_enemy,transposition, enemy_trap, agent_trap,transposition_size,max_depth,character,character_enemy)
    # print(next_move,"next_moove")
    if not next_move == ():
       next_action = dij_show_action(start_agent[0], start_agent[1], next_move[0], next_move[1], gridmap, height, width,score_agent,enemy_trap,character,diccolornumber_agent)
       return next_action
    else:
        return 'n'



def getinfophase2_1(gridmap, height, width, turn, maxturn, character,scoreinitial, scores, trapcount,timelimit):
    global diamond
    global hole
    global start_agent
    global score_agent
    global score_enemy
    global sizedh_minmax
    global walls
    global start_enemy
    global enemy_list
    global transposition
    global transposition_size
    global max_depth
    global agent_trap
    global trapcountinfo
    global holecounter
    global befor_score_agent
    global count_of_hits
    global before_start_agent
    global before_start_enemy
    if turn == 1:
      score_agent = scoreinitial
      score_enemy = scoreinitial

    if character == 'A':
        character_enemy = 'B'
        agent_id = 0
        enemy_id = 1
        befor_score_agent = score_agent
        score_agent = scores[0]
        score_enemy = scores[1]
    else:
        character_enemy = 'A'
        agent_id = 1
        enemy_id = 0
        befor_score_agent = score_agent
        score_enemy = scores[0]
        score_agent = scores[1]

    if turn == 1:

        for i in range(0, height):
            for j in range(0, width):
                s = str(gridmap[i][j])
                if gridmap[i][j] == 'W':
                    walls += 1
                if gridmap[i][j] == 'T' or gridmap[i][j] == 'T'+character_enemy:
                    hole[(i, j, 0)] = True
                if gridmap[i][j] == '1':
                    diamond[(i, j, 10)] = True
                if gridmap[i][j] == '2':
                    diamond[(i, j, 25)] = True
                if gridmap[i][j] == '3':
                    diamond[(i, j, 35)] = True
                if gridmap[i][j] == '4':
                    diamond[(i, j, 75)] = True
                if s.find(character) != -1:
                    start_agent = (i, j)
                if s.find(character_enemy) != -1:
                    start_enemy = (i, j)
        print(start_enemy,"start enemy")
        print(start_agent, "start_agent")
        trapcountinfo=trapcount

        transposition_size = height * width - (walls+len(diamond)+len(hole))

        sizedh_minmax = len(diamond) + len(hole)
    if turn != 1:
        previous_enemy_place = start_enemy
        prevous_enemy_score = score_enemy
        before_start_agent = start_agent
        before_start_enemy = start_enemy
        for i in range(0, height):
            for j in range(0, width):
                s = str(gridmap[i][j])
                if s.find(character) != -1:
                    start_agent = (i, j)
                if s.find(character_enemy) != -1:
                    start_enemy = (i, j)


        # print(befor_score_agent, "befor score agent")
        # print(score_agent, "score_agent")
        print(gridmap[before_start_agent[0]][before_start_agent[1]],"gridmap[before_start_agent[0]][before_start_agent[1]]")
        print(gridmap[start_agent[0]][start_agent[1]],"gridmap[start_agent[0]][start_agent[1]]")
        print(gridmap[before_start_enemy[0]][before_start_enemy[1]],"gridmap[before_start_enemy[0]][before_start_enemy[1]]")
        print(gridmap[start_enemy[0]][start_enemy[1]],"gridmap[start_enemy[0]][start_enemy[1]]== 'T'+character_enemy")


        if (befor_score_agent - score_agent == 21) and ((gridmap[before_start_agent[0]][before_start_agent[1]]=='T' + character and before_start_agent == start_agent)):
            #print("im in hits")
            count_of_hits += 1

        if start_enemy == previous_enemy_place and (start_enemy not in enemy_trap) and gridmap[start_enemy[0]][start_enemy[1]] != 'T'+character_enemy:
            enemy_trap.append(start_enemy)

        if (start_enemy[0], start_enemy[1], 10) in diamond:
            if diccolornumber_enemy['y'] < 15:
                diccolornumber_enemy['y'] += 1
                del diamond[(start_enemy[0], start_enemy[1], 10)]

        if (start_enemy[0], start_enemy[1], 25) in diamond:
            if diccolornumber_enemy['g'] < 8 and score_enemy - 25 > 14:
                diccolornumber_enemy['g'] += 1
                del diamond[(start_enemy[0], start_enemy[1], 25)]

        if (start_enemy[0], start_enemy[1], 35) in diamond:
            if diccolornumber_enemy['r'] < 5 and score_enemy - 35 > 49:
                diccolornumber_enemy['r'] += 1
                del diamond[(start_enemy[0], start_enemy[1], 35)]
        if (start_enemy[0], start_enemy[1], 75) in diamond:
            if diccolornumber_enemy['b'] < 4 and score_enemy - 75 > 139:
                diccolornumber_enemy['b'] += 1
                del diamond[(start_enemy[0], start_enemy[1], 75)]

        if (start_agent[0], start_agent[1], 10) in diamond:
            if diccolornumber_agent['y'] < 15:
                diccolornumber_agent['y'] += 1
                del diamond[(start_agent[0], start_agent[1], 10)]

        if (start_agent[0], start_agent[1], 25) in diamond:
            if diccolornumber_agent['g'] < 8 and score_agent - 25 > 14:
                diccolornumber_agent['g'] += 1
                del diamond[(start_agent[0], start_agent[1], 25)]

        if (start_agent[0], start_agent[1], 35) in diamond:
            if diccolornumber_agent['r'] < 5 and score_agent - 35 > 49:
                diccolornumber_agent['r'] += 1
                del diamond[(start_agent[0], start_agent[1], 35)]

        if (start_agent[0], start_agent[1], 75) in diamond:
            if diccolornumber_agent['b'] < 4 and score_agent - 75 > 139:
               diccolornumber_agent['b'] += 1
               del diamond[(start_agent[0], start_agent[1], 75)]


    # print(character_enemy,"character enemy")
    # print(character,"character")
    depth_minmax = floor(log((10 ** 4) * timelimit, max(sizedh_minmax, 2)))
    if (walls // (height + width)) * 100 < 5 and len(hole) == 0:
        depth_minmax -= 1

    depth_minmax = max(max_depth, depth_minmax)
    if depth_minmax % 2:
        depth_minmax = max(depth_minmax-1, 2)
    # print(depth_minmax, " depth minimax")
    print("-"*15)
    print(maxturn-turn+1)
    print("-" * 15)
    # print(start_enemy,"start_enemy")
    # print(start_agent,"start_agent")
    if character_enemy == 'A':
        enemyturn = maxturn - turn
    else:
        enemyturn = maxturn - turn + 1
    print(start_agent, " start agent")
    print(start_enemy, " start enemy")
    next_move,next_move_enemy,max_depth,maxvalue = minmax1(gridmap, height, width, maxturn-turn+1, enemyturn, diamond, hole, start_agent[0], start_agent[1], start_enemy[0], start_enemy[1], trapcountinfo,depth_minmax, score_agent, score_enemy, diccolornumber_agent,diccolornumber_enemy,transposition, enemy_trap, agent_trap,transposition_size,max_depth,character,character_enemy)
    print(next_move, "next_move agent from minimax")
    print(next_move_enemy, "next_move enemy from minimax")
    print(maxvalue, "max value from minimax")

    next_move_holeOrnot, max_valueholeOrnot = holeornot(gridmap, height,width,holecounter, hole, score_agent, score_enemy, agent_trap, enemy_trap, befor_score_agent,start_agent,start_enemy,next_move,next_move_enemy,character,character_enemy, maxvalue,count_of_hits)
    if max_valueholeOrnot > maxvalue:
        maxvalue = max_valueholeOrnot
        next_move = next_move_holeOrnot
    maxvaluefortrap = float('-inf')
    next_move_trap = tuple()
    trapnumber = len(agent_trap)
    if score_agent >= 35*(trapnumber+1) and trapnumber < trapcountinfo:
        next_move_trap, maxvaluefortrap, flagdiamond = trapornot(gridmap,height, width, next_move, next_move_enemy, maxvalue, score_agent, score_enemy, start_agent, start_enemy, 35 * (trapnumber+1), diccolornumber_agent, diccolornumber_enemy, agent_trap, enemy_trap, character, character_enemy)
        # if (flagdiamond == False) and next_move_trap != ():
        #     maxvaluefortrap = maxvalue + 1
    if next_move_trap != () and maxvaluefortrap > maxvalue:
        next_move = next_move_trap
    if not next_move == ():
       print(next_move, "finaly next move in getinfo")
       next_action = dij_show_action(start_agent[0], start_agent[1], next_move[0], next_move[1], gridmap, height, width,score_agent,score_enemy,enemy_trap,character,character_enemy,diccolornumber_agent)
       print(next_action, "next_action")
       if next_action == 'p':
           agent_trap.append(next_move)
       if next_action == 't':
           holecounter += 1
       else:
           holecounter =max(holecounter-1,0)
           count_of_hits =max(count_of_hits-1,0)


       return next_action

    else:

        next_move, maxvaluefortrap = hitortrap(gridmap, height, width, next_move_enemy, score_agent, score_enemy, maxvalue,start_agent,start_enemy, maxturn-turn+1, enemyturn, agent_trap, enemy_trap,character, character_enemy, diccolornumber_agent, diccolornumber_enemy, trapcountinfo)
        print(next_move, "next move from hitortrap")
        print(maxvaluefortrap, "maxvaluefortrap from hitortrap")
        if next_move != ():
            next_action = dij_show_action(start_agent[0], start_agent[1], next_move[0], next_move[1], gridmap, height,
                                          width, score_agent, score_enemy, enemy_trap, character, character_enemy,
                                          diccolornumber_agent)

            if next_action == 'p':
                agent_trap.append(next_move)
            if next_action == 't':
                holecounter += 1

            #     holeornot(holecounter,hole,score_agent,score_enemy,agent_trap,enemy_trap,befor_score_agent)
            else:
                holecounter = max(holecounter-1, 0)
                count_of_hits = max(count_of_hits-1, 0)

            return next_action
        else:
           return 'n'




def learning_func(gridmap, height, width, turn, maxturn, character,scoreinitial, scores, timelimit):
     global diamond
     global hole
     global walls
     global start_agent
     global diamond_list
     rewards = np.full((height, width), 0.)

     for i in range(0, height):
         for j in range(0, width):
             if gridmap[i][j] == 'W':
                 walls += 1
             if gridmap[i][j] == 'T':
                 hole[(i, j, 0)] = True
                 rewards[i][j] = -1
             if gridmap[i][j] == '1':
                 diamond[(i, j, 10)] = True
                 rewards[i][j] = 10
                 diamond_list.append((i, j, 10))
             if gridmap[i][j] == '2':
                 diamond[(i, j, 25)] = True
                 rewards[i][j] = 25
                 diamond_list.append((i, j, 25))
             if gridmap[i][j] == '3':
                 diamond[(i, j, 35)] = True
                 rewards[i][j] = 35
                 diamond_list.append((i, j, 35))
             if gridmap[i][j] == '4':
                 diamond[(i, j, 75)] = True
                 rewards[i][j] = 75
                 diamond_list.append((i, j, 75))
             if gridmap[i][j][1:] == character:
                 start_agent = (i,j)
             if gridmap[i][j] == 'E' or gridmap[i][j] == 'E'+character:
                 rewards[i][j] = -1

     episodes = 6000
     epsilon = 0.9
     learning_rate = 0.2
     discount_factor = 0.9
     reduce_epsilon = 0.4
     submask_learning = 1 << len(diamond_list)
     #q_values = np.full((height, width,submask_learning, 5), 0.)#full with out
     q_values = {}
     rewards_avrg_episodes={}
     rewards_avrg_episodes['avg']=[]
     rewards_avrg_episodes['episodes']=[]
     rewards_episodes=[]
     SHOW_EVERY = 800

     # for i in range(height):
     #     for j in range(width):
     #         if gridmap[i][j] != 'T':
     #             q_values[i,j,:,4] = float('-inf')

     for i in range(episodes):
        # print("im in episode", i)
        submask_copy = submask_learning-1
        turns = maxturn - turn + 1
        score_agent = scoreinitial[0]
        diamond_copy = diamond.copy()
        rewards_copy = rewards.copy()
        diccolornumber_agent = {'y': 0, 'b': 0, 'g': 0, 'r': 0}
        observation = (start_agent, turns, diamond_copy, diccolornumber_agent)
        location_agent = start_agent
        all_rewards=0
        times=0
        while not qlearning.is_terminal(observation):
            times+=1
            #0=>left,1=>right,2=>up,3=>down,4=>teleport
            location_agent_old = observation[0]
            if (location_agent_old[0], location_agent_old[1], submask_copy) not in q_values:
                q_values[(location_agent_old[0], location_agent_old[1], submask_copy)] = [0.,0.,0.,0.,0.]
                if gridmap[location_agent_old[0]][location_agent_old[1]] != 'T':
                    q_values[(location_agent_old[0], location_agent_old[1], submask_copy)][4] = float ('-inf')
            action_agent = qlearning.getNextAction(observation,gridmap, height, width, hole, score_agent,character, epsilon,q_values,submask_copy)
            location_agent_new = qlearning.getlocation(action_agent, location_agent_old,hole)
            submask_copy_new = submask_copy
            # delete diamond that we get in action from diamond_copy
            reward = qlearning.getreward(rewards_copy, location_agent_new, score_agent,gridmap,diccolornumber_agent,character)
            all_rewards += reward
            observation = (location_agent_new, turns, diamond_copy, diccolornumber_agent)
            if reward == 10:
                del diamond_copy[(observation[0][0], observation[0][1], 10)]
                rewards_copy[observation[0][0]][observation[0][1]] = -1
                index_diamond = diamond_list.index((observation[0][0], observation[0][1], 10))
                submask_copy_new = submask_copy_new & (~(1 << index_diamond))
                diccolornumber_agent['y'] += 1
                score_agent += 10
            if reward == 25:
                del diamond_copy[(observation[0][0], observation[0][1], 25)]
                rewards_copy[observation[0][0]][observation[0][1]] = -1
                index_diamond = diamond_list.index((observation[0][0], observation[0][1], 25))
                submask_copy_new = submask_copy_new & (~(1 << index_diamond))
                diccolornumber_agent['g'] += 1
                score_agent += 25
            if reward == 35:
                del diamond_copy[(observation[0][0], observation[0][1], 35)]
                rewards_copy[observation[0][0]][observation[0][1]] = -1
                index_diamond = diamond_list.index((observation[0][0], observation[0][1], 35))
                submask_copy_new = submask_copy_new & (~(1 << index_diamond))
                diccolornumber_agent['r'] += 1
                score_agent += 35
            if reward == 75:
                 del diamond_copy[(observation[0][0], observation[0][1], 75)]
                 rewards_copy[observation[0][0]][observation[0][1]] = -1
                 index_diamond = diamond_list.index((observation[0][0], observation[0][1], 75))
                 submask_copy_new = submask_copy_new & (~(1 << index_diamond))
                 diccolornumber_agent['b'] += 1
                 score_agent += 75
            if (location_agent_new[0], location_agent_new[1], submask_copy_new) not in q_values:
                q_values[(location_agent_new[0], location_agent_new[1], submask_copy_new)] = [0., 0., 0., 0., 0.]
                if gridmap[location_agent_new[0]][location_agent_new[1]] != 'T':
                    q_values[(location_agent_new[0], location_agent_new[1], submask_copy_new)][4] = float('-inf')
            turns -= 1
            observation = (location_agent_new, turns, diamond_copy, diccolornumber_agent)
            #old_q_value = q_values[location_agent_old[0],location_agent_old[1],submask_copy, action_agent]
            old_q_value = q_values[(location_agent_old[0], location_agent_old[1], submask_copy)][action_agent]
            #temporal_difference = reward + (discount_factor * np.max(q_values[location_agent_new[0], location_agent_new[1], submask_copy_new])) - old_q_value
            #temporal_difference = reward + (discount_factor * max(q_values[(location_agent_new[0], location_agent_new[1], submask_copy_new)])) - old_q_value
            temporal_difference = reward + (discount_factor * np.max(np.array(q_values[(location_agent_new[0], location_agent_new[1], submask_copy_new)]))) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)

            q_values[(location_agent_old[0], location_agent_old[1], submask_copy)][action_agent] = new_q_value
            score_agent -= 1
            submask_copy = submask_copy_new
        print(all_rewards,"rewards for each episodes")
        rewards_episodes.append(all_rewards)
        if not i % SHOW_EVERY:
            average_reward = sum(rewards_episodes[-SHOW_EVERY:]) / SHOW_EVERY
            rewards_avrg_episodes['avg'].append(average_reward)
            rewards_avrg_episodes['episodes'].append(i)
        print(times,"agent truns in each episodes")
        epsilon *= reduce_epsilon
     return q_values,rewards_avrg_episodes


def getinfophase3(gridmap, height, width, turn, maxturn, character,scoreinitial, scores, timelimit):
    global diamond_list
    global score_old
    global q_values

    agentx = 0
    agenty = 0
    for i in range(height):
        for j in range(width):
            if gridmap[i][j][1:] == character:
                agentx = i
                agenty = j
                break
    if turn == 1:
        score_old = scoreinitial[0]
        q_values,rewards_episodes = learning_func(gridmap, height, width, turn, maxturn, character,scoreinitial, scores, timelimit)
        plt.plot(rewards_episodes['episodes'],rewards_episodes['avg'])
        plt.xlabel("epochs")
        plt.ylabel("average rewards")
        plt.show()


    submask_index=0
    for i,item in enumerate(diamond_list):
        if 'E' not in gridmap[item[0]][item[1]]:
            submask_index |= (1 << i)

    score_old = scores
    if ((agentx, agenty, submask_index)) not in q_values:
        q_values[((agentx, agenty, submask_index))] = [0., 0., 0., 0., 0.]
        if gridmap[agentx][agenty] != 'T':
            q_values[(agentx, agenty, submask_index)][4] = float('-inf')
    action = np.argmax(np.array(q_values[(agentx, agenty, submask_index)]))
    # 0=>left,1=>right,2=>up,3=>down,4=>teleport
    if action == 0:
        return 'l'
    if action == 1:
        return 'r'
    if action == 2:
        return 'u'
    if action == 3:
        return 'd'
    if action== 4:
        return 't'