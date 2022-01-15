from dijkstra_level1 import dijkstra
from calculatedFunc import hash_key, sortmoves
from queue import Queue


storedHkeys = Queue()
max_value = float('-inf')
next_move = tuple()
max_depth_new = -1


def minmax(gridmap, height, width, turn_agent, turn_enemy, diamonddic, holedic, agentx, agenty, enemyx, enemyy, trapcount, depth, scoreagent, scoreenemy, diccolornumberagent,diccolornumberenemy,transpositiontable, enemytraps, agenttraps,transpositionsize,max_depth,character,character_enemy):
    global next_move
    global max_value
    current_score_agent = scoreagent
    current_score_enemy = scoreenemy
    visited_diamond = {}
    visited_hole = {}
    max_value = float('-inf')
    next_move = tuple()
    max_depth_new=max_depth

    def alph_beta_minmax(is_max_turn, agentx, agenty, enemyx, enemyy, alpha, beta, level, remain_turn_agent, remain_turn_enemy, score_agent, score_enemy,diccolor_number_copy_agent,diccolor_number_copy_enemy):
        # global current_score_agent
        # global current_score_enemy
        global max_value
        global next_move
        global max_depth_new
        hash_state=hash_key(visited_diamond,visited_hole,agentx,agenty,enemyx,enemyy,remain_turn_agent,remain_turn_enemy,score_agent,score_enemy)
        #transposition value (exact,lowerbound,upperbound,depth)
        #transpositiontable[hash_state][3] >= level or transpositiontable[hash_state][3] <= level because position of  add value
        if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] <= level):
            if transpositiontable[hash_state][0] != float('inf'):
                return transpositiontable[hash_state][0]
            if (transpositiontable[hash_state][1] != float('inf')) and (is_max_turn) :
                if transpositiontable[hash_state][1] <= beta:
                    return  transpositiontable[hash_state][1]
            if (transpositiontable[hash_state][2] != float('inf')) and (not is_max_turn):
                if transpositiontable[hash_state][2] >= alpha:
                    return  transpositiontable[hash_state][2]


        if level == depth:
            print("im in depth == level")
            if len(visited_diamond) == 0:
                value = ((20 * ((score_agent - current_score_agent)-(score_enemy-current_score_enemy))) + ((remain_turn_agent - remain_turn_enemy) * 80)) // 100
            else:
                value = (((20 * ((score_agent - current_score_agent)-(score_enemy-current_score_enemy))) + ((80 * (remain_turn_agent - remain_turn_enemy)))) // 100)
           # print(value,max_value,"value, max_value")
            if value > max_value:
                max_value = value
                for keyvisited, valuevisited in visited_diamond.items():
                    print(keyvisited,valuevisited,"keyvisited , valuevisited")
                    if valuevisited[1] == 0:
                        next_move = keyvisited
                        print("next move", next_move)
                for keyvisited, valuevisited in visited_hole.items():
                    if valuevisited[1] == 0:
                        next_move = (keyvisited[0], keyvisited[1])
                        print("next move", next_move)
            return value

        if (remain_turn_agent == 0 and is_max_turn) or (remain_turn_enemy==0 and (not is_max_turn)):
            print("im in remain turn")
            if len(visited_diamond) == 0:
                value=(((20 * ((score_agent - current_score_agent) - (score_enemy - current_score_enemy))) + (
                (80 * (remain_turn_agent - remain_turn_enemy)))) // 100)
                # value = ((20 * ((score_agent - current_score_agent) - (score_enemy - current_score_enemy))) + (remain_turn_agent - remain_turn_enemy) * 80) // 100
            else:
                value = (((20 * ((score_agent - current_score_agent) - (score_enemy - current_score_enemy))) + (
                    (80 * (remain_turn_agent - remain_turn_enemy)))) // 100)
                # value = (((20 * ((score_agent - current_score_agent)-(score_enemy-current_score_enemy))) + (80 * (remain_turn_agent-remain_turn_enemy))) // 100)
            if value > max_value:
                max_value = value
                for keyvisited, valuevisited in visited_diamond.items():
                    if valuevisited[1] == 0:
                        next_move = keyvisited
                        print("next move", next_move)
                for keyvisited, valuevisited in visited_hole.items():
                    if valuevisited[1] == 0:
                        next_move = (keyvisited[0], keyvisited[1])
                        print("next move", next_move)
            return value

        if len(diamonddic) == len(visited_diamond):
            print("im in len diamond visited diamond")
            value = (((20 * ((score_agent - current_score_agent) - (score_enemy - current_score_enemy))) + (
                (80 * (remain_turn_agent - remain_turn_enemy)))) // 100)
            # value = (((20 * ((score_agent - current_score_agent)-(score_enemy-current_score_enemy))) + (80 * (remain_turn_agent - remain_turn_enemy))) // 100)
            if value > max_value:
                max_value = value
                for keyvisited, valuevisited in visited_diamond.items():
                    if valuevisited[1] == 0:
                        next_move = keyvisited
                        print("next move", next_move)
                for keyvisited, valuevisited in visited_hole.items():
                    if valuevisited[1] == 0:
                        next_move = (keyvisited[0], keyvisited[1])
                        print("next move", next_move)

            return value
        #if dicdistance is empty

        best_value = float('-inf') if is_max_turn else float('inf')
        result_return = float('-inf') if is_max_turn else float('inf')
        print(is_max_turn,"is_max_turn")
        if is_max_turn:
            dicdistance = {}
            print(diamonddic,"diamond dic")
            for initial_diamond in diamonddic:
                goalx = initial_diamond[0]
                goaly = initial_diamond[1]
                manhatan_distance = (abs(goalx-agentx)+abs(goaly-agenty))
                if (initial_diamond[2] == 10) and (diccolor_number_copy_agent['y'] < 15):
                    tuple_distance = dijkstra(gridmap, height, width, agentx, agenty, goalx, goaly, score_agent,enemytraps, character, diccolor_number_copy_agent)
                    dicdistance[(goalx, goaly,10)] = tuple_distance
                if (initial_diamond[2] == 25) and (score_agent - manhatan_distance >= 15) and (diccolor_number_copy_agent['g'] < 8):
                    tuple_distance = dijkstra(gridmap, height, width, agentx, agenty, goalx, goaly, score_agent,enemytraps, character,diccolor_number_copy_agent)
                    dicdistance[(goalx, goaly,25)] = tuple_distance
                if (initial_diamond[2] == 35) and (score_agent - manhatan_distance >= 50) and (diccolor_number_copy_agent['r'] < 5):
                    tuple_distance = dijkstra(gridmap, height, width, agentx, agenty, goalx, goaly, score_agent,enemytraps, character,diccolor_number_copy_agent)
                    dicdistance[(goalx, goaly,35)] = tuple_distance
                if (initial_diamond[2] == 75) and (score_agent - manhatan_distance >= 140) and (diccolor_number_copy_agent['b'] < 4):
                    tuple_distance = dijkstra(gridmap, height, width, agentx, agenty, goalx, goaly, score_agent, enemytraps,character,diccolor_number_copy_agent)
                    dicdistance[(goalx, goaly,75)] = tuple_distance
            # print(dicdistance,"dicdistance")

            # print(diamonddic,"dic_diamond")
            # we dont pass diamonddic in sortmoves check if okey or not
            sort_diamond_list = sortmoves(dicdistance,remain_turn_agent)
            # print(sort_diamond_list,"sort_diamond_list")

            for diamond in sort_diamond_list:
                d = (diamond[0], diamond[1])
                calculatescore = dicdistance[diamond][1]
                calculatedistance = dicdistance[diamond][0]

                if d not in visited_diamond:
                    visited_diamond[d] = (True, level)
                    print(diccolor_number_copy_agent, "dic color number")
                    print(calculatedistance,"calculatedistance")

                    print(calculatescore,"calculatescore")
                    print(remain_turn_agent,"remain_turn_agent")
                    print(level,"level")
                    print(depth,"depth")
                    if (calculatedistance <= remain_turn_agent) and (level + 1 <= depth):
                        # print("im in if ")
                        if (diamond[2] == 10) and (diccolor_number_copy_agent['y'] < 15):
                            print("im in y diamond",diamond[0],diamond[1])
                            diccolor_number_copy_agent['y'] += 1
                            result_return = alph_beta_minmax(not is_max_turn, diamond[0],diamond[1], enemyx, enemyy, alpha, beta, level+1, remain_turn_agent-calculatedistance,remain_turn_enemy, 10+calculatescore , score_enemy,diccolor_number_copy_agent,diccolor_number_copy_enemy)
                            diccolor_number_copy_agent['y'] -= 1

                        if (diamond[2] == 25) and (score_agent >= 15) and (diccolor_number_copy_agent['g'] < 8) and (
                                calculatescore >= 15):
                            print("im in g diamond")
                            diccolor_number_copy_agent['g'] += 1
                            result_return = alph_beta_minmax(not is_max_turn, diamond[0], diamond[1], enemyx, enemyy, alpha, beta, level+1, remain_turn_agent-calculatedistance,remain_turn_enemy, 25+calculatescore , score_enemy,diccolor_number_copy_agent,diccolor_number_copy_enemy)
                            diccolor_number_copy_agent['g'] -= 1

                        if (diamond[2] == 35) and (score_agent >= 50) and (diccolor_number_copy_agent['r'] < 5) and (
                                calculatescore >= 50):
                            print("im in r diamond")
                            diccolor_number_copy_agent['r'] += 1
                            result_return = alph_beta_minmax(not is_max_turn, diamond[0], diamond[1], enemyx, enemyy, alpha, beta, level+1, remain_turn_agent-calculatedistance,remain_turn_enemy, 35+calculatescore , score_enemy,diccolor_number_copy_agent,diccolor_number_copy_enemy)
                            diccolor_number_copy_agent['r'] -= 1

                        if (diamond[2] == 75) and (score_agent >= 140) and (diccolor_number_copy_agent['b'] < 4) and (
                                calculatescore >= 140):
                            print("im in b diamond")
                            diccolor_number_copy_agent['b'] += 1
                            result_return = alph_beta_minmax(not is_max_turn, diamond[0], diamond[1], enemyx, enemyy, alpha, beta, level+1, remain_turn_agent-calculatedistance,remain_turn_enemy, 75+calculatescore , score_enemy,diccolor_number_copy_agent,diccolor_number_copy_enemy)
                            diccolor_number_copy_agent['b'] -= 1

                    best_value = max(best_value, result_return)
                    alpha = max(alpha, best_value)
                    if beta <= alpha:
                        if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] > level):
                            max_depth_new=max(max_depth_new,level)
                            transpositiontable[hash_state] = (transpositiontable[hash_state][0],best_value , transpositiontable[hash_state][2], level)
                        else:
                            max_depth_new = max(max_depth_new, level)
                            if len(transpositiontable) < transpositionsize:
                                storedHkeys.put(hash_state)
                                transpositiontable[hash_state] = (float('inf'), best_value, float('inf'), level)
                            else:
                                storedHkeys.get()
                                storedHkeys.put(hash_state)
                                transpositiontable[hash_state] = (float('inf'), best_value, float('inf'), level)
                        return best_value
                    visited_diamond.pop(d, None)
            dicdistancehole={}
            for initial_hole in holedic:
                goalx=initial_hole[0]
                goaly=initial_hole[1]
                tuple_distance = dijkstra(gridmap, height, width, agentx, agenty, goalx, goaly, score_agent, enemytraps,character,diccolor_number_copy_agent)
                dicdistancehole[(goalx, goaly,0)] = tuple_distance

            sort_hole_list = sortmoves(dicdistancehole,remain_turn_agent)
            for hole in sort_hole_list:
                h = (hole[0], hole[1])

                calculatescore = dicdistancehole[hole][1]
                calculatedistance = dicdistancehole[hole][0]
                visited_hole[(h[0], h[1], level)] = (True, level)
                current_hole = (h[0], h[1], 0)

                # distancehole = dijkstra(gridmap, height, width, agentx, agenty, h[0], h[1], score_agent)

                if (calculatedistance <= remain_turn_agent) and (level + 1 <= depth):
                    value_hole = 0
                    for item_hole in holedic:
                        if item_hole != current_hole:
                            value_hole += alph_beta_minmax(not is_max_turn, item_hole[0], item_hole[1], enemyx, enemyy, alpha, beta, level+1, remain_turn_agent-calculatedistance-1,remain_turn_enemy, calculatescore-1, score_enemy,diccolor_number_copy_agent,diccolor_number_copy_enemy)

                    result_return = (value_hole // (len(holedic) - 1))

                best_value = max(best_value, result_return)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] > level):
                        max_depth_new = max(max_depth_new, level)
                        transpositiontable[hash_state] = (transpositiontable[hash_state][0], best_value, transpositiontable[hash_state][2], level)
                    else:
                        max_depth_new = max(max_depth_new, level)
                        if len(transpositiontable) < transpositionsize:
                            storedHkeys.put(hash_state)
                            transpositiontable[hash_state] = (float('inf'), best_value, float('inf'), level)
                        else:
                            storedHkeys.get()
                            storedHkeys.put(hash_state)
                            transpositiontable[hash_state] = (float('inf'), best_value, float('inf'), level)
                    return best_value
                visited_hole.pop((h[0], h[1], level), None)

            if (result_return == float('-inf')) and (level + 1 <= depth):
                result_return = alph_beta_minmax(not is_max_turn, agentx, agenty, enemyx, enemyy, alpha, beta,
                                                 level + 1, remain_turn_agent, remain_turn_enemy, score_agent,
                                                 score_enemy, diccolor_number_copy_agent, diccolor_number_copy_enemy)
                best_value = max(best_value, result_return)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] > level):
                        max_depth_new = max(max_depth_new, level)
                        transpositiontable[hash_state] = (
                        transpositiontable[hash_state][0], best_value, transpositiontable[hash_state][2], level)
                    else:
                        max_depth_new = max(max_depth_new, level)
                        if len(transpositiontable) < transpositionsize:
                            storedHkeys.put(hash_state)
                            transpositiontable[hash_state] = (float('inf'), best_value, float('inf'), level)
                        else:
                            storedHkeys.get()
                            storedHkeys.put(hash_state)
                            transpositiontable[hash_state] = (float('inf'), best_value, float('inf'), level)
                    return best_value

        else:
            print("im in enemy")
            dicdistanceenemy = {}
            for initial_diamond in diamonddic:
                goalx = initial_diamond[0]
                goaly = initial_diamond[1]
                manhatan_distance = (abs(goalx - agentx) + abs(goaly - agenty))
                if (initial_diamond[2] == 10) and (diccolor_number_copy_agent['y'] < 15):
                    tuple_distance = dijkstra(gridmap, height, width, enemyx, enemyy, goalx, goaly, score_enemy, agenttraps, character_enemy,diccolor_number_copy_enemy)
                    dicdistanceenemy[(goalx, goaly,10)] = tuple_distance
                if (initial_diamond[2] == 25) and (score_agent - manhatan_distance >= 15) and (
                        diccolor_number_copy_agent['g'] < 8):
                    tuple_distance = dijkstra(gridmap, height, width, enemyx, enemyy, goalx, goaly, score_enemy,agenttraps, character_enemy,diccolor_number_copy_enemy)
                    dicdistanceenemy[(goalx, goaly,25)] = tuple_distance
                if (initial_diamond[2] == 35) and (score_agent - manhatan_distance >= 50) and (
                        diccolor_number_copy_agent['r'] < 5):
                    tuple_distance = dijkstra(gridmap, height, width, enemyx, enemyy, goalx, goaly, score_enemy, agenttraps, character_enemy,diccolor_number_copy_enemy)
                    dicdistanceenemy[(goalx, goaly,35)] = tuple_distance
                if (initial_diamond[2] == 75) and (score_agent - manhatan_distance >= 140) and (
                        diccolor_number_copy_agent['b'] < 4):
                    tuple_distance = dijkstra(gridmap, height, width, enemyx, enemyy, goalx, goaly, score_enemy, agenttraps, character_enemy,diccolor_number_copy_enemy)
                    dicdistanceenemy[(goalx, goaly,75)] = tuple_distance

            sort_diamond_list = sortmoves(dicdistanceenemy, remain_turn_enemy)
            print(sort_diamond_list,"sort_diamond_list")
            print(dicdistanceenemy,"dicdistanceenemy")
            for diamond in sort_diamond_list:
                d = (diamond[0], diamond[1])
                calculatescore = dicdistanceenemy[diamond][1]
                calculatedistance = dicdistanceenemy[diamond][0]
                if d not in visited_diamond:
                    visited_diamond[d] = (True, level)
                    print(diccolor_number_copy_agent, "dic color number")
                    print(calculatedistance, "calculatedistance")
                    print(calculatescore, "calculatescore")
                    print(remain_turn_agent, "remain_turn_agent")
                    print(level, "level")
                    print(depth, "depth")
                    if (calculatedistance <= remain_turn_enemy) and (level + 1 <= depth):
                        if (diamond[2] == 10) and (diccolor_number_copy_enemy['y'] < 15):
                            print("im in enemy y")
                            diccolor_number_copy_enemy['y'] += 1
                            result_return = alph_beta_minmax(not is_max_turn, agentx, agenty, diamond[0], diamond[1],
                                                             alpha, beta, level + 1, remain_turn_agent,
                                                             remain_turn_enemy-calculatedistance, score_agent,
                                                             10 +calculatescore, diccolor_number_copy_agent,
                                                             diccolor_number_copy_enemy)
                            diccolor_number_copy_enemy['y'] -= 1

                        if (diamond[2] == 25) and (score_agent >= 15) and (diccolor_number_copy_enemy['g'] < 8) and (
                                calculatescore >= 15):
                            print("im in enemy g")
                            diccolor_number_copy_enemy['g'] += 1
                            result_return = alph_beta_minmax(not is_max_turn, agentx, agenty, diamond[0], diamond[1],
                                                             alpha, beta, level + 1, remain_turn_agent,
                                                             remain_turn_enemy - calculatedistance, score_agent ,
                                                              25 + calculatescore, diccolor_number_copy_agent,
                                                             diccolor_number_copy_enemy)
                            diccolor_number_copy_enemy['g'] -= 1

                        if (diamond[2] == 35) and (score_agent >= 50) and (diccolor_number_copy_enemy['r'] < 5) and (
                                calculatescore >= 50):
                            print("im in enemy r")
                            diccolor_number_copy_enemy['r'] += 1
                            result_return = alph_beta_minmax(not is_max_turn,agentx, agenty, diamond[0], diamond[1],
                                                             alpha, beta, level + 1, remain_turn_agent,
                                                             remain_turn_enemy - calculatedistance, score_agent,
                                                             35 + calculatescore, diccolor_number_copy_agent,
                                                             diccolor_number_copy_enemy)
                            diccolor_number_copy_enemy['r'] -= 1

                        if (diamond[2] == 75) and (score_agent >= 140) and (diccolor_number_copy_enemy['b'] < 4) and (
                                calculatescore >= 140):
                            print("im in enemy b")
                            diccolor_number_copy_enemy['b'] += 1
                            result_return = alph_beta_minmax(not is_max_turn,agentx, agenty, diamond[0], diamond[1],
                                                             alpha, beta, level + 1, remain_turn_agent,
                                                             remain_turn_enemy- calculatedistance, score_agent,
                                                             75 + calculatescore, diccolor_number_copy_agent,
                                                             diccolor_number_copy_enemy)
                            diccolor_number_copy_enemy['b'] -= 1

                    best_value = min(best_value, result_return)
                    beta = min(beta, best_value)
                    if beta <= alpha:
                        if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] > level):
                            max_depth_new = max(max_depth_new, level)
                            transpositiontable[hash_state] = (transpositiontable[hash_state][0], transpositiontable[hash_state][1],best_value , level)
                        else:
                            max_depth_new = max(max_depth_new, level)
                            if len(transpositiontable) < transpositionsize:
                                storedHkeys.put(hash_state)
                                transpositiontable[hash_state] = (float('inf'), float('inf'), best_value, level)
                            else:
                                storedHkeys.get()
                                storedHkeys.put(hash_state)
                                transpositiontable[hash_state] = (float('inf'),float('inf'), best_value, level)

                        return best_value
                    visited_diamond.pop(d, None)


            dicdistanceenemyhole = {}
            for initial_hole in holedic:
                goalx = initial_hole[0]
                goaly = initial_hole[1]
                tuple_distance = dijkstra(gridmap, height, width, enemyx, enemyy, goalx, goaly, score_enemy, agenttraps,character_enemy,diccolor_number_copy_enemy)
                dicdistanceenemyhole[(goalx, goaly,0)] = tuple_distance

            sort_hole_list = sortmoves(dicdistanceenemyhole,remain_turn_enemy)
            for hole in sort_hole_list:
                h = (hole[0], hole[1])
                calculatescore = dicdistanceenemyhole[hole][1]
                calculatedistance = dicdistanceenemyhole[hole][0]
                visited_hole[(h[0], h[1], level)] = (True, level)
                current_hole = (h[0], h[1], 0)
                # distancehole = dijkstra(gridmap, height, width, enemyx, enemyy, h[0], h[1], score_enemy)

                if (calculatedistance <= remain_turn_enemy) and (level + 1 <= depth):
                    value_hole = 0
                    for item_hole in holedic:
                        if item_hole != current_hole:
                            value_hole += alph_beta_minmax(not is_max_turn, agentx, agenty, item_hole[0], item_hole[1],
                                                           alpha, beta, level + 1, remain_turn_agent,
                                                           remain_turn_enemy - calculatedistance - 1, score_agent,
                                                           calculatescore- 1, diccolor_number_copy_agent,
                                                           diccolor_number_copy_enemy)

                    result_return = (value_hole // (len(holedic) - 1))

                best_value = min(best_value, result_return)
                beta = min(beta, best_value)
                if beta <= alpha:
                    if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] > level):
                        max_depth_new = max(max_depth_new, level)
                        transpositiontable[hash_state] = (transpositiontable[hash_state][0], transpositiontable[hash_state][1], best_value, level)
                    else:
                        max_depth_new = max(max_depth_new, level)
                        if len(transpositiontable) < transpositionsize:
                            storedHkeys.put(hash_state)
                            transpositiontable[hash_state] = (float('inf'), float('inf'), best_value, level)
                        else:
                            storedHkeys.get()
                            storedHkeys.put(hash_state)
                            transpositiontable[hash_state] = (float('inf'), float('inf'), best_value, level)
                    return best_value
                visited_hole.pop((h[0], h[1], level), None)
            print(result_return,"result rreturn")
            print(level,"level")
            print(depth,"depth")
            if (result_return == float('inf')) and (level+1 <= depth):
                 print("im in (result_return == -1) and (level+1 <= depth)")
                 result_return = alph_beta_minmax(not is_max_turn, agentx, agenty, enemyx, enemyy,alpha, beta, level + 1, remain_turn_agent,remain_turn_enemy, score_agent, score_enemy, diccolor_number_copy_agent, diccolor_number_copy_enemy)
                 best_value = min(best_value, result_return)
                 beta = min(beta, best_value)
                 if beta <= alpha:
                     if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] > level):
                         max_depth_new = max(max_depth_new, level)
                         transpositiontable[hash_state] = (
                         transpositiontable[hash_state][0], transpositiontable[hash_state][1], best_value, level)
                     else:
                         max_depth_new = max(max_depth_new, level)
                         if len(transpositiontable) < transpositionsize:
                             storedHkeys.put(hash_state)
                             transpositiontable[hash_state] = (float('inf'), float('inf'), best_value, level)
                         else:
                             storedHkeys.get()
                             storedHkeys.put(hash_state)
                             transpositiontable[hash_state] = (float('inf'), float('inf'), best_value, level)
                     return best_value

        #check size for this transposition for update
        if (hash_state in transpositiontable) and (transpositiontable[hash_state][3] > level):
            max_depth_new = max(max_depth_new, level)
            transpositiontable[hash_state]=(best_value,transpositiontable[hash_state][1],transpositiontable[hash_state][2],level)
        else:
            max_depth_new = max(max_depth_new, level)
            if len(transpositiontable) < transpositionsize:
                storedHkeys.put(hash_state)
                transpositiontable[hash_state] = (best_value,float('inf'), float('inf'),  level)
            else:
                storedHkeys.get()
                storedHkeys.put(hash_state)
                transpositiontable[hash_state] = (best_value,float('inf'), float('inf'), level)
        return best_value


    diccolor_number_copy_agent = diccolornumberagent.copy()
    diccolor_number_copy_enemy=diccolornumberenemy.copy()
    alph_beta_minmax(True, agentx, agenty, enemyx, enemyy, float('-inf'), float('inf'), 0, turn_agent,turn_enemy, scoreagent, scoreenemy, diccolor_number_copy_agent,diccolor_number_copy_enemy)
    print(next_move,"next_moveminmax")
    return next_move,max_depth_new
