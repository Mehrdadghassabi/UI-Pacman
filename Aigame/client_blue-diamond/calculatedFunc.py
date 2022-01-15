from json import dumps
import numpy as np


def hash_key(visiteddiamond, visitedhole, agentx, agenty, enemyx, enemyy, remain_turn_agent, remain_turn_enemy, score_agent, score_enemy):
    hash_visiteddiamond = dumps({dumps(i): visiteddiamond[i] for i in sorted(visiteddiamond)})
    hash_visitedhole = dumps({dumps(i): visitedhole[i] for i in sorted(visitedhole)})
    hash_key_list = [hash_visiteddiamond, hash_visitedhole, agentx, agenty, enemyx, enemyy, remain_turn_agent, remain_turn_enemy, score_agent, score_enemy]
    return tuple(hash_key_list)


def sortmoves(dijkstradic,remain_turn, diccolor_number):

    scores = {}
    move_list = []
    # maybe sort all values are better
    dijkstradic_copy = dijkstradic.copy()

    for item in dijkstradic:
        #also we can use manhatani distance in if
        s = item[2]
        if ((s == 10) and (diccolor_number['y'] < 15)) or ((s == 25)  and ((dijkstradic[item][1] >= 15) and (diccolor_number['g'] < 8))) or (
                (s == 35) and ((dijkstradic[item][1] >= 50) and (diccolor_number['r'] < 5))) or ((s== 75) and ((dijkstradic[item][1] >= 140) and (diccolor_number['b'] < 4))) or s == 0:
           value = (20 * (dijkstradic[item][1]+item[2]) + 80 * (remain_turn - dijkstradic[item][0])) // 100
           scores[item] = value
    #     print(item,value,"scorevale")
    # print(len(scores),"sizescores")
    for i in range(min(6, len(scores))):
         max = float('-inf')
         maxlocation = tuple()
         for j in scores:
             if scores[j] > max:
                 #print(j, " j ", scores[j], " scores j")
                 max = scores[j]
                 maxlocation = j

         move_list.append(maxlocation)
         dijkstradic_copy.pop(maxlocation)
        # print(maxlocation, " max_location")
         scores[maxlocation] = float('-inf')
    # print(dijkstradic,"dijkstradic")
    move_list.extend(list(dijkstradic_copy.keys()))
    return move_list




