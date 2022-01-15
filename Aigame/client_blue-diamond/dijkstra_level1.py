from queue import PriorityQueue
from math import inf


def dijkstra(gridmap, height, width, agentx, agenty, goalx, goaly, scoredij,score_enemy,trap, character,character_enemy, diccolornumber):
        visited = {}
        distancelist = {}
        pq = PriorityQueue()
        pq.put((0, agentx, agenty, 0, scoredij))
        distancelist[(agentx, agenty)] = 0
        flag = False
        flag_hit=False
        while not pq.empty():
            temp = pq.get()
            dist = temp[0]
            current_nodex = temp[1]
            current_nodey = temp[2]
            scoredij = temp[4]
            actual_dist = temp[3]
            visited[(current_nodex, current_nodey)] = True
            # print(current_nodex , " " , current_nodey,"current node")
            # print(dist," ", actual_dist , "dist and actual")
            # print(scoredij, " scoredij")
            if (current_nodex == goalx) and (current_nodey == goaly):
                return actual_dist, scoredij

            # up
            flag = False
            flag_hit = False
            if current_nodex -1 >=0 and (gridmap[current_nodex-1][current_nodey] == 'E'+character_enemy or gridmap[current_nodex-1][current_nodey] == 'T'+character_enemy):
                flag_hit=True
            if current_nodex - 1 >= 0:
                if (current_nodex - 1, current_nodey) in trap:
                    flag = True
            if (current_nodex - 1 >= 0) and ((current_nodex - 1, current_nodey) not in visited) and (
                    (((current_nodex-1) == goalx) and (current_nodey == goaly)) or
                    (gridmap[current_nodex - 1][current_nodey] == 'E') or (
                    gridmap[current_nodex - 1][current_nodey] == 'T') or (
                    gridmap[current_nodex - 1][current_nodey] == 'E' + character) or (
                    gridmap[current_nodex - 1][current_nodey] == 'T' + character) or (
                    gridmap[current_nodex - 1][current_nodey] == '1' and diccolornumber['y'] == 15) or (
                    gridmap[current_nodex - 1][current_nodey] == '2' and (scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                    gridmap[current_nodex - 1][current_nodey] == '3' and (scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                    gridmap[current_nodex - 1][current_nodey] == '4' and (scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):

                if (current_nodex - 1, current_nodey) not in distancelist:
                    if flag:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 41
                        pq.put((dist + 41, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                        if scoredij - 1 < score_enemy:
                            distancelist[(current_nodex - 1, current_nodey)] = dist + 21
                            pq.put((dist + 21, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 21))
                        elif scoredij - 1 >= score_enemy:
                            distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                            pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                    else:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))

                else:
                    if flag:
                        if dist + 41 < distancelist[(current_nodex - 1, current_nodey)]:
                            distancelist[(current_nodex - 1, current_nodey)] = dist + 41
                            pq.put((dist + 41, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                            if scoredij - 1 < score_enemy:
                                if dist + 21 < distancelist[(current_nodex - 1, current_nodey)]:
                                    distancelist[(current_nodex - 1, current_nodey)] = dist + 21
                                    pq.put((dist + 21, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 21))
                            elif scoredij - 1 >= score_enemy:
                                if dist + 1 < distancelist[(current_nodex - 1, current_nodey)]:
                                    distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                                    pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                    else:
                        if dist + 1 < distancelist[(current_nodex - 1, current_nodey)]:
                            distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                            pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
            # down
            flag = False
            flag_hit = False

            if current_nodex + 1 < height and (gridmap[current_nodex + 1][current_nodey] == 'E' + character_enemy or gridmap[current_nodex + 1][current_nodey] == 'T' + character_enemy):
                flag_hit = True
            if current_nodex + 1 < height:
                if (current_nodex + 1, current_nodey) in trap:
                    flag = True
            if (current_nodex + 1 < height) and ((current_nodex + 1, current_nodey) not in visited) and (
                    (((current_nodex + 1) == goalx) and (current_nodey == goaly)) or
                    (gridmap[current_nodex + 1][current_nodey] == 'E') or (
                    gridmap[current_nodex + 1][current_nodey] == 'T') or (
                    gridmap[current_nodex + 1][current_nodey] == "E" + character) or (
                    gridmap[current_nodex + 1][current_nodey] == "T" + character) or (
                    gridmap[current_nodex + 1][current_nodey] == '1' and diccolornumber['y'] == 15) or (
                    gridmap[current_nodex + 1][current_nodey] == '2' and (scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                    gridmap[current_nodex + 1][current_nodey] == '3' and (scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                    gridmap[current_nodex + 1][current_nodey] == '4' and (scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):

                if (current_nodex + 1, current_nodey) not in distancelist:
                    if flag:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 41
                        pq.put((dist + 41, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                        if scoredij - 1 < score_enemy:
                            distancelist[(current_nodex + 1, current_nodey)] = dist + 21
                            pq.put((dist + 21, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 21))
                        elif scoredij - 1 >= score_enemy:
                            distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                            pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                    else:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))

                else:
                    if flag:
                        if dist + 41 < distancelist[(current_nodex + 1, current_nodey)]:
                            distancelist[(current_nodex + 1, current_nodey)] = dist + 41
                            pq.put((dist + 41, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                            if scoredij - 1 < score_enemy:
                                if dist + 21 < distancelist[(current_nodex + 1, current_nodey)]:
                                    distancelist[(current_nodex + 1, current_nodey)] = dist + 21
                                    pq.put(
                                        (dist + 21, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 21))
                            elif scoredij - 1 >= score_enemy:
                                if dist + 1 < distancelist[(current_nodex + 1, current_nodey)]:
                                    distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                                    pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                    else:
                        if dist + 1 < distancelist[(current_nodex + 1, current_nodey)]:
                            distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                            pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
            # left
            flag = False
            flag_hit = False
            if current_nodey - 1 >= 0 and (gridmap[current_nodex][current_nodey-1] == 'E' + character_enemy  or gridmap[current_nodex][current_nodey-1] == 'T' + character_enemy):
                flag_hit = True
            if current_nodey - 1 >= 0:
                if (current_nodex, current_nodey - 1) in trap:
                    flag = True
            if (current_nodey - 1 >= 0) and ((current_nodex, current_nodey - 1) not in visited) and (
                    ((current_nodex == goalx) and ((current_nodey -1) == goaly)) or
                    (gridmap[current_nodex][current_nodey - 1] == 'E') or (
                    gridmap[current_nodex][current_nodey - 1] == 'T') or (
                    gridmap[current_nodex][current_nodey - 1] == "E" + character) or (
                    gridmap[current_nodex][current_nodey - 1] == "T" + character) or (
                    gridmap[current_nodex][current_nodey - 1] == '1' and diccolornumber['y'] == 15) or (
                    gridmap[current_nodex][current_nodey - 1] == '2' and (scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                    gridmap[current_nodex][current_nodey-1] == '3' and (scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                    gridmap[current_nodex][current_nodey-1] == '4' and (scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):
                if (current_nodex, current_nodey - 1) not in distancelist:
                    if flag:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 41
                        pq.put((dist + 41, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                        # print("dist",actual_dist,"im in trapi hit")
                        if scoredij - 1 < score_enemy:
                            distancelist[(current_nodex, current_nodey - 1)] = dist + 21
                            pq.put((dist + 21, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 21))
                        elif scoredij - 1 >= score_enemy:
                            distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                            pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                    else:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                        pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))

                else:
                    if flag:
                        if dist + 41 < distancelist[(current_nodex, current_nodey - 1)]:
                            distancelist[(current_nodex, current_nodey - 1)] = dist + 41
                            pq.put((dist + 41, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                        if scoredij - 1 < score_enemy:
                            if dist + 21 < distancelist[(current_nodex, current_nodey - 1)]:
                                distancelist[(current_nodex, current_nodey - 1)] = dist + 21
                                pq.put((dist + 21, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 21))
                        elif scoredij - 1 >= score_enemy:
                            if dist + 1 < distancelist[(current_nodex, current_nodey - 1)]:
                                distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                                pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                    else:
                        if dist + 1 < distancelist[(current_nodex, current_nodey - 1)]:
                            distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                            pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
            # right
            flag = False
            flag_hit = False
            if current_nodey + 1 < width and (gridmap[current_nodex][current_nodey + 1] == 'E' + character_enemy or gridmap[current_nodex][current_nodey + 1] == 'T' + character_enemy):
                flag_hit = True
            if current_nodey + 1 < width:
                if (current_nodex, current_nodey + 1) in trap:
                    flag = True
            if (current_nodey + 1 < width) and ((current_nodex, current_nodey + 1) not in visited) and (
                    ((current_nodex == goalx) and ((current_nodey+1) == goaly)) or
                    (gridmap[current_nodex][current_nodey + 1] == 'E') or (
                    gridmap[current_nodex][current_nodey + 1] == 'T') or (
                    gridmap[current_nodex][current_nodey + 1] == "E" + character) or (
                    gridmap[current_nodex][current_nodey + 1] == "T" + character) or (
                    gridmap[current_nodex][current_nodey + 1] == '1' and diccolornumber['y'] == 15) or (
                    gridmap[current_nodex][current_nodey + 1] == '2' and (scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                    gridmap[current_nodex][current_nodey + 1] == '3' and (scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                    gridmap[current_nodex][current_nodey + 1] == '4' and (scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):

                if (current_nodex, current_nodey + 1) not in distancelist:
                    if flag:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 41
                        pq.put((dist + 41, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                        if scoredij - 1 < score_enemy:
                            distancelist[(current_nodex, current_nodey + 1)] = dist + 21
                            pq.put((dist + 21, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 21))
                        elif scoredij - 1 >= score_enemy:
                            distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                            pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                    else:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                        pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))

                else:
                    if flag:
                        if dist + 41 < distancelist[(current_nodex, current_nodey + 1)]:
                            distancelist[(current_nodex, current_nodey + 1)] = dist + 41
                            pq.put((dist + 41, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 41))
                    elif flag_hit:
                            if scoredij - 1 < score_enemy:
                                if dist + 21 < distancelist[(current_nodex, current_nodey + 1)]:
                                    distancelist[(current_nodex, current_nodey + 1)] = dist + 21
                                    pq.put(
                                        (dist + 21, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 21))
                            elif scoredij - 1 >= score_enemy:
                                if dist + 1 < distancelist[(current_nodex, current_nodey + 1)]:
                                    distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                                    pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                    else:
                        if dist + 1 < distancelist[(current_nodex, current_nodey + 1)]:
                            distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                            pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))

            score_enemy -= 1

        return float('inf'),float('-inf')