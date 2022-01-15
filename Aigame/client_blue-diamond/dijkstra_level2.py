from queue import PriorityQueue,  LifoQueue


def dij_show_way(agentx, agenty, goalx, goaly, gridmap, height, width,scoredij,character):
        visited = {}
        distancelist = {}
        parent = {}
        way = LifoQueue()
        parent[(agentx, agenty)] = (-1,-1,'')
        pq = PriorityQueue()
        pq.put((0, (agentx, agenty)))
        distancelist[(agentx, agenty)] = 0
        #think about trap also
        if agentx == goalx and agenty == goaly:
            way.put('t')
            return way

        while not pq.empty():
            temp = pq.get()
            dist = temp[0]
            current_nodex = temp[1][0]
            current_nodey = temp[1][1]
            visited[(current_nodex, current_nodey)] = True
            if gridmap[agentx][agenty] == 'T'+character and gridmap[goalx][goaly] == 'T':
                way.put('t')
                return way
            if current_nodex == goalx and current_nodey == goaly:
                x = current_nodex
                y = current_nodey
                if gridmap[x][y] == 'T':
                    way.put('t')
                while ((parent[(x, y)][0] != -1) and (parent[(x, y)][1] != -1) and (parent[(x, y)][2] != '')):
                    way.put(parent[(x, y)][2])
                    currentx = parent[(x, y)][0]
                    currenty = parent[(x, y)][1]
                    x = currentx
                    y = currenty
                return way
            # up
            if (current_nodex - 1 >= 0) and ((current_nodex - 1, current_nodey) not in visited) and (
               (gridmap[current_nodex - 1][current_nodey] == 'E') or (gridmap[current_nodex - 1][current_nodey] == 'T')or(gridmap[current_nodex-1][current_nodey]=="E"+character)  or (gridmap[current_nodex-1][current_nodey]=="E"+character) or (gridmap[current_nodex-1][current_nodey]=="T"+character)or(
               (current_nodex - 1 == goalx) and (current_nodey == goaly)) or (gridmap[current_nodex-1][current_nodey] == '1' and scoredij-1 < 0) or (
              gridmap[current_nodex-1][current_nodey] == '2' and scoredij-1 < 15) or (gridmap[current_nodex-1][current_nodey] == '3' and scoredij-1 < 50 ) or (
              gridmap[current_nodex-1][current_nodey] == '4' and scoredij-1 < 140)):

                if (current_nodex - 1, current_nodey) not in distancelist:
                    distancelist[(current_nodex - 1, current_nodey)]=dist+1
                    pq.put((dist + 1, (current_nodex - 1, current_nodey)))
                    parent[(current_nodex-1, current_nodey)] = (current_nodex, current_nodey, 'u')
                else:
                    if dist + 1 < distancelist[(current_nodex - 1, current_nodey)]:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, (current_nodex - 1, current_nodey)))
                        parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')
            # down
            if (current_nodex + 1 < height) and ((current_nodex + 1, current_nodey) not in visited) and (

               (gridmap[current_nodex + 1][current_nodey] == 'E') or (gridmap[current_nodex + 1][current_nodey] == 'T')or(gridmap[current_nodex+1][current_nodey]=="E"+character) or (gridmap[current_nodex+1][current_nodey]=="T"+character)or(
               (current_nodex + 1 == goalx) and (current_nodey == goaly))or (gridmap[current_nodex+1][current_nodey] == '1' and scoredij-1 < 0) or (
              gridmap[current_nodex+1][current_nodey] == '2' and scoredij-1 < 15) or (gridmap[current_nodex+1][current_nodey] == '3' and scoredij-1 < 50 ) or (
              gridmap[current_nodex+1][current_nodey] == '4' and scoredij-1 < 140)):

                if (current_nodex + 1, current_nodey) not in distancelist:
                    distancelist[(current_nodex + 1, current_nodey)]=dist+1
                    pq.put((dist + 1, (current_nodex + 1, current_nodey)))
                    parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
                else:
                    if dist + 1 < distancelist[(current_nodex + 1, current_nodey)]:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, (current_nodex + 1, current_nodey)))
                        parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
            # left
            if (current_nodey - 1 >= 0) and ((current_nodex, current_nodey - 1) not in visited) and (

               (gridmap[current_nodex][current_nodey - 1] == 'E') or (gridmap[current_nodex][current_nodey - 1] == 'T')or(gridmap[current_nodex][current_nodey-1]=="E"+character) or (gridmap[current_nodex][current_nodey-1]=="T"+character)or (
               (current_nodex == goalx) and (current_nodey-1 == goaly))or (gridmap[current_nodex][current_nodey-1] == '1' and scoredij-1 < 0) or (
              gridmap[current_nodex][current_nodey-1] == '2' and scoredij-1 < 15) or (gridmap[current_nodex][current_nodey-1] == '3' and scoredij-1 < 50 ) or (
              gridmap[current_nodex][current_nodey-1] == '4' and scoredij-1 < 140)):

                if (current_nodex, current_nodey - 1) not in distancelist:
                    distancelist[(current_nodex, current_nodey - 1)]=dist+1
                    pq.put((dist + 1, (current_nodex, current_nodey - 1)))
                    parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')
                else:
                    if dist + 1 < distancelist[(current_nodex, current_nodey - 1)]:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                        pq.put((dist + 1, (current_nodex, current_nodey-1)))
                        parent[(current_nodex, current_nodey-1)] = (current_nodex, current_nodey, 'l')

            # right
            if (current_nodey + 1 < width) and ((current_nodex, current_nodey + 1) not in visited) and (
               (gridmap[current_nodex][current_nodey + 1] == 'E') or (gridmap[current_nodex][current_nodey + 1] == 'T') or(gridmap[current_nodex][current_nodey+1]=="E"+character)or (gridmap[current_nodex][current_nodey+1]=="T"+character) or(
               (current_nodex == goalx) and (current_nodey+1 == goaly)) or (gridmap[current_nodex][current_nodey+1] == '1' and scoredij-1 < 0) or (
              gridmap[current_nodex][current_nodey+1] == '2' and scoredij-1 < 15) or (gridmap[current_nodex][current_nodey+1] == '3' and scoredij-1 < 50 ) or (
              gridmap[current_nodex][current_nodey +1] == '4' and scoredij-1 < 140)):
                if (current_nodex, current_nodey + 1) not in distancelist:
                    distancelist[(current_nodex, current_nodey + 1)]=dist+1
                    pq.put((dist + 1, (current_nodex, current_nodey + 1)))
                    parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
                else:
                    if dist + 1 < distancelist[(current_nodex, current_nodey + 1)]:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                        pq.put((dist + 1, (current_nodex, current_nodey + 1)))
                        parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')


            scoredij -= 1


def dij_show_action(agentx, agenty, goalx, goaly, gridmap, height, width, scoredij,score_enemy, trap, character,character_enemy, diccolornumber):
    # print(agentx, agenty , " im in enemy start dijkstra enemy way")
    # print(goalx, goaly, " im in goal enemy dijkstra enemy way")
    visited = {}
    distancelist = {}
    parent = {}
    way = LifoQueue()
    parent[(agentx, agenty)] = (-1, -1, '')
    pq = PriorityQueue()
    pq.put((0, agentx, agenty, 0, scoredij))
    distancelist[(agentx, agenty)] = 0
    #change for trap
    if agentx == goalx and agenty == goaly and (gridmap[agentx][agenty] == 'T' or gridmap[agentx][agenty] == 'T' + character):
        way.put('t')
        way_return = way.get()
        # print(way_return, "wayreturn")
        return way_return
    if gridmap[agentx][agenty] == 'T' + character and gridmap[goalx][goaly] == 'T':
        way.put('t')
        way_return = way.get()
        # print(way_return, "wayreturn")
        return way_return
    if agentx == goalx and agenty == goaly:
        way.put('p')
        way_return = way.get()
        # print("action p choose", "*" * 20)
        return way_return
    flag = False
    while not pq.empty():
        temp = pq.get()
        dist = temp[0]
        current_nodex = temp[1]
        current_nodey = temp[2]
        scoredij = temp[4]
        actual_dist = temp[3]
        visited[(current_nodex, current_nodey)] = True
        if current_nodex == goalx and current_nodey == goaly:
            # print("im in goal = current in dijkstra enemy way")
            x = current_nodex
            y = current_nodey
            if gridmap[x][y] == 'T':
                way.put('t')
            while (parent[(x, y)][0] != -1) and (parent[(x, y)][1] != -1) and (parent[(x, y)][2] != ''):
                way.put(parent[(x, y)][2])
                currentx = parent[(x, y)][0]
                currenty = parent[(x, y)][1]
                x = currentx
                y = currenty
            way_return = way.get()
            # while not way.empty():
            #    print(way.get(),"wayreturn")
            return way_return

        # up
        flag_hit = False
        if current_nodex - 1 >= 0 and (gridmap[current_nodex - 1][current_nodey] == 'E' + character_enemy or gridmap[current_nodex - 1][current_nodey] == 'T' + character_enemy):
            flag_hit = True
        flag = False
        if current_nodex - 1 >= 0:
            if (current_nodex - 1, current_nodey) in trap:
                flag = True
        if (current_nodex - 1 >= 0) and ((current_nodex - 1, current_nodey) not in visited) and (
                (gridmap[current_nodex - 1][current_nodey] == 'E') or (
                 gridmap[current_nodex - 1][current_nodey] == 'T') or (
                 gridmap[current_nodex - 1][current_nodey] == "E"+character) or (
                 gridmap[current_nodex - 1][current_nodey] == "T"+character) or (
                 (current_nodex - 1 == goalx) and (current_nodey == goaly)) or (
                  gridmap[current_nodex - 1][current_nodey] == '1' and diccolornumber['y'] == 15) or (
                  gridmap[current_nodex - 1][current_nodey] == '2' and (
                   scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                  gridmap[current_nodex - 1][current_nodey] == '3' and (
                        scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                  gridmap[current_nodex - 1][current_nodey] == '4' and (
                        scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):

            if (current_nodex - 1, current_nodey) not in distancelist:
                if flag:
                    distancelist[(current_nodex - 1, current_nodey)] = dist + 41
                    pq.put((dist + 41, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 41))
                    parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 21
                        pq.put((dist + 21, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 21))
                        parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')
                    elif scoredij - 1 >= score_enemy:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 1/2
                        pq.put((dist + 1/2, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')
                else:
                    distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                    pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex - 1, current_nodey)]:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 41
                        pq.put((dist + 41, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        if dist + 21 < distancelist[(current_nodex - 1, current_nodey)]:
                            distancelist[(current_nodex - 1, current_nodey)] = dist + 21
                            pq.put((dist + 21, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 21))
                            parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')
                    elif scoredij - 1 >= score_enemy:
                        if dist + 1/2 < distancelist[(current_nodex - 1, current_nodey)]:
                            distancelist[(current_nodex - 1, current_nodey)] = dist + 1/2
                            pq.put((dist + 1/2, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                            parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')
                else:
                    if dist + 1 < distancelist[(current_nodex - 1, current_nodey)]:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex - 1, current_nodey)] = (current_nodex, current_nodey, 'u')


        # down
        flag = False
        flag_hit = False
        if current_nodex + 1 < height and (gridmap[current_nodex + 1][current_nodey] == 'E' + character_enemy or gridmap[current_nodex + 1][current_nodey] == 'T' + character_enemy):
            flag_hit = True
        if current_nodex + 1 < height:
            if (current_nodex + 1, current_nodey) in trap:
                flag = True
        if (current_nodex + 1 < height) and ((current_nodex + 1, current_nodey) not in visited) and (
                (gridmap[current_nodex + 1][current_nodey] == 'E') or (
                gridmap[current_nodex + 1][current_nodey] == 'T') or (
                gridmap[current_nodex + 1][current_nodey] == "E"+character) or (
                gridmap[current_nodex + 1][current_nodey] == "T"+character) or (
                (current_nodex + 1 == goalx) and (current_nodey == goaly)) or (
                gridmap[current_nodex + 1][current_nodey] == '1' and diccolornumber['y'] == 15) or (
                gridmap[current_nodex + 1][current_nodey] == '2' and (
                    scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                gridmap[current_nodex + 1][current_nodey] == '3' and (
                    scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                gridmap[current_nodex + 1][current_nodey] == '4' and (
                    scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):
            if (current_nodex + 1, current_nodey) not in distancelist:
                if flag:
                    distancelist[(current_nodex + 1, current_nodey)] = dist + 41
                    pq.put((dist + 41, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 41))
                    parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 21
                        pq.put((dist + 21, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 21))
                        parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
                    elif scoredij - 1 >= score_enemy:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 1/2
                        pq.put((dist + 1/2, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
                else:
                    distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                    pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex + 1, current_nodey)]:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 41
                        pq.put((dist + 41, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        if dist + 21 < distancelist[(current_nodex + 1, current_nodey)]:
                            distancelist[(current_nodex + 1, current_nodey)] = dist + 21
                            pq.put((dist + 21, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 21))
                            parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
                    elif scoredij - 1 >= score_enemy:
                        if dist + 1/2 < distancelist[(current_nodex + 1, current_nodey)]:
                            distancelist[(current_nodex + 1, current_nodey)] = dist + 1/2
                            pq.put((dist + 1/2, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                            parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')
                else:
                    if dist + 1 < distancelist[(current_nodex + 1, current_nodey)]:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, 'd')

        # left
        flag = False
        flag_hit = False
        if current_nodey - 1 >= 0 and (gridmap[current_nodex][current_nodey - 1] == 'E' + character_enemy or gridmap[current_nodex][current_nodey - 1] == 'T' + character_enemy):
            flag_hit = True
        if current_nodey - 1 >= 0:
            if (current_nodex, current_nodey - 1) in trap:
                flag = True
        if (current_nodey - 1 >= 0) and ((current_nodex, current_nodey - 1) not in visited) and (
                (gridmap[current_nodex][current_nodey - 1] == 'E') or (
                gridmap[current_nodex][current_nodey - 1] == 'T') or (
                gridmap[current_nodex][current_nodey - 1] == "E"+character) or (
                gridmap[current_nodex][current_nodey - 1] == "T"+character) or (
                (current_nodex == goalx) and (current_nodey - 1 == goaly)) or (
                gridmap[current_nodex][current_nodey - 1] == '1' and diccolornumber['y'] == 15) or (
                gridmap[current_nodex][current_nodey - 1] == '2' and (
                scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                gridmap[current_nodex][current_nodey - 1] == '3' and (
                scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                gridmap[current_nodex][current_nodey - 1] == '4' and (
                scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):
            if (current_nodex, current_nodey - 1) not in distancelist:
                if flag :
                    distancelist[(current_nodex, current_nodey - 1)] = dist + 41
                    pq.put((dist + 41, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 41))
                    parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 21
                        pq.put((dist + 21, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 21))
                        parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')
                    elif scoredij - 1 >= score_enemy:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 1 / 2
                        pq.put((dist + 1 / 2, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')
                else:
                    distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                    pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex, current_nodey - 1)]:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 41
                        pq.put((dist + 41, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        if dist + 21 < distancelist[(current_nodex, current_nodey - 1)]:
                            distancelist[(current_nodex, current_nodey - 1)] = dist + 21
                            pq.put((dist + 21, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 21))
                            parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')
                    elif scoredij - 1 >= score_enemy:
                        if dist + 1/2 < distancelist[(current_nodex, current_nodey - 1)]:
                            distancelist[(current_nodex, current_nodey - 1)] = dist + 1/2
                            pq.put((dist + 1/2, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                            parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')
                else:
                    if dist + 1 < distancelist[(current_nodex, current_nodey - 1)]:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                        pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, 'l')


        # right
        flag = False
        flag_hit = False
        if current_nodey + 1 < width and (gridmap[current_nodex][current_nodey + 1] == 'E' + character_enemy or gridmap[current_nodex][current_nodey + 1] == 'T' + character_enemy):
            flag_hit = True
        if current_nodey + 1 < width:
            if (current_nodex, current_nodey + 1) in trap:
                flag = True
        if (current_nodey + 1 < width) and ((current_nodex, current_nodey + 1) not in visited) and (
                (gridmap[current_nodex][current_nodey + 1] == 'E') or (
                gridmap[current_nodex][current_nodey + 1] == 'T') or (
                gridmap[current_nodex][current_nodey + 1] == "E"+character) or (
                gridmap[current_nodex][current_nodey + 1] == "T"+character) or (
                (current_nodex == goalx) and (current_nodey + 1 == goaly)) or (
                gridmap[current_nodex][current_nodey+1] == '1' and diccolornumber['y'] == 15) or (
                gridmap[current_nodex][current_nodey+1] == '2' and (
                scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                gridmap[current_nodex][current_nodey+1] == '3' and (
                scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                gridmap[current_nodex][current_nodey+1] == '4' and (
                scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):
            if (current_nodex, current_nodey + 1) not in distancelist:
                if flag:
                    distancelist[(current_nodex, current_nodey + 1)] = dist + 41
                    pq.put((dist + 41, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 41))
                    parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 21
                        pq.put((dist + 21, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 21))
                        parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
                    elif scoredij - 1 >= score_enemy:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 1/2
                        pq.put((dist + 1/2, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
                else:
                    distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                    pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex, current_nodey + 1)]:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 41
                        pq.put((dist + 41, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
                elif flag_hit:
                    if scoredij - 1 < score_enemy:
                        if dist + 21 < distancelist[(current_nodex, current_nodey + 1)]:
                            distancelist[(current_nodex, current_nodey + 1)] = dist + 21
                            pq.put((dist + 21, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 21))
                            parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
                    elif scoredij - 1 >= score_enemy:
                        if dist + 1/2 < distancelist[(current_nodex, current_nodey + 1)]:
                            distancelist[(current_nodex, current_nodey + 1)] = dist + 1/2
                            pq.put((dist + 1/2, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                            parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
                else:
                    if dist + 1 < distancelist[(current_nodex, current_nodey + 1)]:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                        pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex, current_nodey + 1)] = (current_nodex, current_nodey, 'r')
        score_enemy -= 1


def dijkstrawayenemy(agentx, agenty, goalx, goaly, gridmap, height, width,scoredij,character,diccolornumber, trap, characterenemy, score_enemy):

    # print(agentx, agenty, "agent x, y in dijkstra enemy way")
    # print(goalx, goaly, "goalx, y in dijkstra enemy way")
    visited = {}
    distancelist = {}
    parent = {}
    way = LifoQueue()
    parent[(agentx, agenty)] = (-1, -1, (agentx,agenty,0))
    pq = PriorityQueue()
    pq.put((0, agentx, agenty, 0, scoredij))
    distancelist[(agentx, agenty)] = 0
    if agentx == goalx and agenty == goaly:
        way.put((agentx,agenty,0))
        return way

    if (gridmap[agentx][agenty] == 'T' + character and gridmap[goalx][goaly] == 'T'):
        way.put(())
        return way

    flag = False
    while not pq.empty():
        temp = pq.get()
        dist = temp[0]
        current_nodex = temp[1]
        current_nodey = temp[2]
        scoredij = temp[4]
        actual_dist = temp[3]
        visited[(current_nodex, current_nodey)] = True
        if (current_nodex == goalx) and (current_nodey == goaly):
            # print("im in goal == current in dijkstra enemy way")
            x = current_nodex
            y = current_nodey
            if gridmap[x][y] == 'T':
                way.put(())
            while True:
                way.put(parent[(x, y)][2])
                currentx = parent[(x, y)][0]
                currenty = parent[(x, y)][1]
                x = currentx
                y = currenty
                if x == -1 and y == -1:
                    break

            return way

        # up
        flag = False

        flag_hit = False
        # if current_nodex - 1 >= 0 and gridmap[current_nodex - 1][current_nodey] == 'E' + characterenemy:
        #     flag_hit = True

        if current_nodex - 1 >= 0:
            if (current_nodex - 1, current_nodey) in trap:
                flag = True
        if (current_nodex - 1 >= 0) and ((current_nodex - 1, current_nodey) not in visited) and (
                (gridmap[current_nodex - 1][current_nodey] == 'E') or (
                gridmap[current_nodex - 1][current_nodey] == 'T') or (
                        gridmap[current_nodex - 1][current_nodey] == "E" + character) or(
                gridmap[current_nodex - 1][current_nodey] == "E" + characterenemy)or (
                        gridmap[current_nodex - 1][current_nodey] == "T" + character) or (
                        (current_nodex - 1 == goalx) and (current_nodey == goaly)) or (
                        gridmap[current_nodex - 1][current_nodey] == '1' and diccolornumber['y'] == 15) or (
                        gridmap[current_nodex - 1][current_nodey] == '2' and (
                        scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                        gridmap[current_nodex - 1][current_nodey] == '3' and (
                        scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                        gridmap[current_nodex - 1][current_nodey] == '4' and (
                        scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):

            if (current_nodex - 1, current_nodey) not in distancelist:
                if flag:
                    distancelist[(current_nodex - 1, current_nodey)] = dist + 41
                    pq.put((dist + 41, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 41))

                    parent[(current_nodex-1, current_nodey)] = (current_nodex, current_nodey, (current_nodex-1,
                                                                                               current_nodey,
                                                                                               actual_dist + 1))


                # elif flag_hit:
                #     if scoredij - 1 < score_enemy:
                #         distancelist[(current_nodex - 1, current_nodey)] = dist + 21
                #         pq.put((dist + 21, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 21))
                #         parent[(current_nodex - 1, current_nodey)] = (
                #             current_nodex, current_nodey, (current_nodex - 1, current_nodey, actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         distancelist[(current_nodex - 1, current_nodey)] = dist + 1

                #         parent[(current_nodex -1, current_nodey)] = (
                #             current_nodex, current_nodey, (current_nodex - 1, current_nodey, actual_dist + 1))
                else:
                    distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                    pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex - 1, current_nodey)] = (
                        current_nodex, current_nodey, (current_nodex - 1, current_nodey, actual_dist + 1))

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex - 1, current_nodey)]:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 41
                        pq.put((dist + 41, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex - 1, current_nodey)] = (
                            current_nodex, current_nodey, (current_nodex - 1, current_nodey, actual_dist + 1))

                # elif flag_hit:
                #     if scoredij - 1 < score_enemy:
                #         if dist + 21 < distancelist[(current_nodex - 1, current_nodey)]:
                #             distancelist[(current_nodex - 1, current_nodey)] = dist + 21
                #             pq.put((dist + 21, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 21))
                #             parent[(current_nodex - 1, current_nodey)] = (
                #                 current_nodex, current_nodey, (current_nodex - 1, current_nodey, actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         if dist + 1 < distancelist[(current_nodex - 1, current_nodey)]:
                #             distancelist[(current_nodex - 1, current_nodey)] = dist + 1

                #             pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                #             parent[(current_nodex - 1, current_nodey)] = (
                #                 current_nodex, current_nodey, (current_nodex - 1, current_nodey, actual_dist + 1))
                else:
                    if dist + 1 < distancelist[(current_nodex - 1, current_nodey)]:
                        distancelist[(current_nodex - 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, current_nodex - 1, current_nodey, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex - 1, current_nodey)] = (
                            current_nodex, current_nodey, (current_nodex - 1, current_nodey, actual_dist + 1))

        # down
        flag = False
        flag_hit = False

        # if current_nodex + 1 < height and gridmap[current_nodex + 1][current_nodey] == 'E' + characterenemy:
        #     flag_hit = True
        if current_nodex + 1 < height:
            if (current_nodex + 1, current_nodey) in trap:
                flag = True
        if (current_nodex + 1 < height) and ((current_nodex + 1, current_nodey) not in visited) and (
                (gridmap[current_nodex + 1][current_nodey] == 'E') or (
                gridmap[current_nodex + 1][current_nodey] == 'T') or (

                gridmap[current_nodex - 1][current_nodey] == "E" + characterenemy) or (
                        gridmap[current_nodex + 1][current_nodey] == "E" + character) or (
                        gridmap[current_nodex + 1][current_nodey] == "T" + character) or (
                        (current_nodex + 1 == goalx) and (current_nodey == goaly)) or (
                        gridmap[current_nodex + 1][current_nodey] == '1' and diccolornumber['y'] == 15) or (
                        gridmap[current_nodex + 1][current_nodey] == '2' and (
                        scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                        gridmap[current_nodex + 1][current_nodey] == '3' and (
                        scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                        gridmap[current_nodex + 1][current_nodey] == '4' and (
                        scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):
            if (current_nodex + 1, current_nodey) not in distancelist:
                if flag:
                    distancelist[(current_nodex + 1, current_nodey)] = dist + 41
                    pq.put((dist + 41, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 41))
                    parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                                                                                                 current_nodey,
                                                                                                 actual_dist + 1))
                # elif flag_hit:
                #     if scoredij - 1 < score_enemy:
                #         distancelist[(current_nodex + 1, current_nodey)] = dist + 21
                #         pq.put((dist + 21, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 21))

                #         parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                #                                                                                      current_nodey,
                #                                                                                      actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                #         pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                #         parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                #                                                                                      current_nodey,
                #                                                                                      actual_dist + 1))
                else:
                    distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                    pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                                                                                                 current_nodey,
                                                                                                 actual_dist + 1))

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex + 1, current_nodey)]:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 41
                        pq.put((dist + 41, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                                                                                                     current_nodey,
                                                                                                     actual_dist + 1))

                # elif flag_hit:
                #     if scoredij - 1 < score_enemy:
                #         if dist + 21 < distancelist[(current_nodex + 1, current_nodey)]:
                #             distancelist[(current_nodex + 1, current_nodey)] = dist + 21
                #             pq.put((dist + 21, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 21))
      #             parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                #                                                                                          current_nodey,
                #                                                                                          actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         if dist + 1 < distancelist[(current_nodex + 1, current_nodey)]:
                #             distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                #             pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                #             parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                #                                                                                          current_nodey,
                #                                                                                          actual_dist + 1))

                else:
                    if dist + 1 < distancelist[(current_nodex + 1, current_nodey)]:
                        distancelist[(current_nodex + 1, current_nodey)] = dist + 1
                        pq.put((dist + 1, current_nodex + 1, current_nodey, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex + 1, current_nodey)] = (current_nodex, current_nodey, (current_nodex + 1,
                                                                                                     current_nodey,
                                                                                                     actual_dist + 1))

        # left
        flag = False
        flag_hit = False

        # if current_nodey - 1 < width and gridmap[current_nodex][current_nodey - 1] == 'E' + characterenemy:
        #     flag_hit = True

        if current_nodey - 1 < width:
            if (current_nodex, current_nodey - 1) in trap:
                flag = True
        if (current_nodey - 1 >= 0) and ((current_nodex, current_nodey - 1) not in visited) and (

                (gridmap[current_nodex][current_nodey - 1] == 'E') or (
                gridmap[current_nodex][current_nodey - 1] == 'T') or (
                        gridmap[current_nodex][current_nodey - 1] == "E" + character)
                or (
                        gridmap[current_nodex - 1][current_nodey] == "E" + characterenemy)
                or (
                        gridmap[current_nodex][current_nodey - 1] == "T" + character)  or (
                        (current_nodex == goalx) and (current_nodey - 1 == goaly)) or (
                        gridmap[current_nodex][current_nodey - 1] == '1' and diccolornumber['y'] == 15) or (
                        gridmap[current_nodex][current_nodey - 1] == '2' and (
                        scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                        gridmap[current_nodex][current_nodey - 1] == '3' and (
                        scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                        gridmap[current_nodex][current_nodey - 1] == '4' and (
                        scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):
            if (current_nodex, current_nodey - 1) not in distancelist:
                if flag:
                    distancelist[(current_nodex, current_nodey - 1)] = dist + 41
                    pq.put((dist + 41, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 41))

                    parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, (current_nodex,
                                                                                                 current_nodey - 1,
                                                                                                 actual_dist + 1))

                # elif flag_hit:
                #     # print("dist",actual_dist,"im in trapi hit")
                #     if scoredij - 1 < score_enemy:
                #         distancelist[(current_nodex, current_nodey - 1)] = dist + 21
                #         pq.put((dist + 21, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 21))

                #         parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, (current_nodex,
                #                                                                                      current_nodey - 1,
                #                                                                                      actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                #         pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                #         parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, (current_nodex,
                #                                                                                      current_nodey - 1,
                #                                                                                      actual_dist + 1))
                else:
                    distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                    pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, (current_nodex,
                                                                                                 current_nodey - 1,
                                                                                                 actual_dist + 1))

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex, current_nodey - 1)]:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 41
                        pq.put((dist + 41, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, (current_nodex,
                                                                                                     current_nodey - 1,
                                                                                                     actual_dist + 1))

                # elif flag_hit:
                #     # print("dist", actual_dist, "im in trapi hit")
                #     if scoredij - 1 < score_enemy:
                #         if dist + 21 < distancelist[(current_nodex, current_nodey - 1)]:
                #             distancelist[(current_nodex, current_nodey - 1)] = dist + 21
                #             pq.put((dist + 21, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 21))

                #             parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, (current_nodex,
                #                                                                                          current_nodey - 1,
                #                                                                                          actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         if dist + 1 < distancelist[(current_nodex, current_nodey - 1)]:
                #             distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                #             pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                #             parent[(current_nodex, current_nodey - 1)] = (current_nodex, current_nodey, (current_nodex,
                #                                                                                          current_nodey - 1,
                #                                                                                          actual_dist + 1))

                else:
                    if dist + 1 < distancelist[(current_nodex, current_nodey - 1)]:
                        distancelist[(current_nodex, current_nodey - 1)] = dist + 1
                        pq.put((dist + 1, current_nodex, current_nodey - 1, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex, current_nodey - 1)] = (
                        current_nodex, current_nodey, (current_nodex, current_nodey - 1, actual_dist + 1))

        # right
        flag = False
        flag_hit = False

        # if current_nodey + 1 < width and gridmap[current_nodex][current_nodey + 1] == 'E' + characterenemy:
        #     flag_hit = True

        if current_nodey + 1 < width:
            if (current_nodex, current_nodey + 1) in trap:
                flag = True
        if (current_nodey + 1 < width) and ((current_nodex, current_nodey + 1) not in visited) and (
                (gridmap[current_nodex][current_nodey + 1] == 'E') or (
                gridmap[current_nodex][current_nodey + 1] == 'T')
                or (
                        gridmap[current_nodex - 1][current_nodey] == "E" + characterenemy)
                or (
                        gridmap[current_nodex][current_nodey + 1] == "E" + character) or (
                        gridmap[current_nodex][current_nodey + 1] == "T" + character)  or (
                        (current_nodex == goalx) and (current_nodey + 1 == goaly)) or (
                        gridmap[current_nodex][current_nodey + 1] == '1' and diccolornumber['y'] == 15) or (
                        gridmap[current_nodex][current_nodey + 1] == '2' and (
                        scoredij - 1 < 15 or diccolornumber['g'] == 8)) or (
                        gridmap[current_nodex][current_nodey + 1] == '3' and (
                        scoredij - 1 < 50 or diccolornumber['r'] == 5)) or (
                        gridmap[current_nodex][current_nodey + 1] == '4' and (
                        scoredij - 1 < 140 or diccolornumber['b'] == 4)) or flag or flag_hit):
            if (current_nodex, current_nodey + 1) not in distancelist:
                if flag:
                    distancelist[(current_nodex, current_nodey + 1)] = dist + 41
                    pq.put((dist + 41, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 41))
                    parent[(current_nodex, current_nodey + 1)] = (
                    current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))
                # elif flag_hit:
                #     # print("dist", actual_dist, "im in trapi hit")
                #     if scoredij - 1 < score_enemy:
                #         distancelist[(current_nodex, current_nodey + 1)] = dist + 21
                #         pq.put((dist + 21, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 21))
                #         parent[(current_nodex, current_nodey + 1)] = (
                #         current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         distancelist[(current_nodex, current_nodey + 1)] = dist + 1

                #         pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                #         parent[(current_nodex, current_nodey + 1)] = (
                #         current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))
                else:
                    distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                    pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                    parent[(current_nodex, current_nodey + 1)] = (
                    current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))

            else:
                if flag:
                    if dist + 41 < distancelist[(current_nodex, current_nodey + 1)]:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 41
                        pq.put((dist + 41, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 41))
                        parent[(current_nodex, current_nodey + 1)] = (
                        current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))
                # elif flag_hit:
                #     # print("dist", actual_dist, "im in trapi hit")
                #     if scoredij - 1 < score_enemy:
                #         if dist + 21 < distancelist[(current_nodex, current_nodey + 1)]:
                #             distancelist[(current_nodex, current_nodey + 1)] = dist + 21
                #             pq.put((dist + 21, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 21))
                #             parent[(current_nodex, current_nodey + 1)] = (
                #             current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))
                #     elif scoredij - 1 >= score_enemy:
                #         if dist + 1 < distancelist[(current_nodex, current_nodey + 1)]:
                #             distancelist[(current_nodex, current_nodey + 1)] = dist + 1

                #             pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                #             parent[(current_nodex, current_nodey + 1)] = (
                #             current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))
                else:
                    if dist + 1 < distancelist[(current_nodex, current_nodey + 1)]:
                        distancelist[(current_nodex, current_nodey + 1)] = dist + 1
                        pq.put((dist + 1, current_nodex, current_nodey + 1, actual_dist + 1, scoredij - 1))
                        parent[(current_nodex, current_nodey + 1)] = (
                        current_nodex, current_nodey, (current_nodex, current_nodey + 1, actual_dist + 1))
        # score_enemy -=1
    way.put(())
    return way