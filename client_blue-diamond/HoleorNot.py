
def holeornot(gridmap, height,width,holecounter, hole, score_agent, score_enemy, agent_trap, enemy_trap, befor_score_agent,start_agent,start_enemy,next_move,next_move_enemy,character,character_enemy, maxvalue,count_of_hits):
    holenumber = len(hole)
    #maybe need to decrease maximum_holecounter
    #print(count_of_hits,"countofhole","#"*10)
    maximun_holecounter = holenumber + 2
    nextmove = tuple()
    maxvalueforholeornot = float('-inf')
    value = float('-inf')
    sx=start_agent[0]
    sy=start_agent[1]
    ex=start_enemy[0]
    ey=start_enemy[1]
    print("im in hole or not")
    if next_move_enemy == tuple() or not(gridmap[start_enemy[0]][start_enemy[1]] == 'T'+ character_enemy and gridmap[next_move_enemy[0]][next_move_enemy[1]] == 'T'+character_enemy):
        return nextmove, maxvalueforholeornot
    if gridmap[start_enemy[0]][start_enemy[1]] != 'T' + character_enemy and gridmap[next_move_enemy[0]][next_move_enemy[1]] != 'T' + character_enemy:
        return nextmove, maxvalueforholeornot
    if next_move == tuple():
        return nextmove, maxvalueforholeornot
    # print(gridmap[start_agent[0]][start_agent[1]], "gridmap[start_agent[0]][start_agent[1]] in hole or not")
    # print(gridmap[next_move[0]][next_move[1]], "gridmap[next_move[0]][next_move[1]] in hole or not")
    # print(holecounter,"holecounter")
    # print(gridmap[start_enemy[0]][start_enemy[1]], "gridmap[start_enemy[0]][start_enemy[1]] in hole or not")
    # print(gridmap[next_move_enemy[0]][next_move_enemy[1]], "gridmap[next_move[0]][next_move[1]] in hole or not")
    if gridmap[start_agent[0]][start_agent[1]] == 'T' + character and gridmap[next_move[0]][next_move[1]] == 'T'+character:
        print("im in hole or not start and nextmove = t")

        if (holecounter >= maximun_holecounter) or (count_of_hits >= holenumber//2):
            if gridmap[start_enemy[0]][start_enemy[1]] == 'T' + character_enemy and gridmap[next_move_enemy[0]][next_move_enemy[1]] == 'T'+character_enemy:
                if score_agent < score_enemy:
                    if (sx-1 >= 0) and (gridmap[sx-1][sy] != 'W') and (gridmap[sx-1][sy] != 'T') and (gridmap[sx-1][sy] != 'T'+character_enemy) and ((sx-1,sy) not in enemy_trap):

                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sx-1 hole or not")
                            nextmove = (sx - 1, sy)
                            maxvalueforholeornot = value
                    if (sx + 1 < height) and (gridmap[sx + 1][sy] != 'W') and (gridmap[sx + 1][sy] != 'T') and (
                            gridmap[sx + 1][sy] != 'T' + character_enemy) and ((sx+1, sy) not in enemy_trap):

                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sx+1 hole or not")
                            nextmove = (sx + 1, sy)
                            maxvalueforholeornot = value

                    if (sy - 1 >= 0) and (gridmap[sx][sy-1] != 'W') and (gridmap[sx][sy-1] != 'T') and (
                            gridmap[sx][sy-1] != 'T' + character_enemy) and ((sx, sy-1) not in enemy_trap):

                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sy-1 hole or not")
                            nextmove = (sx, sy - 1)
                            maxvalueforholeornot = value

                    if (sy + 1 < width) and (gridmap[sx][sy + 1] != 'W') and (gridmap[sx][sy + 1] != 'T') and (
                            gridmap[sx][sy + 1] != 'T' + character_enemy) and ((sx, sy + 1) not in enemy_trap):

                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sy+1 hole or not")
                            nextmove = (sx, sy + 1)
                            maxvalueforholeornot = value



                else:
                    print("im in hole or not score > score enemy")
                    if (sx-1 >= 0) and (gridmap[sx-1][sy] == 'T'+character_enemy):
                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sx-1 hole or not down")
                            nextmove = (sx-1, sy)
                            maxvalueforholeornot = value

                    elif (sx + 1 < height) and (gridmap[sx + 1][sy] == 'T' + character_enemy):
                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sx+1 hole or not down")
                            nextmove = (sx + 1, sy)
                            maxvalueforholeornot = value

                    elif (sy - 1 >= 0) and (gridmap[sx][sy-1] == 'T' + character_enemy):
                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sy-1 hole or not down")
                            nextmove = (sx, sy-1)
                            maxvalueforholeornot = value

                    elif (sy + 1 < width) and (gridmap[sx][sy + 1] == 'T' + character_enemy):
                        value = maxvalue + 1
                        if maxvalueforholeornot < value:
                            print(" im in sy+1 hole or not down")
                            nextmove = (sx, sy + 1)
                            maxvalueforholeornot = value

                    else:
                        nextmove = next_move
                        maxvalueforholeornot = maxvalue

    print(nextmove, maxvalueforholeornot, "nextmove and max value in hole or not")
    return nextmove, maxvalueforholeornot
# if  befor_score_agent - score_agent > 20 and holecounter > 1:
#     pass



