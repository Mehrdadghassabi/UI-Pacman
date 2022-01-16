from dijkstraforAll import  dijkstraforall
from dijkstra_level2 import dijkstrawayenemy
from math import inf
def hitortrap(gridmap, height, width, next_move_enemy, score_agent, score_enemy,maxvalue,start_agent, start_enemy, agentturn, enemyturn, agent_trap, enemy_trap, character, character_enemy,diccolornumber_agent, diccolornumber_enemy, trapno):
   maxvaluefortrap = float('-inf')
   nextmove = tuple()
   dicforall, distancediamond, distancehole = dijkstraforall(gridmap, height, width, start_agent[0], start_agent[1],
                                                             score_agent, score_enemy, enemy_trap, character,
                                                             character_enemy, diccolornumber_agent)
   if next_move_enemy != ():
       enemyway = dijkstrawayenemy(start_enemy[0], start_enemy[1], next_move_enemy[0], next_move_enemy[1], gridmap, height, width,score_enemy,character_enemy,diccolornumber_enemy, agent_trap, character, score_agent)
       value = float('-inf')
       trapcount = len(agent_trap) + 1
       leastscore = 35 * (len(agent_trap) + 1)
       flagtrap = False
       if score_agent >= trapcount * 35 and trapcount <= trapno:
            flagtrap = True
       while not enemyway.empty():
           place = enemyway.get()
           if place == ():
               continue
           if dicforall[place[0]][place[1]] == inf:
               #check this if later
               return (), -inf
           place_togo = dicforall[place[0]][place[1]][0]
           flagE = True if gridmap[place[0]][place[1]] == 'E' else False
           flagagent = True if gridmap[place[0]][place[1]] == ('E'+ character) else False
           flagenemy = True if gridmap[place[0]][place[1]] == ('E'+ character_enemy) else False
           if ((place_togo + 2) <= place[2]) and flagtrap and (flagE or flagagent):
                   value = ((40 - 3/2*(place_togo + 2)) * 40) // 100
                   if value >= maxvaluefortrap:
                       print("i choose trap0.............")
                       maxvaluefortrap = value
                       nextmove = (place[0], place[1])

           elif (place_togo) <= place[2] and score_agent > score_enemy and (flagE or flagagent):
               value = ((20 - 3/2*(place_togo )) * 40) // 100
               if value >= maxvaluefortrap:
                   print("i choose trap0.............")
                   maxvaluefortrap = value
                   nextmove = (place[0], place[1])
           elif place_togo == place[2] and score_agent >= score_enemy and flagenemy:
               value = ((20 - 3/2*(place_togo)) * 40) // 100
               if value >= maxvaluefortrap:
                   print("i choose trap0.............")
                   maxvaluefortrap = value
                   nextmove = (place[0], place[1])

   else:
       if score_agent >= score_enemy:
          print(dicforall[start_enemy[0]][start_enemy[1]],"dicforall[start_enemy[0]][start_enemy[1]] in hitortrap")
          if (dicforall[start_enemy[0]][start_enemy[1]][0] != inf):
              print("im in scoreagent scoreenemy")
              nextmove = start_enemy
              maxvaluefortrap = 1

       else:
           sx = start_agent[0]
           sy = start_agent[1]
           ex = start_enemy[0]
           ey = start_enemy[1]
           print(';'*20)
           print((sx - 1, sy) not in enemy_trap, "(sx + 1, sy) not in enemy_trap")
           print((maxvaluefortrap < dicforall[sx - 1][sy][0]), "(maxvaluefortrap < dicforall[sx + 1][sy][0]")
           print(dicforall[ex][ey][0] < dicforall[sx - 1][sy][0], "dicforall[ex][ey][0] < dicforall[sx + 1][sy][0])")
           print(';' * 20)
           if ((sx-1 >= 0) and (dicforall[sx-1][sy][0] != inf)and (gridmap[sx-1][sy] != 'W') and (gridmap[sx-1][sy] != 'E'+character_enemy) and (gridmap[sx-1][sy] != 'T'+character_enemy)and (
                   dicforall[ex][ey][0] <= dicforall[sx-1][sy][0]) and ((sx-1,sy) not in enemy_trap) and (maxvaluefortrap < dicforall[sx-1][sy][0])):
                nextmove = (sx-1, sy)
                maxvaluefortrap = dicforall[sx-1][sy][0]
           if ((sx + 1 < height) and (dicforall[sx+1][sy][0] != inf) and (dicforall[ex][ey][0] <= dicforall[sx + 1][sy][0]) and (gridmap[sx+1][sy] != 'W') and (gridmap[sx+1][sy] != 'E'+character_enemy) and (gridmap[sx+1][sy] != 'T'+character_enemy) and (
                   (sx + 1, sy) not in enemy_trap) and (maxvaluefortrap < dicforall[sx + 1][sy][0])):
               nextmove = (sx + 1, sy)
               maxvaluefortrap = dicforall[sx + 1][sy][0]
           if ((sy - 1 >= 0) and (dicforall[sx][sy-1][0] != inf) and (dicforall[ex][ey][0] <= dicforall[sx][sy-1][0]) and(gridmap[sx][sy-1] != 'W') and (gridmap[sx][sy-1] != 'E'+character_enemy) and (gridmap[sx][sy-1] != 'T'+character_enemy)  and (
                   (sx , sy-1) not in enemy_trap) and (maxvaluefortrap < dicforall[sx ][sy-1][0])):
               nextmove = (sx, sy-1)
               maxvaluefortrap = dicforall[sx][sy-1][0]
           if ((sy + 1 < width) and (dicforall[sx][sy+1][0] != inf) and (dicforall[ex][ey][0] <= dicforall[sx][sy+1][0]) and (gridmap[sx][sy+1] != 'W') and (gridmap[sx][sy+1] != 'E'+character_enemy)and (gridmap[sx][sy+1] != 'T'+character_enemy) and (
                   (sx , sy+1) not in enemy_trap) and (maxvaluefortrap < dicforall[sx][sy+1][0])):
               nextmove = (sx, sy+1)
               maxvaluefortrap = dicforall[sx][sy+1][0]
           print(nextmove, maxvaluefortrap, " next move and max value for trap instead of noop")
   return nextmove, maxvaluefortrap