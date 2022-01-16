import tools



def best_diamond_bfs(src: tuple, graph_dict: dict, my_grid: list, agent_score: int, eaten: list):
    # returns the path to the gem, which we have the most score after eating that

    frontier = [(src, "none")]
    explored = dict()
    gems_and_dists = []
    found_gems = []

    while len(frontier) != 0:
        # print(frontier)
        current = frontier.pop(0)

        explored[current[0]] = current[1]

        children = graph_dict[current[0]].children

        for child in children:
            action = tools.taken_action(current[0], child)
            if ((child, action) not in frontier) and (child not in explored) \
                    and graph_dict[child].content != 'W':

                frontier.append((child, action))

                if tools.is_gem(my_grid, child[0], child[1]) and \
                        (child not in found_gems):

                    goal_path = [tools.taken_action(current[0], child)]
                    tmp_node = tools.get_parent(child, goal_path[0])
                    while tmp_node != src:
                        goal_path.insert(0, explored[tmp_node])
                        tmp_node = tools.get_parent(tmp_node, explored[tmp_node])

                    # before eating score
                    b_e_s = agent_score - len(goal_path)

                    # after eating score
                    a_e_s = b_e_s + tools.gem_value(graph_dict[child].content)

                    if tools.achievable_gem(graph_dict[child].content, b_e_s, eaten):
                        gems_and_dists.append((a_e_s, goal_path, child))
                        found_gems.append(child)

    gems_and_dists.sort(reverse=True)

    if gems_and_dists:
        return gems_and_dists[0]
    else:
        return None, ["noop"], None
