from tools3 import find_elements
import random
import numpy as np
import time
from math import tanh

'''
    maping     point  min  limit
    yellow = 1  10    -     15
    green = 2   25    15    8
    red = 3     35    50    -
    blue = 4    75    140   -
'''
best_answer = None
scores = list()


class Chromosome:
    Genes = None
    Fitness = None

    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness

    def __gt__(self, other):
        return self.Fitness > other.Fitness


def _generate_parent(geneSet, get_fitness, gem_id, h_dict, agent_location, score, turn):
    genes = np.random.permutation(geneSet)
    fitness = get_fitness(genes, gem_id, h_dict, agent_location, score, turn)
    return Chromosome(genes, fitness)


# def calculate_score(genes, gem_id, h_dict, agent_location):
#     global scores
#     score = 45
#     if h_dict[gem_id[genes[0]][0]][1] <= score:
#         score += (h_dict[gem_id[genes[0]][0]][0])
#     score -= manhattan_distance(agent_location[0], agent_location[1], gem_id[genes[0]][1], gem_id[genes[0]][2])
#     for i in range(1, len(genes)):
#         if h_dict[gem_id[genes[i]][0]][1] <= score:
#             score += (h_dict[gem_id[genes[i]][0]][0])
#         score -= manhattan_distance(gem_id[genes[i]][1], gem_id[genes[i - 1]][1], gem_id[genes[i]][2],
#                                     gem_id[genes[i - 1]][2])
#     # print(f"{[(gem_id[genes[i]][1],gem_id[genes[i]][2]) for i in range(len(genes))]}   =>   {score}")
#     scores.append(score)
#     return score


def new_fitness(parent, gem_id, h_dict, agent_location):
    score = 0
    g_dict = {1: 0, 2: 0, 3: 0, 4: 0}
    real_score = 45
    for gem in parent:
        info = calculate_o(agent_location, gem_id[gem][1], gem_id[gem][2])
        t = Gem(gem, gem_id, h_dict, g_dict, real_score)
        # score += (tanh(t * info[0])+1*info[0])
        score += (t * info[0])
        # score += (t*2 + info[0]*7)
        if t < 0:
            t = 0
        real_score += (t - info[1])
        if t != 0:
            g_dict[gem_id[gem][0]] += 1
        agent_location[0] = gem_id[gem][1]
        agent_location[1] = gem_id[gem][2]
    return score


def Gem(gem, gem_id, h_dict, g_dict, score):
    if score >= h_dict[gem_id[gem][0]][1] and g_dict[gem_id[gem][0]] <= h_dict[gem_id[gem][0]][2]:
        return h_dict[gem_id[gem][0]][0]
    return -50


def calculate_o(agent_location, gem_location_x, gem_location_y):
    return (7 + 15 + 2 - manhattan_distance(agent_location[0], agent_location[1], gem_location_x,
                                            gem_location_y)), manhattan_distance(agent_location[0], agent_location[1],
                                                                                 gem_location_x,
                                                                                 gem_location_y)


flag = True


def calculate_score(genes, gem_id, h_dict, agent_location, score, turn):
    global scores
    global flag

    score = score
    steps = turn
    if h_dict[gem_id[genes[0]][0]][1] <= score:
        score += (h_dict[gem_id[genes[0]][0]][0])
    score -= manhattan_distance(agent_location[0], agent_location[1], gem_id[genes[0]][1], gem_id[genes[0]][2])
    steps += manhattan_distance(agent_location[0], agent_location[1], gem_id[genes[0]][1], gem_id[genes[0]][2])
    for i in range(1, len(genes)):
        if steps > 40:
            return score
        if h_dict[gem_id[genes[i]][0]][1] <= score:
            score += (h_dict[gem_id[genes[i]][0]][0])

        score -= manhattan_distance(gem_id[genes[i]][1], gem_id[genes[i]][2], gem_id[genes[i - 1]][1],
                                    gem_id[genes[i - 1]][2])
        steps += manhattan_distance(gem_id[genes[i]][1], gem_id[genes[i]][2], gem_id[genes[i - 1]][1],
                                    gem_id[genes[i - 1]][2])

    # print(f"{[(gem_id[genes[i]][1],gem_id[genes[i]][2]) for i in range(len(genes))]}   =>   {score}")
    return score


def order_xover(a, b, start, stop):
    child = [None] * len(a)
    # Copy a slice from first parent:
    child[start:stop] = a[start:stop]
    # Fill using order from second parent:
    b_ind = stop
    c_ind = stop
    l = len(a)
    while None in child:
        if b[b_ind % l] not in child:
            child[c_ind % l] = b[b_ind % l]
            c_ind += 1
        b_ind += 1
    return child


def order_xover_pair(parent1, parent2, gem_id, h_dict, agent_location, best_score):
    half = len(parent1.Genes) // 2
    start = random.randint(0, len(parent1.Genes) - half)
    stop = start + half
    child1, child2 = order_xover(parent1.Genes, parent2.Genes, start, stop), order_xover(parent2.Genes, parent2.Genes,
                                                                                         start, stop)
    return Chromosome(child1, get_fitness(child1, gem_id, h_dict, agent_location, best_score)), Chromosome(child2,
                                                                                                           get_fitness(
                                                                                                               child2,
                                                                                                               gem_id,
                                                                                                               h_dict,
                                                                                                               agent_location,
                                                                                                               best_score))


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_fitness(genes, gem_id, h_dict, agent_location, score, turn):
    return calculate_score(genes, gem_id, h_dict, agent_location, score, turn)


def get_best(geneSet, get_fitness, gem_id, h_dict, agent_location, score, turn):
    bestparent = list()
    best_answer = None
    for i in range(200):
        bestparent.append(_generate_parent(geneSet, get_fitness, gem_id, h_dict, agent_location, score, turn))
    MaxGeneration = 100
    for i in range(MaxGeneration):
        population = dict()
        bestchild = list()
        for each in bestparent:
            # print([(gem_id[gene][1], gem_id[gene][2]) for gene in each.Genes], each.Fitness)
            result = random.uniform(0, 1)
            # result = np.random.choice([0,1],size=1,p=[.2,.8])
            if result <= .8:
                index1, index2 = np.random.randint(0, len(bestparent), size=2)
                while index1 == index2:
                    index1, index2 = np.random.randint(0, len(bestparent), size=2)
                parent1 = each
                parent2 = bestparent[index1]
                child1, child2 = pmx_pair(parent1, parent2, gem_id, h_dict, agent_location, score, turn)
                bestchild.append(child1)
                bestchild.append(child2)
        choosen_parents = bestchild[0]
        for each in bestchild:
            result = random.uniform(0, 1)
            if result <= .2:
                child = mutate_genes(each.Genes, gem_id, h_dict, agent_location, score, turn)
                bestchild.append(child)
        for each in bestchild:
            if each.Fitness > choosen_parents.Fitness:
                choosen_parents = each
            population[each] = each.Fitness
        for each in bestparent:
            if each.Fitness > choosen_parents.Fitness:
                choosen_parents = each
            population[each] = each.Fitness
        total_score = 0
        min_value = min(population.values())
        for each in population.values():
            total_score += (each + abs(min_value))
        prob = list()
        for each in population.values():
            prob.append((each + abs(min_value)) / total_score)

        bestparent3 = sorted(population.items(), key=lambda population: population[1], reverse=True)
        # print('-------------------------')
        bestparent = Fitness_proportionate_selection(list(population), 100, prob)
        for each in bestparent3[:100]:
            np.append(bestparent, each[0])
        sorted_parent = np.copy(bestparent)
        sorted_parent.sort()
        bestparent2 = sorted(bestparent, key=lambda child: child.Fitness)
        # print('----------------------------------------')
        # print('50 Chossen for next generation')
        # for i in sorted_parent:
        #     print(i.Fitness)
        # print('----------------------------------------')
        # print('\n\n\n\n')
        best = bestparent2[-1]
        if best_answer is None:
            best_answer = best
        if best.Fitness > best_answer.Fitness:
            best_answer = best
        scores.append(best_answer.Fitness)
    # print(best_answer.Genes)
    # print([(gem_id[gene][1], gem_id[gene][2]) for gene in best_answer.Genes], best_answer.Fitness,
    #       'xxxxxxxxxxxxxxxxxxxxxxxxxx')
    return best_answer


def pmx(a, b, start, stop):
    child = [None] * len(a)
    # Copy a slice from first parent:
    child[start:stop] = a[start:stop]
    # Map the same slice in parent b to child using indices from parent a:
    for ind, x in enumerate(b[start:stop]):
        ind += start
        if x not in child:
            while child[ind] != None:
                b = list(b)
                ind = b.index(a[ind])
            child[ind] = x
    # Copy over the rest from parent b
    for ind, x in enumerate(child):
        if x == None:
            child[ind] = b[ind]
    return child


def pmx_pair(a, b, gem_id, h_dict, agent_location, score, turn):
    half = len(a.Genes) // 2
    start = random.randint(0, len(a.Genes) - half)
    stop = start + half
    child1, child2 = pmx(a.Genes, b.Genes, start, stop), pmx(b.Genes, a.Genes, start, stop)
    return Chromosome(child1, get_fitness(child1, gem_id, h_dict, agent_location, score, turn)), Chromosome(child2,
                                                                                                            get_fitness(
                                                                                                                child2,
                                                                                                                gem_id,
                                                                                                                h_dict,
                                                                                                                agent_location,
                                                                                                                score,
                                                                                                                turn))


def mutate_genes(genes, gem_id, h_dict, agent_location, score, turn):
    index1 = random.randrange(0, len(genes))
    index2 = random.randrange(0, len(genes))
    if index1 == index2:
        index1 = random.randrange(0, len(genes))
    genes[index1], genes[index2] = genes[index2], genes[index1]
    fitness = get_fitness(genes, gem_id, h_dict, agent_location, score, turn)
    return Chromosome(genes, fitness)


def Fitness_proportionate_selection(arr, size, probability):
    # for i in range(len(arr)):
    #     print(arr[i].Fitness, probability[i])
    return np.random.choice(arr, size, p=probability)


def define_requirment(map, x, y, score, turn):
    start = time.time()
    requirment = find_elements(map, x, y)
    agent_location = requirment[0]
    gem_total = requirment[1][0]
    gem_sequence = requirment[1][1]
    wall_sequence = requirment[2]
    teleport_sequence = requirment[3]
    initial_score = score
    h_dict = dict()
    h_dict[1] = (10, 0, 15)
    h_dict[2] = (25, 15, 8)
    h_dict[3] = (35, 50, 5)
    h_dict[4] = (75, 140, 4)
    gem_id = dict()
    yellow_gem = 0
    green_gem = 0
    red_gem = 0
    blue_gem = 0
    for each in gem_sequence:
        gem_id[each[0]] = (each[1], each[2], each[3])
        if each[1] == 1:
            yellow_gem += 1
        if each[1] == 2:
            green_gem += 1
        if each[1] == 3:
            red_gem += 1
        if each[1] == 4:
            blue_gem += 1
    best_score = 1
    GenSet = [i for i in range(gem_total)]
    bestparent = (get_best(GenSet, get_fitness, gem_id, h_dict, agent_location, score, turn))
    return cunstruct_sequence(bestparent, gem_id, h_dict)
    # print(bestparent.Fitness,bestparent.Genes)
    # for i in bestparent.Genes:
    #     print(gem_id[i][1],gem_id[i][2])
    # return bestparent.Fitness
    # get_list = list()
    # for i in bestparent.Genes:
    #     get_list.append((gem_id[i][1],gem_id[i][2]))
    # # print(get_list)
    # return get_list

    # print('-------------------------')
    # timeD = time.time() - start
    # print(timeD)


def cunstruct_sequence(Genes, gem_id, h_dict):
    g_dict = {1: 0, 2: 0, 3: 0, 4: 0}
    final_sequence = list()
    for each in Genes.Genes:
        type_gem = gem_id[each][0]
        if g_dict[type_gem] < h_dict[type_gem][2]:
            g_dict[type_gem] += 1
            final_sequence.append(each)
    cordinates = list()
    for each in final_sequence:
        cordinates.append((gem_id[each][1], gem_id[each][2]))
    return cordinates

# def show_plot():
#     x_grid = [i for i in range(len(scores))]
#     plt.figure()
#     plt.plot(x_grid, scores)
#     plt.xlabel('Iteration')
#     plt.ylabel('Best Answer')
#     plt.show()

# def run_testes():
#     l = list()
#     bestparent1 = list()
#     for i in range(100):
#         bestparent1.append(_generate_parent(GenSet, get_fitness, gem_id, h_dict, agent_location, best_score))
#     for i in bestparent1:
#         print(i.Genes , i.Fitness)
#     child1 , child2 = pmx_pair(bestparent1[-1],bestparent1[-2],gem_id, h_dict, agent_location, best_score)
#     print('parent1: ',bestparent1[-1].Genes, bestparent1[-1].Fitness)
#     print('parent2: ',bestparent1[-2].Genes, bestparent1[-2].Fitness)
#     print('child1: ',child1.Genes, child1.Fitness)
#     print('child2: ',child2.Genes, child2.Fitness)
# def Create_Pmx(parent1,parent2, gem_id, h_dict, agent_location, best_score):
#     index = random.randint(0,len(parent1.Genes))
#     gems1 = list(parent1.Genes)[:index] + list(parent2.Genes)[index:]
#     gems2 = list(parent2.Genes)[:index] + list(parent1.Genes)[index:]
#     return Chromosome(gems1, get_fitness(gems1, gem_id, h_dict, agent_location,best_score)), Chromosome(gems2, get_fitness(gems2, gem_id, h_dict, agent_location,best_score))
