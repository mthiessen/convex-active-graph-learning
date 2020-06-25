

import itertools

import graph_tool as gt
import numpy as np
from graph_tool.generation import random_graph

from closure import compute_hull


def simplicial_vertices(g: gt.Graph):
    '''
    returns the (unweighted, undirected) simplicial vertices of g
    '''

    simplicial_vertices = []

    for v in g.vertices():

        #try to find a clique around v
        #TODO: Replace with numpy style
        for x, y in itertools.combinations(g.get_all_neighbors(v), 2):
            if g.edge(x,y) is None:
                break
        else:
            simplicial_vertices.append(int(v))
            #print(len(g.get_all_neighbors(v)))


    return simplicial_vertices


if __name__ == "__main__":
    for i in range(1,100):
        deg_sampler = lambda: np.random.randint(1, i*50)
        g = random_graph(i*100, deg_sampler, directed=False)
        weight = g.new_edge_property("int", val=1)
        s = simplicial_vertices(g)
        print(i*100,len(s))

        if len(s)>0:
            print(np.sum(compute_hull(g, s, weight)>0))
            print(np.sum(compute_hull(g, np.random.randint(0,i*100,len(s)), weight)>0))

        print("=========================")


