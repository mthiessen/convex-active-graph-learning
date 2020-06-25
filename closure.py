import itertools
import queue

import graph_tool as gt
import numpy as np


def dumb_compute_closed_interval(g, S, weight):

    visited_nodes = set(S)

    for i,j in itertools.combinations(S,2):
        for path in gt.topology.all_shortest_paths(g, i,j,weights=weight):
            visited_nodes = visited_nodes.union(path)

    return visited_nodes

def compute_hull(g: gt.Graph, S, weight=None, dist_map=None, comps=None,hist=None,compute_closure=True, already_closed=None):
    """

    :param g:
    :param S:
    :param weight: if = None, unit distance is used, which is faster.
    :param dist_map: n*n array with the pairwise shortest distances. if = None, the function will compute it itself
    :param comps:
    :param hist:
    :param compute_closure: #hull=closure or geodetic set, which is faster
    :return:
    """
    n = g.num_vertices()

    I_S = np.zeros(g.num_vertices(), dtype=np.bool)

    I_S[S] = True

    q = queue.Queue()

    for v in S:
        if already_closed is None or v not in already_closed:
            q.put(v)

    if dist_map is None:
        dist_map = gt.topology.shortest_distance(g, weights=weight).get_2d_array(range(n)).T
        #dist_map = shortest_distance(g, weights=weight).get_2d_array(range(n))
        #is possible but is super slow and memory heavy for some reason. not possible on my 16gb machine with graphs |V| roughly 15k.

    while not q.empty():

        v = q.get()
        if compute_closure:
            starting_nodes = np.arange(g.num_vertices())[I_S]
        else:
            starting_nodes = np.arange(g.num_vertices())[S]
        starting_nodes = starting_nodes#[starting_nodes > v] #assume undirected

        if comps is not None and not g.is_directed():
            vs_comp = comps.a[v]
            vs_comp = np.where(comps.a==vs_comp)[0]

            if np.all(I_S[vs_comp]):
                continue

        #all vertices x s.t. d(v,x)+d(x,s)=d(v,s) for some s \in S. These are exactly the ones on any shortest v-s-paths.
        #visited_nodes = np.any(dist_map[v,:]+dist_map[:,starting_nodes].T==dist_map[v,starting_nodes][:,np.newaxis],axis=0)

        visited_nodes = np.zeros(n, dtype=np.bool)

        #careful this is not linear runtime. but constructing the "predecessor dag" is very slow with the Visitor classes.
        if not g.is_directed():
            #debug= set()
            '''for s in starting_nodes:
                #if s <= v:
                #    continue
                #if already_closed is not None:
                #    if already_closed[v] and already_closed[s]:
                #        #print("yay")
                #        continue
                debug = debug.union(np.where(dist_map[v]+dist_map[s]==dist_map[v,s])[0])
                #visited_nodes[np.where(dist_map[v].a+dist_map[s].a==dist_map[v].a[s])[0]] = True'''

            visited_nodes[np.any(dist_map[v,:]+dist_map[:,starting_nodes].T==dist_map[v,starting_nodes][:,np.newaxis],axis=0)] = True


            #first_mins = starting_nodes[np.argmin(dist_map[:, starting_nodes], axis=1)]
            #second_mins = starting_nodes[np.argpartition(dist_map[:, starting_nodes], 1, axis=1)[:, 1].astype(np.int)]

            #visited_nodes[dist_map[first_mins, range(n)]+ dist_map[range(n),second_mins] == dist_map[first_mins,second_mins]] = True
        else:
            '''if np.issubclass_(dist_map[v].a.dtype, numbers.Integral):
                max_value = np.iinfo(dist_map[v].a.dtype).max
            else:
                max_value = np.finfo(dist_map[v].a.dtype).max
            visited_nodes[
                np.any(dist_map[v, :] + dist_map[:, starting_nodes].T == dist_map[v, starting_nodes][:, np.newaxis],
                       axis=0)] = True'''
            #reachable_starting_nodes = starting_nodes[dist_map[v].a[starting_nodes] < max_value]
            '''for i in range(n):
                if I_S[i]:
                    continue
                if np.any(dist_map[v].a[i] + dist_map[i].a[[reachable_starting_nodes]] == dist_map[v].a[reachable_starting_nodes]):
                    visited_nodes[i] = True'''

            visited_nodes[
                np.any(dist_map[v, :] + dist_map[:, starting_nodes].T == dist_map[v, starting_nodes][:, np.newaxis],
                       axis=0)] = True

        if compute_closure:
            for i in range(n):
                if not I_S[i] and visited_nodes[i]:
                    q.put(i)

        I_S[visited_nodes] = True

        #early stopping if already covered all the connected components of S
        if comps is not None and not g.is_directed():
            if np.sum(I_S) == np.sum(hist[np.unique(comps.get_array()[I_S])]):
                break
        elif np.sum(I_S) == n:
            break

        #print (np.sum(I_S), n)

    return I_S

def compute_shadow(g: gt.Graph, A,B, weight=None, dist_map=None, comps=None,hist=None, B_hulls=None):
    A_closed = compute_hull(g, A, weight, dist_map, comps, hist)
    #B_closed = compute_hull(g, B, weight, dist_map, comps, hist)

    B_closed = np.zeros(g.num_vertices(), dtype=np.bool)
    B_closed[B] = True

    shadow = A_closed.copy()

    for x in range(g.num_vertices()):
        if A_closed[x] or B_closed[x]:
            continue
        if B_hulls is None:
            if np.any(compute_hull(g, np.append(B, x),  weight, dist_map, comps, hist, True) & A_closed):
                shadow[x] = True
        else:
            if np.any(B_hulls[x] & A_closed):
                shadow[x] = True

    return shadow




'''
g = Graph(directed=False)

g.add_vertex(10)

S = [0, 5,2]

weight = g.new_edge_property("int")

e = g.add_edge(0, 1)
weight[e] = 4
e = g.add_edge(1, 2)
weight[e] = 4
e = g.add_edge(0, 3)
weight[e] = 1
e = g.add_edge(3, 4)
weight[e] = 1
e = g.add_edge(4, 5)
weight[e] = 1
e = g.add_edge(5, 6)
weight[e] = 1
e = g.add_edge(6, 7)
weight[e] = 1
e = g.add_edge(7, 8)
weight[e] = 1
e = g.add_edge(8, 9)
weight[e] = 1
e = g.add_edge(9, 2)
weight[e] = 1

#I_S = compute_hull(g, S,weight)
#print(I_S)
#print("==========================")
#I_S = compute_hull(g, S,weight, False)
#print(I_S)

g = Graph(directed=False)

g.add_vertex(5)

S = [0,4]
e = g.add_edge(0, 1)
e = g.add_edge(1, 2)
e = g.add_edge(1, 4)
e = g.add_edge(0, 3)
e = g.add_edge(2, 3)
e = g.add_edge(3, 4)

weight = g.new_edge_property("int", val=1)

I_S = compute_hull(g, S,weight)
print(I_S)
print("==========================")
I_S = compute_hull(g, S,weight,False)
print(I_S)

n = 50
for i in range(100):


    for j in range(1,20):
        print("===============================================")
        np.random.seed(i)
        seed_rng(i)
        S = np.random.choice(n, j)
        print(i)
        print(S)
        print("=====")

        deg_sampler = lambda: np.random.randint(2,10)
        g = random_graph(n,deg_sampler, directed=False)
        weight = g.new_edge_property("int", vals=np.random.randint(1,10,g.num_edges()))

        I_S = compute_hull(g, S,weight)
        hull = np.array(I_S)
        print(I_S)
        print("==========================")
        I_S = compute_hull(g, S,weight,False)
        I_S_dumb = dumb_compute_closed_interval(g, S, weight)
        if np.any(I_S_dumb != set(np.where(I_S)[0])):
            graph_draw(g, vertex_text=g.vertex_index, output="dag.png", output_size=(1000, 1000), edge_text=weight,
                       vertex_font_size=20, edge_font_size=20)
            I_S_dumb = dumb_compute_closed_interval(g, S, weight)
            exit(1)
        print(I_S)
        previous_I_S_dumb = set()
        for _ in range(n):
            I_S = compute_hull(g, np.arange(g.num_vertices())[I_S==True],weight,False)
            I_S_dumb = dumb_compute_closed_interval(g,I_S_dumb,weight)
            print(I_S)
            if np.any(I_S_dumb != set(np.where(I_S)[0])):
                exit(2)

            if previous_I_S_dumb == I_S_dumb:
                break

            previous_I_S_dumb = set(I_S_dumb)

        if np.any(hull != I_S):
            exit(3)


#graph_draw(g,vertex_text=g.vertex_index,output="two-nodes.png",  output_size=(1000,1000), vertex_font_size=20)'''