import itertools
import os
from queue import Queue

import numpy as np
import graph_tool
import graph_tool.topology
import scipy

import closure
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull
from labelpropgation import label_propagation
from simplicial_vertices import simplicial_vertices
from synthetic_spcs import natural_keys


def local_global_strategy(Y, W, alpha=0.5, iterations=200, eps=0.000001):
    np.fill_diagonal(W,0)
    D = np.sum(W, axis=0)
    if np.any(D == 0):
        D += D[D>0].min()/2
    Dhalfinverse = 1 / np.sqrt(D)
    Dhalfinverse = np.diag(Dhalfinverse)
    S = np.dot(np.dot(Dhalfinverse, W), Dhalfinverse)

    F = np.zeros((Y.shape[0], Y.shape[1]))
    oldF = np.ones((Y.shape[0], Y.shape[1]))
    oldF[:Y.shape[1], :Y.shape[1]] = np.eye(Y.shape[1])
    i = 0
    while (np.abs(oldF - F) > eps).any() or i >= iterations:
        oldF = F
        F = np.dot(alpha * S, F) + (1 - alpha) * Y

    result = np.zeros(Y.shape[0])
    #uniform argmax
    for i in range(Y.shape[0]):
        result[i] = np.random.choice(np.flatnonzero(F[i] == F[i].max()))

    return result

    #return np.argmax(F, axis=1)

def label_propagation2(W, known_labels, labels):
    W = np.exp(-W *W/ 2) #similarity
    Y = np.zeros((W.shape[0],labels.size))

    for i,label in enumerate(labels):
        Y[known_labels == label,i] = 1

    return local_global_strategy(Y,W)


def mssp(g: graph_tool.Graph, weight_prop: graph_tool.EdgePropertyMap, L, known_labels):
    n= g.num_vertices()
    dist_map = np.ones((n,n))*np.inf

    for i,j in itertools.combinations(L,2):
        if known_labels[i] != known_labels[j]:
            dist_map[i,j] = graph_tool.topology.shortest_distance(g, i, j, weight_prop)


    i,j = np.unravel_index(dist_map.argmin(), dist_map.shape)

    if weight_prop is None:
        total_weight = g.num_edges() + 1
    else:
        total_weight = np.sum(weight_prop.a) + 1

    if dist_map[i,j] < total_weight:

        path,_ = graph_tool.topology.shortest_path(g, i, j, weight_prop)
        mid_point = path[len(path)//2]
        return mid_point
    else:
        return None


def s2(g: graph_tool.Graph, weight_prop: graph_tool.EdgePropertyMap, labels, budget=20, use_adjacency=False):
    L = set()

    n = g.num_vertices()

    known_labels = -np.ones(n)*np.inf

    W = graph_tool.topology.shortest_distance(g, weights=weight_prop).get_2d_array(range(n)) #original distance map

    x = np.random.choice(list(set(range(n)).difference(L)))
    while budget>0:
        known_labels[x] = labels[x]
        L.add(x)
        if len(L) == n:
            break
        budget -= 1
        to_remove = []
        for e in g.get_out_edges(x):
            if known_labels[e[1]] > -np.inf and known_labels[e[1]] != known_labels[x]:
                to_remove.append(e)

        for e in to_remove:
            g.remove_edge(g.edge(e[0],e[1]))

        mid_point = mssp(g, weight_prop, L, known_labels)

        if mid_point is not None:
            x = int(mid_point)
        else:
            x = np.random.choice(list(set(range(n)).difference(L)))
        prediction = label_propagation(W, known_labels, labels, use_adjacency=use_adjacency)

        print("accuracy", np.sum(prediction == labels) / labels.size)


if __name__ == "__main__":
    np.random.seed(43)
    files = os.listdir("res/new_synthetic/")
    files.sort(key=natural_keys)
    for filename in files:
        for label_idx in range(10):
            if ".csv" not in filename:
                continue
            instance = filename.split(".")[0]
            print("======================================================")
            print("file", instance, "label", label_idx)
            edges = np.genfromtxt("res/new_synthetic/" + instance + ".csv", delimiter=",", dtype=np.int)[:, :2]
            n = np.max(edges) + 1
            g = graph_tool.Graph(directed=False)
            g.add_vertex(n)
            g.add_edge_list(edges)
            weight_prop = g.new_edge_property("double", val=1)

            y = np.zeros(n)
            add_string = ""
            if label_idx > 4:
                add_string = "_simplicial_start"
            y[np.genfromtxt("res/new_synthetic/labels/" + instance + "_" + str(label_idx) + add_string + "_positive.csv",
                            dtype=np.int)] = True

            a = s2(g, weight_prop, y)
            print(a)
            print(y)
            print(np.sum(a != y)/n, "%")