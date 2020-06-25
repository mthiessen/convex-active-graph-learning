import json
import os
import pickle

import graph_tool as gt
import numpy as np
import scipy

import shortest_shortest_path_querying
import simplicial_vertices
from DijkstraVisitingMostNodes import shortest_path_cover_logn_apx
from closure import compute_hull
from shortest_shortest_path_querying import local_global_strategy, label_propagation
from spcquerrying import spc_querying_naive, spc_querying_with_closure, spc_querying_with_shadow
from synthetic_spcs import natural_keys
import pandas as pd

def spc_querying_experiments(g: gt.Graph, weight_prop: gt.EdgePropertyMap, spc, labels):
    print("correct labels: ", labels)
    print("================naive=================")
    a, b = spc_querying_naive(g, spc, labels)
    print("pred: ", a)
    print("queries: ", b, np.sum(b))
    print("correct: ", np.sum(a==labels),np.sum(a == labels)/g.num_vertices())
    print("================interval================")
    a, b = spc_querying_with_closure(g, spc, weight_prop, labels, False)
    print("pred: ", a)
    print("queries: ", b, np.sum(b))
    print("correct: ", np.sum(a == labels), np.sum(a == labels) / g.num_vertices())
    print("================closure================")
    #a, b = spc_querying_with_closure(g, spc, weight_prop, labels)
    #print("pred: ", a)
    #print("queries: ", b, np.sum(b))
    #print("correct: ", np.sum(a == labels), np.sum(a == labels) / g.num_vertices())
    #print("================s2================")
    new_labels = np.zeros(g.num_vertices())
    new_labels[labels == np.unique(labels)[1]] = 1
    s2_labelling = shortest_shortest_path_querying.s2(g, weight_prop, labels, int(np.sum(b)))
    print("accuracy s2 after label_prop: ", np.sum(s2_labelling == new_labels) / g.num_vertices())

def spc_semi_supervised_experiments(g: gt.Graph, weight_prop: gt.EdgePropertyMap, labels):
    np.random.seed(1)
    dist_map = gt.topology.shortest_distance(g, weights=weight_prop)
    W = dist_map.get_2d_array(range(g.num_vertices()))  # original distance map
    new_labels = np.zeros(g.num_vertices())
    new_labels[labels == np.unique(labels)[1]] = 1
    for budget in [10,20,50,100]:
        print("========================================================")
        print("budget: ", budget, "|V|=", g.num_vertices())
        print("==================s2=====================")
        overall_labelling = shortest_shortest_path_querying.s2(g, weight_prop, labels, budget)
        print("accuracy after label_prop: ", np.sum(overall_labelling == new_labels) / g.num_vertices())
        for _ in range(5):
            starting_vertices = np.random.choice(range(g.num_vertices()), budget, replace=False)

            known_labels = -np.ones(g.num_vertices())*np.inf
            known_labels[starting_vertices] = labels[starting_vertices]

            pos_label, neg_label = np.unique(labels)

            pos = np.where(known_labels==pos_label)[0]
            neg = np.where(known_labels==neg_label)[0]
            print("=============without hull===================")
            print("label propagation")
            overall_labelling = label_propagation(W, known_labels, np.unique(labels))
            print("accuracy after label_prop: ", np.sum(overall_labelling == new_labels)/g.num_vertices())

            print("=============interval============")
            pos_hull = compute_hull(g, pos, weight_prop, dist_map, compute_closure=False)
            neg_hull = compute_hull(g, neg, weight_prop, dist_map, compute_closure=False)
            print("pos", pos.size)
            print("hull size: ", np.sum(pos_hull))
            print("hull correctness overall", np.sum(pos_hull&(labels==pos_label)))
            mask = np.ones(g.num_vertices(),dtype=np.bool)
            mask[pos] = False
            print("hull correctness on new vertices", np.sum(pos_hull[mask] & (labels == pos_label)[mask]))
            known_labels[pos_hull] = pos_label

            print("neg", neg.size)
            print("hull size: ", np.sum(neg_hull))
            print("hull correctness overall", np.sum(neg_hull & (labels == neg_label)))
            mask = np.ones(g.num_vertices(), dtype=np.bool)
            mask[neg] = False
            print("hull correctness on new vertices", np.sum(neg_hull[mask] & (labels == neg_label)[mask]))
            known_labels[neg_hull] = neg_label

            print("label propagation")
            overall_labelling = label_propagation(W, known_labels, np.unique(labels))
            print("accuracy after label_prop: ", np.sum(overall_labelling==new_labels)/g.num_vertices())

            print("==============closure=================")
            pos_hull = compute_hull(g, pos, weight_prop, dist_map)
            neg_hull = compute_hull(g, neg, weight_prop, dist_map)
            print("pos", pos.size)
            print("hull size: ", np.sum(pos_hull))
            print("hull correctness overall", np.sum(pos_hull & (labels == pos_label)))
            mask = np.ones(g.num_vertices(), dtype=np.bool)
            mask[pos] = False
            print("hull correctness on new vertices", np.sum(pos_hull[mask] & (labels == pos_label)[mask]))
            known_labels[pos_hull] = pos_label

            print("neg", neg.size)
            print("hull size: ", np.sum(neg_hull))
            print("hull correctness overall", np.sum(neg_hull & (labels == neg_label)))
            mask = np.ones(g.num_vertices(), dtype=np.bool)
            mask[neg] = False
            print("hull correctness on new vertices", np.sum(neg_hull[mask] & (labels == neg_label)[mask]))
            print("label propagation")
            known_labels[neg_hull] = neg_label

            overall_labelling = label_propagation(W, known_labels, np.unique(labels))
            print("accuracy after label_prop: ", np.sum(overall_labelling == new_labels) / g.num_vertices())

def is_convex(dir, prefix, target_column, weighted=False):
    print(dir)
    np.random.seed(0)
    edges = np.genfromtxt(dir+prefix+'_edges.csv', skip_header=True, dtype=np.int, delimiter=',')

    df = pd.read_csv(dir+prefix+'_target.csv')#.sort_values('new_id')
    print(dir, "weighted", weighted)

    weight=1
    if weighted:
        if 'twitch' in dir:
            weight = np.zeros(edges.shape[0])
            max = df.iloc[:,1].max()
            min = df.iloc[:,1].min()
            df.iloc[:,1] =(df.iloc[:,1] - min)/(max - min)
            max = df.iloc[:, 3].max()
            min = df.iloc[:, 3].min()
            df.iloc[:, 3] = (df.iloc[:, 3] - min) / (max - min)


            for i, e in enumerate(edges):
                weight[i] = (df.iloc[e[0],1]-df.iloc[e[1],1])**2 + (df.iloc[e[0],3]-df.iloc[e[1],3])**2

        elif 'facebook' in dir:
            attributes = json.load(open('res/git/'+dir+'/facebook_features.json'))
            weight = np.zeros(edges.shape[0])
            for i, e in enumerate(edges):
                weight[i] = len(set(attributes[str(e[0])]).symmetric_difference(attributes[str(e[1])]))

    labels,_ = pd.factorize(df.iloc[:, target_column])

    new_n = 4000
    pos_label, neg_label = np.unique(labels)
    pos = np.where(labels == pos_label)[0]
    neg = np.where(labels == neg_label)[0]

    g = gt.Graph(directed=False)

    g.add_edge_list(edges)

    '''d = g.get_out_degrees(range(g.num_vertices()))
    
    

    d_pos = d[pos].argsort()[-new_n//2:][::-1]
    d_neg = d[neg].argsort()[-new_n//2:][::-1]

    d = np.append(d_pos, d_neg)

    g2 = gt.Graph(directed=False)

    edges =edges[np.isin(edges[:,0],d)&np.isin(edges[:,1],d)]

    indexes = np.unique(edges)
    labels = labels[indexes]
    for i, idx in enumerate(indexes):
        edges[edges==idx] = i

    g2.add_edge_list(edges)

    comp = gt.topology.label_largest_component(g2)
    d = np.where(comp.a == 1)[0]
    labels = labels[d]
    g3 = gt.Graph(directed=False)

    edges = edges[np.isin(edges[:, 0], d) & np.isin(edges[:, 1], d)]

    for i, idx in enumerate(np.unique(edges)):
        edges[edges == idx] = i
    g3.add_edge_list(edges)
    g = g3'''



    if weighted:
        weight = g.new_edge_property("double", vals=weight)
    else:
        weight = g.new_edge_property("double", val=1)

    comps, hist = gt.topology.label_components(g)
    #print(hist)
    #dist_map = gt.shortest_distance(g, weights=weight)
    simple = simplicial_vertices.simplicial_vertices(g)
    gt.stats.remove_self_loops(g)
    print("n=",g.num_vertices(), "simplicial=", len(simple))
    #spc = shortest_path_cover_logn_apx(g, weight)
    if weighted:
        weighted_str = "_weigted_"
    else:
        weighted_str = ""
    #pickle.dump(spc, open(dir+'spc'+weighted_str+'.p', 'wb'))
    spc = pickle.load(open(dir+'spc'+weighted_str+'.p', 'rb'))

    weight = None


    pos = np.where(labels == pos_label)[0]
    neg = np.where(labels == neg_label)[0]

    print("pos", len(pos))
    print("neg", len(neg))
    spc_semi_supervised_experiments(g,weight, labels)



    p_interval = compute_hull(g, pos, weight, compute_closure=False)
    n_interval = compute_hull(g, neg, weight, compute_closure=False)

    print("pos_interval size: ", np.sum(p_interval))
    print("neg_interval size: ", np.sum(n_interval))
    print("intersection of intervals size: ", np.sum(p_interval&n_interval))

    p_hull = compute_hull(g, pos, weight)
    n_hull= compute_hull(g, neg, weight)

    print("pos_hull size: ", np.sum(p_hull))
    print("neg_hull size: ", np.sum(n_hull))
    print("intersection of hulls size: ", np.sum(p_hull & n_hull))

    #spc_querying_experiments(g, weight, spc, labels)

def benchmark(dataset):
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    # X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))

    n = 1500

    y = y[:n]
    dists = scipy.spatial.distance.cdist(X, X)
    W = dists[:n, :n]  # np.exp(-(dists) ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    W[W > np.quantile(W, 0.004)] = np.inf
    # W2 = np.copy(W) less edges is slower strangely
    # W2[W2 <= 0.1] = 0

    weights = W[(W < np.inf) & (W > 0)].flatten()
    edges = np.array(np.where((W < np.inf) & (W > 0))).T

    np.random.seed(0)

    g = gt.Graph()

    # construct actual graph
    g.add_vertex(n)
    g.add_edge_list(edges)
    weight_prop = g.new_edge_property("double", val=1)

    comps, hist = gt.topology.label_components(g)

    #print(len(simplicial_vertices(g)))

    spc = pickle.load(open("res/benchmark/spc_1_q_0.004_weighted_False.p", "rb"))

    #spc_semi_supervised_experiments(g, weight_prop, y)
    spc_querying_experiments(g, weight_prop, spc, y)


if __name__ == "__main__":
    benchmark(1)