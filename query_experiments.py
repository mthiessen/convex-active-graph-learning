import itertools

import graph_tool
import graph_tool.topology
import numpy as np
import pickle
import pandas as pd
import gzip
import scipy.spatial.distance

from labelpropgation import label_propagation
from shortest_shortest_path_querying import s2
from spcquerrying import budgeted_spc_querying, budgeted_heuristic_querying

def prepare_cora(directed=False):
    print("directed", directed)
    edges = np.genfromtxt('res/cora/cora.edges', dtype=np.int, delimiter=',')[:, :2] - 1
    labels = np.genfromtxt('res/cora/cora.node_labels', dtype=np.int, delimiter=',')[:, 1]

    g = graph_tool.Graph(directed=directed)

    g.add_edge_list(edges)
    vfilt = graph_tool.topology.label_largest_component(g, directed=False)

    labels = labels[vfilt.a.astype(np.bool)]
    g = graph_tool.GraphView(g, vfilt=vfilt)
    g.purge_vertices()

    weight_prop = g.new_edge_property("int", val=1)
    #spc = shortest_path_cover_logn_apx(g, weight_prop)
    spc = pickle.load(open("res/cora/largest_comp_new_spc_" + str(directed) + ".p", "rb"))

    print("spc", len(spc))

    pickle.dump(spc, open("res/cora/largest_comp_new_spc_" + str(directed) + ".p", "wb"))

    #spc = pickle.load(open("res/cora/largest_comp_new_spc_" + str(directed) + ".p", "rb"))

    return g, labels, spc


def prepare_citeseer(directed=False, weighted=False):
    print("directed", directed)
    attributes_df = pd.read_csv('res/citeseer/citeseer.content', sep="\t", header=None, dtype=np.str)
    features = attributes_df.iloc[:, 1:-1].to_numpy(dtype=np.int)
    labels, _ = pd.factorize(attributes_df.iloc[:, -1])
    new_ids, old_ids = pd.factorize(attributes_df.iloc[:, 0])

    edges_df = pd.read_csv('res/citeseer/citeseer.cites', sep="\t", header=None, dtype=np.str)
    edges_df = edges_df[edges_df.iloc[:, 0].apply(lambda x: x in old_ids)]
    edges_df = edges_df[edges_df.iloc[:, 1].apply(lambda x: x in old_ids)]
    renamed = edges_df.replace(old_ids, new_ids)
    edges = renamed.to_numpy(dtype=np.int)
    edges = np.fliplr(edges)
    g = graph_tool.Graph(directed=directed)

    g.add_edge_list(edges)

    vfilt = graph_tool.topology.label_largest_component(g, directed=False)

    labels = labels[vfilt.a.astype(np.bool)]
    g = graph_tool.GraphView(g, vfilt=vfilt)
    g.purge_vertices()

    weight = np.sum(np.abs(features[edges[:, 0]] - features[edges[:, 1]]), axis=1)
    if weighted:
        weight_prop = g.new_edge_property("int", vals=weight)
    else:
        weight_prop = g.new_edge_property("int", val=1)

    #spc = shortest_path_cover_logn_apx(g, weight_prop)
    spc = pickle.load(open("res/citeseer/largest_comp_new_spc_directed_"+str(directed)+"_weighted_"+str(weighted)+".p", "rb"))

    print("spc", len(spc))

    #pickle.dump(spc, open("res/citeseer/largest_comp_new_spc_directed_"+str(directed)+"_weighted_"+str(weighted)+".p", "wb"))

    return g, labels, spc

def prepare_rmnist(weighted=False):
    print("weighted", weighted)
    f = gzip.open("res/rmnist/rmnist_200/rmnist_200.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
    X = np.array(training_data[0])
    y = np.array(training_data[1])

    n = X.shape[0]

    dists = scipy.spatial.distance.cdist(X, X)
    # y = y[:n]

    labels = y

    W = dists[:n, :n]  # np.exp(-(dists) ** 2 / (2 * sigma ** 2))

    k = 5

    for i in range(n):
        W[i, np.argsort(W[i])[(k+1):]] = np.inf #k and itself with zero weight

    np.fill_diagonal(W, 0)
    # W[W > np.quantile(W, q)] = np.inf
    # W2 = np.copy(W) less edges is slower strangely
    # W2[W2 <= 0.1] = 0

    weights = W[(W < np.inf) & (W > 0)].flatten()
    edges = np.array(np.where((W < np.inf) & (W > 0))).T
    edges = edges[edges[:, 0] < edges[:, 1]]

    # return

    g = graph_tool.Graph(directed=False)

    # construct actual graph
    g.add_vertex(n)
    g.add_edge_list(edges)

    vfilt = graph_tool.topology.label_largest_component(g, directed=False)

    labels = labels[vfilt.a.astype(np.bool)]
    g = graph_tool.GraphView(g, vfilt=vfilt)
    g.purge_vertices()

    if weighted:
        weight_prop = g.new_edge_property("double", vals=weights)
    else:
        weight_prop = g.new_edge_property("int", val=1)

    #spc = shortest_path_cover_logn_apx(g, weight_prop)
    spc = pickle.load(open("res/rmnist/rmnist_200/correct_largest_comp_5knn_spc_" + str(0.05) + "_" + str(weighted) + ".p","rb"))
    #pickle.dump(spc, open("res/rmnist/rmnist_200/new_5knn_spc_" + str(0.05) + "_" + str(weighted) + ".p", "wb"))
    #pickle.dump(spc, open("res/rmnist/rmnist_200/correct_largest_comp_5knn_spc_" + str(0.05) + "_" + str(weighted) + ".p", "wb"))

    print(len(spc))

    return g, labels, spc

def prepare_coil(weighted=False):
    print("weighted", weighted)
    dataset = 3
    X = np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',X.tab')
    # X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    y = (np.genfromtxt('res/benchmark/SSL,set=' + str(dataset) + ',y.tab'))
    n = X.shape[0]

    dists = scipy.spatial.distance.cdist(X, X)
    y = y[:n]

    W = dists[:n, :n]  # np.exp(-(dists) ** 2 / (2 * sigma ** 2))
    k = 10

    for i in range(n):
        W[i, np.argsort(W[i])[(k+1):]] = np.inf  #k and itself with zero weight

    np.fill_diagonal(W, 0)

    weights = W[(W < np.inf) & (W > 0)].flatten()
    edges = np.array(np.where((W < np.inf) & (W > 0))).T
    edges = edges[edges[:, 0] < edges[:, 1]]

    g = graph_tool.Graph(directed=False)

    # construct actual graph
    g.add_vertex(n)
    g.add_edge_list(edges)

    vfilt = graph_tool.topology.label_largest_component(g, directed=False)

    y = y[vfilt.a.astype(np.bool)]
    g = graph_tool.GraphView(g, vfilt=vfilt)
    g.purge_vertices()

    if weighted:
        weight_prop = g.new_edge_property("double", vals=weights)
    else:
        weight_prop = g.new_edge_property("int", val=1)

    #spc = shortest_path_cover_logn_apx(g, weight_prop)
    spc = pickle.load(open("res/coil/largest_comp_10knn_spc_" + "_" + str(weighted) + ".p","rb"))
    #pickle.dump(spc, open("res/coil/largest_comp_10knn_spc_" + "_" + str(weighted) + ".p", "wb"))

    print(len(spc))

    return g, y, spc

def do_query_experiment_spc(g, spc, y):
    for hull_as_maxmization in [False]:
        for hull_in_between in [False]:
            for use_adjacency in [False, True]:
                print("======================================================================")
                print("hull_as_maxmization",hull_as_maxmization,"hull_in_between",hull_in_between,"use_adjacency", use_adjacency)
                budgeted_heuristic_querying(g, y, compute_hulls_between_queries=hull_in_between, hull_as_optimization=hull_as_maxmization, use_adjacency=use_adjacency)

def do_qery_experiment_sample(g, spc, y):
    print("random sample")
    sample = []
    dist_map = graph_tool.topology.shortest_distance(g).get_2d_array(range(g.num_vertices())).T.astype(np.double)
    dist_map[dist_map>g.num_vertices()] = np.inf
    known_labels = -np.inf * np.ones(g.num_vertices())
    for use_adjacency in [False,True]:
        for i in range(50):
            print(i)
            sample.append(np.random.choice(list(set(range(g.num_vertices())).difference(sample))))
            known_labels[sample] = y[sample]
            result = label_propagation(dist_map, known_labels, y, use_adjacency=use_adjacency)

            print(np.sum(result == y) / g.num_vertices())

def do_query_experiment_s2(g, spc, y):
    for use_adjacency in [False, True]:
        print("======================================================================")
        print("use_adjacency", use_adjacency)
        s2(g, None, y, 50, use_adjacency)

def are_convex(labelled_path):
    error = 0
    for c in np.unique(labelled_path):
        subpath = np.where(labelled_path==c)[0]
        if subpath.size != subpath[-1]-subpath[0]+1:
            error += subpath.size

    return error

def check_convexity_of_spc(g,spc, y):
    convex_paths = 0

    already_checked = set()

    for p in spc:



        if are_convex(y[p]) == 0:
            convex_paths += 1

        already_checked.update(p)
    print("num convex", convex_paths, convex_paths / len(spc))

def count_convex_paths(g, spc, y):
    count_convex = 0
    count_all_paths = 0

    sample_size = 2
    for s in itertools.combinations(range(g.num_vertices()), sample_size):
        paths = graph_tool.topology.all_shortest_paths(g, s[0], s[1])
        for path in paths:
            count_all_paths += 1
            if are_convex(y[path]) == 0:
                count_convex += 1

        if count_all_paths > 20*count_convex:
            print("=====================================================================")
            print("no chance. almost no convex paths")
            print("all", count_all_paths, "convex", count_convex)
            print("=====================================================================")
            break

    print("convex paths", count_convex, "all paths", count_all_paths)

def shorten_spc(spc):
    already_checked = set()
    for i, p in enumerate(spc):

        # cut out start end end of the path, as already counted
        start = 0
        for x in p:
            if x in already_checked:
                start += 1
            else:
                break

        end = len(p)
        for x in reversed(p):
            if x in already_checked:
                end -= 1
            else:
                break

        spc[i] = p[start:end]

    return spc

if __name__=="__main__":

    np.random.seed(0)


    np.random.seed(0)

    for run in range(10):
        print("=======================================================")
        print("run", run)
        for test in [4]:
            print("=====================================")
            print("test", test)
            if test == 1:
                g, y, spc = prepare_citeseer()
            elif test == 2:
                g, y, spc = prepare_rmnist(True)
            elif test == 3:
                g, y, spc = prepare_cora()
            elif test == 4:
                g, y, spc = prepare_coil(True)
            #do_qery_experiment_sample(g, spc, y)
            # check_convexity_of_spc(g, spc, y)
            spc = shorten_spc(spc)
            #do_qery_experiment_sample(g, spc, y)
            do_query_experiment_spc(g, spc, y)
            #check_convexity_of_spc(g, spc, y)
'''
    prepare_citeseer() #spc 667
    prepare_citeseer(True) #spc 1135
    #worthless all "paths" are just edges
    prepare_citeseer(False,True)
    prepare_citeseer(True,True)

    prepare_cora() spc 646
    prepare_cora(True) spc 1277

    prepare_rmnist() spc 351
    prepare_rmnist(True) spc 351

    prepare_coil() spc 463
    prepare_coil(True) spc 462

'''