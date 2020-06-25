import os
import random
from queue import Queue

import numpy as np
import graph_tool
import graph_tool.topology as gt_topology

import closure
from closure import compute_hull
from labelpropgation import label_propagation

import pickle


def binarySearch(arr, l, r, something, known_labels, depth=0):
    label_budget = 0
    while np.any(known_labels == -np.inf):

        mid = l + (r - l) // 2

        #if mid == l:
        #    return mid, label_budget

        # Check if x is present at mid
        if known_labels[mid] == -np.inf:
            label_budget += 1
            known_labels[mid] = arr[mid]

        #if known_labels[mid+1] >= 0:
            #break

        if known_labels[mid] == known_labels[0]:
            known_labels[l:mid+1] = known_labels[l]
            l = mid
        elif known_labels[mid] == known_labels[-1]:
            known_labels[mid:r+1] = known_labels[r]
            r = mid
        else:
            #new class --> recurse!
            l_label_budget,_ = binarySearch(arr[:mid+1], 0, mid, something, known_labels[:mid+1], depth+1)
            r_label_budget,_ = binarySearch(arr[mid:], 0, known_labels.size -mid-1, something,known_labels[mid:], depth+1)


            label_budget += l_label_budget + r_label_budget
            break

    # If we reach here, then the element was not present
    return label_budget, known_labels

def spc_querying_naive(g : graph_tool.Graph, paths, y, trust_own_predictions=True, weight=None, closed_interval=False):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    known_labels = -np.ones(g.num_vertices())*np.inf
    budget = np.zeros(g.num_vertices())
    for i, path in enumerate(paths):
        if not trust_own_predictions or known_labels[path[0]] == -np.inf:
            budget[i] += 1
            known_labels[path[0]] = y[path[0]]
        if not trust_own_predictions or known_labels[path[-1]] == -np.inf:
            budget[i] += 1
            known_labels[path[-1]] = y[path[-1]]

        if known_labels[path[0]] == known_labels[path[-1]]:
            known_labels[path] = known_labels[path[0]]
        else:
            label_budget, new_labels = binarySearch(y[path], 0, len(path)-1, known_labels[path[0]], known_labels[path])
            known_labels[path] = new_labels
            budget[i] += label_budget
        if closed_interval:
            p =closure.compute_hull(g, np.where(known_labels==np.unique(y)[0])[0], weight, compute_closure=False)
            n = closure.compute_hull(g, np.where(known_labels==np.unique(y)[1])[0], weight, compute_closure=False)

            known_labels[p] = np.unique(y)[0]
            known_labels[n] = np.unique(y)[1]

    return known_labels, budget

def spc_querying_naive_multiclass(g : graph_tool.Graph, paths, y, trust_own_predictions=True, weight=None):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    known_labels = -np.ones(g.num_vertices())*np.inf
    budget = np.zeros(g.num_vertices())
    for i, path in enumerate(paths):
        if not trust_own_predictions or known_labels[path[0]] == -np.inf:
            budget[i] += 1
            known_labels[path[0]] = y[path[0]]
        if not trust_own_predictions or known_labels[path[-1]] == -np.inf:
            budget[i] += 1
            known_labels[path[-1]] = y[path[-1]]

        if known_labels[path[0]] == known_labels[path[-1]]:
            known_labels[path] = known_labels[path[0]]
        else:
            mid, label_budget = binarySearch(y[path], 0, len(path)-1, known_labels[path[0]], known_labels[path])
            budget[i] += label_budget
            known_labels[path[0:mid+1]] = known_labels[path[0]]
            known_labels[path[mid+1:]] = known_labels[path[-1]]


    return known_labels, budget


def spc_querying_naive_one_convex(g : graph_tool.Graph, paths, y, convex_label, epsilon=0.5, weight=None, binary_search=False,closed_interval=False):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    print("epsilon", epsilon)
    known_labels = -np.ones(g.num_vertices())*np.inf
    budget = np.zeros(g.num_vertices())

    non_convex_label = np.unique(y)
    non_convex_label = non_convex_label[int(np.where(non_convex_label==convex_label)[0]+1)%2]
    for i, full_path in enumerate(paths):

        if np.any(known_labels[full_path] == convex_label):
            smallest = np.min(np.where(known_labels[full_path] == convex_label)[0])
            biggest = np.max(np.where(known_labels[full_path] == convex_label)[0])

            if np.any(known_labels[full_path[:smallest]] == non_convex_label):
                known_labels[full_path[:np.max(np.where(known_labels[full_path[:smallest]] == non_convex_label)[0])]] = non_convex_label

            if np.any(known_labels[full_path[biggest:]] == non_convex_label):
                known_labels[full_path[np.min(np.where(known_labels[full_path[biggest:]] == non_convex_label)[0]):]] = non_convex_label

        path = np.array(full_path)[known_labels[full_path] == -np.inf]

        for z in range(1,int(np.ceil(1/epsilon))):
            j = int(z*(np.ceil(epsilon*len(path))))
            while j < len(path) and known_labels[path[j]] != -np.inf:
                j += 1
            if j >= len(path):
                break

            if np.sum(np.where(known_labels==-np.inf)[0]) <= epsilon*len(path):
                conv_region = np.where(known_labels[path] == convex_label)[0]
                if conv_region.size > 0:
                    known_labels[path] = known_labels[path[0]]
                    known_labels[np.min(conv_region):np.max(conv_region)+1] = convex_label
                break

            known_labels[path[j]] = y[path[j]]
            budget[i] += 1

        if np.any(known_labels[path] == convex_label):
            smallest = np.min(np.where(known_labels[path] == convex_label)[0])
            biggest = np.max(np.where(known_labels[path] == convex_label)[0])
            if binary_search:
                l_path = path[:smallest+1]
                if known_labels[l_path[0]] == -np.inf:
                    known_labels[l_path[0]] = y[l_path[0]]
                    budget[i] += 1
                label_budget, new_labels = binarySearch(y[l_path], 0, len(l_path) - 1, known_labels[l_path[0]], known_labels[l_path])
                known_labels[l_path] = new_labels
                budget[i] += label_budget

                r_path = path[biggest:]
                if known_labels[r_path[-1]] == -np.inf:
                    known_labels[r_path[-1]] = y[r_path[-1]]
                    budget[i] += 1
                label_budget, new_labels = binarySearch(y[r_path], 0, len(r_path) - 1, known_labels[r_path[0]], known_labels[r_path])
                known_labels[r_path] = new_labels
                budget[i] += label_budget
            else:
                j_minus = smallest -1
                while j_minus > 0 and known_labels[path[j_minus]] == -np.inf:
                    j_minus -= 1
                j_plus = biggest+ 1
                while j_plus < len(path) and known_labels[path[j_plus]] == -np.inf:
                    j_plus += 1

                if known_labels[path[j_minus + (smallest - j_minus)//2]] == -np.inf:
                    known_labels[path[j_minus + (smallest - j_minus)//2]] = y[path[j_minus + (smallest - j_minus)//2]]
                    budget[i] += 1
                if known_labels[path[biggest + (j_plus - biggest) // 2]] == -np.inf:
                    known_labels[path[biggest + (j_plus - biggest) // 2]] = y[path[biggest + (j_plus - biggest) // 2]]
                    budget[i] += 1

                smallest = np.min(np.where(known_labels[path] == convex_label)[0])
                biggest = np.max(np.where(known_labels[path] == convex_label)[0])

                known_labels[path[smallest:biggest+1]] = convex_label

                if smallest > 0:
                    known_labels[path[:smallest-1]] = non_convex_label
                if biggest < len(path)-1:
                    known_labels[path[biggest+1:]] = non_convex_label
        else:
            known_labels[path] = non_convex_label

        convex_class = closure.compute_hull(g, np.where(known_labels == convex_label)[0], weight)
        known_labels[convex_class] = convex_label
    return known_labels, budget

def spc_querying_with_closure(g: graph_tool.Graph, paths, weights, y, whole_hull=True, exit_if_non_convex_partition=False):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    np.random.seed(55)
    #these two lines make repetitive closure computation a lot faster
    dist_map = graph_tool.topology.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices())).T
    comps, hist = graph_tool.topology.label_components(g)

    known_labels = -np.ones(g.num_vertices())
    num_of_known_labels = 0
    budget = 0

    pos_value, neg_value = np.unique(y)

    next_candidate_queues = [Queue() for _ in paths]
    left = np.zeros(len(paths), dtype=np.int)
    right = np.array([len(p)-1 for p in paths], dtype=np.int)
    queue_idxs = list(range(len(paths)))

    n = g.num_vertices()

    for i,path in enumerate(paths):
        next_candidate_queues[i].put(0)
        if len(path) > 1:
            next_candidate_queues[i].put(len(path)-1)

    starting_idx = np.random.choice(np.where(right>0)[0])
    starting_path = paths[starting_idx]

    budget += 2
    l = next_candidate_queues[starting_idx].get()
    r = next_candidate_queues[starting_idx].get()
    known_labels[starting_path[l]] = y[starting_path[l]]
    known_labels[starting_path[r]] = y[starting_path[r]]

    if known_labels[starting_path[0]] == known_labels[starting_path[-1]]:
        #color the hull of the path in the color of the endpoints
        path_closure = np.where(compute_hull(g, starting_path, weights, dist_map, comps, hist, whole_hull))[0]
        known_labels[path_closure] = known_labels[starting_path[0]]
        num_of_known_labels = len(path_closure)
        del queue_idxs[starting_idx]
    else:
        if (len(starting_path)>=3):
            next_candidate_queues[starting_idx].put(l + (r - l)//2)
        else:
            del queue_idxs[starting_idx]
        num_of_known_labels = 2

    pos = np.where(known_labels==pos_value)[0]
    neg = np.where(known_labels==neg_value)[0]

    candidates = np.zeros(len(paths), dtype=np.int)

    candidates[queue_idxs] = [next_candidate_queues[queue_idx].get() for queue_idx in queue_idxs] #this is always relative to the path

    candidate_pos_hulls = np.zeros((len(paths),n), dtype=np.bool)
    if len(pos) > 0:
        candidate_pos_hulls[queue_idxs] = [compute_hull(g, np.append(pos, paths[idx][candidates[idx]]), weights, dist_map, comps, hist, whole_hull) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_pos_hulls[idx][paths[idx][candidates[idx]]] = True
    candidate_neg_hulls = np.zeros((len(paths),n), dtype=np.bool)
    if len(neg) > 0:
        candidate_neg_hulls[queue_idxs] = [compute_hull(g, np.append(neg, paths[idx][candidates[idx]]), weights, dist_map, comps, hist, whole_hull) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_neg_hulls[idx][paths[idx][candidates[idx]]] = True
    pos_gains = np.zeros(len(paths))
    neg_gains = np.zeros(len(paths))

    while num_of_known_labels < n:
        to_remove = []
        changed = []
        for idx in queue_idxs:
            while known_labels[paths[idx][candidates[idx]]] >= 0:
                if not next_candidate_queues[idx].empty():
                    candidates[idx] = next_candidate_queues[idx].get()
                else:
                    maybe_remove = refill_queue_for_candidate(idx, candidates[idx], candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
                    if maybe_remove is not None:
                        to_remove.append(maybe_remove)
                        break
                    else:
                        candidates[idx] = next_candidate_queues[idx].get()
                changed.append(idx)

        for i in changed:
            candidate_pos_hulls[i] = compute_hull(g, np.append(pos, paths[i][candidates[i]]), weights, dist_map, comps, hist, whole_hull, already_closed=pos)
            candidate_neg_hulls[i] = compute_hull(g, np.append(neg, paths[i][candidates[i]]), weights, dist_map, comps, hist, whole_hull, already_closed=neg)

        for i in to_remove:
            queue_idxs.remove(i)
            if exit_if_non_convex_partition and np.sum(known_labels[paths[i]] >= 0) != len(paths[i]):
                exit(555)

        pos_gains[queue_idxs] = np.sum(candidate_pos_hulls[queue_idxs], axis=1) - len(pos)
        neg_gains[queue_idxs] = np.sum(candidate_neg_hulls[queue_idxs], axis=1) - len(neg)

        heuristic = np.average(np.array([pos_gains[queue_idxs], neg_gains[queue_idxs]]), axis=0)

        candidate_idx = queue_idxs[np.argmax(heuristic)]
        candidate_vertex = candidates[candidate_idx]

        if exit_if_non_convex_partition and known_labels[paths[candidate_idx][candidate_vertex]] == y[paths[candidate_idx][candidate_vertex]]:
            exit(9)
        known_labels[paths[candidate_idx][candidate_vertex]] = y[paths[candidate_idx][candidate_vertex]]

        budget += 1

        if known_labels[paths[candidate_idx][candidate_vertex]] == pos_value:
            pos =np.where(candidate_pos_hulls[candidate_idx])[0]
            known_labels[pos]  = pos_value
            #only recompute pos hulls, the negatives won't change
            candidate_pos_hulls[queue_idxs] = [compute_hull(g, np.append(pos, paths[idx][candidates[idx]]), weights, dist_map, comps, hist, whole_hull, already_closed=pos) for idx in queue_idxs]
        else:
            neg = np.where(candidate_neg_hulls[candidate_idx])[0]
            known_labels[neg] = neg_value
            # only recompute pos hulls, the negatives won't change
            candidate_neg_hulls[queue_idxs] = [compute_hull(g, np.append(neg, paths[idx][candidates[idx]]), weights, dist_map, comps, hist, whole_hull, already_closed=neg) for idx in queue_idxs]

        if next_candidate_queues[candidate_idx].empty():

            maybe_remove = refill_queue_for_candidate(candidate_idx, candidate_vertex, candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
            if maybe_remove is None:
                candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()
            else:
                queue_idxs.remove(candidate_idx)
        else:
            candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()

        candidate_pos_hulls[candidate_idx] = compute_hull(g, np.append(pos, paths[candidate_idx][candidates[candidate_idx]]), weights, dist_map, comps, hist, whole_hull, already_closed=pos)
        candidate_neg_hulls[candidate_idx] = compute_hull(g, np.append(neg, paths[candidate_idx][candidates[candidate_idx]]),weights, dist_map, comps, hist, whole_hull, already_closed=neg)

        pos = np.where(compute_hull(g, np.where(known_labels==pos_value)[0], weights, dist_map, comps, hist))[0]
        neg = np.where(compute_hull(g, np.where(known_labels==neg_value)[0], weights, dist_map, comps, hist))[0]

        num_of_known_labels = len(pos) + len(neg)

        print(num_of_known_labels, n)

    return known_labels, budget

def spc_querying_with_shadow(g: graph_tool.Graph, paths, weights, y):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''
    np.random.seed(55)
    #these two lines make repetitive closure computation a lot faster
    dist_map = graph_tool.topology.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices())).T
    comps, hist = graph_tool.topology.label_components(g)

    known_labels = -np.ones(g.num_vertices())
    num_of_known_labels = 0
    budget = 0

    pos_value, neg_value = np.unique(y)

    next_candidate_queues = [Queue() for _ in paths]
    left = np.zeros(len(paths), dtype=np.int)
    right = np.array([len(p)-1 for p in paths], dtype=np.int)
    queue_idxs = list(range(len(paths)))

    n = g.num_vertices()

    for i,path in enumerate(paths):
        next_candidate_queues[i].put(0)
        if len(path) > 1:
            next_candidate_queues[i].put(len(path)-1)

    starting_idx = np.random.choice(np.where(right>0)[0])
    starting_path = paths[starting_idx]

    budget += 2
    l = next_candidate_queues[starting_idx].get()
    r = next_candidate_queues[starting_idx].get()
    known_labels[starting_path[l]] = y[starting_path[l]]
    known_labels[starting_path[r]] = y[starting_path[r]]

    if known_labels[starting_path[0]] == known_labels[starting_path[-1]]:
        #color the hull of the path in the color of the endpoints
        path_closure = np.where(compute_hull(g, starting_path, weights, dist_map, comps, hist))[0]
        known_labels[path_closure] = known_labels[starting_path[0]]
        num_of_known_labels = len(path_closure)
        del queue_idxs[starting_idx]
    else:
        if (len(starting_path)>=3):
            next_candidate_queues[starting_idx].put(l + (r - l)//2)
        else:
            del queue_idxs[starting_idx]
        num_of_known_labels = 2

    pos = np.where(known_labels==pos_value)[0]
    neg = np.where(known_labels==neg_value)[0]

    candidates = np.zeros(len(paths), dtype=np.int)

    candidates[queue_idxs] = [next_candidate_queues[queue_idx].get() for queue_idx in queue_idxs] #this is always relative to the path

    candidate_pos_hulls = np.zeros((len(paths),n), dtype=np.bool)
    temp_pos_hulls =  np.zeros((n,n), dtype=np.bool)
    if len(pos) > 0:
        candidate_pos_hulls[queue_idxs] = [closure.compute_hull(g, np.append(pos, paths[idx][candidates[idx]]), weights, dist_map, comps, hist) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_pos_hulls[idx][paths[idx][candidates[idx]]] = True
    candidate_neg_hulls = np.zeros((len(paths),n), dtype=np.bool)
    temp_neg_hulls = np.zeros((n, n), dtype=np.bool)
    if len(neg) > 0:
        candidate_neg_hulls[queue_idxs] = [closure.compute_hull(g, np.append(neg, paths[idx][candidates[idx]]), weights, dist_map, comps, hist) for idx in queue_idxs]
    else:
        for idx in queue_idxs:
            candidate_neg_hulls[idx][paths[idx][candidates[idx]]] = True
    pos_gains = np.zeros(len(paths))
    neg_gains = np.zeros(len(paths))

    while num_of_known_labels < n:
        to_remove = []
        changed = []
        for idx in queue_idxs:
            while known_labels[paths[idx][candidates[idx]]] >= 0:
                if not next_candidate_queues[idx].empty():
                    candidates[idx] = next_candidate_queues[idx].get()
                else:
                    maybe_remove = refill_queue_for_candidate(idx, candidates[idx], candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
                    if maybe_remove is not None:
                        to_remove.append(maybe_remove)
                        break
                    else:
                        candidates[idx] = next_candidate_queues[idx].get()
                changed.append(idx)

        for i in range(n):
            temp_pos_hulls[i] = closure.compute_hull(g, np.append(pos, i), weights, dist_map, comps, hist, True, pos if len(pos) > 0 else None)
            temp_neg_hulls[i] = closure.compute_hull(g, np.append(neg, i), weights, dist_map, comps, hist, True, neg if len(neg) > 0 else None)

        for i in changed:
            candidate_pos_hulls[i] = closure.compute_shadow(g, np.append(pos, paths[i][candidates[i]]), neg, weights, dist_map, comps, hist, B_hulls=temp_neg_hulls)
            candidate_neg_hulls[i] = closure.compute_shadow(g, np.append(neg, paths[i][candidates[i]]), pos, weights, dist_map, comps, hist, B_hulls=temp_pos_hulls)

        for i in to_remove:
            queue_idxs.remove(i)
            if np.sum(known_labels[paths[i]] >= 0) != len(paths[i]):
                exit(555)

        pos_gains[queue_idxs] = np.sum(candidate_pos_hulls[queue_idxs], axis=1) - len(pos)
        neg_gains[queue_idxs] = np.sum(candidate_neg_hulls[queue_idxs], axis=1) - len(neg)

        heuristic = np.average(np.array([pos_gains[queue_idxs], neg_gains[queue_idxs]]), axis=0)

        candidate_idx = queue_idxs[np.argmax(heuristic)]
        candidate_vertex = candidates[candidate_idx]

        if known_labels[paths[candidate_idx][candidate_vertex]] == y[paths[candidate_idx][candidate_vertex]]:
            exit(9)
        known_labels[paths[candidate_idx][candidate_vertex]] = y[paths[candidate_idx][candidate_vertex]]

        budget += 1

        if known_labels[paths[candidate_idx][candidate_vertex]] == pos_value:
            pos =np.where(candidate_pos_hulls[candidate_idx])[0]
            known_labels[pos]  = pos_value
            #only recompute pos hulls, the negatives won't change
            candidate_pos_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(pos, paths[idx][candidates[idx]]), neg, weights, dist_map, comps, hist, temp_neg_hulls) for idx in queue_idxs]
            candidate_neg_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(neg, paths[idx][candidates[idx]]), pos, weights, dist_map, comps, hist, temp_pos_hulls) for idx in queue_idxs]

        else:
            neg =np.where(candidate_neg_hulls[candidate_idx])[0]
            known_labels[neg] = neg_value
            # only recompute pos hulls, the negatives won't change
            candidate_pos_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(pos, paths[idx][candidates[idx]]), neg, weights, dist_map, comps, hist, temp_neg_hulls) for idx in queue_idxs]

            candidate_neg_hulls[queue_idxs] = [closure.compute_shadow(g, np.append(neg, paths[idx][candidates[idx]]), pos, weights, dist_map, comps, hist, temp_pos_hulls) for idx in queue_idxs]

        if next_candidate_queues[candidate_idx].empty():

            maybe_remove = refill_queue_for_candidate(candidate_idx, candidate_vertex, candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right)
            if maybe_remove is None:
                candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()
            else:
                queue_idxs.remove(candidate_idx)
        else:
            candidates[candidate_idx] = next_candidate_queues[candidate_idx].get()

        candidate_pos_hulls[candidate_idx] = closure.compute_shadow(g, np.append(pos, paths[candidate_idx][candidates[candidate_idx]]), neg, weights, dist_map, comps, hist, temp_neg_hulls)
        candidate_neg_hulls[candidate_idx] = closure.compute_shadow(g, np.append(neg, paths[candidate_idx][candidates[candidate_idx]]), pos, weights, dist_map, comps, hist, temp_pos_hulls)

        #pos = np.where(known_labels==pos_value)[0]
        #neg = np.where(known_labels==neg_value)[0]
        pos = np.where(compute_hull(g, np.where(known_labels==pos_value)[0], weights, dist_map, comps, hist))[0]
        neg = np.where(compute_hull(g, np.where(known_labels==neg_value)[0], weights, dist_map, comps, hist))[0]
        num_of_known_labels = len(pos) + len(neg)

        print(num_of_known_labels, n)

    return known_labels, budget

def refill_queue_for_candidate(candidate_idx, candidate_vertex, candidates, known_labels, left, next_candidate_queues, paths, queue_idxs, right):
    l = left[candidate_idx]
    r = right[candidate_idx]
    if candidate_vertex != l and candidate_vertex != r:

        if known_labels[paths[candidate_idx][candidate_vertex]] == known_labels[paths[candidate_idx][l]]:
            left[candidate_idx] = candidate_vertex
        else:
            right[candidate_idx] = candidate_vertex
    mid = left[candidate_idx] + (right[candidate_idx] - left[candidate_idx]) // 2
    if mid != left[candidate_idx] and mid != right[candidate_idx]:
        next_candidate_queues[candidate_idx].put(mid)
        return None
    else:
        return candidate_idx

def binarySearchGenerator(known_labels, path, l, r, depth=0):
    if depth == 0:
        if bool(random.getrandbits(1)):
            yield l
            yield r
        else:
            yield r
            yield l
    while l < r:

        mid = l + (r - l) // 2
        if mid == l:
            break
        #if mid == l:
        #    return mid, label_budget

        # Check if x is present at mid
        if known_labels[path[mid]] == -np.inf:
            yield mid

        if known_labels[path[mid]] == known_labels[path[0]]:
            l = mid
        elif known_labels[path[mid]] == known_labels[path[-1]]:
            r = mid
        else:
            #new class --> recurse!
            binarySearchGenerator(known_labels, path, mid, r, depth+1)
            binarySearchGenerator(known_labels, path, l, mid, depth+1)
            break

def helper_sum_sizes(candidate_hulls, classes_hulls):
    arr = np.inf*np.ones(len(classes_hulls))

    for i, key in enumerate(candidate_hulls.keys()):
        arr[i] = (np.sum(candidate_hulls[key]))# - np.sum(classes_hulls[key]))

    return np.average(arr)

def budgeted_spc_querying(g : graph_tool.Graph, paths, y, weights=None, budget=50,  compute_hulls_between_queries=False, hull_as_optimization=False, use_adjacency=False):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''

    if use_adjacency:
        dist_map = graph_tool.topology.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices())).T

        adjacency = dist_map.copy()
        adjacency[adjacency > 1] = 0
    else:
        #to prevent overflow etc.
        dist_map = graph_tool.topology.shortest_distance(g, weights=weights).get_2d_array(
            range(g.num_vertices())).T.astype(np.double)
        dist_map[dist_map > g.num_vertices()] = np.inf

    #hack to allow both endpoints as candidates:
    #new_spc = paths.copy()
    #for p in paths:
    #    new_spc.append(p[::-1])

    #paths = new_spc

    comps, hist = graph_tool.topology.label_components(g)
    n = g.num_vertices()
    classes = np.unique(y)
    known_labels = -np.ones(g.num_vertices())*np.inf

    candidates = np.zeros(len(paths), dtype=np.int)
    candidate_generators = np.zeros(len(paths), dtype=np.object)
    for i, path in enumerate(paths):
        candidate_generators[i] = binarySearchGenerator(known_labels, path, 0, len(path)-1)
        candidates[i] = next(candidate_generators[i])

    candidate_hulls = np.zeros(len(paths), dtype=np.object)
    candidate_hull_sizes = np.zeros(len(paths))
    classes_hull_sizes = np.zeros(len(paths))
    known_classes = dict()
    classes_hulls = dict()

    deg = g.degree_property_map("total").a
    deg = deg*deg

    for j, candidate in enumerate(candidates):
        candidate_hulls[j] = dict()

    for c in classes:
        known_classes[c] = set()
        classes_hulls[c] = dict()
        for j, candidate in enumerate(candidates):
            temp = np.zeros(n, dtype=np.bool)
            classes_hulls[c] = temp.copy() #empty hulls
            temp[paths[j][candidate]] = True
            candidate_hulls[j][c] = temp #singleton hull
    for z in range(budget):
        #compute most promising vertex
        for p in range(len(paths)):
            if known_labels[paths[p][candidates[p]]] == -np.inf:
                candidate_hull_sizes[p] = helper_sum_sizes(candidate_hulls[p], classes_hulls)
            else:
                candidate_hull_sizes[p] = -1

        maximizers = np.where(candidate_hull_sizes == np.max(candidate_hull_sizes))[0]

        #prefer not queried paths
        if np.any(candidates[maximizers] == 0):
            maximizers = maximizers[np.where(candidates[maximizers] == 0)[0]]
            p_star = np.random.choice(maximizers)
        else:
            p_star = np.random.choice(maximizers)
        candidate = paths[p_star][candidates[p_star]]

        #query it
        known_labels[candidate] = y[candidate]

        #update data structures
        known_classes[known_labels[candidate]].add(candidate)
        classes_hulls[known_labels[candidate]] = candidate_hulls[p_star][known_labels[candidate]]



        for j in range(len(candidates)):
            path = paths[j]
            while known_labels[path[candidates[j]]] != -np.inf or path[candidates[j]] in classes_hulls[known_labels[candidate]]:
                try:
                    candidates[j] = next(candidate_generators[j])
                except StopIteration:
                    break
            #if not candidate_hulls[j][c][candidate]:
            #if not classes_hulls[c][path[candidates[j]]]:
                #classes_hulls_c_set = set(np.where(classes_hulls[c])[0])
                #old_hull_with_new_candidate = list(classes_hulls_c_set)
                #old_hull_with_new_candidate.append(path[candidates[j]])
            for c in classes:
                candidate_hulls[j][c] = compute_hull(g, list(known_classes[c].union([path[candidates[j]]])), weights, dist_map, comps, hist, hull_as_optimization)#, classes_hulls_c_set)

        '''if compute_hulls_between_queries:
            for c in classes:
                known_labels[np.where(compute_hull(g, np.where(known_labels == c)[0], weights, dist_map, comps, hist))[0]] = c'''

        if compute_hulls_between_queries:
            known_labels_augmented = known_labels.copy()
            known_classes_hulls_temp = np.zeros((n, len(classes)), dtype=np.bool)
            for i, c in enumerate(classes):
                known_classes_hulls_temp[:,i] = compute_hull(g, np.where(known_labels_augmented == c)[0], weights, dist_map, comps, hist, compute_closure=False)

            for i, c in enumerate(classes):
                only_c = known_classes_hulls_temp[:,i] & ~(np.sum(known_classes_hulls_temp[:,np.arange(len(classes))!=i],axis=1).astype(bool))
                known_labels_augmented[only_c] = c

        else:
            known_labels_augmented = known_labels

        if use_adjacency:
            prediction = label_propagation(adjacency, known_labels_augmented, y, use_adjacency=use_adjacency)
        else:
            prediction = label_propagation(dist_map, known_labels_augmented, y, use_adjacency=use_adjacency)
        print("======")
        print(z+1, np.sum(known_labels>-np.inf))
        print("accuracy", np.sum(prediction==y)/y.size)
        #print(known_classes)
        
    return known_labels


def budgeted_heuristic_querying(g: graph_tool.Graph, y, weights=None, budget=50, compute_hulls_between_queries=False,
                          hull_as_optimization=False, use_adjacency=False):
    '''

    :param g:
    :param paths: list of paths
    :param y: ground truth
    :param weight:
    :return:
    '''

    deg = g.degree_property_map("total").a
    #deg = deg*deg
    if use_adjacency:
        dist_map = graph_tool.topology.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices())).T

        adjacency = dist_map.copy()
        adjacency[adjacency > 1] = 0
    else:
        # to prevent overflow etc.
        dist_map = graph_tool.topology.shortest_distance(g, weights=weights).get_2d_array(
            range(g.num_vertices())).T.astype(np.double)
        dist_map[dist_map > g.num_vertices()] = np.inf

    # hack to allow both endpoints as candidates:
    # new_spc = paths.copy()
    # for p in paths:
    #    new_spc.append(p[::-1])

    # paths = new_spc

    comps, hist = graph_tool.topology.label_components(g)
    n = g.num_vertices()
    classes = np.unique(y)
    known_labels = -np.ones(g.num_vertices()) * np.inf

    candidate_hulls = np.zeros(n, dtype=np.object)
    candidate_hull_sizes = np.zeros(n)
    known_classes = dict()
    classes_hulls = dict()
    for j in range(n):
        candidate_hulls[j] = dict()

    for c in classes:
        known_classes[c] = set()
        classes_hulls[c] = dict()
        classes_hulls[c] = np.zeros(n, np.bool)
        for j in range(n):
            one_hot = np.zeros(n, dtype=np.bool)
            one_hot[j] = True
            candidate_hulls[j][c] = one_hot  # singleton hull
    for z in range(budget):
        # compute most promising vertex
        for p in range(n):
            if known_labels[p] == -np.inf:
                candidate_hull_sizes[p] = helper_sum_sizes(candidate_hulls[p], classes_hulls)
            else:
                candidate_hull_sizes[p] = -1

        maximizers = np.where(candidate_hull_sizes == np.max(candidate_hull_sizes))[0]


        #overlap of classes
        classes_hulls_overlap = np.sum(np.array([key_index_array[1] for key_index_array in classes_hulls.items()]), axis=0)
        #classes_hulls_overlap[classes_hulls_overlap<=1] = 0
        maximizers = maximizers[np.where(classes_hulls_overlap[maximizers] == np.min(classes_hulls_overlap[maximizers]))[0]]

        #maximizers = maximizers[np.where(deg[maximizers] == np.max(deg[maximizers]))[0]]

        p_star = np.random.choice(maximizers)

        # query it
        known_labels[p_star] = y[p_star]

        # update data structures
        known_classes[known_labels[p_star]].add(p_star)
        classes_hulls[known_labels[p_star]] = candidate_hulls[p_star][known_labels[p_star]]

        for j in range(n):

            if known_labels[j] == -np.inf:# and not classes_hulls[c][j]:
                # if not candidate_hulls[j][c][candidate]:
                # if not classes_hulls[c][path[candidates[j]]]:
                # classes_hulls_c_set = set(np.where(classes_hulls[c])[0])
                # old_hull_with_new_candidate = list(classes_hulls_c_set)
                # old_hull_with_new_candidate.append(path[candidates[j]])
                c = known_labels[p_star]
                candidate_hulls[j][c] = compute_hull(g, list(known_classes[c].union([j])), weights,
                                                        dist_map, comps, hist,
                                                         hull_as_optimization)  # , classes_hulls_c_set)


                test = np.zeros(n, dtype=np.bool)

                for p1 in list(known_classes[c].union([j])):
                    for p2 in list(known_classes[c].union([j])):
                        test[dist_map[p1,:]+ dist_map[:,p2] == dist_map[p1,p2]] = True



        '''if compute_hulls_between_queries:
            for c in classes:
                known_labels[np.where(compute_hull(g, np.where(known_labels == c)[0], weights, dist_map, comps, hist))[0]] = c'''

        if compute_hulls_between_queries:
            known_labels_augmented = known_labels.copy()
            known_classes_hulls_temp = np.zeros((n, len(classes)), dtype=np.bool)
            for i, c in enumerate(classes):
                known_classes_hulls_temp[:, i] = compute_hull(g, np.where(known_labels_augmented == c)[0], weights,
                                                              dist_map, comps, hist, compute_closure=False)

            for i, c in enumerate(classes):
                only_c = known_classes_hulls_temp[:, i] & ~(
                    np.sum(known_classes_hulls_temp[:, np.arange(len(classes)) != i], axis=1).astype(bool))
                known_labels_augmented[only_c] = c

        else:
            known_labels_augmented = known_labels

        if use_adjacency:
            prediction = label_propagation(adjacency, known_labels_augmented, y, use_adjacency=use_adjacency)
        else:
            prediction = label_propagation(dist_map, known_labels_augmented, y, use_adjacency=use_adjacency)
        print("=====")
        print(z + 1, np.sum(known_labels > -np.inf))
        print(np.sum(np.array([i[1] for i in list(classes_hulls.items())]),axis=1))
        print("accuracy", np.sum(prediction == y) / y.size)
        #print(known_classes)

    return known_labels

if __name__ == "__main__":
    np.random.seed(43)
    files = os.listdir("res/new_synthetic/")
    #files.sort()
    for filename in files:
        for label_idx in range(10):
            #if "500_2" not in filename and "500_3" not in filename and "500_4" not in filename:
            #    continue
            instance = filename.split(".")[0]
            print("======================================================")
            print("file", instance, "label", label_idx)
            edges = np.genfromtxt("res/new_synthetic/" + instance + ".csv", delimiter=",", dtype=np.int)[:, :2]
            n = np.max(edges)+1
            g = graph_tool.Graph(directed=False)
            g.add_vertex(n)
            g.add_edge_list(edges)
            weight_prop = g.new_edge_property("double", val=1)

            y = np.zeros(n)
            add_string = ""
            if label_idx > 4:
                add_string = "_simplicial_start"
            y[np.genfromtxt("res/new_synthetic/labels/" + instance + "_"+str(label_idx)+add_string+"_positive.csv", dtype=np.int)] = True

            #spc = shortest_path_cover_logn_apx(g, weight_prop)#pickle.load(open("res/new_synthetic/spc/" + instance + ".p", "rb"))
            spc = pickle.load(open("res/new_synthetic/spc/" + instance + ".p", "rb"))
            g.set_directed(False)

            a,b = spc_querying_naive(g, spc, y)
            #a,b =spc_querying_with_closure(g, spc, weight_prop,y )

            #print(a)
            #print(y)
            #if not np.all(a==y):
            #    exit(22)
            #print("shadow: #queries:",np.sum(b), b)
            budgeted_heuristic_querying(g, y,weight_prop, hull_as_optimization=True)

            print("random!!!")
            sample = []
            dist_map = graph_tool.topology.shortest_distance(g).get_2d_array(range(g.num_vertices())).T
            for i in range(20):
                sample.append(np.random.randint(0,n,))
                known_labels = -np.inf * np.ones(g.num_vertices())
                known_labels[sample] = y[sample]
                result = label_propagation(dist_map, known_labels, y)

                print(np.sum(result == y) / g.num_vertices())
            continue
            print(a)



            print(y)
            if not np.all(a == y):
                exit(22)
            print("#queries:", b)

            a, b = spc_querying_with_shadow(g, spc, weight_prop, y)
            print(a)
            print(y)
            if not np.all(a == y):
                exit(22)
            print("#queries:",np.sum(b), b)
            print("===========s2=================")
            a = s2(g, weight_prop, y, budget=b)
            print("s2 acc:", np.sum(a==y)/n)
