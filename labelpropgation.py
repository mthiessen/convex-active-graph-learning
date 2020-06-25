import numpy as np


def label_propagation(W, known_labels, labels, iterations=10, delta=0.00001, use_adjacency=False):
    if not use_adjacency:
        W = np.exp(-W * W / 2) #similarity
    labels = np.unique(labels)
    Y = np.zeros((W.shape[0],labels.size))

    given_idx = np.where(known_labels > -np.inf)[0]

    #build one-hot label matrix
    for i,label in enumerate(labels):
        Y[known_labels == label,i] = 1

    D = np.sum(W, axis=1)

    eps = min(np.min(D), 0.000001)

    D_inverse_dot_W = 1/(D)[:,np.newaxis]*W

    oldY = np.ones((Y.shape[0], Y.shape[1]))
    i = 0
    while (np.abs(oldY - Y) > delta).any() and i <= iterations:
        oldY = Y
        Y = np.dot(D_inverse_dot_W, Y)
        Y[given_idx] = oldY[given_idx]
        i += 1


    # uniform argmax
    #for i in range(Y.shape[0]):
    #    result[i] = np.random.choice(np.flatnonzero(Y[i] == Y[i].max()))
    #maybe for the future
    #Y[not_given_idx] = np.dot(np.dot(np.linalg.inv(np.eye(len(not_given_idx)) - D_inverse_dot_W[not_given_idx][:,not_given_idx]), D_inverse_dot_W[not_given_idx][:,given_idx]), Y[given_idx])

    result = labels[np.argmax(Y, axis=1)]

    return result