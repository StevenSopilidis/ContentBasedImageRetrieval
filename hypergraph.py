import numpy as np
from rank_normalization import find_rank
import math

def compute_membership_measure(ranked_lists, k):
    """
    Compute the membership measure r(ei, vj) for each vertex vj and hyperedge ei.

    Parameters:
        ranked_lists: Dict containing ranked lists for each object.
        k: Number of neighbors to consider.

    Returns:
        dict: Membership measure values for each hyperedge and vertex.
    """
    membership = {}
    for i, ranked_list_i in ranked_lists.items():
        membership[i] = {}
        neighbors_i = [x[0] for x in ranked_list_i[:k]]
        for j in range(len(ranked_lists)):
            if j == i:
                continue
            membership[i][j] = 0
            neighbors_j = [x[0] for x in ranked_lists[j][:k]]
            for x in neighbors_i:
                for y in neighbors_j:
                    wp_i_x = 1 - np.log2(1 + next(rank for obj, rank in ranked_list_i if obj == x))
                    wp_x_j = 1 - np.log2(1 + next(rank for obj, rank in ranked_lists[x] if obj == y))
                    membership[i][j] += wp_i_x * wp_x_j
    return membership

def construct_incidence_matrix(membership, num_vertices):
    """
    Construct the incidence matrix H for the hypergraph.

    Parameters:
        membership: Membership measures for hyperedges and vertices.
        num_vertices: Total number of vertices.

    Returns:
        np.array: The incidence matrix H.
    """
    num_hyperedges = len(membership)
    H = np.zeros((num_hyperedges, num_vertices))
    for i, members in membership.items():
        for j, value in members.items():
            H[i][j] = value
    return H

def compute_hyperedge_weights(H, k):
    """
    Compute the weights of hyperedges based on the incidence matrix.

    Parameters:
        H: Incidence matrix.
        k: Number of neighbors to consider.

    Returns:
        np.array: Weights for each hyperedge.
    """
    weights = np.zeros(H.shape[0])
    for i in range(H.shape[0]):
        neighbors = np.argsort(-H[i])[:k]
        weights[i] = np.sum(H[i][neighbors])
    return weights