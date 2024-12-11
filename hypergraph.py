import numpy as np
from rank_normalization import find_rank

def construct_hypergraph(normalized_ranks: dict, k: int) -> np.array:
    """
    Function that constructs hypergraph 

    Parameters:
        normalized_ranks: reciprocal rank normalized ranked_lists of objects
        k: number of neighbors in the graph

    Returns:
        np.array: numpy array that represents the hypergraph
    """

    n = len(normalized_ranks.keys())
    incidence_matrix = np.zeros((n, n))

    for i in range(n):
        neighbors_i = sorted(enumerate(normalized_ranks[i]), key=lambda x: x[1])[:k]

        for j in neighbors_i:
            neighbors_j = sorted(enumerate(normalized_ranks[j[0]]), key=lambda x: x[1])[:k]

            for x in neighbors_j:
                wp_ij = 1 - np.log(find_rank(normalized_ranks[i], j[0]) + 1) / np.log(k)
                wp_jx = 1 - np.log(find_rank(normalized_ranks[j[0]], x[0]) + 1) / np.log(k)

                incidence_matrix[i, x[0]] += wp_ij * wp_jx

    return incidence_matrix
