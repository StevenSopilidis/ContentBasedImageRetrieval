import numpy as np
from rank_normalization import find_rank
import math

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

        for j in range(n):
            neighbors_j = sorted(enumerate(normalized_ranks[j]), key=lambda x: x[1])[:k]

            if find_rank(neighbors_i, j) == -math.inf: # check if v_j does not belong ot e_i
                continue

            # calculate r(e_i, v_j)
            for x in neighbors_j:
                

    print(incidence_matrix)

    return incidence_matrix
