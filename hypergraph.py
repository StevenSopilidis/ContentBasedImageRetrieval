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

    for i in range(n): # iterate through hyper-edges
        neighbors_i = sorted(normalized_ranks[i], key=lambda x: x[1])[:k]

        for j in range(n): # iterate through nodes

            # calculate r(e_i, v_j)
            for o_x in neighbors_i:
                neighbors_ox = sorted(normalized_ranks[o_x[0]], key=lambda x: x[1])[:k]
                
                for o_j in neighbors_ox:
                    t_ix = find_rank(neighbors_i, o_x[0])
                    t_xj = find_rank(neighbors_ox, o_j[0])
                    
                    wp_ix = 1 - math.log(t_ix + 1) / math.log(k)
                    wp_xj = 1 - math.log(t_xj + 1) / math.log(k)

                    incidence_matrix[i, j] = wp_ix * wp_xj
                    
    print(incidence_matrix)

    return incidence_matrix
