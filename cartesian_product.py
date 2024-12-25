import numpy as np

def get_pairwise_similarity(ei: int, vi: int, vj: int, w: np.array, h: np.array) -> float:
    """
    Function that calcutes pairwise similarity relationship
    
    Parameters:
        ei: index of hyperedge
        vi: index of first vertex
        vj: index of second vertex
        w: weigh matrix of hypergraph
        h: hypergraph
        
    Returns:
        float: which represents the pairwise relationship
    """
    
    return w[ei] * h[ei, vi] * h[ei, vj]

def get_similarity_measure(vi: int, vj: int, w: np.array, h: np.array) -> float:
    """
    Function that calculates similarity measure c(vi, vj)
    
    Parameters:
        vi: index of first vertex
        vj: index of second vertex
        w: weigh matrix of hypergraph
        h: hypergraph
        
    Returns:
        float: represents similarity measure c(vi, vj)
    """
    
    n = w.shape[0] # number of hyperedges
    sum = 0.0
    for i in range(n):
        sum += get_pairwise_similarity(i, vi, vj, w, h)
        
    return sum

def get_new_weight_matrix(w: np.array, h: np.array) -> np.array:
    """
    Function for getting new weight matrix given the weight matrix of the hypergraph and the hypergraph
    
    Parameters:
        w: weigh matrix of hypergraph
        h: hypergraph
        
    Returns:
        np.array: new weight matrix
    """
    
    n = h.shape[1] # number of vertices
    c = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            c[i, j] = get_similarity_measure(i, j, w, h)
            
    return c * c