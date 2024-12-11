import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def get_ranked_lists(query_features: np.array, dataset_features: np.array) -> dict:
    """
    Function that calculates the ranked lists t_i for every datapoint and also
    for the query object t_q

    Parameters:
        query_features: features of query object
        dataset_features: features of dataset objects

    Returns:
        dict: dictionary where the key is the index of the object and the value is the ranked list
        and the element where key == L is the ranked list of the query object
    """
    query_object_similarity_scores = euclidean_distances([query_features], dataset_features)[0]

    # Assign Ranks Based on Similarity
    ranked_indices = np.argsort(query_object_similarity_scores)[::-1]  # Sort by descending similarity
    ranked_list = [(idx, rank + 1) for rank, idx in enumerate(ranked_indices)]

    # calculate t_i for every iε[L]
    ranked_lists = {}
    for query_idx in range(len(dataset_features)):
        scores = euclidean_distances([dataset_features[query_idx]], dataset_features)[0] # calculate scores
        sorted_indices = np.argsort(scores)[::-1]  # sort them in descending order
        ranked_lists[query_idx] = [(idx, rank + 1) for rank, idx in enumerate(sorted_indices)]

    # add query object ranked list to the dict
    ranked_lists[len(dataset_features)] = ranked_list

    return ranked_lists


def find_rank(ranked_list: np.array, target_idx: int) -> int:
    """
    Function that finds rank of specified object inside provided ranked list

    Parameters:
        ranked_list: ranked list t_i to search target_idx in 
        target_idx: index of object whose rank we are trying to find

    Returns:
        int: rank of the object inside the ranked_list
    """

    for idx, rank in ranked_list:
        if idx == target_idx:
            return rank
        
    return len(ranked_list)

def rank_normalization(ranked_lists: dict, L: int) -> dict:
    """
    Perform reciprocal rank normalization on a set of ranked lists using the formula
    P_n(i,j) = 2*L - (t_i(j) + t_j[i])

    Parameters:
        ranked_lists: dict where the key is the index of the element and the value
                      is the ranked_list t_i. Each element of the ranked_list t_i is
                      a tuple where the first element is the index of the object and the second
                      is the rank of that object
        L: The maximum size of ranked lists to consider

    Returns:
        dict: normalized similarity scores
    """
    normalized_similarity_scores = {}

    for idx, ranked_list in ranked_lists.items():
        normalized_similarity_scores[idx] = {}
        normalized_similarity_scores[idx] = []
        for item_idx, rank in ranked_list:
            # 2*L - (t_idx(item_idx) + t_item_idx(idx))
            reciprocal_rank = 2*L - (rank + find_rank(ranked_lists[item_idx], idx))
            normalized_similarity_scores[idx].append((item_idx, reciprocal_rank))
    
    return normalized_similarity_scores