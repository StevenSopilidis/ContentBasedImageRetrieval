from utils import FeatureExtractor, load_images, load_image
from rank_normalization import get_ranked_lists, rank_normalization
from hypergraph import compute_hyperedge_weights, compute_membership_measure, construct_incidence_matrix
from cartesian_product import get_new_weight_matrix
import os
import numpy as np

directory_path = "./data"
k = 12
T = 1

images = load_images(directory_path)

L = len(images)

extractor = FeatureExtractor()
dataset_features = np.array([extractor.extract(image) for image in images])

print("----> Extracted features from dataset")

query_image = load_image("./query.jpg")
query_features = extractor.extract(query_image)

# A) RANK NORMALIZATION
indices = np.arange(len(dataset_features))

ranked_lists = get_ranked_lists(query_features, dataset_features)



for t in range(T):
    print(f"----> Iteration {t}/{T}")

    # normalize 
    normalized_scores = rank_normalization(ranked_lists, L)
    print("Scores normalized")

    # Construct hypergraphy
    membership = compute_membership_measure(ranked_lists, k) # membership r(ei, vj) between edges and vertices
    hypergraph = construct_incidence_matrix(membership, len(dataset_features) + 1) # constructed hypergraph
    hyperedge_weights = compute_hyperedge_weights(hypergraph, k) # W
    print("Hypergraph constructed")

    # Hypergraph-based Similarities
    Sh = hypergraph @ hypergraph.T
    Sv = hypergraph.T @ hypergraph
    S = Sh * Sv  # Element-wise product
    print("Calculated element-wise product")


    # Cartesian Product-Based Similarity
    C = get_new_weight_matrix(hyperedge_weights, hypergraph)
    print("Calculated cartesian product-based product")


    # Compute Affinity Matrix
    W = C * S
    print("Calculated affinity matrix")


    # Generate New Ranked Lists
    new_ranked_lists_dict = {}
    for i in range(len(dataset_features) + 1):
        ranked_indices = np.argsort(-W[i])[:k]
        new_ranked_lists_dict[i] = [(idx, W[i][idx]) for idx in ranked_indices]

    ranked_lists_dict = new_ranked_lists_dict


# get the ranked list of last element (aka our query image)
query_image_key = max(ranked_lists_dict.keys())
ranked_list = ranked_lists_dict[query_image_key]

for i, _ in ranked_list:
    image = images[i]
    image.show()