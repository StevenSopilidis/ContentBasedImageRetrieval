from utils import load_image, FeatureExtractor
from rank_normalization import get_ranked_lists, rank_normalization
from hypergraph import compute_hyperedge_weights, compute_membership_measure, construct_incidence_matrix
from cartesian_product import get_new_weight_matrix
import os
import numpy as np

directory_path = "./data"
k = 2
T = 20
data = os.listdir(directory_path)
image_files = [os.path.join(directory_path, f) for f in data if os.path.isfile(os.path.join(directory_path, f))]
images = [load_image(file) for file in image_files]

L = len(images)

extractor = FeatureExtractor()
dataset_features = np.array([extractor.extract(image) for image in images])

query_image = load_image("./query.jpg")
query_features = extractor.extract(query_image)

# A) RANK NORMALIZATION
indices = np.arange(len(dataset_features))

ranked_lists = get_ranked_lists(query_features, dataset_features)
normalized_scores = rank_normalization(ranked_lists, L)



for t in range(T):
    # Construct hypergraphy
    membership = compute_membership_measure(ranked_lists, k) # membership r(ei, vj) between edges and vertices
    hypergraph = construct_incidence_matrix(membership, len(dataset_features) + 1) # constructed hypergraph
    hyperedge_weights = compute_hyperedge_weights(hypergraph, k) # W

    # Hypergraph-based Similarities
    Sh = hypergraph @ hypergraph.T
    Sv = hypergraph.T @ hypergraph
    S = Sh * Sv  # Element-wise product

    # Cartesian Product-Based Similarity
    C = get_new_weight_matrix(hyperedge_weights, hypergraph)

    # Compute Affinity Matrix
    W = C * S

    # Generate New Ranked Lists
    new_ranked_lists_dict = {}
    for i in range(len(dataset_features)):
        ranked_indices = np.argsort(-W[i])[:k]
        new_ranked_lists_dict[i] = [(idx, W[i][idx]) for idx in ranked_indices]

    ranked_lists_dict = new_ranked_lists_dict

print(ranked_lists_dict)