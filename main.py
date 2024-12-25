from utils import load_image, FeatureExtractor
from rank_normalization import get_ranked_lists, rank_normalization
from hypergraph import compute_hyperedge_weights, compute_membership_measure, construct_incidence_matrix
from cartesian_product import get_new_weight_matrix
import os
import numpy as np

directory_path = "./data"
k = 2
data = os.listdir(directory_path)
image_files = [os.path.join(directory_path, f) for f in data if os.path.isfile(os.path.join(directory_path, f))]
images = [load_image(file) for file in image_files]

L = len(images)

extractor = FeatureExtractor()
dataset_features = np.array([extractor.extract(image) for image in images])

query_image = load_image("./query.jpg")
query_features = extractor.extract(query_image)

# A) RANK NORMALIZATION
ranked_lists = get_ranked_lists(query_features, dataset_features)
normalized_scores = rank_normalization(ranked_lists, L)

# Construct hypergraphy
membership = compute_membership_measure(ranked_lists, k) # membership r(ei, vj) between edges and vertices
hypergraph = construct_incidence_matrix(membership, len(dataset_features) + 1) # constructed hypergraph
hyperedge_weights = compute_hyperedge_weights(hypergraph, k) # W

# # C) Hyperedges similarities
Sh = hypergraph @ hypergraph.T
Sv = hypergraph.T @ hypergraph
S = Sh * Sv

# # D) Cartesian Product of Hyperedge elements
w = get_new_weight_matrix(hyperedge_weights, hypergraph)

print(w)