from utils import load_image, FeatureExtractor
from rank_normalization import get_ranked_lists, rank_normalization
from hypergraph import construct_hypergraph
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

# B) Hypergraph construction
hypergraph, w = construct_hypergraph(normalized_scores, k)

# C) Hyperedges similarities
Sh = hypergraph @ hypergraph.T
Sv = hypergraph.T @ hypergraph
S = Sh * Sv

# D) Cartesian Product of Hyperedge elements
w = get_new_weight_matrix(w, hypergraph)