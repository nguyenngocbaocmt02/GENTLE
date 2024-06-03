import numpy as np
import torch
from igraph import Graph
import igraph
import torch.nn as nn
def prim(graph):
    """
    Implementation of Prim's algorithm using igraph.

    Parameters:
    - graph: A weighted, connected, undirected graph in the form of an adjacency matrix.

    Returns:
    - mst_edges: List of edges in the minimum spanning tree.
    - mst_weight: Total weight of the minimum spanning tree.
    """
    g = Graph.Weighted_Adjacency(graph.tolist(), mode=igraph.ADJ_UPPER, attr="weight", loops=False)
    mst = g.spanning_tree(weights=g.es["weight"], return_tree=True)
    print(mst)
    mst_edges = [(e.source, e.target, e["weight"]) for e in mst.es]
    mst_weight = sum(e["weight"] for e in mst.es)

    return mst_edges, mst_weight

class LinearInterpolationModel(nn.Module):
    def __init__(self, cdf_values, y_values):
        super(LinearInterpolationModel, self).__init__()
        self.register_buffer('cdf_values', torch.tensor(cdf_values, dtype=torch.float32))
        self.register_buffer('y_values', torch.tensor(y_values, dtype=torch.float32))
        self.range_len = len(y_values) - 2

    def forward(self, x):
        x0 = torch.searchsorted(self.y_values, x, right=True).clamp(0, self.range_len)
        x1 = x0 + 1

        y0 = self.cdf_values[x0]
        y1 = self.cdf_values[x1]

        delta_y_values = self.y_values[x1] - self.y_values[x0]
        interpolated_values = y0 + (x - self.y_values[x0]) * (y1 - y0) / delta_y_values

        return torch.clamp(interpolated_values, 0.0, 1.0)

def preprocess_dataset(x, y):
    preprocessed_x = []
    preprocessed_y = []

    grouped_by_x = {}
    for i in range(len(x)):
        current_x = tuple(x[i])
        if current_x not in grouped_by_x:
            grouped_by_x[current_x] = []
        grouped_by_x[current_x].append(y[i])

    for current_x, y_values in grouped_by_x.items():
        num_samples = len(y_values)
        sorted_indices = np.argsort(y_values)
        sorted_uniform_values = np.sort(np.random.rand(num_samples))
        #if num_samples == 47:
        #    print(num_samples)
        #    np.savez('preprocessed_data.npz', check=sorted_uniform_values)
        for i, index in enumerate(sorted_indices):
            new_x = np.concatenate(([sorted_uniform_values[i]], current_x))
            preprocessed_x.append(new_x)
            preprocessed_y.append(y_values[index])
            #    if num_samples == 47:
            #    preprocessed_x.append(new_x)
            #    preprocessed_y.append(y_values[index])
    return np.array(preprocessed_x), np.array(preprocessed_y)
