from absl import app, flags

import gzip
import pickle

import scipy.io as scio
import numpy as np
import networkx as nx


flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('resolution', 1.0, 'Resolution for Louvain algorithm.')
flags.DEFINE_integer('cluster_k', 15, 'Cluster k for asyn_fluidc.')
FLAGS = flags.FLAGS


def load_topology(mat_filename: str) -> np.ndarray:
    """

    :param mat_filename: matlab file
    :return: adjacency matrix
    """

    mat_graph = scio.loadmat(mat_filename)
    topology = mat_graph['A']
    print(f'shape of graph: {topology.shape}.')
    return topology


def find_components(topology: np.ndarray) -> np.ndarray:
    g = nx.from_numpy_array(topology, create_using=nx.Graph)
    gcc = np.array(list(max(nx.connected_components(g), key=len)))
    g.remove_nodes_from(np.setdiff1d(np.arange(topology.shape[0]), gcc))
    g = nx.convert_node_labels_to_integers(g)
    topology = nx.to_numpy_array(g)
    print(f'shape of graph after reserving connected components: {topology.shape}.')
    return topology


file_map = {
    'TECB': 'data/real/pruned_TECB.mat',
    'TYB': 'data/real/pruned_TYB.mat',
    'Human': 'data/real/PPI_Human.mat',
    'Yeast': 'data/real/PPI_Yeast.mat',
    'Brain': 'data/real/Brain.mat',
}


dynamic_map = {
    'gene': ['TECB', 'TYB', 'Human', 'Yeast'],
    'neuron': ['Brain'],
}


def main(_):
    print('-' * 80)
    topology_map = {}
    for k, v in file_map.items():
        print(f'Generate graph for {k}:')
        topology = load_topology(v)
        a = 1.0
        topology = a * topology
        topology = find_components(topology)
        topology_map[k] = topology
        print(topology.shape)

        print('-' * 80)

    for k, v in dynamic_map.items():
        print(f'Save graph for {k}: {v}')
        graph = {}
        clustered_graph = {}
        for k_ in v:
            topology_ = topology_map[k_]
            graph[k_] = topology_
            if k != 'neuron':
                cluster_k = max(int(topology_.shape[0]/FLAGS.cluster_k), 1)
                communities = nx.community.asyn_fluidc(nx.from_numpy_array(topology_), k=cluster_k, seed=FLAGS.seed)
            else:
                communities = nx.community.louvain_communities(nx.from_numpy_array(topology_),
                                                               resolution=FLAGS.resolution, seed=FLAGS.seed)
            # sort communities by size
            communities = sorted(communities, key=lambda x: len(x))
            print(f'{k_}')
            print(f'community size: {[len(c) for c in communities]}')
            for i, c in enumerate(communities):
                subgraph = nx.subgraph(nx.from_numpy_array(topology_), c)
                clustered_graph[f'{k_}_{i}'] = nx.to_numpy_array(subgraph)

        save_filename = f'data/real/real_{k}.gz'
        with gzip.open(save_filename, 'wb') as f:
            pickle.dump(graph, f)

        save_filename = f'data/real/real_clustered_{k}_large.gz'
        with gzip.open(save_filename, 'wb') as f:
            pickle.dump(clustered_graph, f)


if __name__ == '__main__':
    app.run(main)
