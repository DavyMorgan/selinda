import networkx as nx
import numpy as np
from absl import logging
from absl.flags import FLAGS
from networkx.utils import powerlaw_sequence

from tqdm.rich import trange


def check_graph(g: nx.Graph, seed: int, seed_step: int, graph_type: str) -> (bool, int):
    if nx.is_connected(g):
        success = True
    else:
        print(f'{graph_type} graph generation failed with seed {seed}')
        seed += seed_step
        success = False
    return success, seed


def generate_graph(graph_type: str, n: int, avg_degree: int,
                   mean_edge_weight: float, gamma: float, nu: float, seed: int,
                   seed_step: int = 1000) -> np.ndarray:
    """
    Generate a graph with the given parameters.
    :param graph_type: type of graph
    :param n: number of nodes
    :param avg_degree: average degree of the graph
    :param mean_edge_weight: weight of each edge
    :param gamma: power law exponent for degree
    :param nu: power law exponent for edge weight
    :param seed: random seed
    :param seed_step: step size for random seed

    :return: graph adjacency matrix
    """
    if graph_type.startswith('ER'):
        m = int(n * avg_degree / 2)
        success = False
        while not success:
            g = nx.gnm_random_graph(n, m, seed=seed, directed=False)
            success, seed = check_graph(g, seed, seed_step, 'ER')
    elif graph_type.startswith('SF'):
        success = False
        while not success:
            degree_sequence = np.array(powerlaw_sequence(n, gamma, seed=seed))
            degree_sequence = avg_degree * degree_sequence / degree_sequence.mean()
            degree_sequence = np.array([max(int(round(d)), 0) for d in degree_sequence])
            if degree_sequence.sum() % 2 == 1:
                degree_sequence[np.argmax(degree_sequence)] += 1
            g = nx.configuration_model(degree_sequence, seed=seed)
            g = nx.Graph(g)
            g.remove_edges_from(nx.selfloop_edges(g))
            success, seed = check_graph(g, seed, seed_step, 'SF')
    elif graph_type.startswith('BA'):
        m = int(avg_degree / 2)
        g = nx.barabasi_albert_graph(n, m, seed=seed)
    elif graph_type.startswith('COMMUNITY'):
        success = False
        while not success:
            prob_out = avg_degree / n / 5
            prob_in = 10 * prob_out
            g = nx.random_partition_graph([n // 2, n - n // 2], prob_in, prob_out, seed=seed)
            success, seed = check_graph(g, seed, seed_step, 'COMMUNITY')
    elif graph_type.startswith('SMALL_WORLD'):
        g = nx.connected_watts_strogatz_graph(n, avg_degree, 0.4, tries=1000, seed=seed)
    else:
        raise ValueError(f'Unknown graph type {graph_type}')

    if graph_type.endswith('-uniform'):
        edge_weight = {e: mean_edge_weight for e in g.edges}
    elif graph_type.endswith('-power'):
        num_edges = len(g.edges)
        edge_weight_sequence = np.array(powerlaw_sequence(num_edges, nu, seed=seed + 1))
        edge_weight_sequence = mean_edge_weight * edge_weight_sequence / edge_weight_sequence.mean()
        edge_weight = dict(zip(g.edges, edge_weight_sequence))
    else:
        raise ValueError(f'Unknown graph type {graph_type}')
    nx.set_edge_attributes(g, edge_weight, 'weight')
    return nx.to_numpy_array(g, dtype=np.float64, weight='weight')


def generate_multiple_graphs(graphs, graph_type, n, avg_degree, edge_weight, gamma, nu, assortative=False):
    num_graphs = FLAGS.num_graphs
    logging.info('\n' + '=' * 80)
    logging.info(f'Generating {num_graphs} {graph_type} graphs '
                 f'with {n} nodes, {avg_degree} average degree, '
                 f'{edge_weight} edge weight, {gamma} gamma, {nu} nu, and seed {FLAGS.seed}.\n')
    for i in trange(num_graphs):
        seed = FLAGS.seed + i
        graph = generate_graph(graph_type, n, avg_degree, edge_weight, gamma, nu, seed)
        key = f'{graph_type}_node-{n}_degree-{avg_degree}_' \
              f'weight-{edge_weight}_gamma-{gamma}_nu-{nu}' \
              f'_seed-{seed}'
        graphs[key] = graph

    return graphs


