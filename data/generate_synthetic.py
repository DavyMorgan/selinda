import gzip
import pickle

from absl import app, flags

from data.utils import generate_multiple_graphs

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_graphs', 10, 'Number of graphs to generate.')
flags.DEFINE_bool('generate_n', False, 'Generate graphs of different scales.')
flags.DEFINE_bool('generate_ba_n', False, 'Generate BA graphs of different scales.')
flags.DEFINE_bool('generate_community_n', False, 'Generate graphs with communities of different scales.')
flags.DEFINE_bool('generate_small_world_n', False, 'Generate small world graphs of different scales.')
flags.DEFINE_bool('generate_case', False, 'Generate er graphs for case study.')
FLAGS = flags.FLAGS


def main(_):
    # generate graphs of different scales
    if FLAGS.generate_n:
        graph_types = ['ER-uniform', 'SF-uniform']
        edge_weight = 1.0
        nu = 2.0
        avg_degree = 6
        gamma = 2.0
        n = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        graphs = dict()
        for graph_type in graph_types:
            for n_ in n:
                graphs = generate_multiple_graphs(graphs, graph_type, n_, avg_degree, edge_weight, gamma, nu)
        filename = 'synthetic/synthetic_n_degree-6.gz'
        with gzip.open(f'data/{filename}', 'wb') as f:
            pickle.dump(graphs, f)

    # generate BA graphs of different scales
    if FLAGS.generate_ba_n:
        graph_types = ['BA-uniform']
        edge_weight = 1.0
        nu = None
        avg_degree = 6
        gamma = None
        n = [80, 100, 120, 140, 160, 180, 200]
        graphs = dict()
        for graph_type in graph_types:
            for n_ in n:
                graphs = generate_multiple_graphs(graphs, graph_type, n_, avg_degree, edge_weight, gamma, nu)
        filename = 'synthetic/synthetic_n_ba.gz'
        with gzip.open(f'data/{filename}', 'wb') as f:
            pickle.dump(graphs, f)

    # generate graphs with community structure of different scales
    if FLAGS.generate_community_n:
        graph_types = ['COMMUNITY-uniform']
        edge_weight = 1.0
        nu = None
        avg_degree = 6
        gamma = None
        n = [80, 100, 120, 140, 160, 180, 200]
        graphs = dict()
        for graph_type in graph_types:
            for n_ in n:
                graphs = generate_multiple_graphs(graphs, graph_type, n_, avg_degree, edge_weight, gamma, nu)
        filename = 'synthetic/synthetic_n_community.gz'
        with gzip.open(f'data/{filename}', 'wb') as f:
            pickle.dump(graphs, f)

    # generate small-world graphs of different scales
    if FLAGS.generate_small_world_n:
        graph_types = ['SMALL_WORLD-uniform']
        edge_weight = 1.0
        nu = None
        avg_degree = 6
        gamma = None
        n = [80, 100, 120, 140, 160, 180, 200]
        graphs = dict()
        for graph_type in graph_types:
            for n_ in n:
                graphs = generate_multiple_graphs(graphs, graph_type, n_, avg_degree, edge_weight, gamma, nu)
        filename = 'synthetic/synthetic_n_small_world.gz'
        with gzip.open(f'data/{filename}', 'wb') as f:
            pickle.dump(graphs, f)

    # generate graphs for resilience capacity case study
    if FLAGS.generate_case:
        graph_type = 'ER-uniform'
        edge_weight = 1.0
        nu = None
        avg_degree = 5
        gamma = None
        n = 15
        graphs = dict()
        graphs = generate_multiple_graphs(graphs, graph_type, n, avg_degree, edge_weight, gamma, nu)
        filename = 'synthetic/synthetic_case.gz'
        with gzip.open(f'data/{filename}', 'wb') as f:
            pickle.dump(graphs, f)


if __name__ == '__main__':
    app.run(main)
