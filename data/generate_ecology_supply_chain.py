import networkx as nx
from absl import app, flags
import pandas as pd
from scipy import stats

from data.utils import generate_projection_net

flags.DEFINE_bool('generate_ecology', False, 'Generate ecology networks.')
flags.DEFINE_bool('generate_supply_chain', False, 'Generate supply chain networks.')
flags.DEFINE_bool('save', True, 'Save the generated graphs.')
flags.DEFINE_bool('save_sorted', True, 'Save the sorted companies.')
FLAGS = flags.FLAGS


def get_supplier_customer_id(node_id, x1, x2):
    x1 = x1.replace('nan', '')
    x2 = x2.replace('nan', '')
    if x1:
        x1 = x1.replace(';', ',').split(',')
        x1 = [x for x in x1 if x != node_id]
    else:
        x1 = []
    if x2:
        x2 = x2.replace(';', ',').split(',')
        x2 = [x for x in x2 if x != node_id]
    else:
        x2 = []
    return x1 + x2


def main(_):
    print('-' * 80)

    if FLAGS.generate_ecology:
        result = []
        file_ids = [1, 3, 6, 7, 9, 2, 4, 10, 11, 57, 16]
        for file_id in file_ids:
            print(f'Generate graph for {file_id}:')
            bipartite_graph_filename = f'data/ecology_supply_chainecology_supply_chain/network_{file_id}.csv'
            abundance_filename = f'data/ecology_supply_chainecology_supply_chain/abundance_{file_id}.csv'
            bipartite_graph = pd.read_csv(bipartite_graph_filename, header=None).values
            abundance = pd.read_csv(abundance_filename, header=None).values
            graph = generate_projection_net(bipartite_graph, axis=1)
            graph = nx.from_numpy_array(graph)
            print(f'Graph nodes: {graph.number_of_nodes()}')
            nx.set_node_attributes(graph, dict(enumerate(abundance.flatten())), 'abundance')
            degree = [d for n, d in graph.degree()]
            abundance = list(nx.get_node_attributes(graph, 'abundance').values())
            corr = stats.pearsonr(degree, abundance)[0]
            print(f'Correlation between degree and abundance: {corr}')
            result.append((graph.number_of_nodes(), corr))
        result = pd.DataFrame(result, columns=['nodes', 'corr'])
        if FLAGS.save:
            result.to_csv('data/ecology_supply_chain/ecology_corr.csv', index=False)
        print(f'Corr mean and std: {result["corr"].mean()}, {result["corr"].std()}')

    if FLAGS.generate_supply_chain:
        result = []
        result_low = []
        result_medium = []
        result_high = []
        supply_chain_filename = 'data/ecology_supply_chain/macrodata_basic.xlsx'
        supply_chain = pd.read_excel(supply_chain_filename, dtype=str)
        supply_chain = supply_chain[['年份', '股票代码', '股票简称', '供应商股票代码', '供应商关联-股票代码', '供应商采购额_万元',
                                     '客户股票代码', '客户关联-股票代码', '客户销售额_万元', '上游供应商', '下游客户', '行业名称']]
        supply_chain = supply_chain.dropna(subset=['年份', '股票代码'])
        supply_chain = supply_chain.dropna(subset=['供应商采购额_万元', '客户销售额_万元'], how='all')

        result_sorted = []

        years = supply_chain['年份'].unique().tolist()
        for year in years:
            print(f'Generate graph for {year}:')
            year_supply_chain = supply_chain[supply_chain['年份'] == year]
            graph = nx.Graph()
            for _, row in year_supply_chain.iterrows():
                node_id = row['股票代码']
                company_name = row['股票简称']
                industry_name = row['行业名称']
                supplier1 = str(row['供应商股票代码'])
                supplier2 = str(row['供应商关联-股票代码'])
                supplier_id = get_supplier_customer_id(node_id, supplier1, supplier2)
                supplier_company_name = row['上游供应商']
                customer1 = str(row['客户股票代码'])
                customer2 = str(row['客户关联-股票代码'])
                customer_id = get_supplier_customer_id(node_id, customer1, customer2)
                customer_company_name = row['下游客户']
                if len(supplier_id) == 0 and len(customer_id) == 0:
                    continue
                if node_id not in graph:
                    graph.add_node(node_id)
                    graph.nodes[node_id]['input'] = 0.0
                    graph.nodes[node_id]['output'] = 0.0
                    graph.nodes[node_id]['company_name'] = company_name
                    graph.nodes[node_id]['industry_name'] = industry_name
                else:
                    graph.nodes[node_id]['company_name'] = company_name
                    graph.nodes[node_id]['industry_name'] = industry_name
                for supplier in supplier_id:
                    if supplier not in graph:
                        graph.add_node(supplier)
                        graph.nodes[supplier]['input'] = 0.0
                        graph.nodes[supplier]['output'] = 0.0
                        graph.nodes[supplier]['company_name'] = supplier_company_name
                        graph.nodes[supplier]['industry_name'] = ''
                    if not graph.has_edge(supplier, node_id):
                        graph.add_edge(supplier, node_id)
                    graph.nodes[supplier]['output'] += float(row['供应商采购额_万元'])/len(supplier_id)
                    graph.nodes[node_id]['input'] += float(row['供应商采购额_万元'])/len(supplier_id)
                for customer in customer_id:
                    if customer not in graph:
                        graph.add_node(customer)
                        graph.nodes[customer]['input'] = 0.0
                        graph.nodes[customer]['output'] = 0.0
                        graph.nodes[customer]['company_name'] = customer_company_name
                        graph.nodes[customer]['industry_name'] = ''
                    if not graph.has_edge(node_id, customer):
                        graph.add_edge(node_id, customer)
                    graph.nodes[customer]['input'] += float(row['客户销售额_万元'])/len(customer_id)
                    graph.nodes[node_id]['output'] += float(row['客户销售额_万元'])/len(customer_id)
            throughput = {node: graph.nodes[node]['input'] + graph.nodes[node]['output'] for node in graph.nodes}
            nx.set_node_attributes(graph, throughput, 'throughput')
            selected_components = [c for c in nx.connected_components(graph) if 12 <= len(c) <= 61]
            selected_components_low = [c for c in nx.connected_components(graph) if 10 < len(c) <= 20]
            selected_components_medium = [c for c in nx.connected_components(graph) if 20 < len(c) <= 50]
            selected_components_high = [c for c in nx.connected_components(graph) if len(c) > 50]

            def compute_corr(c):
                sub_graph = graph.subgraph(c)
                degree_ = [d for n, d in sub_graph.degree()]
                throughput_ = list(nx.get_node_attributes(sub_graph, 'throughput').values())
                corr_ = stats.pearsonr(degree_, throughput_)[0]
                print(f'Component size: {len(c)}, correlation between degree and throughput: {corr_}')
                degree_times_throughput = [d * t for d, t in zip(degree_, throughput_)]
                sorted_companies_ = [(sub_graph.nodes[node]["company_name"], sub_graph.nodes[node]["industry_name"], node)
                                     for node in sorted(c,
                                                        key=lambda x: degree_times_throughput[list(c).index(x)],
                                                        reverse=True)]
                return corr_, sorted_companies_

            for component in selected_components:
                corr, sorted_companies = compute_corr(component)
                result.append((len(component), corr))
                result_sorted.append(sorted_companies)
            for component in selected_components_low:
                corr, _ = compute_corr(component)
                result_low.append((len(component), corr))
            for component in selected_components_medium:
                corr, _ = compute_corr(component)
                result_medium.append((len(component), corr))
            for component in selected_components_high:
                corr, _ = compute_corr(component)
                result_high.append((len(component), corr))

        # sort result_sorted according to result
        zipped_lists = zip(result, result_sorted)
        sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[0][1])
        result_sorted = [(corr, element) for corr, element in sorted_zipped_lists]

        result = pd.DataFrame(result, columns=['nodes', 'corr'])
        if FLAGS.save:
            result.to_csv('data/ecology_supply_chain/supply_chain_corr.csv', index=False)
        print(f'Corr mean and std: {result["corr"].mean()}, {result["corr"].std()}')

        result_low = pd.DataFrame(result_low, columns=['nodes', 'corr'])
        if FLAGS.save:
            result_low.to_csv('data/ecology_supply_chain/supply_chain_corr_low.csv', index=False)
        print(f'Corr mean and std: {result_low["corr"].mean()}, {result_low["corr"].std()}')

        result_medium = pd.DataFrame(result_medium, columns=['nodes', 'corr'])
        if FLAGS.save:
            result_medium.to_csv('data/ecology_supply_chain/supply_chain_corr_medium.csv', index=False)
        print(f'Corr mean and std: {result_medium["corr"].mean()}, {result_medium["corr"].std()}')

        result_high = pd.DataFrame(result_high, columns=['nodes', 'corr'])
        if FLAGS.save:
            result_high.to_csv('data/ecology_supply_chain/supply_chain_corr_high.csv', index=False)
        print(f'Corr mean and std: {result_high["corr"].mean()}, {result_high["corr"].std()}')

        if FLAGS.save_sorted:
            # save json
            import json
            with open('data/ecology_supply_chain/supply_chain_sorted.json', 'w') as f:
                json.dump(result_sorted, f, ensure_ascii=False, indent=2)

    print('-' * 80)


if __name__ == '__main__':
    app.run(main)
