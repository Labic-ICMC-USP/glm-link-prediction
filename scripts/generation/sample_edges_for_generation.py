from argparse import ArgumentParser
import pickle
import json
import numpy as np
from glm_based_event_analysis.random_walks.sampler import RandomWalkSampler

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Sample edges for event generation.")
    parser.add_argument('--graph', type=str, required=True, help="Path to the graph 'pkl' file.")
    parser.add_argument('--num_nodes', type=int, default=1000, help="Number of source nodes to sample.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output json file where sampled edges will be saved.")

    # argumentos opcionais para amostragem
    parser.add_argument("--num_walks_per_node", type=int, default=10, help="Number of random walks to perform per source node.")
    parser.add_argument("--walk_length", type=int, default=3, help="Length of each random walk.")
    parser.add_argument("--bias", type=str, choices=['none', "degree", "edge_weight"], default='none', help="Bias to use for random walk sampling.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility.")

    return parser 

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # mostrando argumentos
    print("Sampling arguments:")
    for arg, value in vars(args).items():
        print(f"\t {arg}: {value}")

    # configurando seed
    np.random.seed(args.seed)

    # carregando grafo base
    with open(args.graph, 'rb') as f:
        G = pickle.load(f)

    # preparando sampler
    sampler = RandomWalkSampler(bias=args.bias, num_walks_per_node=args.num_walks_per_node, walk_length=args.walk_length + 1, inverted=True)

    # amostrando nós de origem com base na data
    nodes_sorted_by_time = sorted(G.nodes(data=True), key=lambda x: x[1]['when'], reverse=True)
    sampled_nodes = [node[0] for node in nodes_sorted_by_time[:args.num_nodes]]

    # invertendo o grafo base para amostrar eventos passados
    assert G.is_directed(), "O grafo deve ser direcionado para amostrar eventos passados." 
    G_inverted = G.reverse()  

    # preparando serialização
    test_data = []
    for node in sampled_nodes:
        # nao adicionar o proprio evento origem
        walks = [sampler.sample_random_walk(node, G_inverted)[:-1] for _ in range(args.num_walks_per_node)]
        test_data.append(
            {
                "source_node": node,
                "random_walks": walks
            }
        )
    
    with open(args.output_file, 'w') as f:
        json.dump(test_data, f, indent=2)

        

