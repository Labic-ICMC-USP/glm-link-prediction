from tqdm import tqdm
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.utils import negative_sampling

def filter_edges_by_weight_range(edge_list: list[tuple[int, int, dict]], min_weight: float, max_weight: float, weight_key: str = 'weight') -> list[tuple[int, int, dict]]:
    '''
    Filtra as arestas com base em um intervalo de peso especificado. 
    Espera uma lista de arestas originadas pelo G.edges(data=True) e retorna apenas aquelas cujo peso (definido por weight_key) esteja entre min_weight e max_weight.
    Parâmetros:
    - edge_list: Lista de arestas no formato (u, v, d), onde u e v são os nós de origem e destino, e d é um dicionário de atributos da aresta.
    - min_weight: O limite inferior do intervalo de peso (inclusive).
    - max_weight: O limite superior do intervalo de peso (exclusive).
    - weight_key: A chave no dicionário de atributos da aresta que contém o peso a ser filtrado (padrão é 'weight').    
    Retorna:    
    - Uma lista de arestas que atendem ao critério de peso especificado, no mesmo formato (u, v, d).   
    '''
    return [edge for edge in edge_list if min_weight <= edge[2][weight_key] < max_weight]

def sample_topk_edges(edge_list: list[tuple[int, int, dict]], G: nx.Graph, k: int, sort_edges_by: str = "weight") -> list:
    '''
    Amostra as top k arestas de uma lista de arestas, garantindo que a remoção de cada aresta não aumente o número de componentes conectados no grafo.
    Parâmetros:
-   - edge_list: Lista de arestas no formato (u, v, d), onde u e v são os nós de origem e destino, e d é um dicionário de atributos da aresta.
    - G: O grafo original.
    - k: O número de arestas a serem amostradas.
    - sort_edges_by: A chave pelo qual as arestas serão ordenadas (padrão é "weight").
    Retorna:
    - Uma lista de arestas amostradas, no mesmo formato (u, v, d).
    '''

    sampled_edges = []
    sampled_source_nodes = set()
    sampled_target_nodes = set()

    pbar = tqdm(range(0, len(edge_list)), desc="Sampling edges. Tested edges")

    original_connected_components = nx.number_connected_components(G)
    reference_G = G.copy()


    if sort_edges_by == "when":
        # ordenando com base na data da aresta de destino!
        sorted_edges = sorted(edge_list, key=lambda x: G.nodes[x[1]][sort_edges_by], reverse=True)
    elif sort_edges_by in ["weight", "what_sim", "where_sim", "when_sim"]:
        sorted_edges = sorted(edge_list, key=lambda x: x[2][sort_edges_by], reverse=True)
    else:
        raise ValueError(f"Unsupported sort_edges_by value: {sort_edges_by}")

    # Remove edges from the graph until the number of test examples is matched
    with tqdm(total=k, desc="- Sampling edges") as pbar:
        for edge in sorted_edges:

            u, v, _ = edge

            # all target nodes will be removed for training
            # therefore, they cannot be a source node in any other edge, otherwise we would be leaking information through the neighborhood
            if v in sampled_source_nodes:
                continue

            reference_G.remove_edge(u, v)

            # Check if the removal of an edge increases the number of connected components
            if (nx.number_connected_components(reference_G) > original_connected_components):
                reference_G.add_edge(u, v)
            else:
        
                sampled_edges.append(edge)
                sampled_source_nodes.add(u)
                sampled_target_nodes.add(v)
                pbar.update(1)

            if len(sampled_edges) >= k:
                break
        
        # note que esse processo pode resultar em menos de k arestas
    
    return sampled_edges

def sample_edges_by_weight(edge_list: list[tuple[int, int, dict]], k: int, weight_key: str = 'weight') -> list:
    '''
    Amostra k arestas de uma lista de arestas com base em seus pesos, onde a probabilidade de cada aresta ser amostrada é proporcional ao seu peso (definido por weight_key).
    Parâmetros:
    - edge_list: Lista de arestas no formato (u, v, d), onde u e v são os nós de origem e destino, e d é um dicionário de atributos da aresta.
    - k: O número de arestas a serem amostradas.
    - weight_key: A chave no dicionário de atributos da aresta que contém o peso a ser usado para amostragem (padrão é 'weight').
    Retorna:
    - Uma lista de arestas amostradas, no mesmo formato (u, v, d).
    '''
    weights = np.array([edge[2][weight_key] for edge in edge_list])
    probabilities = weights / weights.sum()
    sampled_indices = np.random.choice(len(edge_list), size=k, replace=False, p=probabilities)
    return [edge_list[i] for i in sampled_indices]

def sample_neg_edges(G: nx.Graph, num_neg_samples: int) -> list[tuple[int, int]]:
    '''
    Sample negative edges from the graph.
    Parameters:
    - G: The input graph from which to sample negative edges.
    - num_neg_samples: The number of negative edges to sample.
    Returns:
    - A list of negative edges, where each edge is represented as a tuple (u, v) of node IDs.
    '''

    G_data = from_networkx(G)
    seqid2eventid = {seqid: eventid for seqid, eventid in enumerate(G.nodes)}

    neg_samples = negative_sampling(
        G_data.edge_index, 
        num_nodes=G_data.num_nodes, 
        num_neg_samples=num_neg_samples,
        force_undirected=True
    )
    seqid_pairs = neg_samples.t().tolist()

    return [(seqid2eventid[u], seqid2eventid[v]) for u, v in seqid_pairs]