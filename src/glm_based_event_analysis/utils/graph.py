import networkx as nx

def get_node_metadata(G: nx.Graph, nodeid: str) -> dict:

        """Obtains the metadadata of a node, from its attributes in the reference graph."""

        node_data = G.nodes[nodeid]
        node_data['event_id'] = nodeid
        curr_date = node_data.get('when', None)

        if isinstance(curr_date, (str)):
            node_data['when'] = curr_date
        else:
            node_data['when'] = curr_date.strftime("%Y-%m-%d-%H:%M:%S")

        return node_data

def remove_edges_from_graph(G: nx.Graph, eval_edges: list[dict]) -> nx.Graph:
    """
    Returns a new graph by removing test edges and their target nodes from the original graph.
    Args:
        G (nx.Graph): The original graph containing all nodes and edges.
        eval_edges (list[dict]): A list of dictionaries, where each dictionary contains information about an edge to be removed for evaluation. 
            Each dictionary should have the keys 'u', 'v', and 'label', where 'u' and 'v' are the nodes of the edge, and 'label' indicates whether the edge is positive or negative.
    Returns:
        nx.Graph: A new graph that is a copy of the original graph but with the specified edges and their target nodes removed.
    """

    # removendo arestas de teste e nós alvo 
    G_train = G.copy()

    for edge_info in eval_edges:
        # remover apenas arestas que existem
        v = edge_info["v"]

        # se a aresta é negativa, nao removemos o nó alvo, pois ele pode aparecer na vizinhança de outros nós
        if edge_info["label"] == False: continue

        # remoção do nó (garante que ele nao apareça na vizinhança)
        # remover o no alvo remove a aresta junto
        if G_train.has_node(v):
            G_train.remove_node(v)
    
    return G_train