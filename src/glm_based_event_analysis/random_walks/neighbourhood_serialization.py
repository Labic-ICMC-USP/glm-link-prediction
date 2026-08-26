import numpy as np
import json
import networkx as nx
from typing import Optional
from glm_based_event_analysis.random_walks.sampler import RandomWalkSampler


class NeighbourhoodSerializer:
    '''Class responsible for sampling and serializing the neighbourhood of a given node in a graph, to be used as additional context for link prediction tasks.'''

    def __init__(self, config: dict, G_ref: nx.Graph):

        # grafo de consulta
        self.G_ref: nx.Graph = G_ref # não é necessário copiar. não deve ser modificado. usado para obter atributos de no

        # gerais
        general_config = config["general"]
        self.use_ego_graph: bool = general_config.get("use_ego_graph", False) # amostrar ou não um ego graph antes de amostrar caminhadas.
        self.use_support_anchors: bool = general_config.get("use_support_anchors", False)
        self.n_support_anchors: int = general_config.get("n_support_anchors", 10)

        # ego graph 
        ego_graph_config = config["ego_graph"]
        self.ego_graph_radius: int = ego_graph_config.get("radius", 3)

        # random walks
        random_walks_config = config["random_walks"]
        inverted = random_walks_config.get("inverted", False)
        self.random_walk_sampler: RandomWalkSampler = RandomWalkSampler.from_config(random_walks_config)

        # configurando grafo de ref
        if inverted:
            self.G_ref = self.G_ref.reverse(copy=False)

    def _sample_ego_graph(self, anchor_node_id: str) -> nx.Graph:
        """Retorna o ego graph do nó dado, com o raio especificado."""
        return nx.ego_graph(self.G_ref, anchor_node_id, radius=self.ego_graph_radius, center=True) # TODO: vale a pena incluir o no centro?

    def _sample_support_anchors(self, G: nx.Graph, anchor_node_id: str, n_support_anchors: int) -> list[str]:
        """Retorna uma lista de nós âncora de suporte. Amostrados baseado em seu grau."""

        # candidate nodes é um subconjunto do G_ref
        candidate_nodes = list(G.nodes())

        anchor_node_metadata = self._get_node_metadata(anchor_node_id)
        anchor_node_date = anchor_node_metadata["when"]

        # filtrando nós que ocorreram antes do nó âncora
        candidate_nodes = [node_id for node_id in candidate_nodes if self._get_node_metadata(node_id)["when"] <= anchor_node_date]

        degrees = np.array([G.degree(node) for node in candidate_nodes])
        degrees = degrees + 1e-6 # para evitar divisão por zero
        weights = degrees / degrees.sum()
        support_anchors = np.random.choice(candidate_nodes, size=n_support_anchors, replace=False, p=weights)
        support_anchors = [anchor_node_id] + support_anchors.tolist() # incluindo o nó âncora como suporte
        return support_anchors
    
    def _get_node_metadata(self, node_id: str) -> dict:
        """Obtém os metadados do nó dado."""

        node_attr = self.G_ref.nodes[node_id]
        curr_date = node_attr.get('when', None)
        if isinstance(curr_date, (str)):
            node_attr['when'] = curr_date
        else:
            node_attr['when'] = curr_date.strftime("%Y-%m-%d-%H:%M:%S")

        # padronizando a ordem dos atributos 
        node_data = {
            "what": node_attr["what"],
            "who": node_attr["who"],
            "when": node_attr["when"],
            "where": node_attr["where"],
            "why": node_attr["why"],
            "how": node_attr["how"],
        } 

        return node_data

    def _obtain_walk_metadata(self, walk: list[str]) -> list[dict]:
        """Obtém os metadados de cada nó na caminhada aleatória."""

        walk_with_metadata = []

        for node_id in walk:
            node_data = self._get_node_metadata(node_id)
            walk_with_metadata.append(node_data)

        return walk_with_metadata

    def _serialize_random_walks(self, random_walks_with_metadata: list[list[dict]]) -> str:
        """Serializa as random walks em uma string."""

        # serializando para string (json)
        serialized_random_walks = json.dumps(random_walks_with_metadata, indent=2)

        return serialized_random_walks

    def get_neighbourhood_representation(self, anchor_node_id: str, return_json_str: bool = True) -> dict | str:
        """Retorna uma representação da vizinhança do nó dado, incluindo o ego graph e as random walks."""

        G_ref = self._sample_ego_graph(anchor_node_id) if self.use_ego_graph else self.G_ref
        num_walks_per_node = self.random_walk_sampler.num_walks_per_node

        # se nao usar nós de suporte, as caminhadas aleatórias são amostradas a partir do nó âncora
        # idem, caso o ego graph for muito pequeno
        random_walks_data = []
        # em ambos os casos, omitir o nó âncora
        if not self.use_support_anchors or (G_ref.number_of_nodes() <= self.n_support_anchors):
            walks = [self.random_walk_sampler.sample_random_walk(anchor_node_id, G_ref) for _ in range(num_walks_per_node)]

            # omitindo nó de origem
            if not self.random_walk_sampler.inverted:
                walks = [walk[1:] for walk in walks]
            else:
                walks = [walk[:-1] for walk in walks]

            random_walks_data = [{'source': anchor_node_id, 'walk': self._obtain_walk_metadata(walk), "walk_ids": walk } for walk in walks]
        else:
            support_anchors = self._sample_support_anchors(G_ref, anchor_node_id, self.n_support_anchors)
            for support_anchor in support_anchors:
                walks = [self.random_walk_sampler.sample_random_walk(support_anchor, G_ref) for _ in range(num_walks_per_node)]

                # omitindo nó de origem
                if not self.random_walk_sampler.inverted:
                    walks = [walk[1:] for walk in walks]
                else:
                    walks = [walk[:-1] for walk in walks]
                random_walks_data.extend([{'source': support_anchor, 'walk': self._obtain_walk_metadata(walk), "walk_ids": walk } for walk in walks])

        return self._serialize_random_walks(random_walks_data) if return_json_str else random_walks_data