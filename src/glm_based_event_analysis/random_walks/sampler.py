import numpy as np
import networkx as nx

class RandomWalkSampler:
    '''Class responsible for sampling random walks from a given node in a graph.'''

    def __init__(self, bias: str = "none", 
                       num_walks_per_node: int = 10, 
                       walk_length: int = 5, 
                       inverted: bool = False):
        
        self.bias: str = bias
        self.num_walks_per_node: int = num_walks_per_node
        self.walk_length: int = walk_length
        self.inverted: bool = inverted # whetger to reverse the order of the random walk (useful for sampling past events)
    
    @classmethod
    def from_config(cls, config: dict):
       
        bias = config.get("bias", "none")
        num_walks_per_node = config.get("num_walks", 10)
        walk_length = config.get("walk_length", 5)
        inverted = config.get("inverted", False)

        return cls(bias=bias, num_walks_per_node=num_walks_per_node, walk_length=walk_length, inverted=inverted)

    def sample_random_walk(self, anchor_node_id: str, G: nx.Graph) -> list[str]:
        """Returns a list of nodes representing a random walk starting from the given node.
        Parameters:
        - anchor_node_id: the node from which the random walk starts.
        - G: the graph from which to sample the random walk. Can be a subgraph of the original graph.
        - include_anchor: whether to include the anchor node in the returned walk.
        """

        random_walk = [anchor_node_id]

        for _ in range(self.walk_length-1):
            current_node = random_walk[-1]
            current_node_neighbors = list(G.neighbors(current_node))
            
            # caso o grafo seja direcionado, é possivel chegar num folha -> impossível expandir.
            if len(current_node_neighbors) == 0:
                break 

            if self.bias == "none":
                next_node = np.random.choice(current_node_neighbors)
            elif self.bias == "degree":
                # bias para nós de maior grau
                degrees = np.array([G.degree(neighbor) for neighbor in current_node_neighbors])
                probabilities = degrees / degrees.sum()
                next_node = np.random.choice(current_node_neighbors, p=probabilities)
            elif self.bias == "edge_weight":
                # bias para arestas de maior peso
                # assume que os pesos existam na propriedade 'weight' das arestas
                weights = np.array([G[current_node][neighbor]['weight'] for neighbor in current_node_neighbors]) + 1e-6 # para evitar divisão por zero
                probabilities = weights / weights.sum()
                next_node = np.random.choice(current_node_neighbors, p=probabilities)
            else:
                raise ValueError(f"Bias '{self.bias}' unsuported.")

            random_walk.append(next_node)
        
        if self.inverted:
            random_walk.reverse()
        
        return random_walk