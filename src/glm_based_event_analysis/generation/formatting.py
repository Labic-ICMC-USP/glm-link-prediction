import json
import networkx as nx
from glm_based_event_analysis.utils.graph import get_node_metadata
from glm_based_event_analysis.link_prediction.datasets import EVENT_STR_TEMPLATE

def prepare_json_str(node_walk: list[str], G: nx.Graph) -> str:
    """
    Prepares a list of JSON strings for a given list of edges and a graph.
    Each JSON string contains the metadata of the two nodes connected by the edge.
    
    Args:
        node_walk (list[str]): A list of node IDs representing a walk in the graph.
        G (nx.Graph): The graph containing the nodes and their metadata.
    
    Returns:
        str: A JSON string containing the metadata of the nodes in the walk.
    """
    
    node_walk_with_metadata = []

    for node_id in node_walk:
        node_metadata = get_node_metadata(G, node_id)
        node_walk_with_metadata.append(node_metadata)   

    return json.dumps(node_walk_with_metadata, indent=None)[:-2] # desconsiderando o final da formatação json


def prepare_simple_str(node_walk: list[str], G: nx.Graph) -> str:
    """
    Prepares a simple string representation for a given list of edges and a graph.
    Each string contains the metadata of the two nodes connected by the edge in a simple format.
    
    Args:
        node_walk (list[str]): A list of node IDs representing a walk in the graph.
        G (nx.Graph): The graph containing the nodes and their metadata.
    
    Returns:
        str: A simple string containing the metadata of the nodes in the walk.
    """
    
    node_walk_with_metadata = []

    for node_id in node_walk:
        node_metadata = get_node_metadata(G, node_id)
        node_str = EVENT_STR_TEMPLATE.format(
                id=node_id,
                who=node_metadata.get("who", ""),
                what=node_metadata.get("what", ""),
                when=node_metadata.get("when", ""),
                where=node_metadata.get("where", ""),
                why=node_metadata.get("why", ""),
                how=node_metadata.get("how", "")
            )
        node_walk_with_metadata.append(node_str)

    return "\n".join(node_walk_with_metadata)

def is_event_valid(event: dict | None) -> bool:
    """
    Check if the event dictionary contains all required keys: "event_id", "what", "when", "where", "how", "who", and "why".
    """

    if not event or not isinstance(event, dict): return False
    
    required_keys = {"event_id", "what", "when", "where", "how", "who", "why"}
    return required_keys.issubset(event.keys())

def format_event(event: dict) -> dict:
    """
    Format an event dictionary to ensure a pre-defined key order.
    """

    formatted_event = {}
    keys_order = ["what", "where", "when", "who", "how", "why", "event_id"] # ordem usada para chaves da predição de link. TODO: padronizar no futuro
    for key in keys_order:
        formatted_event[key] = event.get(key, None)
    
    return formatted_event
    
