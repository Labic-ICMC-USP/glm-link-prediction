from yfiles_jupyter_graphs import GraphWidget
from typing import Optional
import seaborn as sns
import networkx as nx
from pyvis.network import Network

def visualize_jupyter_graph(G: nx.Graph, communities: Optional[list] = None) -> GraphWidget:
  
  # Create widget
    w = GraphWidget()

    # Load NetworkX graph
    w.import_graph(G)

    #showing only the id and date
    w.node_tooltip_mapping = lambda node: f"ID: {node}"

    # highlighting communities
    if communities is not None:
        color_palette = sns.color_palette("hls", len(communities)).as_hex()
        community_color_mapping = {}
        for i, community in enumerate(communities):
            for node in community:
                community_color_mapping[node] = color_palette[i] 
                
        w.node_color_mapping = lambda node: community_color_mapping[node["properties"]["label"]] 

    return w

def generate_pyvis_graph(G: nx.Graph, communities: Optional[list] = None, output_path: str = "event_net_demo.html"):

    net = Network(notebook=True,cdn_resources='remote')

    if communities is not None:
        color_palette = sns.color_palette("hls", len(communities)).as_hex()
        community_color_mapping = {}
        for i, community in enumerate(communities):
            for node in community:
                community_color_mapping[node] = color_palette[i] 

    for node in G.nodes():
        net.add_node(
            node,
            size=5 + G.degree[node],
            title="what: " + G.nodes[node]['what'] + "\n" +\
                "where: " + G.nodes[node]['where'] + "\n" +\
                "when: " + str(G.nodes[node]['when'].date()),
            color=community_color_mapping[node] if communities is not None else "gray"

        )

    for edge in G.edges():
        net.add_edge(edge[0], edge[1])

    # Use force-directed physics 
    net.force_atlas_2based()

    # net.show("network.html")
    net.write_html(output_path)