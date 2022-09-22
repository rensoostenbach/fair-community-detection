import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def draw_graph(G: nx.Graph, pos: dict, communities=None):
    """Draw a graph, with the choice of including communities or not"""

    plt.figure(figsize=(15, 10))
    plt.axis("off")
    if communities is not None:
        # Coloring every node such that communities have the same color
        node_color_pred = [0] * len(G.nodes)
        for idx, community in enumerate(communities):
            for node in community:
                node_color_pred[node] = idx

        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            node_size=100,
            cmap=plt.cm.RdYlBu,
            node_color=node_color_pred,
            linewidths=0.5,
            edgecolors="black",
        )
        nx.draw_networkx_edges(G=G, pos=pos, alpha=0.3)
    else:
        nx.draw_networkx_nodes(
            G=G, pos=pos, node_size=100, linewidths=0.5, edgecolors="black"
        )
        nx.draw_networkx_edges(G=G, pos=pos, alpha=0.3)

    plt.show()


def remove_community(node_u: int, communities: list):
    """
    Remove the community that node_u belongs to in the list of communities

    :param node_u: The ID of the node for which we want to remove its community
    :param communities: List of communities
    :return communities: List of communities without the community that node_u belongs to
    """
    for idx, community in enumerate(communities):
        for node in community:
            if node == node_u:
                com_to_delete = idx
    communities = communities[:com_to_delete] + communities[com_to_delete + 1 :]
    return communities


def small_large_communities(communities, percentile):
    """
    Decide which communities are small ones and large ones, based on a percentile cutoff value.

    :param communities: List of communities
    :param percentile: The percentile on which the cutoff value is based (e.g. 75 for 75th percentile)
    :return small_large: Dictionary indicating per node whether it is in a small or large community
    """
    lengths = np.array([len(community) for community in communities])
    small_large = {}

    for idx, community in enumerate(communities):
        for node in community:
            if lengths[idx] >= np.percentile(lengths, percentile):
                small_large[node] = "large"
            else:
                small_large[node] = "small"
    return small_large
