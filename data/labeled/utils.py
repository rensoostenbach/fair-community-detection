import networkx as nx
import pandas as pd
import numpy as np


def relabel_graph(G: nx.Graph):
    """
    Relabel a graph such that the nodes start at 0 and end at len(G.nodes) - 1

    :param G: The NetworkX Graph we are relabeling
    :return G_relabeled: The relabeled Graph
    """
    nodes = G.nodes
    new_labels = list(range(len(nodes)))
    mapping = {}

    for old_node, new_node in zip(nodes, new_labels):
        mapping[old_node] = new_node

    G_relabeled = nx.relabel_nodes(G=G, mapping=mapping)

    # Now also relabel the communities via the mapping

    for node in G_relabeled.nodes:
        new_community = []
        for key, value in G_relabeled.nodes[node].items():  # Value is the list containing the community
            for old_node in value:
                new_community.append(mapping[old_node])
        G_relabeled.nodes[node]["community"] = new_community

    return G_relabeled


def remove_overlapping_nodes(G: nx.Graph):
    """
    Remove nodes that are overlapping (in multiple communities) from the network
    :param G: The NetworkX Graph
    """
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    df = pd.DataFrame(communities)
    value, count = np.unique(np.array(df).flatten(), return_counts=True)
    non_nans = ~np.isnan(value)
    value = value[non_nans]
    counts = count[non_nans]
    nodes_to_remove = value[np.where(counts > 1)]

    H = G.copy()  # Avoid networkx frozen graph can't be modified error
    H.remove_nodes_from(nodes_to_remove)

    for node in H.nodes:
        H.nodes[node]["community"] = list(set(G.nodes[node]["community"]).difference(set(nodes_to_remove)))

    return H
