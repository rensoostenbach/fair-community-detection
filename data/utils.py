import networkx as nx
import pandas as pd
import numpy as np
import csv


def process_stanford_graph(edgelist_file: str, community_file: str, outname: str):
    G = nx.read_edgelist(f"{edgelist_file}", nodetype=int)

    communities = {}
    with open(f"{community_file}") as file:
        csv_reader = csv.reader(file, delimiter="\t")
        for row in csv_reader:
            for node in row:
                communities[int(node)] = {"community": [int(x) for x in row]}

    nx.set_node_attributes(G, communities)

    nodes_to_remove = []
    for node in G.nodes:
        if not G.nodes[node]:  # If it does not have a community
            nodes_to_remove.append(node)

    G.remove_nodes_from(nodes_to_remove)

    # Process overlapping nodes/communities if there are any, we remove them as a whole
    gt_communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    overlapping = sum([len(com) for com in gt_communities]) > len(G.nodes)
    if overlapping:
        G = remove_overlapping_nodes(G=G)

    # We take the largest connected component if we have an unconnected Graph
    if not nx.is_connected(G):
        print(f"G is unconnected, so we take the largest connected component")
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])

    G_relabeled = relabel_graph(G=G)

    nx.write_gpickle(G_relabeled, f"{outname}.pickle")
    print(f"Graph written as {outname}.pickle")


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
        for key, value in G_relabeled.nodes[
            node
        ].items():  # Value is the list containing the community
            for old_node in value:
                try:
                    new_community.append(mapping[old_node])
                except KeyError:  # Happens for an old_node that has been deleted
                    continue
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
        H.nodes[node]["community"] = list(
            set(G.nodes[node]["community"]).difference(set(nodes_to_remove))
        )

    return H
