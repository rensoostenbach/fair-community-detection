import networkx as nx
import pandas as pd
import numpy as np
import csv
import random
from tqdm import tqdm


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

    # We take the largest connected component if we have an unconnected Graph
    if not nx.is_connected(G):
        print(f"G is unconnected, so we take the largest connected component")
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])

    # Process overlapping nodes/communities if there are any
    gt_communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    overlapping = sum([len(com) for com in gt_communities]) > len(G.nodes)
    if overlapping:
        G = process_overlapping_nodes_communities(G=G)

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


def process_overlapping_nodes_communities(G: nx.Graph):
    """
    CHANGE ME
    :param G:
    :return:
    """
    gt_communities = {frozenset(G.nodes[v]["community"]) for v in G}
    df = pd.DataFrame(gt_communities)
    value, count = np.unique(np.array(df).flatten(), return_counts=True)
    non_nans = ~np.isnan(value)
    value = value[non_nans]
    counts = count[non_nans]
    overlapping_nodes = value[np.where(counts > 1)]

    # Adjust gt_communities such that we check every community if it has an overlapping node.
    # If it is the first occurence of that node, we skip it. Else, we remove it from the community

    list_gt_communities = [set(x) for x in gt_communities]
    # Randomly shuffle the above list so we traverse through the communities in a different order for every new run
    random.shuffle(list_gt_communities)
    seen_nodes = set()
    for comm in tqdm(list_gt_communities):
        for overlapping_node in overlapping_nodes:
            if overlapping_node in comm and overlapping_node not in seen_nodes:
                seen_nodes.add(overlapping_node)
            elif overlapping_node in comm and overlapping_node in seen_nodes:
                comm.remove(overlapping_node)

    H = G.copy()  # Avoid networkx frozen graph can't be modified error
    for node in tqdm(H.nodes):
        new_community = find_community(node_u=node, communities=list_gt_communities)
        H.nodes[node]["community"] = list(new_community)

    return H


def find_community(node_u: int, communities: list):
    """
    Find the community that node_u belongs to.
    :param node_u:
    :param communities:
    :return:
    """
    for idx, community in enumerate(communities):
        for node in community:
            if node == node_u:
                return community


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
