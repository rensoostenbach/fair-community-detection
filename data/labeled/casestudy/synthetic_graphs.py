import typing

import networkx as nx
import random
from itertools import product
import numpy as np

from utils import draw_graph, small_large_communities, dense_nondense_communities


def create_two_community_graphs(num_nodes_G1: int, num_nodes_G2: int):
    """
    Create two complete graphs with each a certain number of nodes specified

    :param num_nodes_G1: Number of nodes in the first graph
    :param num_nodes_G2: Number of nodes in the second graph
    :param mu_density: Whether we are varying in mu or density
    :return: Both connected graphs as NetworkX Graphs
    """
    connected_G1 = nx.complete_graph(num_nodes_G1)
    connected_G2 = nx.complete_graph(range(num_nodes_G1, num_nodes_G1 + num_nodes_G2))

    for node in connected_G1.nodes:
        connected_G1.nodes[node]["community"] = set(list(connected_G1.nodes))
    for node in connected_G2.nodes:
        connected_G2.nodes[node]["community"] = set(list(connected_G2.nodes))

    return connected_G1, connected_G2


def remove_intra_community_edges(G: nx.Graph, fraction: float):
    """
    Randomly remove (intra-community) edges in a graph based on a given fraction.

    :param G: The NetworkX Graph in which we want to remove edges
    :param fraction: The fraction of edges we want to remove
    :return G: The NetworkX Graph with removed edges
    """
    to_remove_G = random.sample(G.edges(), k=int(len(G.edges) * fraction))
    G.remove_edges_from(to_remove_G)
    return G


def add_inter_community_edges(G: nx.Graph, G1_nodes: list, G2_nodes: list, mu: float):
    """
    Add inter-community edges based on the mu parameter.

    Difference between mu here and mu in LFR benchmark is as follows:
    Mu here considers the total amount of edges in the graph,
    while mu in the LFR benchmark considers inter-community edges incident to each node.

    :param G: NetworkX Graph of the network in which we want to add inter-community edges
    :param G1_nodes: List of nodes in the first community
    :param G2_nodes: List of nodes in the second community
    :param mu: Fraction of inter-community edges, NOT exactly the same as in the LFR benchmark!
    :return G: The NetworkX Graph with inter-community edges added (if possible).
    """
    try:
        inter_community_edges = random.sample(
            set(product(G1_nodes, G2_nodes)),
            int((len(G.edges) / (1 - mu))) - len(G.edges),
        )
        G.add_edges_from(inter_community_edges)

        return G

    except ValueError:  # Occurs when there are less inter-community edges possible than mu parameter wants
        print(
            f"Tried to sample {int((len(G.edges) / (1 - mu))) - len(G.edges)} "
            f"inter-community edges, but there are only {len(set(product(G1_nodes, G2_nodes)))} possibilities."
        )


def varying_mu_values(
    num_nodes_G1: int, num_nodes_G2: int, mus: typing.Union[list, np.ndarray]
):
    """
    This function is used for generating networks with varying mu values and pre-specified community sizes
    for two communities only.
    Difference between mu here and mu in LFR benchmark is as follows:
    Mu here considers the total amount of edges in the graph,
    while mu in the LFR benchmark considers inter-community edges incident to each node.

    TODO: Think about whether I should do mu exactly the same in LFR (if possible)
    Not doing that for now, see my notes in Obsidian on LFR benchmark graphs

    :param num_nodes_G1: Number of nodes in the first community
    :param num_nodes_G2: Number of nodes in the second community
    :param mus: Fractions of inter-community edges, NOT exactly the same as in the LFR benchmark!
    :return graphs: List of all generated graphs
    """
    graphs = []
    for mu in mus:
        connected_G1, connected_G2 = create_two_community_graphs(
            num_nodes_G1=num_nodes_G1, num_nodes_G2=num_nodes_G2
        )

        G1 = remove_intra_community_edges(G=connected_G1, fraction=mu)
        G2 = remove_intra_community_edges(G=connected_G2, fraction=mu)
        G = nx.compose(G1, G2)
        G = add_inter_community_edges(
            G=G, G1_nodes=list(G1.nodes), G2_nodes=list(G2.nodes), mu=mu
        )
        graphs.append(G)
    return graphs


def varying_denseness(
    num_nodes: int,
    densenesses_G1: typing.Union[list, np.ndarray],
    densenesses_G2: typing.Union[list, np.ndarray],
    inter_community_edges: float,
):
    """
    This function is used for generating communities with a specified denseness per community.
    A denseness value of 1 means that every node in a community is connected to every other node
    in the same community, and lower means that a part of the intra-community edges has been removed.

    TODO: Think about which LFR parameter this relates to, and whether it should be changed to make it the same as LFR
    Not doing that for now, see my notes in Obsidian on LFR benchmark graphs

    :param num_nodes: Number of nodes in both communities (always the same for now)
    :param densenesses_G1: The denseness of the first community
    :param densenesses_G2: The denseness of the second community
    :param inter_community_edges: Fraction of inter-community edges
    :return graphs: List of all generated graphs
    """
    graphs = []
    for denseness_G1 in densenesses_G1:
        for denseness_G2 in densenesses_G2:
            connected_G1, connected_G2 = create_two_community_graphs(
                num_nodes_G1=num_nodes, num_nodes_G2=num_nodes
            )

            G1 = remove_intra_community_edges(G=connected_G1, fraction=1 - denseness_G1)
            G2 = remove_intra_community_edges(G=connected_G2, fraction=1 - denseness_G2)

            G = nx.compose(G1, G2)
            G = add_inter_community_edges(
                G=G,
                G1_nodes=list(G1.nodes),
                G2_nodes=list(G2.nodes),
                mu=inter_community_edges,
            )
            graphs.append(G)
    return graphs


def mislabel_nodes(G: nx.Graph, mislabel_comm_nodes: dict, size_percentile=90, density_cutoff=0.5):
    """
    Pretty sure this function only works for two communities because of comm_types.index(comm_type) call
    TODO: Rewrite this function such that num_nodes and where_to_mislabel become a dictionary
     that indicates per type how many to remove so we don't need two calls of this function
    :param G:
    :param mislabel_comm_nodes: Dict of community types and number of nodes to mislabel
    :param size_percentile:
    :param density_cutoff:
    :return:
    """
    communities = list({frozenset(G.nodes[v]["community"]) for v in G})
    where_to_mislabel = set(mislabel_comm_nodes.keys())
    if len(where_to_mislabel.intersection({"small", "large"})) > 0:
        node_comm_types, comm_types = small_large_communities(
            communities=communities, percentile=size_percentile
        )
    elif len(where_to_mislabel.intersection({"sparse", "dense"})) > 0:  # dense or sparse
        node_comm_types, comm_types = dense_nondense_communities(
            G=G, communities=communities, cutoff=density_cutoff
        )
    else:  # random
        num_nodes = mislabel_comm_nodes["random"]
        random_nodes = random.sample(G.nodes, k=num_nodes)
        for random_node in random_nodes:
            node_old_community = G.nodes[random_node]["community"]
            node_old_community_removed = node_old_community.difference({random_node})
            node_new_community = set(G.nodes).difference(node_old_community_removed)
            for node in node_old_community_removed:
                G.nodes[node]['community'] = node_old_community_removed
            for node in node_new_community:
                G.nodes[node]['community'] = node_new_community
        return G

    for comm_type in where_to_mislabel:
        communities = list({frozenset(G.nodes[v]["community"]) for v in G})
        comm_to_mislabel = comm_types.index(comm_type)
        # Randomly sample num_nodes that need to be mislabeled
        nodes_to_mislabel = random.sample(communities[comm_to_mislabel], k=mislabel_comm_nodes[comm_type])
        other_community = communities[1-comm_to_mislabel]  # Get the other community
        # Now other_community will always be the new community for the set of nodes that will be mislabeled
        new_community_original_nodes = set(nodes_to_mislabel).union(set(other_community))

        new_community_mislabeled_nodes = set(communities[comm_to_mislabel]).difference(set(nodes_to_mislabel))

        for node in G.nodes:
            if node in new_community_original_nodes:
                G.nodes[node]['community'] = new_community_original_nodes
            else:  # Mislabeled nodes
                G.nodes[node]['community'] = new_community_mislabeled_nodes

    return G
