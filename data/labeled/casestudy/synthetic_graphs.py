import typing

import networkx as nx
import random
from itertools import product

import numpy as np

from metrics.own_metric import fair_unfair_nodes, fairness_gt
from utils import draw_graph


def create_two_community_graphs(num_nodes_G1: int, num_nodes_G2: int):
    """
    Create two complete graphs with each a certain number of nodes specified

    :param num_nodes_G1: Number of nodes in the first graph
    :param num_nodes_G2: Number of nodes in the second graph
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

    :param num_nodes_G1: Number of nodes in the first community
    :param num_nodes_G2: Number of nodes in the second community
    :param mus: The mu values we want to iterate over
    :return: Calls calculate_fairness_metric(G) which will print fairness metric per mu value
    """
    for mu in mus:
        print(f"Running with mu = {mu}")

        connected_G1, connected_G2 = create_two_community_graphs(
            num_nodes_G1=num_nodes_G1, num_nodes_G2=num_nodes_G2
        )

        G1 = remove_intra_community_edges(G=connected_G1, fraction=mu)
        G2 = remove_intra_community_edges(G=connected_G2, fraction=mu)
        G = nx.compose(G1, G2)
        G = add_inter_community_edges(
            G=G, G1_nodes=list(G1.nodes), G2_nodes=list(G2.nodes), mu=mu
        )

        calculate_fairness_metric(G)


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

    :param num_nodes: Number of nodes in both communities (always the same for now)
    :param denseness_G1: The denseness of the first community
    :param denseness_G2: The denseness of the second community
    :param inter_community_edges: Fraction of inter-community edges
    :return: Calls calculate_fairness_metric(G) which will print fairness metric
    """
    for denseness_G1 in densenesses_G1:
        for denseness_G2 in densenesses_G2:
            print(f"Denseness G1: {denseness_G1}, denseness G2: {denseness_G2}")

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

            calculate_fairness_metric(G)


def calculate_fairness_metric(G: nx.Graph):
    """
    Calculate and print out the fairness metric for a given graph with communities.

    :param G: The NetworkX Graph with communities we want to compute the fairness for
    :return: Prints out the fairness metric for the given graph
    """
    try:
        communities = {frozenset(G.nodes[v]["community"]) for v in G}
        gt_communities = list(communities)

        #  Not important drawing stuff, just for myself
        pos = nx.spring_layout(G)  # compute graph layout
        draw_graph(G, pos=pos, communities=communities)

        fair_nodes, unfair_nodes = fair_unfair_nodes(G, gt_communities)
        print(
            f"Fairness metric: {fairness_gt(fair_nodes=fair_nodes, unfair_nodes=unfair_nodes)} \n"
        )

    except TypeError:  # Occurs when the ValueError in add_inter_community_edges is triggered
        pass
