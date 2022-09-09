import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def draw_graph(G: nx.Graph, pos: dict, communities=None):
    """Draw a graph, with the choice of including communities or not"""

    plt.figure(figsize=(15, 10))
    plt.axis("off")
    if communities is not None:
        # Coloring every node such that communities have the same color
        node_color_pred = [0] * len(G.nodes)
        for idx, community in enumerate(communities.communities):
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


def random_sample_edges(nodes: np.ndarray, edges: np.ndarray, alpha: float):
    """
    Take a random subsample of the edges based on the alpha parameter.

    :param nodes: A numpy array containing the nodes in the original graph
    :param edges: A numpy array containing the edges in the original graph
    :param alpha: A float describing how much we should sample, 0 > alpha > 1
    :return sampled_G: The sampled networkx Graph
    :return sampled_edges: The sampled edges
    :return nonsampled_edges: The edges that were in the original graph, but not in the sampled Graph
    """
    num_samples = round(alpha * edges.shape[0])

    random_indices = np.arange(0, edges.shape[0])
    np.random.shuffle(random_indices)

    sampled_edges = edges[random_indices[:num_samples]]
    sampled_G = nx.Graph()
    sampled_G.add_nodes_from(nodes)
    sampled_G.add_edges_from(sampled_edges)
    nonsampled_edges = edges[random_indices[num_samples:]]

    return sampled_G, sampled_edges, nonsampled_edges


def potentional_edge_list(G, nodes):
    """
    TODO: Write docstring?
    Totnutoe gaat deze functie uit van undirected edges! Dat doet de paper ook
    """
    all_potential_edges = np.array(list(combinations(nodes, 2)))
    existing_edges = np.array(G.edges)
    # From https://stackoverflow.com/questions/39321615/numpy-array-set-difference
    all_potential_edges_rows = all_potential_edges.view(
        [("", all_potential_edges.dtype)] * all_potential_edges.shape[1]
    )
    existing_edges_rows = existing_edges.view(
        [("", existing_edges.dtype)] * existing_edges.shape[1]
    )
    all_potentional_edges_without_existing = (
        np.setdiff1d(all_potential_edges_rows, existing_edges_rows)
        .view(all_potential_edges.dtype)
        .reshape(-1, all_potential_edges.shape[1])
    )
    return all_potentional_edges_without_existing


def modularity_obj_function(communities, G, edges):
    modularity = 0
    for community in communities.communities:
        subgraph = G.subgraph(community)

        l_r = len(subgraph.edges)
        M = edges.shape[0]
        d_r = sum(x[1] for x in subgraph.degree)
        modularity += (l_r / M) - ((d_r / (2 * M)) ** 2)

    return modularity


def score_function(G_without_ij, G_with_ij, communities, edges):
    mod_G_without_ij = modularity_obj_function(
        communities=communities, G=G_without_ij, edges=edges
    )
    mod_G_with_ij = modularity_obj_function(
        communities=communities, G=G_with_ij, edges=edges
    )
    score = mod_G_with_ij - mod_G_without_ij
    return score
