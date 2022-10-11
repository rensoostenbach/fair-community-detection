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


def small_large_communities(communities: list, percentile: int):
    """
    Decide which communities are small ones and large ones, based on a percentile cutoff value.

    :param communities: List of ground-truth communities
    :param percentile: The percentile on which the cutoff value is based (e.g. 75 for 75th percentile)
    :return small_large: Dictionary indicating per node whether it is in a small or large community
    :return small_large_coms: List containing per community whether it is small or large
    """
    sizes = np.array([len(community) for community in communities])
    small_large = {}
    small_large_coms = []

    for idx, community in enumerate(communities):
        if sizes[idx] >= np.percentile(sizes, percentile):
            small_large_coms.append("large")
        else:
            small_large_coms.append("small")
        for node in community:
            if sizes[idx] >= np.percentile(sizes, percentile):
                small_large[node] = "large"
            else:
                small_large[node] = "small"
    return small_large, small_large_coms


def dense_nondense_communities(G: nx.Graph, communities: list, cutoff: float):
    """
    Decide which communities are dense ones and non-dense ones, based on a percentile cutoff value.

    :param G: The NetworkX Graph from which we can extract the edges
    :param communities: List of ground-truth communities
    :param cutoff: The cutoff on which the cutoff value is based (0.5 for 50% of intra-community edges)
    :return dense_nondense: Dictionary indicating per node whether it is in a dense or non-dense community
    :return dense_nondense_coms: List containing per community whether it is dense or non-dense
    """
    intra_com_edges = np.array([G.subgraph(communities[idx]).size() for idx, community in enumerate(communities)])
    # Need to divide above numbers by maximum amount of edges possible in community
    sizes = [len(community) for community in communities]
    max_possible_edges = np.array([(size * (size - 1)) / 2 for size in sizes])
    densities = np.array(intra_com_edges/(max_possible_edges))
    dense_nondense = {}
    dense_nondense_coms = []

    for idx, community in enumerate(communities):
        if densities[idx] >= cutoff:
            dense_nondense_coms.append("dense")
        else:
            dense_nondense_coms.append("non-dense")
        for node in community:
            if densities[idx] >= cutoff:
                dense_nondense[node] = "dense"
            else:
                dense_nondense[node] = "non-dense"

    return dense_nondense, dense_nondense_coms


def mapping(gt_communities: list, pred_coms: list):
    """

    :param gt_communities:
    :param pred_coms:
    :return achieved_distribution:
    :return mapping_list:
    """
    achieved_distribution = []
    mapping_list = []
    for idx, real_com in enumerate(gt_communities):
        most_similar_community_idx, jaccard_score = find_max_jaccard(
            real_com=real_com, pred_coms=pred_coms
        )
        num_correct_nodes = len(
            set(pred_coms[most_similar_community_idx]).intersection(set(real_com))
        )
        achieved_distribution.append(num_correct_nodes)
        mapping_list.append((most_similar_community_idx, jaccard_score))

    # TODO: Maybe change mapping_list such that we do as we told Akrati, wait for her response.
    return achieved_distribution, mapping_list


def jaccard_similarity(com1: list, com2: list):
    """

    :param com1:
    :param com2:
    :return:
    """
    return len(set(com1).intersection(com2)) / len(set(com1).union(com2))


def find_max_jaccard(real_com: list, pred_coms: list):
    """

    :param real_com:
    :param pred_coms:
    :return:
    """
    jaccard_score = 0
    most_similar_community = None
    for idx, predicted_community in enumerate(pred_coms):
        if jaccard_similarity(real_com, predicted_community) > jaccard_score:
            jaccard_score = jaccard_similarity(real_com, predicted_community)
            most_similar_community = idx

    return most_similar_community, jaccard_score


def split_distribution(distribution: list, comm_types: list):
    """

    :param distribution:
    :param comm_types:
    :return:
    """
    dist_small = []
    dist_large = []
    for com_idx, small_large in enumerate(comm_types):
        if small_large == "large":
            dist_large.append(distribution[com_idx])
        else:  # small
            dist_small.append(distribution[com_idx])
    return dist_small, dist_large


def transform_to_ytrue_ypred(gt_communities: list, pred_coms: list, mapping_list: list):
    n = sum([len(com) for com in gt_communities])
    y_true = [None] * n
    y_pred = [None] * n

    for com, nodes in enumerate(gt_communities):
        for node in nodes:
            y_true[node] = com

    # It can occur that a predicted community is never most similar to a ground-truth community
    # In this case, we keep that label and make sure that f1_score labels parameter does not have that label

    # It can also occur that a predicted community is most similar with multiple ground-truth communities
    # TODO: Transform mapping_list such that we do as we told Akrati? Wait for her response
    # In this case, we simply that the ground-truth community that comes first in mapping_list

    mapping_list_coms = [x[0] for x in mapping_list]  # Only retrieve the communities, not the jaccard scores
    for com, nodes in enumerate(pred_coms):
        for node in nodes:
            try:
                y_pred[node] = mapping_list_coms.index(com)
            except ValueError:
                y_pred[node] = com

    return y_true, y_pred
