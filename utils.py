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


def plot_fairness(emd: list, f1: list, acc: list, x_axis: list, xlabel: str, title="Fairness scores per ..."):
    plt.plot(x_axis, emd, label="EMD Fairness", marker=".")
    plt.plot(x_axis, f1, label="F1 Fairness", marker=".")
    plt.plot(x_axis, acc, label="Accuracy fairness", marker=".")
    plt.legend()
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Fairness scores")
    plt.title(f"{title}")
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


def dense_sparse_communities(G: nx.Graph, communities: list, cutoff: float):
    """
    Decide which communities are dense ones and sparse ones, based on a percentile cutoff value.

    :param G: The NetworkX Graph from which we can extract the edges
    :param communities: List of ground-truth communities
    :param cutoff: The cutoff on which the cutoff value is based (0.5 for 50% of intra-community edges)
    :return dense_sparse: Dictionary indicating per node whether it is in a dense or sparse community
    :return dense_sparse_coms: List containing per community whether it is dense or sparse
    """
    intra_com_edges = np.array(
        [
            G.subgraph(communities[idx]).size()
            for idx, community in enumerate(communities)
        ]
    )
    # Need to divide above numbers by maximum amount of edges possible in community
    sizes = [len(community) for community in communities]
    max_possible_edges = np.array([(size * (size - 1)) / 2 for size in sizes])
    densities = np.array(intra_com_edges / (max_possible_edges))
    dense_sparse = {}
    dense_sparse_coms = []

    for idx, community in enumerate(communities):
        if densities[idx] >= cutoff:
            dense_sparse_coms.append("dense")
        else:
            dense_sparse_coms.append("sparse")
        for node in community:
            if densities[idx] >= cutoff:
                dense_sparse[node] = "dense"
            else:
                dense_sparse[node] = "sparse"

    return dense_sparse, dense_sparse_coms


def modify_mapping_list(mapping_list: list):
    """
    Modify the mapping list such that a predicted community is not connect to two ground truth communities.
    Instead, we regard the ground-truth community which is most similar to the predicted community as the right one, and
    regard all other ground-truth communities that are mapped to that predicted community as completely misclassified.

    :param mapping_list: List of tuples where x[0] indicates the predicted community and x[1] the Jaccard score
    :return most_similar_pred_coms: List of most similar predicted communities, -1 in case of not being the most
                                    similar ground-truth community to a predicted community.
    """

    most_similar_pred_coms = [x[0] for x in mapping_list]
    jaccards = [x[1] for x in mapping_list]
    seen = set()
    duplicates = []
    if len(set(most_similar_pred_coms)) < len(
        most_similar_pred_coms
    ):  # There are duplicates that need to be handled
        for com in most_similar_pred_coms:
            if com in seen:
                duplicates.append(com)
            else:
                seen.add(com)
        for duplicate in set(duplicates):  # Process every duplicate
            indices = set(
                [i for i, x in enumerate(most_similar_pred_coms) if x == duplicate]
            )  # Get the idx of duplicates
            max_jaccard = 0
            for index in indices:
                if jaccards[index] > max_jaccard:
                    max_jaccard = jaccards[index]
                # Find the index that max jaccard belongs to, and set the most similar pred com to -1 for the others
            max_jaccard_index = jaccards.index(max_jaccard)
            indices.remove(max_jaccard_index)
            for index in indices:
                most_similar_pred_coms[index] = -1

    else:
        return most_similar_pred_coms

    return most_similar_pred_coms


def mapping(gt_communities: list, pred_coms: list):
    """

    :param gt_communities:
    :param pred_coms:
    :return achieved_distribution:
    :return mapping_list: Index indicates gt-community and value indicates the pred comm that is most similar
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

    mapping_list = modify_mapping_list(mapping_list=mapping_list)
    # Change the achieved_distribution such that it becomes 0 in the places where mapping_list == -1
    gt_comm_misclassified_indices = [i for i, x in enumerate(mapping_list) if x == -1]
    for index in gt_comm_misclassified_indices:
        achieved_distribution[index] = 0

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


def split_types(distribution_fraction: list, comm_types: list):
    """

    :param distribution_fraction:
    :param comm_types:
    :return:
    """

    type1 = []
    type2 = []
    unique_comm_types = np.unique(comm_types)
    for com_idx, comm_type in enumerate(comm_types):
        if comm_type == unique_comm_types[0]:
            type1.append(distribution_fraction[com_idx])
        else:
            type2.append(distribution_fraction[com_idx])
    return type1, type2


def transform_to_ytrue_ypred(gt_communities: list, pred_coms: list, mapping_list: list):
    n = sum([len(com) for com in gt_communities])
    y_true = [None] * n
    y_pred = [None] * n

    for com, nodes in enumerate(gt_communities):
        for node in nodes:
            y_true[node] = com

    # It can occur that a predicted community is never most similar to a ground-truth community
    # In this case, we label all the nodes in those predicted communities with -1 (ValueError part)

    # It can also occur that a predicted community is most similar with multiple ground-truth communities
    # In this case, mapping_list will have a -1 value for the ground-truth communities that are not most similar
    # with that predicted community

    for com, nodes in enumerate(pred_coms):
        for node in nodes:
            try:
                y_pred[node] = mapping_list.index(com)
            except ValueError:
                y_pred[node] = -1

    return y_true, y_pred
