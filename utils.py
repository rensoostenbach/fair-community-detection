import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter


def draw_graph(G: nx.Graph, pos: dict, filename: str, communities=None):
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

    plt.savefig(f"plots/{filename}.png")
    plt.close()  # Use plot.show() if we want to show it


def lineplot_fairness(
    emd: list,
    f1: list,
    acc: list,
    x_axis: list,
    xlabel: str,
    title="Fairness scores per ...",
):
    """
    Draw a line plot that indicates fairness for various synthetic situations

    :param emd: List containing EMD fairness scores
    :param f1: List containing F1 fairness scores
    :param acc: List containing Acc fairness scores
    :param x_axis: List, values for the x-axis
    :param xlabel: String, x-axis label
    :param title: String, title of the plot
    :return: Matplotlib plot
    """
    plt.plot(x_axis, emd, label="EMD Fairness", marker=".")
    plt.plot(x_axis, f1, label="F1 Fairness", marker=".")
    plt.plot(x_axis, acc, label="Accuracy fairness", marker=".")
    plt.legend()
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Fairness scores")
    plt.ylim(bottom=0)
    plt.title(f"{title}")
    plt.show()


def plot_heatmap(
    data: np.array,
    title: str,
):
    """
    Plot a heatmap for the synthetic case of misclassifying nodes in the minor or major community.

    :param data: Tabular data (NumPy array or Pandas DataFrame) containing the fairness values
    :param title: String, title of the plot
    :return: Matplotlib plot
    """
    ax = sns.heatmap(data, cmap="crest_r")
    ax.invert_yaxis()
    ax.set_xlabel("Number of misclassified nodes in major community")
    ax.set_ylabel("Number of misclassified nodes in minor community")
    plt.title(f"{title}")
    plt.show()


def scatterplot_fairness(
    fairness_scores: dict,
    accuracy_scores: dict,
    fairness_type: int,
    accuracy_type: int,
):
    """

    :param fairness_scores: Dictionary containing fairness scores per method and fairness type
    :param accuracy_scores: Dictionary containing accuracy values per method and accuracy type
    :param fairness_type: 0 for EMD, 1 for F1, 2 for ACC
    :param accuracy_type: 0 for ARI, 1 for VI
    :return: Matplotlib plot
    """
    for method, scores in fairness_scores.items():
        fairness_score = [x[fairness_type] for x in scores]
        acc_score_list = [x for x in accuracy_scores[method]][accuracy_type]
        acc_score = [x.score for x in acc_score_list]
        plt.scatter(np.mean(fairness_score), np.mean(acc_score), label=f"{method}")

    plt.xlabel(f"Average Fairness score of type {fairness_type}")
    plt.ylabel(f"Accuracy of type {accuracy_type}")
    plt.title(f"Accuracy vs Fairness")
    plt.xlim(0, 1)
    if accuracy_type != 1:
        plt.ylim(0, 1)
    else:  # Variation of information, need to set different bound than 1.
        max_vi = 0
        for score in accuracy_scores.values():
            matchingresult_per_method = score[1]
            vi_per_method = [x.score for x in matchingresult_per_method]
            for vi in vi_per_method:
                if vi > max_vi:
                    max_vi = vi
        plt.ylim(0, max_vi)
    plt.legend()
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


def dense_sparse_communities(G: nx.Graph, communities: list, percentile: int):
    """
    Decide which communities are dense ones and sparse ones, based on a percentile cutoff value.

    :param G: The NetworkX Graph from which we can extract the edges
    :param communities: List of ground-truth communities
    :param percentile: The percentile on which the cutoff value is based (e.g. 75 for 75th percentile)
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
    densities = np.array(intra_com_edges / max_possible_edges)
    dense_sparse = {}
    dense_sparse_coms = []

    for idx, community in enumerate(communities):
        if densities[idx] >= np.percentile(densities, percentile):
            dense_sparse_coms.append("dense")
        else:
            dense_sparse_coms.append("sparse")
        for node in community:
            if densities[idx] >= np.percentile(densities, percentile):
                dense_sparse[node] = "dense"
            else:
                dense_sparse[node] = "sparse"

    return dense_sparse, dense_sparse_coms


def modify_mapping_list(mapping_list: list):
    """
    Modify the mapping list such that a predicted community is not connected to two ground truth communities.
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
            try:
                max_jaccard_index = jaccards.index(max_jaccard)
                indices.remove(max_jaccard_index)
            except KeyError:  # Occurs when there are multiple jaccards that have the same score
                max_jaccard_indices = [
                    i for i, x in enumerate(jaccards) if x == max_jaccard
                ]
                # We remove the first element that intersects with indices and max_jaccard_indices
                idx_to_remove = list(indices.intersection(set(max_jaccard_indices)))[0]
                indices.remove(idx_to_remove)
            for index in indices:
                most_similar_pred_coms[index] = -1

    else:
        return most_similar_pred_coms

    return most_similar_pred_coms


def mapping(gt_communities: list, pred_coms: list):
    """
    Map the ground-truth communities to their most similar predicted community.

    :param gt_communities: List containing the ground-truth communities
    :param pred_coms: List contianing the predicted communities
    :return achieved_distribution: List containing per ground-truth community how many nodes were correctly classified
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
    Compute the Jaccard similarity between two communities.

    :param com1: List containing the nodes of the first community
    :param com2: List containing the nodes of the second community
    :return: float, the Jaccard similarity
    """
    return len(set(com1).intersection(com2)) / len(set(com1).union(com2))


def find_max_jaccard(real_com: list, pred_coms: list):
    """
    Find the maximum Jaccard value between a ground-truth community and all predicted communities.

    :param real_com: The nodes of the ground-truth community
    :param pred_coms: A list containing the predicted communities
    :return: Tuple, the most similar predicted community together with the Jaccard score
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
    Split scores per community into scores per fairness type

    :param distribution_fraction: List containing the scores per community, either as distributions or fractions
    :param comm_types: List containing the fairness type of each community
    :return: type1, type2: Scores per fairness type
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
    """
    Transform the ground-truth communties and predicted communities to y_true and y_pred sklearn format,
    so we can calculate the F1 score.

    :param gt_communities: List containing the ground-truth communities
    :param pred_coms: List containing the predicted communities
    :param mapping_list: List of mapping from ground-truth community (index) to predicted community (value)
    :return: y_true, y_pred. Lists of the true communities and predicted communities
    """
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


def classify_graph(G: nx.Graph, percentile: int):
    """
    Classify an LFR Graph to see whether it is suitable for testing fairness.

    In order to be suitable for small-large fairness:
    - The network has at least 3 large communities and 3 small communities based on the percentile
    - The largest community is at least 5x(?) larger than the smallest community.

    In order to be suitable for dense-sparse fairness:
    - The network has at least 3 dense communities and 3 sparse communities based on the percentile
    - The most dense community is at least 5x(?) denser than the most sparse community.

    :param G:
    :param percentile:
    :return classified_graph: Tuple consisting of the graph in both places if it is suitable for both fairness types.
                              Else, the graph is in either of the two places, or in neither.
    """
    communities = list({frozenset(G.nodes[v]["community"]) for v in G})

    _, comm_types_size = small_large_communities(
        communities=communities, percentile=percentile
    )
    size_counters = Counter(comm_types_size).values()
    _, comm_types_density = dense_sparse_communities(
        G=G, communities=communities, percentile=percentile
    )
    density_counters = Counter(comm_types_density).values()

    intra_com_edges = np.array(
        [
            G.subgraph(communities[idx]).size()
            for idx, community in enumerate(communities)
        ]
    )
    sizes = [len(community) for community in communities]
    max_possible_edges = np.array([(size * (size - 1)) / 2 for size in sizes])
    densities = np.array(intra_com_edges / max_possible_edges)

    size_bool = max(sizes) / min(sizes) >= 5 and all(x >= 3 for x in size_counters)
    density_bool = max(densities) / min(densities) >= 5 and all(
        x >= 3 for x in density_counters
    )

    if size_bool and density_bool:
        return G, G
    elif size_bool and not density_bool:
        return G, None
    elif not size_bool and density_bool:
        return None, G
    else:
        return None, None
