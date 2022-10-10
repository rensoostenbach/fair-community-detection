import networkx as nx
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance

from utils import (
    mapping,
    small_large_communities,
    dense_nondense_communities,
    split_distribution,
    transform_to_ytrue_ypred,
)


def fairness_score(type1_score: float, type2_score: float):
    """
    Compare the scores of two types of communities via taking the absolute difference and dividing over their sum.
    Value closer to 0, the better.

    :param type1_score:
    :param type2_score:
    :return:
    """
    if (
        type1_score == type2_score == 0
    ):  # Would give division by 0 error. Occurs when classification is perfect.
        return 0
    else:
        return abs(type1_score - type2_score) / (type1_score + type2_score)


def accuracy_fairness(
    achieved_fractions: list,
    achieved_fractions_type1: list,
    achieved_fractions_type2: list,
):
    """

    :param achieved_fractions:
    :param achieved_fractions_type1:
    :param achieved_fractions_type2:
    :return:
    """
    simple_fairness_metric = np.average(achieved_fractions)
    simple_fairness_metric_types = fairness_score(
        type1_score=np.average(achieved_fractions_type1),
        type2_score=np.average(achieved_fractions_type2),
    )
    print(f"Accuracy fairness overall: {simple_fairness_metric}")
    print(f"Accuracy fairness compared: {simple_fairness_metric_types}")


def f1_fairness(gt_communities: list, pred_coms: list, mapping: dict):  # TODO:
    # Not sure if I should use sklearn implementation, or simply build a confusion matrix myself
    # to calculate the harmonic mean of precision and recall

    # Using sklearn implementation requires me to write some code te get y_true and y_pred
    y_true, y_pred = transform_to_ytrue_ypred(gt_communities, pred_coms, mapping)


    return f1_score(y_true, y_pred, average=None)


def emd_fairness(
    real_fractions: list,
    achieved_fractions: list,
    real_fractions_type1: list,
    real_fractions_type2: list,
    achieved_fractions_type1: list,
    achieved_fractions_type2: list,
):
    """

    :param real_fractions:
    :param achieved_fractions:
    :param real_fractions_type1:
    :param real_fractions_type2:
    :param achieved_fractions_type1:
    :param achieved_fractions_type2:
    :return:
    """
    # Same as accuracy fairness
    fairness_emd = 1 - wasserstein_distance(
        u_values=real_fractions, v_values=achieved_fractions
    )

    fairness_emd_type1 = wasserstein_distance(
        u_values=real_fractions_type1, v_values=achieved_fractions_type1
    )
    fairness_emd_type2 = wasserstein_distance(
        u_values=real_fractions_type2, v_values=achieved_fractions_type2
    )

    fairness_emd_types = fairness_score(
        type1_score=fairness_emd_type1, type2_score=fairness_emd_type2
    )

    print(f"EMD fairness overall: {fairness_emd}")
    print(f"EMD fairness compared: {fairness_emd_types}")


def calculate_fairness_metrics(
    G: nx.Graph,
    gt_communities: list,
    pred_communities: list,
    fairness_type: str,
    size_percentile=90,
    density_cutoff=0.5,
):
    """
    Calculate and print out the fairness metric for a given LFR graph with ground-truth and predicted communities.

    :param G: The NetworkX graph for which we want to compute the fairness
    :param gt_communities: List of ground-truth communities
    :param pred_communities: List of communities as predicted by CD method
    :param fairness_type: String indicating small/large, dense/non-dense
    :param size_percentile: Integer percentile of the small-large cutoff
    :param density_cutoff: Float from 0 to 1 indication the cutoff ratio for density
    :return:
    """
    # Distributions / fractions
    real_distribution = [len(community) for community in gt_communities]
    real_fractions = [1] * len(gt_communities)

    achieved_distribution, mapping_dict = mapping(
        gt_communities=gt_communities, pred_coms=pred_communities
    )
    achieved_fractions = list(
        np.array(achieved_distribution) / np.array(real_distribution)
    )

    # Decide which type of fairness we are looking into
    if fairness_type == "small_large":
        node_comm_types, comm_types = small_large_communities(
            communities=gt_communities, percentile=size_percentile
        )
    else:  # fairness_type == "density" --> Perhaps TODO: Add a third option for fairness type
        node_comm_types, comm_types = dense_nondense_communities(
            G=G, communities=gt_communities, cutoff=density_cutoff
        )

    real_dist_type1, real_dist_type2 = split_distribution(
        distribution=real_distribution,
        comm_types=comm_types,
    )
    achieved_dist_type1, achieved_dist_type2 = split_distribution(
        distribution=achieved_distribution,
        comm_types=comm_types,
    )

    real_fractions_type1 = [1] * len(real_dist_type1)
    real_fractions_type2 = [1] * len(real_dist_type2)

    achieved_fractions_type1 = list(
        np.array(achieved_dist_type1) / np.array(real_dist_type1)
    )
    achieved_fractions_type2 = list(
        np.array(achieved_dist_type2) / np.array(real_dist_type2)
    )

    accuracy_fairness(
        achieved_fractions=achieved_fractions,
        achieved_fractions_type1=achieved_fractions_type1,
        achieved_fractions_type2=achieved_fractions_type2,
    )
    test = f1_fairness(gt_communities=gt_communities, pred_coms=pred_communities, mapping=mapping_dict)
    emd_fairness(
        real_fractions=real_fractions,
        achieved_fractions=achieved_fractions,
        real_fractions_type1=real_fractions_type1,
        real_fractions_type2=real_fractions_type2,
        achieved_fractions_type1=achieved_fractions_type1,
        achieved_fractions_type2=achieved_fractions_type2,
    )

    return node_comm_types
