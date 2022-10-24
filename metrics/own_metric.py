import networkx as nx
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance

from utils import (
    mapping,
    small_large_communities,
    dense_sparse_communities,
    split_types,
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


def f1_fairness(gt_communities: list, pred_coms: list, mapping_list: list):
    # Using sklearn implementation requires me to write some code te get y_true and y_pred
    y_true, y_pred = transform_to_ytrue_ypred(gt_communities, pred_coms, mapping_list)

    return f1_score(
        y_true, y_pred, average=None, labels=list(range(len(gt_communities)))
    )


def score_per_comm_to_fairness(score_per_comm: list, comm_types: list):
    type1_score, type2_score = split_types(score_per_comm, comm_types=comm_types)
    return fairness_score(
        type1_score=np.average(type1_score), type2_score=np.average(type2_score)
    )


def emd_fairness(real_fractions: list, achieved_fractions: list, comm_types: list):
    """
    :param real_fractions:
    :param achieved_fractions:
    :param comm_types:
    :return:
    """
    real_fractions_type1, real_fractions_type2 = split_types(
        distribution_fraction=real_fractions,
        comm_types=comm_types,
    )
    achieved_fractions_type1, achieved_fractions_type2 = split_types(
        distribution_fraction=achieved_fractions,
        comm_types=comm_types,
    )

    fairness_emd_type1 = wasserstein_distance(
        u_values=real_fractions_type1, v_values=achieved_fractions_type1
    )
    fairness_emd_type2 = wasserstein_distance(
        u_values=real_fractions_type2, v_values=achieved_fractions_type2
    )

    score = fairness_score(
        type1_score=fairness_emd_type1, type2_score=fairness_emd_type2
    )

    print(f"Achieved fractions type 1: {np.round(achieved_fractions_type1, 2)}")
    print(f"Achieved fractions type 2: {np.round(achieved_fractions_type2, 2)}")

    return score


def calculate_fairness_metrics(
    G: nx.Graph,
    gt_communities: list,
    pred_communities: list,
    fairness_type: str,
    percentile=75,
):
    """
    Calculate and print out the fairness metric for a given LFR graph with ground-truth and predicted communities.

    :param G: The NetworkX graph for which we want to compute the fairness
    :param gt_communities: List of ground-truth communities
    :param pred_communities: List of communities as predicted by CD method
    :param fairness_type: String indicating size, density
    :param percentile: Integer percentile of the small-large or density cutoff
    :return:
    """
    # Distributions / fractions
    real_distribution = [len(community) for community in gt_communities]
    real_fractions = [1] * len(gt_communities)

    achieved_distribution, mapping_list = mapping(
        gt_communities=gt_communities, pred_coms=pred_communities
    )
    achieved_fractions = list(
        np.array(achieved_distribution) / np.array(real_distribution)
    )

    # Decide which type of fairness we are looking into
    if fairness_type == "size":
        node_comm_types, comm_types = small_large_communities(
            communities=gt_communities, percentile=percentile
        )
    else:  # fairness_type == "density" -->  TODO Perhaps: Add a third option for fairness type
        node_comm_types, comm_types = dense_sparse_communities(
            G=G, communities=gt_communities, percentile=percentile
        )

    f1_per_comm = f1_fairness(
        gt_communities=gt_communities,
        pred_coms=pred_communities,
        mapping_list=mapping_list,
    )

    emd_fairness_score = emd_fairness(
        real_fractions=real_fractions,
        achieved_fractions=achieved_fractions,
        comm_types=comm_types,
    )
    f1_fairness_score = score_per_comm_to_fairness(
        score_per_comm=f1_per_comm, comm_types=comm_types
    )
    accuracy_fairness_score = score_per_comm_to_fairness(
        score_per_comm=achieved_fractions, comm_types=comm_types
    )

    return emd_fairness_score, f1_fairness_score, accuracy_fairness_score
