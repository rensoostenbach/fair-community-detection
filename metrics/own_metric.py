import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
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
    Value closer to 1, the better.

    :param type1_score: Score of fairness type 1
    :param type2_score: Score of fairness type 2
    :return: Float, fairness score that compares fairness type 1 and 2.
    """
    if (
        type1_score == type2_score == 0
    ):  # Would give division by 0 error. Occurs when classification is perfect.
        return 1
    else:
        return 1 - (abs(type1_score - type2_score) / (type1_score + type2_score))


def f1_fairness(gt_communities: list, pred_coms: list, mapping_list: list):
    """
    Compute the F1 score per community

    :param gt_communities: List containing the ground-truth communities
    :param pred_coms: List containing the predicted communities
    :param mapping_list: List of mapping from ground-truth community (index) to predicted community (value)
    :return: F1 score per community
    """
    # Using sklearn implementation requires me to write some code te get y_true and y_pred
    y_true, y_pred = transform_to_ytrue_ypred(
        gt_communities=gt_communities, pred_coms=pred_coms, mapping_list=mapping_list
    )

    return precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(gt_communities))),
        zero_division=0,
    )


def score_per_comm_to_fairness(score_per_comm: list, comm_types: list):
    """
    Transform the score per community to a fairness score

    :param score_per_comm: Scores per community
    :param comm_types: Fairness type per community
    :return: Float, fairness score
    """
    type1_score, type2_score = split_types(score_per_comm, comm_types=comm_types)
    return fairness_score(
        type1_score=np.average(type1_score), type2_score=np.average(type2_score)
    )


def emd_fairness(real_fractions: list, achieved_fractions: list, comm_types: list):
    """
    :param real_fractions: List containing 1's for the length of the number of ground-truth communities there are
    :param achieved_fractions: List containing the fractions of correctly classified nodes per ground-truth community
    :param comm_types: Fairness type per community
    :return: Float, EMD Fairness
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

    return score


def calculate_fairness_metrics(
    G: nx.Graph,
    gt_communities: list,
    pred_communities: list,
    fairness_type: str,
    percentile=75,
    alpha=0,
    interpret_results=False,
):
    """
    Calculate the fairness metric for a given LFR graph with ground-truth and predicted communities.

    :param G: The NetworkX graph for which we want to compute the fairness
    :param gt_communities: List of ground-truth communities
    :param pred_communities: List of communities as predicted by CD method
    :param fairness_type: String indicating size, density
    :param percentile: Integer percentile of the small-large or density cutoff
    :param alpha: Weighting parameter between 0 and 1. Can be either float (global) or dict (relative to community type)
    :param interpret_results: Boolean indicating whether to return intermediate results to interpret them
    :return: Tuple containing all three fairness scores
    """

    if isinstance(alpha, (float, int)):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha should be between 0 and 1.")
    elif isinstance(alpha, dict):
        for value in alpha.values():
            if not 0 <= value <= 1:
                raise ValueError("Alpha should be between 0 and 1.")

    # Distributions / fractions
    real_distribution = [len(community) for community in gt_communities]
    real_fractions = [1] * len(gt_communities)

    achieved_distribution, mapping_list = mapping(
        gt_communities=gt_communities, pred_coms=pred_communities
    )
    achieved_fractions = list(
        np.array(achieved_distribution) / np.array(real_distribution)
    )
    achieved_fractions_unweighted = achieved_fractions

    # Decide which type of fairness we are looking into
    if fairness_type == "size":
        node_comm_types, comm_types = small_large_communities(
            communities=gt_communities, percentile=percentile
        )
    else:  # fairness_type == "density" -->  TODO Perhaps: Add a third option for fairness type
        node_comm_types, comm_types = dense_sparse_communities(
            G=G, communities=gt_communities, percentile=percentile
        )

    # Calculating all scores here, depending on the weighting for FCC and F1 fairness.
    # No weighting needed for EMD
    emd_fairness_score = emd_fairness(
        real_fractions=real_fractions,
        achieved_fractions=achieved_fractions,
        comm_types=comm_types,
    )

    TP_per_comm = np.array(achieved_distribution)
    FN_per_comm_squared = (
        np.array(real_distribution) - np.array(achieved_distribution)
    ) ** 2

    unweighted_fcc = score_per_comm_to_fairness(
        score_per_comm=achieved_fractions_unweighted, comm_types=comm_types
    )
    achieved_fractions_weighted = TP_per_comm / (TP_per_comm + FN_per_comm_squared)
    weighted_fcc = score_per_comm_to_fairness(
        score_per_comm=achieved_fractions_weighted, comm_types=comm_types
    )

    precision, recall, unweighted_f1_per_comm, support = f1_fairness(
        gt_communities=gt_communities,
        pred_coms=pred_communities,
        mapping_list=mapping_list,
    )

    unweighted_f1 = score_per_comm_to_fairness(
        score_per_comm=unweighted_f1_per_comm, comm_types=comm_types
    )

    y_true, y_pred = transform_to_ytrue_ypred(
        gt_communities=gt_communities,
        pred_coms=pred_communities,
        mapping_list=mapping_list,
    )
    conf_matrix = confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=list(range(len(gt_communities)))
    )
    FP_per_comm_squared = (conf_matrix.sum(axis=0) - np.diag(conf_matrix)) ** 2
    weighted_f1_per_comm = (
        2 * TP_per_comm / (2 * TP_per_comm + FP_per_comm_squared + FN_per_comm_squared)
    )
    weighted_f1 = score_per_comm_to_fairness(
        score_per_comm=weighted_f1_per_comm, comm_types=comm_types
    )

    # Use the alpha parameter to calculate the final scores
    if isinstance(alpha, (float, int)):
        fcc_fairness_score = ((1 - alpha) * unweighted_fcc) + (alpha * weighted_fcc)
        f1_fairness_score = ((1 - alpha) * unweighted_f1) + (alpha * weighted_f1)
    else:  # Alpha is a dict
        fcc_fairness_scores = [None] * len(comm_types)
        f1_fairness_scores = [None] * len(comm_types)
        for i, comm_type in enumerate(comm_types):
            fcc_fairness_scores[i] = (
                (1 - alpha[comm_type]) * achieved_fractions_unweighted[i]
            ) + (alpha[comm_type] * achieved_fractions_weighted[i])
            f1_fairness_scores[i] = (
                (1 - alpha[comm_type]) * unweighted_f1_per_comm[i]
            ) + (alpha[comm_type] * weighted_f1_per_comm[i])
        fcc_fairness_score = score_per_comm_to_fairness(
            score_per_comm=fcc_fairness_scores, comm_types=comm_types
        )
        f1_fairness_score = score_per_comm_to_fairness(
            score_per_comm=f1_fairness_scores, comm_types=comm_types
        )

    if interpret_results:
        fractions_type1, fractions_type2 = split_types(
            distribution_fraction=achieved_fractions, comm_types=comm_types
        )
        f1_type1, f1_type2 = split_types(
            distribution_fraction=unweighted_f1_per_comm, comm_types=comm_types
        )
        precision_type1, precision_type2 = split_types(
            distribution_fraction=precision, comm_types=comm_types
        )
        recall_type1, recall_type2 = split_types(
            distribution_fraction=recall, comm_types=comm_types
        )
        return (
            emd_fairness_score,
            f1_fairness_score,
            fcc_fairness_score,
            weighted_f1,
            weighted_fcc,
            fractions_type1,
            fractions_type2,
            f1_type1,
            f1_type2,
            precision_type1,
            precision_type2,
            recall_type1,
            recall_type2,
            mapping_list,
            comm_types,
        )
    else:
        return emd_fairness_score, f1_fairness_score, fcc_fairness_score
