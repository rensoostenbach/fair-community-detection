import pickle
import networkx as nx
import ray
from datetime import datetime
from itertools import product
from cdlib.algorithms import (
    eigenvector,
    label_propagation,
    leiden,
    louvain,
    spinglass,
    infomap,
)
from cdlib.evaluation import adjusted_rand_index, variation_of_information
from cdlib import NodeClustering

from metrics.own_metric import calculate_fairness_metrics

"""
Currently works for DBLP, LiveJournal, YouTube, all petsters, blogcatalog, flickr
Not using Ray because I can't fit that much into my RAM on my local machine.
"""

# TODO: Can make this file run all datasets in a for loop

PERCENTILE = 75
CD_METHODS = [eigenvector, label_propagation, leiden, louvain]
FAIRNESS_TYPES = ["size", "density"]
CONFIGS = list(product(CD_METHODS, FAIRNESS_TYPES))


def evaluate_method(config):
    cd_method = config[0]
    fairness_type = config[1]
    print(
        f"{datetime.now()}: Starting with {cd_method.__name__} for fairness type {fairness_type}"
    )

    emd = []
    f1 = []
    fcc = []
    ari = []
    vi = []

    with open(f"data/labeled/real_world/DBLP/DBLP_top5000.pickle", "rb") as graph_file:

        G = pickle.load(graph_file)

    G.remove_edges_from(nx.selfloop_edges(G))

    print(f"Graph characteristics: {str(G)}")

    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)
    cdlib_communities = NodeClustering(communities=gt_communities, graph=G)

    # CD part
    pred_coms = cd_method(g_original=G)

    (
        emd_fairness_score,
        f1_fairness_score,
        fcc_fairness_score,
    ) = calculate_fairness_metrics(
        G=G,
        gt_communities=gt_communities,
        pred_communities=pred_coms.communities,
        fairness_type="size",
        percentile=PERCENTILE,
    )

    emd.append(emd_fairness_score)
    f1.append(f1_fairness_score)
    fcc.append(fcc_fairness_score)
    ari.append(adjusted_rand_index(cdlib_communities, pred_coms))
    vi.append(variation_of_information(cdlib_communities, pred_coms))

    fairness_scores = (emd, f1, fcc)
    evaluation_scores = (ari, vi)

    with open(
        f"results/DBLP/{fairness_type}-{cd_method.__name__}-fairness-scores.pickle",
        "wb",
    ) as handle:
        pickle.dump(fairness_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"results/DBLP/{fairness_type}-{cd_method.__name__}-evaluation-scores.pickle",
        "wb",
    ) as handle:
        pickle.dump(evaluation_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"{datetime.now()}: Done with {cd_method.__name__} for fairness type {fairness_type}"
    )


if __name__ == "__main__":
    for configuration in CONFIGS:
        evaluate_method(configuration)
