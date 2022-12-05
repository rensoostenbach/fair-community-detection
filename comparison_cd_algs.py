import pickle
import networkx as nx
import pandas as pd
from datetime import datetime
from itertools import product
from sklearn.cluster import KMeans
from cdlib.algorithms import *
import ray
from node2vec import Node2Vec
from cdlib.evaluation import adjusted_rand_index, variation_of_information
from cdlib import NodeClustering

from metrics.own_metric import calculate_fairness_metrics
from utils import transform_sklearn_labels_to_communities
"""
This file will be used to compare all CD Methods on their fairness and accuracy.
"""

# SCAN en mcode: node_coverage is niet perse 1.
# head_tail: Process finished with exit code -1073741571 (0xC00000FD)

CD_METHODS = {
    louvain: None,
    leiden: None,
    rb_pots: None,
    rber_pots: None,
    cpm: None,
    significance_communities: None,
    surprise_communities: None,
    greedy_modularity: None,
    der: None,
    label_propagation: None,
    async_fluid: [None],
    infomap: None,
    walktrap: None,
    girvan_newman: [3],
    em: [None],
    # scan: [0.7, 3],
    gdmp2: None,
    spinglass: None,
    eigenvector: None,
    agdl: [None, 4],
    sbm_dl: None,
    sbm_dl_nested: None,
    markov_clustering: None,
    edmot: None,
    chinesewhispers: None,
    ga: None,
    belief: None,
    threshold_clustering: None,
    # mod_m: {"query_node": None},  # TODO: Need to find out what to do with this
    # mod_r: {"query_node": None},  # TODO: Need to find out what to do with this
    # head_tail: None,
    kcut: None,
    gemsec: None,
    scd: None,
    pycombo: None,
    paris: None,
    ricci_community: None,
    spectral: None,
    # mcode: None,
    # r_spectral_clustering: {
    #     "n_clusters": None,
    #     "method": [
    #         "vanilla",
    #         "regularized_with_kmeans",
    #         "sklearn_spectral_embedding",
    #         "sklearn_kmeans",
    #     ],  # TODO: Take into account that we need to do 4 runs, one for each different method value
    # },
    Node2Vec: None,
}
FAIRNESS_TYPES = ["size", "density"]
PERCENTILE = 75
CONFIGS = list(product(CD_METHODS.keys(), FAIRNESS_TYPES))

ray.init(num_cpus=8)


@ray.remote
def evaluate_method(config):
    cd_method = config[0]
    fairness_type = config[1]

    print(
        f"{datetime.now()}: Starting with {cd_method.__name__} for fairness type {fairness_type}"
    )
    with open(f"data/labeled/lfr/{fairness_type}_seeds.txt") as seeds_file:
        seeds = [line.rstrip() for line in seeds_file][0]

    emd = []
    f1 = []
    acc = []
    ari = []
    vi = []
    for seed in seeds:
        with open(
            f"data/labeled/lfr/{fairness_type}_graph_{seed}.pickle", "rb"
        ) as graph_file:
            G = pickle.load(graph_file)
            G.remove_edges_from(nx.selfloop_edges(G))

            communities = {frozenset(G.nodes[v]["community"]) for v in G}
            gt_communities = list(communities)
            cdlib_communities = NodeClustering(communities=gt_communities, graph=G)

            # CD part
            try:
                if cd_method.__name__ == "Node2Vec":
                    g_emb = Node2Vec(G)
                    mdl = g_emb.fit()
                    emb_df = (
                        pd.DataFrame(
                            [mdl.wv.get_vector(str(n)) for n in G.nodes()],
                            index=G.nodes
                        )
                    )
                    kmeans = KMeans(n_clusters=len(gt_communities), random_state=0).fit(emb_df)
                    pred_coms = transform_sklearn_labels_to_communities(labels=kmeans.labels_)
                    pred_coms = NodeClustering(communities=pred_coms, graph=G)
                else:
                    if CD_METHODS[cd_method] is None:
                        pred_coms = cd_method(g_original=G)
                    else:  # Fill in other parameters
                        for idx, value in enumerate(CD_METHODS[cd_method]):
                            if value is None:
                                CD_METHODS[cd_method][idx] = len(gt_communities)
                        pred_coms = cd_method(G, *CD_METHODS[cd_method])
            except Exception:  # In case of convergence error or something else
                emd.append(None)
                f1.append(None)
                acc.append(None)
                ari.append(None)
                vi.append(None)
                continue

            (
                emd_fairness_score,
                f1_fairness_score,
                accuracy_fairness_score,
            ) = calculate_fairness_metrics(
                G=G,
                gt_communities=gt_communities,
                pred_communities=pred_coms.communities,
                fairness_type="size",
                percentile=PERCENTILE,
            )

            emd.append(emd_fairness_score)
            f1.append(f1_fairness_score)
            acc.append(accuracy_fairness_score)
            ari.append(adjusted_rand_index(cdlib_communities, pred_coms))
            vi.append(variation_of_information(cdlib_communities, pred_coms))

    fairness_scores = (emd, f1, acc)
    evaluation_scores = (ari, vi)

    with open(
        f"results/Comparison test 1 dataset/{fairness_type}-{cd_method.__name__}-fairness-scores.pickle",
        "wb",
    ) as handle:
        pickle.dump(fairness_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"results/Comparison test 1 dataset/{fairness_type}-{cd_method.__name__}-evaluation-scores.pickle",
        "wb",
    ) as handle:
        pickle.dump(evaluation_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"{datetime.now()}: Done with {cd_method.__name__} for fairness type {fairness_type}"
    )


if __name__ == "__main__":
    for configuration in CONFIGS:
        # evaluate_method(configuration)
        evaluate_method.remote(configuration)
