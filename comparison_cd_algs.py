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
from metrics.ground_truth import modified_ari, modified_f_measure
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
    # girvan_newman: [
    #     3
    # ],  # Took 20 minutes for ONE dataset! Very slow, not sure if I should include it
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
    # mod_m: {"query_node": None},  # Only gives the community of the query node. Not sure if thats interesting, perhaps
    # mod_r: {"query_node": None},  # I should write code to query all nodes individually, and then combine the results
    # head_tail: None,
    kcut: None,
    gemsec: None,
    scd: None,
    pycombo: None,
    paris: None,
    ricci_community: None,
    spectral: None,
    # mcode: None,
    "r_spectral_clustering-vanilla": None,
    "r_spectral_clustering-regularized_with_kmeans": None,
    "r_spectral_clustering-sklearn_spectral_embedding": None,
    "r_spectral_clustering-sklearn_kmeans": None,
    Node2Vec: None,
}
FAIRNESS_TYPES = ["size", "density"]
PERCENTILE = {"size": 75, "density": 50}
CONFIGS = list(product(CD_METHODS.keys(), FAIRNESS_TYPES))

ray.init()


@ray.remote
def evaluate_method(config):
    cd_method = config[0]
    fairness_type = config[1]

    try:
        print(
            f"{datetime.now()}: Starting with {cd_method.__name__} for fairness type {fairness_type}"
        )
    except AttributeError:  # happens for r_spectral_clustering
        print(
            f"{datetime.now()}: Starting with {cd_method} for fairness type {fairness_type}"
        )

    with open(f"data/labeled/lfr/{fairness_type}_seeds.txt") as seeds_file:
        seeds = [line.rstrip() for line in seeds_file][0]  # TODO: Change me for HPC, or maybe keep it as a first test

    emd = []
    f1 = []
    fcc = []
    ari = []
    vi = []
    ari_m = []
    F_measure_m = []
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
                if cd_method in [
                    "r_spectral_clustering-vanilla",
                    "r_spectral_clustering-regularized_with_kmeans",
                    "r_spectral_clustering-sklearn_spectral_embedding",
                    "r_spectral_clustering-sklearn_kmeans",
                ]:
                    method = cd_method.split("-")[1]  # Grab the element after the - icon
                    cd_method_function = r_spectral_clustering
                    pred_coms = cd_method_function(g_original=G, method=method)
                elif cd_method.__name__ == "Node2Vec":
                    g_emb = Node2Vec(G)
                    mdl = g_emb.fit()
                    emb_df = pd.DataFrame(
                        [mdl.wv.get_vector(str(n)) for n in G.nodes()], index=G.nodes
                    )
                    kmeans = KMeans(n_clusters=len(gt_communities), random_state=0).fit(
                        emb_df
                    )
                    pred_coms = transform_sklearn_labels_to_communities(
                        labels=kmeans.labels_
                    )
                    pred_coms = NodeClustering(communities=pred_coms, graph=G)

                else:
                    if CD_METHODS[cd_method] is None:
                        pred_coms = cd_method(g_original=G)
                    else:  # Fill in other parameters
                        for idx, value in enumerate(CD_METHODS[cd_method]):
                            if value is None:
                                CD_METHODS[cd_method][idx] = len(gt_communities)
                        pred_coms = cd_method(G, *CD_METHODS[cd_method])
            except Exception as e:  # In case of convergence error or something else
                print(repr(e))
                emd.append(None)
                f1.append(None)
                fcc.append(None)
                ari.append(None)
                vi.append(None)
                continue

            (
                emd_fairness_score,
                f1_fairness_score,
                fcc_fairness_score,
            ) = calculate_fairness_metrics(
                G=G,
                gt_communities=gt_communities,
                pred_communities=pred_coms.communities,
                fairness_type=fairness_type,
                percentile=PERCENTILE[fairness_type],
            )

            emd.append(emd_fairness_score)
            f1.append(f1_fairness_score)
            fcc.append(fcc_fairness_score)

            ari.append(adjusted_rand_index(first_partition=cdlib_communities, second_partition=pred_coms))
            vi.append(variation_of_information(first_partition=cdlib_communities, second_partition=pred_coms))
            ari_m.append(modified_ari(pred_coms=pred_coms.communities, real_coms=gt_communities, G=G))
            F_measure_m.append(modified_f_measure(pred_coms=pred_coms.communities, real_coms=gt_communities, G=G))

    fairness_scores = (emd, f1, fcc)
    evaluation_scores = (ari, vi, ari_m, F_measure_m)

    try:
        with open(
            f"results/Initial comparison/{fairness_type}-{cd_method.__name__}-fairness-scores.pickle",
            "wb",
        ) as handle:
            pickle.dump(fairness_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(
            f"results/Initial comparison/{fairness_type}-{cd_method.__name__}-evaluation-scores.pickle",
            "wb",
        ) as handle:
            pickle.dump(evaluation_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"{datetime.now()}: Done with {cd_method.__name__} for fairness type {fairness_type}"
        )
    except AttributeError:  # Happens for r_spectral_clustering
        with open(
            f"results/Initial comparison/{fairness_type}-{cd_method}-fairness-scores.pickle",
            "wb",
        ) as handle:
            pickle.dump(fairness_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(
            f"results/Initial comparison/{fairness_type}-{cd_method}-evaluation-scores.pickle",
            "wb",
        ) as handle:
            pickle.dump(evaluation_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"{datetime.now()}: Done with {cd_method} for fairness type {fairness_type}"
        )


if __name__ == "__main__":
    for configuration in CONFIGS:
        # evaluate_method(configuration)
        evaluate_method.remote(configuration)
