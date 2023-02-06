from cdlib.algorithms import mod_m
from cdlib.evaluation import *
from cdlib import NodeClustering
import networkx as nx
import pickle

from metrics.own_metric import calculate_fairness_metrics
from metrics.ground_truth import modified_ari, modified_f_measure

fairness_type = "size"

with open(f"data/labeled/lfr/{fairness_type}_seeds.txt") as seeds_file:
    seeds = [line.rstrip() for line in seeds_file][0]

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
        pred_coms = mod_m(g_original=G, query_node=0)

        print("debug")

        (
            emd_fairness_score,
            f1_fairness_score,
            fcc_fairness_score,
        ) = calculate_fairness_metrics(
            G=G,
            gt_communities=gt_communities,
            pred_communities=pred_coms.communities,
            fairness_type=fairness_type,
            percentile=75,
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
