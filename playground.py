import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from cdlib import algorithms

from metrics.own_metric import fairness, fairness_small_large
from metrics.ground_truth import purity, inverse_purity
from utils import draw_graph, small_large_communities


n = 1000
tau1 = 3
tau2 = 1.5
mu = 0.3

G = LFR_benchmark_graph(
    n=n, tau1=tau1, tau2=tau2, mu=mu, min_degree=4, min_community=25, seed=0
)
G.remove_edges_from(nx.selfloop_edges(G))

communities = {frozenset(G.nodes[v]["community"]) for v in G}
gt_communities = list(communities)

#  Not important drawing stuff, just for myself
# Coloring every node such that communities have the same color
# node_color = [0] * n
# pos = nx.spring_layout(G)  # compute graph layout
# draw_graph(G, pos=pos, communities=communities)

pred_coms = algorithms.louvain(g_original=G)

(
    fairness_score,
    fair_pred_nodes,
    unfair_pred_nodes,
    fair_real_nodes,
    unfair_real_nodes,
) = fairness(G=G, pred_coms=pred_coms.communities, real_coms=gt_communities)

print(f"Our own (fairness) metric: {fairness_score}")
print(f"Purity: {purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}")
print(
    f"Inverse purity: {inverse_purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}"
)

small_large = small_large_communities(communities=gt_communities, percentile=90)
print(
    f"Small/large fairness of Louvain: {fairness_small_large(small_large=small_large, unfair_nodes=unfair_pred_nodes)}"
)

print("debugger here")
