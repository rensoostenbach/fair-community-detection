import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from cdlib import algorithms

from metrics.ground_truth import purity, inverse_purity
from metrics.own_metric import calculate_fairness_metrics
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

# CD part
pred_coms = algorithms.louvain(g_original=G, randomize=0)

#  Not important drawing stuff, just for myself
# pos = nx.spring_layout(G)  # compute graph layout
# draw_graph(G, pos=pos, communities=communities)

calculate_fairness_metrics(
    G=G,
    gt_communities=gt_communities,
    pred_communities=pred_coms.communities,
    fairness_type="small_large",
    size_percentile=90,
)

print(f"Purity: {purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}")
print(
    f"Inverse purity: {inverse_purity(pred_coms=pred_coms.communities, real_coms=gt_communities)}"
)

print("debugger here")
