import networkx as nx
import numpy as np

from utils import draw_graph
from data.labeled.casestudy.synthetic_graphs import (
    varying_mu_values,
    varying_denseness,
    calculate_fairness_metric,
)
from metrics.own_metric import fair_unfair_nodes, fairness_gt


# Varying mu values for pre-specified community sizes
num_small = 30
num_large = 30
mus = np.linspace(0.01, 0.5, 15)
# varying_mu_values(num_nodes_G1=num_small, num_nodes_G2=num_large, mus=mus)


# Varying denseness between two communities
num_nodes = 10
densenesses_G1 = [0.6, 0.7, 0.8, 0.9]
densenesses_G2 = [0.65, 0.75, 0.85, 0.9]
inter_community_edges = 0.1
# varying_denseness(
#     num_nodes=num_nodes,
#     densenesses_G1=densenesses_G1,
#     densenesses_G2=densenesses_G2,
#     inter_community_edges=inter_community_edges,
# )


# Fairness metric on karate club: 0.9705882352941176 --> One unfair node (maybe node 3 as in barabasi chapter?)
# See myplot.png, node with label 8 is labeled as unfair
G = nx.karate_club_graph()
mr_hi = set()
officer = set()
for node in G.nodes:
    if G.nodes[node]["club"] == "Mr. Hi":
        mr_hi.add(node)
    else:
        officer.add(node)
gt_communities = list([frozenset(mr_hi), frozenset(officer)])
communities = set(gt_communities)

#  Not important drawing stuff, just for myself
pos = nx.spring_layout(G)  # compute graph layout
draw_graph(G, pos=pos, communities=communities)

fair_nodes, unfair_nodes = fair_unfair_nodes(G, gt_communities)
print(
    f"Fairness metric: {fairness_gt(fair_nodes=fair_nodes, unfair_nodes=unfair_nodes)} \n"
)
