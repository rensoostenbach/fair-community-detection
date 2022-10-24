import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import numpy as np
import matplotlib.pyplot as plt

from utils import draw_graph

""" 
This playground was used to get a feeling of how the LFR benchmark works.
By experimenting with this, I found out that it is very sensitive, and I
will work with the LFR benchmark in a different way than initially thought
(more detailed notes are in Obsidian)
"""


def print_densities(graph: nx.Graph, communities: list):
    intra_com_edges = np.array(
        [
            graph.subgraph(communities[idx]).size()
            for idx, community in enumerate(communities)
        ]
    )
    # Need to divide above numbers by maximum amount of edges possible in community
    sizes = [len(community) for community in communities]
    max_possible_edges = np.array([(size * (size - 1)) / 2 for size in sizes])
    densities = np.array(intra_com_edges / max_possible_edges)
    print(f"Densities of these communities: {densities}")


# This shows that communities become more and more cluttered for larger values of mu
# Probably shouldn't go higher than 0.3, maybe even lower
# However, becoming more or less cluttered might not be a good reason for unfairness,
# like community sizes or community denseness
# n = 250
# tau1 = 3
# tau2 = 1.1
# mus = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
#
# for mu in mus:
#     G = LFR_benchmark_graph(
#         n=n, tau1=tau1, tau2=tau2, mu=mu, min_degree=3, min_community=int(0.1*n), seed=0
#     )
#     G.remove_edges_from(nx.selfloop_edges(G))
#
#     communities = {frozenset(G.nodes[v]["community"]) for v in G}
#     gt_communities = list(communities)
#     print_densities(graph=G, communities=gt_communities)
#     #  Not important drawing stuff, just for myself
#     pos = nx.spring_layout(G)  # compute graph layout
#     draw_graph(G, pos=pos, communities=communities)

# This shows that higher values of tau1 have a more "even" degree distribution
# Lower values of tau produce graphs with a few nodes having lots of links
# tau1s = [2, 2.25, 2.5, 2.75, 3]
# tau2 = 1.1
# n = 250
# mu = 0.2
#
# for tau1 in tau1s:
#     try:
#         G = LFR_benchmark_graph(
#             n=n, tau1=tau1, tau2=tau2, mu=mu, min_degree=3, min_community=25, seed=7
#         )
#     except nx.ExceededMaxIterations:
#         continue
#     G.remove_edges_from(nx.selfloop_edges(G))
#
#     communities = {frozenset(G.nodes[v]["community"]) for v in G}
#     gt_communities = list(communities)
#     degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
#     dmax = max(degree_sequence)
#     fig, ax = plt.subplots()
#     ax.bar(*np.unique(degree_sequence, return_counts=True))
#     ax.set_title(f"Degree histogram with tau1={tau1}")
#     ax.set_xlabel("Degree")
#     ax.set_ylabel("# of Nodes")
#     plt.show()

#     #  Not important drawing stuff, just for myself
#     # pos = nx.spring_layout(G)  # compute graph layout
#     # draw_graph(G, pos=pos, communities=communities)


# This shows that increasing min_degree subsequently increases the denseness in the whole graph
# But not necessarily per community!
# tau1 = 3
# tau2 = 1.5
# n = 250
# mu = 0.1
# min_degrees = list(range(3, 10))
#
# for min_degree in min_degrees:
#     try:
#         G = LFR_benchmark_graph(
#             n=n, tau1=tau1, tau2=tau2, mu=mu, min_degree=min_degree, min_community=20, seed=1
#         )
#     except nx.ExceededMaxIterations:
#         continue
#     G.remove_edges_from(nx.selfloop_edges(G))
#
#     communities = {frozenset(G.nodes[v]["community"]) for v in G}
#     gt_communities = list(communities)
#     print(f"There are {len(gt_communities)} communities in this setup when min_degree={min_degree}.")
#     print_densities(graph=G, communities=gt_communities)
#
#     #  Not important drawing stuff, just for myself
#     pos = nx.spring_layout(G)  # compute graph layout
#     draw_graph(G, pos=pos, communities=communities)


# This shows that higher values of tau2 create a more "even" distribution of community sizes
# Probably lower values of tau2 are interesting to us.

# VERY sensitive though, below setup changes massively when we go from seed=0 to seed=1

tau1 = 3
tau2s = np.linspace(start=1.1, stop=2, num=20)
n = 1000
mu = 0.1

for tau2 in tau2s:
    try:
        G = LFR_benchmark_graph(
            n=n, tau1=tau1, tau2=tau2, mu=mu, min_degree=3, min_community=25, seed=1
        )
    except nx.ExceededMaxIterations:
        continue
    G.remove_edges_from(nx.selfloop_edges(G))

    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)
    print(
        f"There are {len(gt_communities)} communities in this setup when tau2={tau2}."
    )
    print(f"Sizes of these communities: {[len(comm) for comm in gt_communities]}")

    #  Not important drawing stuff, just for myself
    # pos = nx.spring_layout(G)  # compute graph layout
    # draw_graph(G, pos=pos, communities=communities)
