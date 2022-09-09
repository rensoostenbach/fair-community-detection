import pickle
import networkx as nx
import numpy as np
from cdlib import algorithms

from utils import (
    draw_graph,
    random_sample_edges,
    potentional_edge_list,
    modularity_obj_function,
    score_function,
)

# load the data
infile = open("data/CommunityFitNet_updated.pickle", "rb")
df = pickle.load(infile)

# let's try to recreate the experiment in Box 1 from ghasemian2019evaluating
for graph in df.itertuples(index=False):
    print(graph[1])
    graph_idx = graph[0]
    nodes = graph[12]
    edges = graph[13]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    sampled_G, sampled_edges, nonsampled_edges = random_sample_edges(
        nodes=nodes, edges=edges, alpha=0.8
    )

    #  Not important drawing stuff, just for myself
    pos = nx.spring_layout(G)  # compute graph layout
    draw_graph(G, pos=pos, communities=None)

    # Running an algorithm for detecting communities
    coms = algorithms.louvain(g_original=G)

    draw_graph(G, pos=pos, communities=coms)
    # End of not important drawing stuff

    # Continue with trying to calculate the score for missing edges?
    potential_edges = potentional_edge_list(G=sampled_G, nodes=nodes)

    # Now we need to randomly choose 10000 pairs of missing links (non_sampled_edges) and non-links (potential_edges)
    for i in range(10000):
        missing_link = nonsampled_edges[np.random.randint(nonsampled_edges.shape[0]), :]
        non_link = potential_edges[np.random.randint(potential_edges.shape[0]), :]

        G_with_missing_link = sampled_G.copy()
        G_with_missing_link.add_edge(missing_link[0], missing_link[1])
        score_with_missing_link = score_function(
            G_without_ij=sampled_G,
            G_with_ij=G_with_missing_link,
            communities=coms,
            edges=edges,
        )

        G_with_non_link = sampled_G.copy()
        G_with_non_link.add_edge(non_link[0], non_link[1])
        score_with_non_link = score_function(
            G_without_ij=sampled_G,
            G_with_ij=G_with_non_link,
            communities=coms,
            edges=edges,
        )

    if graph_idx > 5:
        break
