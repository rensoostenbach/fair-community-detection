import pickle
import networkx as nx
import numpy as np
from cdlib import algorithms

from utils import (
    draw_graph,
    random_sample_edges,
    potentional_edge_list,
    modularity_obj_function,
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

    sampled_G = random_sample_edges(nodes=nodes, edges=edges, alpha=0.8)

    #  Not important drawing stuff, just for myself
    pos = nx.spring_layout(G)  # compute graph layout
    draw_graph(G, pos=pos, communities=None)

    # Running an algorithm for detecting communities
    coms = algorithms.louvain(g_original=G)

    draw_graph(G, pos=pos, communities=coms)
    # End of not important drawing stuff

    # Continue with trying to calculate the score for missing edges?
    potential_edges = potentional_edge_list(G=sampled_G, nodes=nodes)
    for node_i, node_j in potential_edges:
        # TODO: modularity uitrekenen met en zonder edge i,j voor de score functie s_ij
        modularity_without = modularity_obj_function(communities=coms, G=sampled_G, edges=edges)

        G_with = sampled_G.add_edge(node_i, node_j)
        modularity_with = modularity_obj_function(communities=coms, G=G_with, edges=edges)
    if graph_idx > 5:
        break
