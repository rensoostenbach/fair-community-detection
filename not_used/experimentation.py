import pickle
import networkx as nx
import numpy as np
from cdlib import algorithms

from utils import (
    draw_graph,
    random_sample_edges,
    potentional_edge_list,
    score_function,
    compute_tpr_fpr,
    plot_roc_curve,
)

# load the data
infile = open("data/nonlabeled/CommunityFitNet_updated.pickle", "rb")
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

    #  Not important drawing stuff, just for myself
    pos = nx.spring_layout(G)  # compute graph layout
    draw_graph(G, pos=pos, communities=None)

    # Running an algorithm for detecting communities
    coms = algorithms.louvain(g_original=G)

    draw_graph(G, pos=pos, communities=coms)
    # End of not important drawing stuff

    alphas = np.linspace(0.1, 0.9, 9)
    all_tpr = []
    all_fpr = []

    for alpha in alphas:
        sampled_G, sampled_edges, nonsampled_edges = random_sample_edges(
            nodes=nodes, edges=edges, alpha=alpha
        )

        # Continue with trying to calculate the score for missing edges?
        potential_edges = potentional_edge_list(G=sampled_G, nodes=nodes)

        # Now we need to randomly choose 10000 pairs of missing links (non_sampled_edges) and non-links (potential_edges)
        iterations = 1000
        scores_with_missing_link = np.zeros(iterations)
        scores_with_non_link = np.zeros(iterations)
        for i in range(iterations):
            missing_link = nonsampled_edges[np.random.randint(nonsampled_edges.shape[0]), :]
            non_link = potential_edges[np.random.randint(potential_edges.shape[0]), :]

            G_with_missing_link = sampled_G.copy()
            G_with_missing_link.add_edge(missing_link[0], missing_link[1])
            scores_with_missing_link[i] = score_function(
                    G_without_ij=sampled_G,
                    G_with_ij=G_with_missing_link,
                    communities=coms,
                    edges=edges,
                )

            G_with_non_link = sampled_G.copy()
            G_with_non_link.add_edge(non_link[0], non_link[1])
            scores_with_non_link[i] = score_function(
                    G_without_ij=sampled_G,
                    G_with_ij=G_with_non_link,
                    communities=coms,
                    edges=edges,
                )

        tpr, fpr = compute_tpr_fpr(scores_with_missing_link, scores_with_non_link)
        all_tpr.append(tpr)
        all_fpr.append(fpr)

    # auc_score = auc(all_fpr, all_tpr)
    plot_roc_curve(all_tpr, all_fpr)


    if graph_idx > 5:
        break
