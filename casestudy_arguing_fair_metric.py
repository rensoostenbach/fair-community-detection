import copy
import networkx as nx
import numpy as np
from cdlib import algorithms

from utils import draw_graph, plot_fairness
from data.labeled.casestudy.synthetic_graphs import (
    varying_mu_values,
    varying_denseness,
    mislabel_nodes,
)
from metrics.own_metric import calculate_fairness_metrics


# Varying mu values for pre-specified community sizes
num_small = 30
num_large = 30
mus = np.linspace(0.01, 0.5, 15)
# graphs = varying_mu_values(num_nodes_G1=num_small, num_nodes_G2=num_large, mus=mus)


# Varying denseness between two communities
# However, below code keeps a fixed graph and varies the number of misclassified nodes

num_dense_nodes = 40
num_sparse_nodes = 50
densenesses_G1 = [0.3]
densenesses_G2 = [0.1]
density_cutoff = 0.2
inter_community_edges = 0.05
graphs = varying_denseness(
    num_nodes_G1=num_dense_nodes,
    num_nodes_G2=num_sparse_nodes,
    densenesses_G1=densenesses_G1,
    densenesses_G2=densenesses_G2,
    inter_community_edges=inter_community_edges,
)

for G in graphs:
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)

    #  Not important drawing stuff, just for myself
    # pos = nx.spring_layout(G)  # compute graph layout
    # draw_graph(G, pos=pos, communities=communities)

    emd = []
    f1 = []
    acc = []
    misclassified_nodes = list(range(0, 22, 2))

    for num_misclassified_nodes in misclassified_nodes:
        print(f"Number of misclassified nodes: {num_misclassified_nodes}")
        mislabel_comm_nodes = {"sparse": num_misclassified_nodes}
        G_mislabeled = mislabel_nodes(
            G=copy.deepcopy(G),
            mislabel_comm_nodes=mislabel_comm_nodes,
            density_cutoff=density_cutoff,
        )
        mislabeled_communities = {
            frozenset(G_mislabeled.nodes[v]["community"]) for v in G_mislabeled
        }

        #  Not important drawing stuff, just for myself
        # draw_graph(G_mislabeled, pos=pos, communities=mislabeled_communities)

        (
            emd_fairness_score,
            f1_fairness_score,
            accuracy_fairness_score,
        ) = calculate_fairness_metrics(
            G=G,
            gt_communities=gt_communities,
            pred_communities=list(mislabeled_communities),
            fairness_type="density",
            density_cutoff=density_cutoff,
        )

        emd.append(emd_fairness_score)
        f1.append(f1_fairness_score)
        acc.append(accuracy_fairness_score)

    plot_fairness(
        emd=emd,
        f1=f1,
        acc=acc,
        x_axis=misclassified_nodes,
        xlabel="Number of misclassified nodes",
        title="Fairness score per number of misclassified nodes",
    )


# Varying small community sizes and only mislabeling in small
num_smalls = [20, 25, 30, 35, 40, 45, 50]
num_large = 100
emd = []
f1 = []
acc = []
for num_small in num_smalls:
    graphs = varying_mu_values(
        num_nodes_G1=num_small, num_nodes_G2=num_large, mus=[0.3]
    )
    for G in graphs:
        communities = {frozenset(G.nodes[v]["community"]) for v in G}
        gt_communities = list(communities)

        #  Not important drawing stuff, just for myself
        # pos = nx.spring_layout(G)  # compute graph layout
        # draw_graph(G, pos=pos, communities=communities)

        print(f"Size of small community: {num_small}")
        mislabel_comm_nodes = {"small": 10}
        G_mislabeled = mislabel_nodes(
            G=copy.deepcopy(G),
            mislabel_comm_nodes=mislabel_comm_nodes,
            size_percentile=75,
        )
        mislabeled_communities = {
            frozenset(G_mislabeled.nodes[v]["community"]) for v in G_mislabeled
        }

        #  Not important drawing stuff, just for myself
        # draw_graph(G_mislabeled, pos=pos, communities=mislabeled_communities)

        (
            emd_fairness_score,
            f1_fairness_score,
            accuracy_fairness_score,
        ) = calculate_fairness_metrics(
            G=G,
            gt_communities=gt_communities,
            pred_communities=list(mislabeled_communities),
            fairness_type="small_large",
            size_percentile=75,
        )

        emd.append(emd_fairness_score)
        f1.append(f1_fairness_score)
        acc.append(accuracy_fairness_score)

plot_fairness(
    emd=emd,
    f1=f1,
    acc=acc,
    x_axis=num_smalls,
    xlabel="Size of small community",
    title="Fairness score per small community sizes",
)
