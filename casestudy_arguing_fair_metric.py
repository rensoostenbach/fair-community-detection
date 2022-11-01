import copy
import networkx as nx
import numpy as np

from utils import draw_graph, lineplot_fairness, plot_heatmap
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
# This produces a nice graph, but manually mislabeling in density does not make that much sense

# num_dense_nodes = 40
# num_sparse_nodes = 50
# densenesses_G1 = [0.3]
# densenesses_G2 = [0.1]
# density_cutoff = 0.2
# inter_community_edges = 0.05
# graphs = varying_denseness(
#     num_nodes_G1=num_dense_nodes,
#     num_nodes_G2=num_sparse_nodes,
#     densenesses_G1=densenesses_G1,
#     densenesses_G2=densenesses_G2,
#     inter_community_edges=inter_community_edges,
# )
#
# for G in graphs:
#     communities = {frozenset(G.nodes[v]["community"]) for v in G}
#     gt_communities = list(communities)
#
#     #  Not important drawing stuff, just for myself
#     pos = nx.spring_layout(G)  # compute graph layout
#     draw_graph(G, pos=pos, communities=communities)
#
#     emd = []
#     f1 = []
#     acc = []
#     misclassified_nodes = list(range(0, 22, 2))
#
#     for num_misclassified_nodes in misclassified_nodes:
#         print(f"Number of misclassified nodes: {num_misclassified_nodes}")
#         mislabel_comm_nodes = {"sparse": num_misclassified_nodes}
#         G_mislabeled = mislabel_nodes(
#             G=copy.deepcopy(G),
#             mislabel_comm_nodes=mislabel_comm_nodes,
#             density_cutoff=density_cutoff,
#         )
#         mislabeled_communities = {
#             frozenset(G_mislabeled.nodes[v]["community"]) for v in G_mislabeled
#         }
#
#         #  Not important drawing stuff, just for myself
#         draw_graph(G_mislabeled, pos=pos, communities=mislabeled_communities)
#
#         (
#             emd_fairness_score,
#             f1_fairness_score,
#             accuracy_fairness_score,
#         ) = calculate_fairness_metrics(
#             G=G,
#             gt_communities=gt_communities,
#             pred_communities=list(mislabeled_communities),
#             fairness_type="density",
#             percentile=75,
#         )
#
#         emd.append(emd_fairness_score)
#         f1.append(f1_fairness_score)
#         acc.append(accuracy_fairness_score)
#
#     plot_fairness(
#         emd=emd,
#         f1=f1,
#         acc=acc,
#         x_axis=misclassified_nodes,
#         xlabel="Number of misclassified nodes",
#         title="Fairness score per number of misclassified nodes",
#     )


# Varying small community sizes and only mislabeling in small
num_smalls = [10, 15, 20, 25, 30, 35, 40, 45, 50]
num_large = 100
emd = []
f1 = []
acc = []
for num_small in num_smalls:
    graphs = varying_mu_values(
        num_nodes_G1=num_small, num_nodes_G2=num_large, mus=[0.15]
    )
    for G in graphs:
        communities = {frozenset(G.nodes[v]["community"]) for v in G}
        gt_communities = list(communities)

        #  Not important drawing stuff, just for myself
        # pos = nx.spring_layout(G)  # compute graph layout
        # draw_graph(G, pos=pos, communities=communities)

        mislabel_comm_nodes = {
            "small": 10
        }  # Could make num_small/2 for example, which shows that fairness is constant
        G_mislabeled = mislabel_nodes(
            G=copy.deepcopy(G),
            mislabel_comm_nodes=mislabel_comm_nodes,
            percentile=75,
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
            fairness_type="size",
            percentile=75,
        )

        emd.append(emd_fairness_score)
        f1.append(f1_fairness_score)
        acc.append(accuracy_fairness_score)

lineplot_fairness(
    emd=emd,
    f1=f1,
    acc=acc,
    x_axis=num_smalls,
    xlabel="Size of small community",
    noline=False,
    title=f"Fairness score per small community sizes\n"
    f"(N_large: {num_large} nodes, 10 nodes are misclassified in small community)",
)

# Varying number of misclassified nodes in small or large community
num_small = 25
num_large = 50

graph = varying_mu_values(num_nodes_G1=num_small, num_nodes_G2=num_large, mus=[0.15])

for G in graph:
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)

    #  Not important drawing stuff, just for myself
    # pos = nx.spring_layout(G)  # compute graph layout
    # draw_graph(G, pos=pos, filename=f"initial_graph", communities=communities)

    misclassified_nodes = list(range(0, 21))
    small_large = ["small", "large"]

    for size in small_large:
        emd = []
        f1 = []
        acc = []
        for num_misclassified_nodes in misclassified_nodes:
            mislabel_comm_nodes = {size: num_misclassified_nodes}

            G_mislabeled = mislabel_nodes(
                G=copy.deepcopy(G),
                mislabel_comm_nodes=mislabel_comm_nodes,
                percentile=75,
            )
            mislabeled_communities = {
                frozenset(G_mislabeled.nodes[v]["community"]) for v in G_mislabeled
            }

            #  Not important drawing stuff, just for myself
            # draw_graph(
            #     G_mislabeled,
            #     pos=pos,
            #     filename=f"large_{num_misclassified_nodes_large}_small_{num_misclassified_nodes_small}",
            #     communities=mislabeled_communities,
            # )

            (
                emd_fairness_score,
                f1_fairness_score,
                accuracy_fairness_score,
            ) = calculate_fairness_metrics(
                G=G,
                gt_communities=gt_communities,
                pred_communities=list(mislabeled_communities),
                fairness_type="size",
                percentile=75,
            )

            emd.append(emd_fairness_score)
            f1.append(f1_fairness_score)
            acc.append(accuracy_fairness_score)

        lineplot_fairness(
            emd=emd,
            f1=f1,
            acc=acc,
            x_axis=misclassified_nodes,
            xlabel=f"Number of misclassified nodes in {size} community",
            noline=False,
            title=f"Fairness score per number of misclassified nodes in {size} community\n"
            f"N_small={num_small}, N_large={num_large}",
        )

# Varying number of misclassified nodes in small and large community
num_small = 25
num_large = 50

graph = varying_mu_values(num_nodes_G1=num_small, num_nodes_G2=num_large, mus=[0.15])

for G in graph:
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    gt_communities = list(communities)

    #  Not important drawing stuff, just for myself
    # pos = nx.spring_layout(G)  # compute graph layout
    # draw_graph(G, pos=pos, filename="initial_graph", communities=communities)

    misclassified_nodes = list(range(0, 21))
    small_large = ["small", "large"]
    emd = []
    f1 = []
    acc = []

    for num_misclassified_nodes_large in misclassified_nodes:
        for num_misclassified_nodes_small in misclassified_nodes:
            mislabel_comm_nodes = {
                "small": num_misclassified_nodes_small,
                "large": num_misclassified_nodes_large,
            }

            G_mislabeled = mislabel_nodes(
                G=copy.deepcopy(G),
                mislabel_comm_nodes=mislabel_comm_nodes,
                percentile=75,
            )
            mislabeled_communities = {
                frozenset(G_mislabeled.nodes[v]["community"]) for v in G_mislabeled
            }

            #  Not important drawing stuff, just for myself
            # draw_graph(
            #     G_mislabeled,
            #     pos=pos,
            #     filename=f"large_{num_misclassified_nodes_large}_small_{num_misclassified_nodes_small}",
            #     communities=mislabeled_communities,
            # )

            (
                emd_fairness_score,
                f1_fairness_score,
                accuracy_fairness_score,
            ) = calculate_fairness_metrics(
                G=G,
                gt_communities=gt_communities,
                pred_communities=list(mislabeled_communities),
                fairness_type="size",
                percentile=75,
            )

            emd.append(
                (
                    num_misclassified_nodes_large,
                    num_misclassified_nodes_small,
                    emd_fairness_score,
                )
            )
            f1.append(
                (
                    num_misclassified_nodes_large,
                    num_misclassified_nodes_small,
                    f1_fairness_score,
                )
            )
            acc.append(
                (
                    num_misclassified_nodes_large,
                    num_misclassified_nodes_small,
                    accuracy_fairness_score,
                )
            )

    # Creating array for F1 heatmap
    f1_arr = np.empty((len(misclassified_nodes), len(misclassified_nodes)))
    for large, small, score in f1:
        f1_arr[large][small] = score

    plot_heatmap(
        data=f1_arr,
        title="F1 Fairness values for misclassifying\n"
        "nodes in major vs minor community",
    )

    acc_arr = np.empty((len(misclassified_nodes), len(misclassified_nodes)))
    for large, small, score in acc:
        acc_arr[large][small] = score

    plot_heatmap(
        data=acc_arr,
        title="Acc Fairness values for misclassifying\n"
        "nodes in major vs minor community",
    )
