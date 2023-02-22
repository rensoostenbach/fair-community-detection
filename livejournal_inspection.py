import pickle
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import small_large_communities, dense_sparse_communities

FAIRNESS_TYPES = ["size", "density"]
PERCENTILE = {"size": 75, "density": 50}

for fairness_type in FAIRNESS_TYPES:
    with open(
            f"data/labeled/real_world/LiveJournal/LiveJournal_top5000_overlapping_processed_0.pickle",
            "rb",
    ) as graph_file:
        G = pickle.load(graph_file)
        G.remove_edges_from(nx.selfloop_edges(G))

        communities = {frozenset(G.nodes[v]["community"]) for v in G}
        gt_communities = list(communities)

        # Decide which type of fairness we are looking into
        if fairness_type == "size":
            node_comm_types, comm_types = small_large_communities(
                communities=gt_communities, percentile=PERCENTILE[fairness_type]
            )
        else:  # fairness_type == "density"
            node_comm_types, comm_types = dense_sparse_communities(
                G=G, communities=gt_communities, percentile=PERCENTILE[fairness_type]
            )

        intra_com_edges = np.array(
            [
                G.subgraph(gt_communities[idx]).size()
                for idx, community in enumerate(gt_communities)
            ]
        )
        # Need to divide above numbers by maximum amount of edges possible in community
        sizes = [len(community) for community in gt_communities]
        # For LiveJournal: max_possible_edges can be 0 in the case when a community has a single node. This results in
        # np.nan, causing issues. Instead, we set the densities of these nans to 0.
        max_possible_edges = np.array([(size * (size - 1)) / 2 for size in sizes])
        densities = np.array(intra_com_edges / max_possible_edges)
        densities = np.nan_to_num(densities)

        df = pd.DataFrame(columns=["comm_type"])
        df["comm_type"] = comm_types
        df["Community size"] = sizes
        df["Community density"] = densities

        g = sns.displot(df, x="Community size", kde=True, kde_kws={'clip': (0.0, 1500)})
        plt.title(f"Distribution of community sizes")
        plt.tight_layout()
        plt.savefig(f"plots/LiveJournal_sizes.png", dpi=150)

        g = sns.displot(df, x="Community density", kde=True, kde_kws={'clip': (0.0, 1.0)})
        plt.title(f"Distribution of community densities")
        plt.tight_layout()
        plt.savefig(f"plots/LiveJournal_densities.png", dpi=150)
        print(f"Count per group type: {df.groupby(['comm_type']).sum()}")
