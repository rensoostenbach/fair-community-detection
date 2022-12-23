import pandas as pd
import networkx as nx
from data.utils import remove_overlapping_nodes, relabel_graph

edges_df = pd.read_csv("data/edges.csv", names=["source", "target"])
communities_df = pd.read_csv("data/group-edges.csv", names=["node", "community"])

G = nx.from_pandas_edgelist(edges_df)

communities = {}

for comm in communities_df["community"].unique():
    community = communities_df[communities_df["community"] == comm]["node"]
    for node in community:
        communities[node] = {"community": community}

nx.set_node_attributes(G, communities)

# Process overlapping nodes/communities if there are any, we remove them as a whole
gt_communities = list({frozenset(G.nodes[v]["community"]) for v in G})
overlapping = sum([len(com) for com in gt_communities]) > len(G.nodes)
if overlapping:
    G = remove_overlapping_nodes(G=G)  # This is now not a connected Graph anymore

# We take the largest connected component because we have an unconnected Graph
print(f"Is G connected? {nx.is_connected(G)}")
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])

G_relabeled = relabel_graph(G=G0)


nx.write_gpickle(G_relabeled, "flickr.pickle")
