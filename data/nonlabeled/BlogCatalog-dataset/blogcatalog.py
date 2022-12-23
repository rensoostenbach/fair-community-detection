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

nodes_to_remove = []
for node in G.nodes:
    if not G.nodes[node]:  # If it does not have a community
        nodes_to_remove.append(node)

G.remove_nodes_from(nodes_to_remove)

# Process overlapping nodes/communities if there are any, we remove them as a whole
gt_communities = list({frozenset(G.nodes[v]["community"]) for v in G})
overlapping = sum([len(com) for com in gt_communities]) > len(G.nodes)
if overlapping:
    G = remove_overlapping_nodes(G=G)  # This remains a connected Graph

G_relabeled = relabel_graph(G=G)

nx.write_gpickle(G_relabeled, "blogcatalog.pickle")
