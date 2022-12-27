import pandas as pd
import networkx as nx
import numpy as np

from data.utils import relabel_graph

# Preprocessed with Notepad++, removed every occurrence of \"
df = pd.read_csv("ent.petster-friendships-dog-uniq", delim_whitespace=True, encoding="ansi")

# Only keep instances that are in a race with at least 100 samples
min_100_df = df[df["dat.race"].map(df["dat.race"].value_counts()).gt(99)]
nodes = min_100_df["ent"]
# Transform this to a NetworkX Graph
edges = pd.read_csv(
    "out.petster-friendships-dog-uniq",
    delim_whitespace=True,
    encoding="ansi",
    names=["source", "target"],
)
G = nx.from_pandas_edgelist(edges)
G = G.subgraph(nodes)

# Now do the node attributes, perhaps not most efficient way
races = np.unique(min_100_df["dat.race"])
races_dict = {key: [] for key in races}
for row in min_100_df.itertuples():
    races_dict[row[6]].append(row.ent)  # 6 is the race

node_attributes = {}
for key, value in races_dict.items():
    for node in value:
        node_attributes[node] = {"community": value}

nx.set_node_attributes(G, node_attributes)

G_relabeled = relabel_graph(G=G)

nx.write_gpickle(G_relabeled, "petster-dog.pickle")
