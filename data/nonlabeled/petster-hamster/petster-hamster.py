import pandas as pd
import networkx as nx
import numpy as np

from data.utils import relabel_graph

df = pd.read_csv("ent.petster-hamster", delim_whitespace=True, encoding="ansi")
# Preprocess hometown such that we obtain the country
df["dat.country"] = df["dat.hometown"].str.split(",").str[-1]
# Remove the ; for US and Canada
df["dat.country"] = df["dat.country"].str.split(";").str[-1].str.strip()

# Only keep instances that are in a country with at least 5 samples
min_5_df = df[df["dat.country"].map(df["dat.country"].value_counts()).gt(4)]
nodes = min_5_df["ent"]
# Transform this to a NetworkX Graph
edges = pd.read_csv(
    "out.petster-hamster",
    delim_whitespace=True,
    encoding="ansi",
    names=["source", "target"],
)
G = nx.from_pandas_edgelist(edges)
G = G.subgraph(nodes)

# Now do the node attributes, perhaps not most efficient way
countries = np.unique(min_5_df["dat.country"])
countries_dict = {key: [] for key in countries}
for row in min_5_df.itertuples():
    countries_dict[row[13]].append(row.ent)

node_attributes = {}
for key, value in countries_dict.items():
    for node in value:
        node_attributes[node] = {"community": value}

nx.set_node_attributes(G, node_attributes)

G_relabeled = relabel_graph(G=G)

nx.write_gpickle(G_relabeled, "petster-hamster.pickle")

# TODO: Make a different Graph using race as community label
