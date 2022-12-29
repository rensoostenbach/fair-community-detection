import networkx as nx
import pandas as pd
import os
import re
import numpy as np

from data.utils import relabel_graph

with open("cit-HepTh.txt", "rb") as edgelist:
    G = nx.read_edgelist(edgelist)

journal_refs = []
unique_journal_refs = set()

for root, dirs, files in os.walk(os.curdir):
    for file in files:
        if file.endswith(".abs"):
            journal_ref = None
            with open(f"{root}/{file}", "rb") as abstract:
                journal_ref = re.search(
                    r"Journal-ref: (.*?)\s\(", str(abstract.read()), re.M
                )
                if journal_ref is not None:
                    journal_refs.append((file.split(".")[0], journal_ref.groups()[0]))
                    unique_journal_refs.add(journal_ref.groups()[0])

df = pd.DataFrame(data=journal_refs, columns=["id", "journal-ref"])
df["journal"] = df["journal-ref"].str.replace(
    r"\d+", ""
)  #  Remove the digits which are volume numbers

# Only keep instances that are in a race with at least 100 samples
min_100_df = df[df["journal"].map(df["journal"].value_counts()).gt(99)]
nodes = min_100_df["id"]

G_subgraph = G.subgraph(nodes)

# Now do the node attributes, perhaps not most efficient way
journals = np.unique(min_100_df["journal"])
journals_dict = {key: [] for key in journals}
for row in min_100_df.itertuples():
    journals_dict[row.journal].append(row.id)

node_attributes = {}
for key, value in journals_dict.items():
    for node in value:
        node_attributes[node] = {"community": value}

nx.set_node_attributes(G_subgraph, node_attributes)

G_relabeled = relabel_graph(G=G_subgraph)

nx.write_gpickle(G_relabeled, "cit-hepth.pickle")
