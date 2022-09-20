import networkx as nx
import csv

from utils import draw_graph

G = nx.read_edgelist("email-directed/email-Eu-core.txt", nodetype=int)

communities = {}
community_keys = set()

with open("email-directed/email-Eu-core-department-labels.txt") as file:
    csv_reader = csv.reader(file, delimiter=" ")
    for row in csv_reader:
        node = int(row[0])
        community_label = int(row[1])
        communities[node] = {"community": community_label}
        community_keys.add(community_label)

nx.set_node_attributes(G, communities)

nice_communities = {key: [] for key in list(community_keys)}
for node in G.nodes:
    nice_communities[G.nodes[node]["community"]].append(node)

# Checking the size of each community
for community, nodes in nice_communities.items():
    print(f"Community {community} has size {len(nodes)}")

pos = nx.spectral_layout(G)  # compute graph layout
draw_graph(G=G, pos=pos)
