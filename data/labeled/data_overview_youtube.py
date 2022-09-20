import networkx as nx
import csv

from utils import draw_graph

G = nx.read_edgelist("Youtube/com-youtube.ungraph.txt", nodetype=int)

communities = {}
community_sizes = []

with open("Youtube/com-youtube.all.cmty.txt") as file:
    csv_reader = csv.reader(file, delimiter="\t")
    for idx, row in enumerate(csv_reader):
        community_sizes.append(len(row))
        for node in row:
            communities[int(node)] = {"community": idx}

nx.set_node_attributes(G, communities)

print(sorted(community_sizes, reverse=True))

# pos = nx.spring_layout(G)  # compute graph layout
# draw_graph(G=G, pos=pos)
