import networkx as nx
import csv
from data.labeled.utils import relabel_graph, remove_overlapping_nodes

G = nx.read_edgelist("com-lj.ungraph.txt", nodetype=int)

communities = {}
with open("com-lj.top5000.cmty.txt") as file:
    csv_reader = csv.reader(file, delimiter="\t")
    for row in csv_reader:
        for node in row:
            communities[int(node)] = {"community": [int(x) for x in row]}

nx.set_node_attributes(G, communities)

nodes_to_remove = []
for node in G.nodes:
    if not G.nodes[node]:  # If it does not have a community
        nodes_to_remove.append(node)

G.remove_nodes_from(nodes_to_remove)

# We take the largest connected component because we have an unconnected Graph
print(f"Is G connected? {nx.is_connected(G)}")
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])

# Process overlapping nodes/communities if there are any, we remove them as a whole
gt_communities = list({frozenset(G0.nodes[v]["community"]) for v in G0})
overlapping = sum([len(com) for com in gt_communities]) > len(G0.nodes)
if overlapping:
    G0 = remove_overlapping_nodes(G=G0)

G_relabeled = relabel_graph(G=G0)

nx.write_gpickle(G_relabeled, "LiveJournal_top5000.pickle")
