import networkx as nx

G = nx.read_gml("football.gml")
print("debug")

communities = set()
# TODO: communities hieruit halen, undirected maken, en plotten om te kijken of het interessant kan zijn
for node in G.nodes:
    communities.add(G.nodes[node]["value"])

num_communities = len(communities)
communities = [[] for i in range(num_communities)]

for node in G.nodes:
    communities[G.nodes[node]["value"]].append(node)

for community in communities:
    print(len(community))

# Minimum comm size is 5, max 15. So not really interesting
