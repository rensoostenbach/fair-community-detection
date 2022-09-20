import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(G: nx.Graph, pos: dict, communities=None):
    """Draw a graph, with the choice of including communities or not"""

    plt.figure(figsize=(15, 10))
    plt.axis("off")
    if communities is not None:
        # Coloring every node such that communities have the same color
        node_color_pred = [0] * len(G.nodes)
        for idx, community in enumerate(communities.communities):
            for node in community:
                node_color_pred[node] = idx

        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            node_size=100,
            cmap=plt.cm.RdYlBu,
            node_color=node_color_pred,
            linewidths=0.5,
            edgecolors="black",
        )
        nx.draw_networkx_edges(G=G, pos=pos, alpha=0.3)
    else:
        nx.draw_networkx_nodes(
            G=G, pos=pos, node_size=100, linewidths=0.5, edgecolors="black"
        )
        nx.draw_networkx_edges(G=G, pos=pos, alpha=0.3)

    plt.show()
