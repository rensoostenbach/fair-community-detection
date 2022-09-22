import networkx as nx
from utils import remove_community


def jaccard_distance(G: nx.Graph, node_u: int, node_v: int):
    """
    Compute the Jaccard distance between two nodes as described in Shao et al. 2015

    :param G: A NetworkX graph
    :param node_u: ID of node u in the graph G
    :param node_v: ID of node v in the graph G
    :return: The Jaccard distance between nodes u and v
    """
    neighbors_u = set(G.neighbors(node_u))
    neighbors_v = set(G.neighbors(node_v))
    return 1 - (len(neighbors_u.intersection(neighbors_v))) / (
        len(neighbors_u.union(neighbors_v))
    )


def fairness_per_node(G: nx.Graph, node_u: int, communities: list, com_a: list):
    """
    We compute our own fairness metric here for a single node u

    :param G: A NetworkX graph
    :param node_u: Node u in the graph G
    :param communities: List of all the (detected or ground-truth) communities
    :param com_a: The list of nodes that are in the same community as node_u
    :return bool: Boolean indicating if node_u meets the fairness constraints
    """

    left_sum = 0
    right_sum = 0
    for node_v in com_a:
        left_sum += jaccard_distance(G, node_u, node_v)

    com_b = remove_community(node_u=node_u, communities=communities)
    # Transform the list of frozensets to a flat list of all nodes
    com_b = [node for community in com_b for node in community]

    for node_v in com_b:
        right_sum += jaccard_distance(G=G, node_u=node_u, node_v=node_v)

    left_side = (1 / (len(com_a) - 1)) * left_sum
    right_side = (1 / len(com_b)) * right_sum

    return left_side <= right_side


def fairness(G: nx.Graph, pred_coms: list, real_coms: list):
    """
    Compute the fairness via our own proposed metric.

    :param G: A NetworkX graph
    :param pred_coms: List of the predicted communities
    :param real_coms: List of the real communities
    :return: A number between 0 and 1 indicating the fairness of the predicted
             communities compared to the "real" communities
    """
    fair_pred_nodes = set()
    for idx, community in enumerate(pred_coms):
        for node in community:
            fair_pred_nodes.add(
                fairness_per_node(
                    G=G, node_u=node, communities=pred_coms, com_a=community
                )
            )

    fair_real_nodes = set()
    for idx, community in enumerate(real_coms):
        for node in community:
            fair_real_nodes.add(
                fairness_per_node(
                    G=G, node_u=node, communities=real_coms, com_a=community
                )
            )

    return (len(fair_pred_nodes.intersection(fair_real_nodes))) / (
        len(fair_pred_nodes.union(fair_real_nodes))
    )
