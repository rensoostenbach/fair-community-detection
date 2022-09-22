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


def fair_unfair_nodes(G: nx.Graph, communities: list):
    """
    Compute the set of fair and unfair nodes

    :param G: A NetworkX graph
    :param communities: List of all the (detected or ground-truth) communities
    :return fair_nodes: Set of the fair nodes as determined by the fairness constraint
    :return unfair_nodes: Set of the unfair nodes  as determined by the fairness constraint
    """
    fair_nodes = set()
    unfair_nodes = set()
    for idx, community in enumerate(communities):
        for node in community:
            if fairness_per_node(
                G=G, node_u=node, communities=communities, com_a=community
            ):
                fair_nodes.add(node)
            else:
                unfair_nodes.add(node)

    return fair_nodes, unfair_nodes


def fairness(G: nx.Graph, pred_coms: list, real_coms: list):
    """
    Compute the fairness via our own proposed metric.

    :param G: A NetworkX graph
    :param pred_coms: List of the predicted communities
    :param real_coms: List of the real communities
    :return: A number between 0 and 1 indicating the fairness of the predicted
             communities compared to the "real" communities
    """
    fair_pred_nodes, unfair_pred_nodes = fair_unfair_nodes(G=G, communities=pred_coms)
    fair_real_nodes, unfair_real_nodes = fair_unfair_nodes(G=G, communities=real_coms)

    fairness_score = (len(fair_pred_nodes.intersection(fair_real_nodes))) / (
        len(fair_pred_nodes.union(fair_real_nodes))
    )

    return (
        fairness_score,
        fair_pred_nodes,
        unfair_pred_nodes,
        fair_real_nodes,
        unfair_real_nodes,
    )
