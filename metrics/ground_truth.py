import networkx as nx
from utils import find_community, mapping


def purity(pred_coms, real_coms):
    max_same_total = 0
    for pred_com in pred_coms:
        max_same_total += purity_single_community(pred_com, real_coms)

    N = len([node for community in pred_coms for node in community])
    return max_same_total / N


def purity_single_community(pred_com, real_coms):
    max_same = 0
    for real_com in real_coms:
        if len(set(pred_com).intersection(real_com)) > max_same:
            max_same = len(set(pred_com).intersection(real_com))
    return max_same


def inverse_purity(pred_coms, real_coms):
    return purity(pred_coms=real_coms, real_coms=pred_coms)


def f_measure(pred_coms, real_coms):
    return (
        2
        * purity(pred_coms=pred_coms, real_coms=real_coms)
        * inverse_purity(pred_coms=pred_coms, real_coms=real_coms)
    ) / (
        purity(pred_coms=pred_coms, real_coms=real_coms)
        + inverse_purity(pred_coms=pred_coms, real_coms=real_coms)
    )


def nodal_weights(node_u, real_coms, G: nx.Graph):
    # Take subgraph of community, then look at G.degree
    real_community_idx = find_community(node_u=node_u, communities=real_coms)
    real_community = real_coms[real_community_idx]
    G_subgraph = G.subgraph(real_community)
    nodes = [x[0] for x in G_subgraph.degree]
    degrees = [x[1] for x in G_subgraph.degree]
    d_int_u = degrees[nodes.index(node_u)]  # number of links within community
    max_degree_v = max([x[1] for x in G.degree])  # Max degree of whole network
    w_u = d_int_u / max_degree_v
    return w_u


def total_nodal_weights(G: nx.Graph, real_coms: list):
    w = 0
    for node in G.nodes:
        w += nodal_weights(node_u=node, real_coms=real_coms, G=G)
    return w


def modified_purity_node(
    node_u: int, pred_coms: list, real_coms: list, mapping_list: list
):
    real_comm = find_community(node_u=node_u, communities=real_coms)
    pred_comm = find_community(node_u=node_u, communities=pred_coms)
    return int(mapping_list[real_comm] == pred_comm)


def modified_purity(pred_coms, real_coms, G: nx.Graph):
    modified_purity_score = 0
    w = total_nodal_weights(G=G, real_coms=real_coms)
    _, mapping_list = mapping(gt_communities=real_coms, pred_coms=pred_coms)
    for comm in real_coms:
        for node in comm:
            mod_pur_node = modified_purity_node(
                node_u=node,
                pred_coms=pred_coms,
                real_coms=real_coms,
                mapping_list=mapping_list,
            )
            w_u = nodal_weights(node_u=node, real_coms=real_coms, G=G)
            modified_purity_score += (w_u / w) * mod_pur_node
    return modified_purity_score


def modified_inverse_purity(pred_coms: list, real_coms: list, G: nx.Graph):
    return modified_purity(pred_coms=real_coms, real_coms=pred_coms, G=G)


def modified_f_measure(pred_coms: list, real_coms: list, G: nx.Graph):
    return (
        2
        * modified_purity(pred_coms=pred_coms, real_coms=real_coms, G=G)
        * modified_inverse_purity(pred_coms=pred_coms, real_coms=real_coms, G=G)
    ) / (
        modified_purity(pred_coms=pred_coms, real_coms=real_coms, G=G)
        + modified_inverse_purity(pred_coms=pred_coms, real_coms=real_coms, G=G)
    )


# TODO: Modified ARI
#  plan van aanpak: Gewoon de approach van de paper overnemen, niks van sklearn o.i.d.
#  Ziet ernaar uit dat dat wel goed te doen zou moeten zijn
def paired_nodal_weights(nodes_subset: list or set, real_coms: list, G: nx.Graph):
    W_s = 0
    for node_u in nodes_subset:
        w_u = nodal_weights(node_u=node_u, real_coms=real_coms, G=G)
        for node_v in nodes_subset:
            w_v = nodal_weights(node_u=node_v, real_coms=real_coms, G=G)
            W_s += w_u * w_v
    return W_s


def modified_ari(pred_coms: list, real_coms: list, G: nx.Graph):
    W_omega_c = 0
    W_omega = 0
    W_c = 0

    for pred_com in pred_coms:
        W_omega += paired_nodal_weights(nodes_subset=pred_com, real_coms=real_coms, G=G)

    for real_com in real_coms:
        W_c += paired_nodal_weights(nodes_subset=real_com, real_coms=real_coms, G=G)

    for pred_com in pred_coms:
        for real_com in real_coms:
            nodes_subset = set(pred_com).intersection(set(real_com))
            W_omega_c += paired_nodal_weights(
                nodes_subset=nodes_subset, real_coms=real_coms, G=G
            )

    W_V = paired_nodal_weights(nodes_subset=G.nodes, real_coms=real_coms, G=G)

    upper_part = W_omega_c - (W_omega * W_c / W_V)
    lower_part = 0.5 * (W_omega + W_c) - (W_omega * W_c / W_V)

    ARI_m = upper_part / lower_part
    return ARI_m
