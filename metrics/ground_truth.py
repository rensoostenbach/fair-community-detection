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
