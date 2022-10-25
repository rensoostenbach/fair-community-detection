import inspect
from cdlib.algorithms import *
import node2vec
# TODO: This file will serve as a starting point for the actual experiments in my thesis
#  I can transform the playground and work from there
#  But first, figure out for every method that does not either have a single argument or number of comms as 2nd argument
#  what kind of arguments they have, and what values to use.

CD_METHODS = [
    louvain,
    leiden,
    rb_pots,
    rber_pots,
    cpm,
    significance_communities,
    surprise_communities,
    greedy_modularity,
    der,
    label_propagation,
    async_fluid,
    infomap,
    walktrap,
    girvan_newman,  # Has level as second argument, docs example uses 3
    em,
    scan,  # Has epsilon and mu as arguments, docs example uses 0.7 and 3
    gdmp2,
    spinglass,
    eigenvector,
    agdl,
    frc_fgsn,
    sbm_dl,
    sbm_dl_nested,
    markov_clustering,
    edmot,
    chinesewhispers,
    siblinarity_antichain,
    ga,
    belief,
    threshold_clustering,
    lswl_plus,
    lswl,
    mod_m,
    mod_r,
    head_tail,
    kcut,
    gemsec,
    scd,
    pycombo,
    paris,
    principled_clustering,
    ricci_community,
    spectral,
    mcode,
    r_spectral_clustering,
    node2vec
    # bayan # Need to update cdlib for bayan to work, and maybe install Gurobi solver, or we disregard it
]

for method in CD_METHODS:
    # Retrieve all arguments we need to fill in, first one is always the graph
    args = [p for p in inspect.signature(method).parameters.values() if isinstance(p.default, type)]
    if len(args) == 1:
        print(f"Method {method.__name__} has only graph: {args}")
    else:
        # find out what other things we need to fill in
        print(f"Method {method.__name__} has more arguments: {args}")
