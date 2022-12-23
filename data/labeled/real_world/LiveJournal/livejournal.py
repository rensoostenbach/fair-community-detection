from data.utils import process_stanford_graph

process_stanford_graph(
    edgelist_file="com-lj.ungraph.txt",
    community_file="com-lj.top5000.cmty.txt",
    outname="LiveJournal_top5000",
)
