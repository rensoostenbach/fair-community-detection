from data.utils import process_stanford_graph

process_stanford_graph(
    edgelist_file="com-dblp.ungraph.txt",
    community_file="com-dblp.top5000.cmty.txt",
    outname="DBLP_top5000",
)
