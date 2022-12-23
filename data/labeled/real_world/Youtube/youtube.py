from data.utils import process_stanford_graph

process_stanford_graph(
    edgelist_file="com-youtube.ungraph.txt",
    community_file="com-youtube.all.cmty.txt",
    outname="Youtube",
)
