from data.utils import process_stanford_graph

process_stanford_graph(
    edgelist_file="com-amazon.ungraph.txt",
    community_file="com-amazon.top5000.cmty.txt",
    outname="Amazon_top5000",
)
