from data.utils import process_stanford_graph

process_stanford_graph(
    edgelist_file="com-amazon.ungraph.txt",
    community_file="com-amazon.top5000.cmty.txt",
    outname="Amazon_top5000",
)

# This graph only has one community left --> Probably because we remove so many nodes that one comm is left
