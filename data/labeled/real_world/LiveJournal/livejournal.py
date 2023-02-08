from data.utils import process_stanford_graph

if __name__ == '__main__':
    for i in range(100):
        outname = f"LiveJournal_top5000_overlapping_processed_{i}"
        process_stanford_graph(
            edgelist_file="com-lj.ungraph.txt",
            community_file="com-lj.top5000.cmty.txt",
            outname=outname,
        )
