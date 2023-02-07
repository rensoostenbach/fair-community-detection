from data.utils import process_stanford_graph

outname = "LiveJournal_top5000_overlapping_processed"

if __name__ == '__main__':

    process_stanford_graph(
        edgelist_file="com-lj.ungraph.txt",
        community_file="com-lj.top5000.cmty.txt",
        outname=outname,
    )
