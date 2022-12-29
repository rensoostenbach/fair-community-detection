import pickle
import os
from utils import scatterplot_fairness


results = []
results_directory = "DBLP"
for root, dirs, files in os.walk(results_directory):
    for file in files:
        if file.endswith(".pickle"):
            results.append(os.path.join(root, file))


size_fairness_scores = {}
size_evaluation_scores = {}
density_fairness_scores = {}
density_evaluation_scores = {}
for result in results:
    filename_splitted = os.path.basename(result).split("-")
    with open(result, "rb") as pickled_results:
        pickled_scores = pickle.load(pickled_results)
        eval(f"{filename_splitted[0]}_{filename_splitted[2]}_scores")[
            filename_splitted[1]
        ] = pickled_scores

# Now we make scatterplots of accuracy vs fairness
for fairness_type in ["density", "size"]:
    for evaluation_metric in ["ARI", "VI", "ARI_m", "Purity_m"]:
        for fairness_metric in ["EMD", "F1", "FCC"]:
            scatterplot_fairness(
                fairness_scores=eval(f"{fairness_type}_fairness_scores"),
                evaluation_scores=eval(f"{fairness_type}_evaluation_scores"),
                fairness_metric=fairness_metric,
                evaluation_metric=evaluation_metric,
                title=f"{evaluation_metric} vs {fairness_metric} {fairness_type} Fairness",
                filename=f"{results_directory}/Scatterplot_{fairness_type}_{evaluation_metric}_{fairness_metric}",
            )
