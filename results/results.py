import pickle
import os
from utils import scatterplot_fairness, match_result_to_underlying_method

OPTIMIZATION = [
    "louvain",
    "leiden",
    "cpm",
    "significance_communities",
    "surprise_communities",
    "greedy_modularity",
    "paris",
    "pycombo",
    "rb_pots",
    "rber_pots",
]
PROPAGATION = [
    "label_propagation",
    "agdl",
    "async_fluid",
]
SPECTRAL = [
    "kcut",
    "r_spectral_clustering-vanilla",
    "r_spectral_clustering-regularized_with_kmeans",
    "r_spectral_clustering-sklearn_spectral_embedding",
    "r_spectral_clustering-sklearn_kmeans",
    "spectral",
    "eigenvector",
]
# BETWEENNESS = [
#     None  # Not using either of the two methods, should remove it here and in my thesis
# ]
DYNAMICS = ["infomap", "spinglass", "walktrap"]
MATRIX = ["markov_clustering", "chinesewhispers"]
REPRESENTATIONAL = ["Node2Vec", "der",
                    # "gemsec",
                    "ricci_community"
                    ]
OTHER = ["sbm_dl", "sbm_dl_nested",
         "belief",
         "edmot", "em",
         # "ga",
         "scd"]

for fairness_type in ["density", "size"]:
    ALL_UNDERLYING_FAIRNESS = [
        [{key: None for key in OPTIMIZATION}, "Optimization"],
        [{key: None for key in PROPAGATION}, "Propagation"],
        [{key: None for key in SPECTRAL}, "Spectral properties"],
        [{key: None for key in DYNAMICS}, "Dynamics"],
        [{key: None for key in MATRIX}, "Matrix approach"],
        [{key: None for key in REPRESENTATIONAL}, "Representational learning"],
        [{key: None for key in OTHER}, "Other"],
    ]
    ALL_UNDERLYING_EVALUATION = [
        [{key: None for key in OPTIMIZATION}, "Optimization"],
        [{key: None for key in PROPAGATION}, "Propagation"],
        [{key: None for key in SPECTRAL}, "Spectral properties"],
        [{key: None for key in DYNAMICS}, "Dynamics"],
        [{key: None for key in MATRIX}, "Matrix approach"],
        [{key: None for key in REPRESENTATIONAL}, "Representational learning"],
        [{key: None for key in OTHER}, "Other"],
    ]

    results = []
    results_directory = "LFR results"
    for root, _, files in os.walk(results_directory):
        for file in files:
            filename_splitted = os.path.basename(file).split("-")
            fairness_type_file = filename_splitted[0]
            if file.endswith(".pickle") and fairness_type_file == fairness_type:
                method = filename_splitted[1]
                score_type = filename_splitted[2]

                if (
                    method == "r_spectral_clustering"
                ):  # Quick fix for having a - in the name
                    method = filename_splitted[1] + "-" + filename_splitted[2]
                    score_type = filename_splitted[3]

                with open(f"{root}/{file}", "rb") as pickled_results:
                    pickled_scores = pickle.load(pickled_results)

                if score_type == "fairness":
                    ALL_UNDERLYING_FAIRNESS = match_result_to_underlying_method(
                        cd_method=method,
                        scores=pickled_scores,
                        underlying_methods=ALL_UNDERLYING_FAIRNESS,
                    )
                else:  # evaluation
                    ALL_UNDERLYING_EVALUATION = match_result_to_underlying_method(
                        cd_method=method,
                        scores=pickled_scores,
                        underlying_methods=ALL_UNDERLYING_EVALUATION,
                    )

    for idx, scores in enumerate(
        zip(ALL_UNDERLYING_FAIRNESS, ALL_UNDERLYING_EVALUATION)
    ):
        for evaluation_metric in ["ARI", "VI", "F_measure_m"]:
            for fairness_metric in ["EMD", "F1", "FCC"]:
                scatterplot_fairness(
                    fairness_scores=ALL_UNDERLYING_FAIRNESS[idx][0],
                    evaluation_scores=ALL_UNDERLYING_EVALUATION[idx][0],
                    fairness_metric=fairness_metric,
                    evaluation_metric=evaluation_metric,
                    title=f"{evaluation_metric} vs {fairness_metric} - {fairness_type.capitalize()} Fairness"
                    f" for underlying method {ALL_UNDERLYING_FAIRNESS[idx][1]}",
                    filename=f"{results_directory}/Scatterplot_{fairness_type}_{evaluation_metric}_"
                    f"{fairness_metric}_{ALL_UNDERLYING_FAIRNESS[idx][1]}",
                )
