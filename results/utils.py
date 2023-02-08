import numpy as np
import matplotlib.pyplot as plt


def scatterplot_fairness(
    fairness_scores: dict,
    evaluation_scores: dict,
    fairness_metric: str,
    evaluation_metric: str,
    title: str,
    filename: str,
):
    """

    :param fairness_scores: Dictionary containing fairness scores per method and fairness type
    :param evaluation_scores: Dictionary containing accuracy values per method and accuracy type
    :param fairness_metric: String indicating fairness type: EMD, F1, FCC
    :param evaluation_metric: String indicating evaluation metrics: ARI, VI, Purity_m
    :param filename: Filename of plot that will be saved
    :return: Matplotlib plot
    """
    fairness_metrics = {"EMD": 0, "F1": 1, "FCC": 2}
    evaluation_metrics = {"ARI": 0, "VI": 1, "Purity_m": 2}

    for method, scores in fairness_scores.items():
        if (
            len(scores[fairness_metrics[fairness_metric]]) > 1
        ):  # Not a real-world dataset, thus LFR datasets
            fairness_score = scores[fairness_metrics[fairness_metric]]
            fairness_score = [x for x in fairness_score if x is not None]
            evaluation_score_list = [x for x in evaluation_scores[method]][
                evaluation_metrics[evaluation_metric]
            ]
            eval_score = [x.score for x in evaluation_score_list if x is not None]

        else:  # Real-world dataset
            fairness_score = scores[fairness_metrics[fairness_metric]]
            evaluation_score_list = [x for x in evaluation_scores[method]][
                evaluation_metrics[evaluation_metric]
            ]
            eval_score = [x.score for x in evaluation_score_list if x is not None]

        plt.errorbar(
            x=np.mean(fairness_score),
            y=np.mean(eval_score),
            xerr=np.std(fairness_score),
            yerr=np.std(eval_score),
            label=f"{method}",
            fmt="o",
        )

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel(f"Average Fairness score of type {fairness_metric}")
    plt.ylabel(f"Accuracy of type {evaluation_metric}")
    plt.title(title)
    plt.xlim(0, 1)
    if evaluation_metric != "VI":
        plt.ylim(0, 1)
    else:  # Variation of information, need to set different bound than 1.
        max_vi = 0
        for score in evaluation_scores.values():
            matchingresult_per_method = score[1]
            vi_per_method = [
                x.score for x in matchingresult_per_method if x is not None
            ]
            for vi in vi_per_method:
                if vi > max_vi:
                    max_vi = vi
        plt.ylim(0, max_vi)
    plt.savefig(f"plots/{filename}.png", bbox_inches="tight")
    plt.close()  # Use plot.show() if we want to show it


def match_result_to_underlying_method(
    cd_method: str, scores: tuple, underlying_methods: list
):
    for underlying_method in underlying_methods:
        if cd_method in underlying_method[0].keys():
            underlying_method[0][cd_method] = scores
            return underlying_methods
