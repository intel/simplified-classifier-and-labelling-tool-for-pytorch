""" Code for plotting CM.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def plot_confusion_matrix(
    cm,
    grp_names=None,
    categories="auto",
    path_to_save="./cm.png",
    count=True,
    percent=True,
    cbar=True,
    sum_stats=True,
    fig_size=None,
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cm:            confusion matrix to be passed in
    grp_names:     List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    empty = ["" for i in range(cm.size)]

    if grp_names and len(grp_names) == cm.size:
        grp_labels = ["{}\n".format(value) for value in grp_names]
    else:
        grp_labels = empty

    if count:
        grp_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    else:
        grp_counts = empty

    if percent:
        grp_percentages = [
            "{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)
        ]
    else:
        grp_percentages = empty

    sq_labels = []
    for value1, value2, value3 in zip(grp_labels, grp_counts, grp_percentages):
        sq_labels.append(f"{value1}{value2}{value3}".strip())
    sq_labels = np.asarray(sq_labels).reshape(cm.shape[0], cm.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cm) / float(np.sum(cm))

        # if it is a binary confusion matrix, show some more stats
        if len(cm) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cm[1, 1] / sum(cm[:, 1])
            recall = cm[1, 1] / sum(cm[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = (
                "\nAccuracy={:0.3f}"
                "\nPrecision={:0.3f}"
                "\nRecall={:0.3f}"
                "\nF1 Score={:0.3f}".format(accuracy, precision, recall, f1_score)
            )
        else:
            stats_text = "\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # set figure parameters according to set arguments
    if fig_size == None:
        # Get default figure size if not set
        fig_size = plt.rcParams.get("figure.figsize")

    # HEATMAP VISUALIZATION
    plt.figure(figsize=fig_size)
    sb.heatmap(
        cm,
        annot=sq_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    plt.ylabel("True label")
    plt.xlabel("Predicted label" + stats_text)

    if title:
        plt.title(title)

    plt.savefig(path_to_save)
