"""Plotting helpers for sentiment and attention analysis."""
import numpy as np
from matplotlib import pyplot as plt

SENTIMENT_DISPLAY_NAMES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]


def plot_preds_labels_dists(hist_preds, hist_labels, label_out, is_equal_label_dist=False):
    """Bar chart of predictions vs gold labels (5-class sentiment)."""
    bar_width = 0.20
    fig, ax = plt.subplots()
    list_preds = sorted(hist_preds.items())
    x_preds, y_preds = zip(*list_preds)
    list_labels = sorted(hist_labels.items())
    x_labels, y_labels = zip(*list_labels)
    labels = ["Predictions", "Gold-Labels"]
    shift = 0.5
    rects1 = plt.bar(
        np.arange(len(SENTIMENT_DISPLAY_NAMES)) - (shift * bar_width),
        y_preds,
        color="#7eb0d5",
        label=labels[0],
        width=bar_width,
    )
    rects2 = plt.bar(
        np.arange(len(SENTIMENT_DISPLAY_NAMES)) + (shift * bar_width),
        y_labels,
        color="#bd7ebe",
        label=labels[1],
        width=bar_width,
    )
    plt.xticks(np.arange(len(y_preds)), SENTIMENT_DISPLAY_NAMES, rotation=0)
    if is_equal_label_dist:
        plt.title("Sentiment Analysis Distribution: Preds vs. Labels (Equal)")
    else:
        plt.title("Sentiment Analysis Distribution: Preds vs. Labels (Non-Equal)")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                1.005 * height,
                "%d" % float(height),
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    ax.legend()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, 120))
    plt.ylabel("The number of predictions/labels")
    fig.tight_layout()
    plt.savefig(label_out + "_" + "preds" + ".png", format="png", dpi=300)
