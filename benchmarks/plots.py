from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay


def plot_reconerr_comparision(reconerr, input_, labels, start=0, end=None, title=None):
    r"""
    Plots the reconstruction error with respect to the input and output labels.
    """
    end = end or len(reconerr)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7))
    ax[0].plot(reconerr[start:end], color="b", label="reconstruction error")
    ax[0].legend(shadow=True)
    ax[1].plot(input_[start:end], label="input data")
    ax[1].legend(shadow=True)
    ax[2].plot(labels[start:end], color="g", label="labels")
    ax[2].legend(shadow=True)
    if title:
        ax[0].set_title(title)

    return fig


def plot_roc_curve(y_true, y_pred, model_name, title=None):
    _ = RocCurveDisplay.from_predictions(y_true, y_pred, name=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Baseline (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title:
        plt.title(title)
    plt.legend()
