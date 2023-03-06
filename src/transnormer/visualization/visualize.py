from numpy.typing import ArrayLike
from typing import Optional, List
import matplotlib.pyplot as plt


def plot_loss(
    loss_train: ArrayLike,
    loss_test: Optional[ArrayLike] = None,
    epochs_train: Optional[ArrayLike] = None,
    epochs_test: Optional[ArrayLike] = None,
    title: str = "Loss per epoch",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    legend: List[str] = ["Training Loss", "Test Loss"],
) -> None:
    """Visualize loss history"""

    # Create count of the number of epochs
    if epochs_train is None:
        epochs_train = range(1, len(loss_train) + 1)

    plt.title(title)
    # Plot train loss as a dashed red line
    plt.plot(epochs_train, loss_train, "r--")
    # Plot test loss as well
    if loss_test is not None:
        if epochs_test is None:
            epochs_test = range(1, len(loss_test) + 1)
        # Plot tes loss as a solid blue line
        plt.plot(epochs_test, loss_test, "b-")
        # Add legend
        plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
