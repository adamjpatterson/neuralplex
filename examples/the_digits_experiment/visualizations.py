"""Plot data and evaluation results supplied by the digits notebook."""

import matplotlib.pyplot as plt


def plot_predictions(images, labels, predictions=None, title='Sample digits'):
    """Return a figure of the first twelve supplied images, without selecting successes."""
    count = min(12, len(images))
    if count == 0:
        raise ValueError('Supply at least one image')
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis('off')
        if index >= len(images):
            continue
        axis.imshow(images[index], cmap='gray', vmin=0, vmax=16)
        label = f'True: {labels[index]}'
        if predictions is not None:
            label += f'   Predicted: {predictions[index]}'
        axis.set_title(label, fontsize=10)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(matrix):
    """Return a figure; rows are true digits and columns are predicted digits."""
    figure, axis = plt.subplots(figsize=(7, 6))
    heatmap = axis.imshow(matrix, cmap='Blues')
    axis.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit',
             ylabel='True digit', title='Neural-pleX: held-out confusion matrix')
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(matrix[row, column]), ha='center', va='center',
                      color='white' if matrix[row, column] > matrix.max() / 2 else 'black')
    figure.colorbar(heatmap, ax=axis)
    figure.tight_layout()
    return figure
