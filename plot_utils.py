import os
import numpy as np
import matplotlib.pyplot as plt


font_size = 16


def plot(x, y=None, memorization_val=None, class_names=None, output_dir=None, output_file=None, add_mem_scores=False):
    num_plots_per_row = 3
    plot_rows = 3
    plot_size = 3
    fig, ax = plt.subplots(plot_rows, num_plots_per_row, figsize=(plot_size * num_plots_per_row, plot_size * plot_rows),
                           sharex=True, sharey=True)

    input = x.cpu().numpy()
    is_grayscale = input.shape[1] == 1
    input = input[:, 0, :, :] if is_grayscale else np.transpose(input, (0, 2, 3, 1))
    if class_names is not None:
        y = [class_names[int(i)] for i in y]

    for idx in range(len(input)):
        ax[idx // num_plots_per_row, idx % num_plots_per_row].imshow(input[idx], cmap='gray' if is_grayscale else None)
        if y is not None:
            if add_mem_scores:
                ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(
                    f"{'Label: ' if class_names is None else ''}{y[idx]}" + (
                        f"\n(Mem: {memorization_val[idx]:.4f})" if memorization_val is not None else ""), color='g')
            else:
                ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(
                    f"{'Label: ' if class_names is None else ''}{y[idx].replace('_', ' ').title()}", color='k',
                    fontsize=font_size)

        if idx == plot_rows * num_plots_per_row - 1:
            break

    for a in ax.ravel():
        a.set_axis_off()
        a.set_yticklabels([])
        a.set_xticklabels([])

    fig.tight_layout()
    if output_file is not None:
        plt.savefig(os.path.join(output_dir, output_file), dpi=300, bbox_inches="tight")
    plt.close('all')
