import os

import numpy as np
import matplotlib.pyplot as plt


font_size = 16
num_plots_per_row = 3
plot_rows = 3


def log_wandb_img(image_loc):
    import wandb
    image_name = os.path.splitext(os.path.split(image_loc)[1])[0]
    wandb.log({image_name: wandb.Image(image_loc)})


def plot(x, y=None, class_names=None, output_dir=None, output_file=None, diff_image=False, log_wandb=False):
    """Plot the first 9 images of x in a grid, titled by (named) labels y."""
    plot_size = 3
    fig, ax = plt.subplots(plot_rows, num_plots_per_row, figsize=(plot_size * num_plots_per_row, plot_size * plot_rows),
                           sharex=True, sharey=True)

    input = x.cpu().numpy()
    is_grayscale = input.shape[1] == 1
    input = input[:, 0, :, :] if is_grayscale else np.transpose(input, (0, 2, 3, 1))
    if class_names is not None:
        y = [class_names[int(i)] for i in y]

    if diff_image:
        input = np.abs(input)

        # Scale the maximum value to 1 for better visibility
        max_vals = input.reshape(len(input), -1).max(axis=1)
        max_vals[max_vals == 0] = 1.

        if is_grayscale:
            input = input / max_vals[:, None, None]
        else:
            input = input / max_vals[:, None, None, None]

    for idx in range(len(input)):
        ax[idx // num_plots_per_row, idx % num_plots_per_row].imshow(input[idx], cmap='gray' if is_grayscale else None)
        if y is not None:
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
        if log_wandb:
            log_wandb_img(os.path.join(output_dir, output_file))

    plt.close('all')


def plot_probe_examples(probe_imgs, triggered_sets, dataset, train_set, rank, output_dir, log_wandb=False):
    """Plot the clean probes and, for each triggered set (name -> ImageSet), its images and trigger diffs."""
    print("Backdoor examples")
    for name, image_set in triggered_sets.items():
        plot(image_set.images, image_set.labels, class_names=train_set.classes,
             output_dir=output_dir, output_file=f"backdoor_{dataset}_{name}_{rank}.png")
        plot(image_set.diff, image_set.labels, class_names=train_set.classes,
             output_dir=output_dir, output_file=f"backdoor_{dataset}_{name}_diff_{rank}.png", diff_image=True)

    print("Clean examples")
    output_file = f"clean_{dataset}_{rank}.png"
    plot(probe_imgs.images, probe_imgs.labels, class_names=train_set.classes,
         output_file=output_file, output_dir=output_dir)
    if log_wandb:
        log_wandb_img(os.path.join(output_dir, output_file))
