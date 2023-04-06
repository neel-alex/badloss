import os
import natsort

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


font_size = 16

num_plots_per_row = 3
plot_rows = 3
num_queue_plots = num_plots_per_row * plot_rows

line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
marker_colors = ["tab:gray", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink",
                 "tab:olive",
                 "tab:brown", "tab:cyan"]
plot_train_test_sets = False
linewidth = 5.0
alpha = 0.7


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


def plot_probe_examples(probes, dataset, train_set, attack_types, rank, output_dir):
    print("Backdoor examples")
    plot(probes["backdoor"], probes["backdoor_labels"], None, class_names=train_set.classes,
         output_file=f"backdoor_{dataset}_{rank}.png", output_dir=output_dir)

    # In[ ]:

    # Plot updated backdoors
    print("Updated backdoor examples")
    for attack_type in attack_types:
        plot(probes[f"backdoor_{attack_type}"], probes[f"backdoor_{attack_type}_labels"], None,
             class_names=train_set.classes, output_file=f"backdoor_{dataset}_{attack_type}_{rank}.png",
             output_dir=output_dir)

    # In[ ]:

    print("Clean examples")
    plot(probes["clean"], probes["clean_labels"], None, class_names=train_set.classes,
         output_file=f"clean_{dataset}_{rank}.png", output_dir=output_dir)


def plot_probe_ex(x, y, probs, output_file=None):
    plot_size = 3
    fig, ax = plt.subplots(plot_rows, num_plots_per_row, figsize=(plot_size * num_plots_per_row, plot_size * plot_rows), sharex=True, sharey=True)

    for idx in range(len(x)):
        ax[idx // num_plots_per_row, idx % num_plots_per_row].imshow(x[idx])
        # ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(y[idx])
        if probs is not None:
            ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(f"{y[idx]} (PD: {probs[idx]:.3f})")
        else:
            ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(f"{y[idx]}")

        if idx == plot_rows * num_plots_per_row - 1:
            break

    for a in ax.ravel():
        a.set_axis_off()

        # Turn off tick labels
        a.set_yticklabels([])
        a.set_xticklabels([])

    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches=0.0, pad_inches=0)
    plt.close()


def some_plot(statistics, log_predictions, label_map_dict, include_plot_title, dataset, main_proc, output_dir):
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
    marker_colors = ["tab:gray", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink",
                     "tab:olive",
                     "tab:brown", "tab:cyan"]
    for val_included in [True, False]:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        x_vals = list(range(1, len(statistics["test"]) + 1))
        for idx, k in enumerate(natsort.natsorted(list(statistics.keys()))):
            if k == "predictions":
                continue
            if not log_predictions and k == "train":
                continue
            if not val_included and "_val" in k:
                continue
            if not plot_train_test_sets and ("train" in k or "test" in k):
                continue
            # line = plt.plot(x_vals, [x["acc"] for x in statistics[k]], linewidth=2., marker=marker_list[idx % len(marker_list)],
            #                 color=marker_colors[idx % len(marker_colors)], alpha=0.75, markeredgecolor='k', label=label_map_dict[k])
            line = plt.plot(x_vals, [x["acc"] for x in statistics[k]], linewidth=linewidth,
                            color=marker_colors[idx % len(marker_colors)],
                            alpha=alpha, label=label_map_dict[k])
            line[0].set_color(marker_colors[idx % len(marker_colors)])
            line[0].set_linestyle(line_styles[idx % len(line_styles)])

        plt.legend(prop={'size': font_size})
        plt.xlabel("Epochs", fontsize=font_size)
        plt.ylabel("Accuracy (%)", fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        if include_plot_title:
            plt.title(f"Training accuracy dynamics computed for ResNet-50 (CIFAR-100)", fontsize=font_size)
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"probe_acc_{dataset}{'_val' if val_included else ''}.png")
        if main_proc and output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")


def some_other_plot(statistics, log_predictions, label_map_dict, include_plot_title, dataset, main_proc, output_dir):
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
    marker_colors = ["tab:gray", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink",
                     "tab:olive",
                     "tab:brown", "tab:cyan"]
    for val_included in [True, False]:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        x_vals = list(range(1, len(statistics["test"]) + 1))
        for idx, k in enumerate(natsort.natsorted(list(statistics.keys()))):
            if k == "predictions":
                continue
            if not log_predictions and k == "train":
                continue
            if not val_included and "_val" in k:
                continue
            line = plt.plot(x_vals, [x["loss"] for x in statistics[k]], linewidth=linewidth,
                            color=marker_colors[idx % len(marker_colors)],
                            alpha=alpha, label=label_map_dict[k])
            line[0].set_color(marker_colors[idx % len(marker_colors)])
            line[0].set_linestyle(line_styles[idx % len(line_styles)])

        plt.legend(prop={'size': font_size})
        plt.xlabel("Epochs", fontsize=font_size)
        plt.ylabel("Loss", fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        if include_plot_title:
            plt.title(f"Training loss dynamics computed for ResNet-50 (CIFAR-100)", fontsize=font_size)
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"probe_loss_{dataset}{'_val' if val_included else ''}.png")
        if main_proc and output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")


def make_normalizers(num_train_probes, train_set, discarded_idx, unique_probe_identity):
    normalizers = {k: (num_train_probes if k != "train" else (len(train_set) - len(discarded_idx))) for k in
                   unique_probe_identity}
    print("Normalizers:", normalizers)
    return normalizers


def yet_another_plot(statistics, normalizers, epoch_cumulative_scores, epoch_cumulative_scores_first_learned,
                     label_map_dict, include_plot_title, dataset, main_proc, output_dir):
    # Normalization should only happen for num_train_probes (val probes are separate)


    for val_included in [True, False]:
        for iden, epoch_scores in enumerate([epoch_cumulative_scores, epoch_cumulative_scores_first_learned]):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 6)

            for idx, k in enumerate(natsort.natsorted(list(epoch_scores.keys()))):
                if not val_included and "_val" in k:
                    continue
                if not plot_train_test_sets and ("train" in k or "test" in k):
                    continue

                y = epoch_scores[k]
                x = np.arange(len(y))
                x_vals = list(range(1, len(statistics["test"]) + 1))
                y_norm = [(float(i) / normalizers[k]) * 100. for i in y]
                # line = plt.plot(x, y_norm, linewidth=2., marker=marker_list[idx % len(marker_list)],
                #                 color=marker_colors[idx % len(marker_colors)], alpha=0.75, markeredgecolor='k', label=label_map_dict[k])
                line = plt.plot(x_vals, y_norm, linewidth=linewidth, color=marker_colors[idx % len(marker_colors)],
                                alpha=alpha, label=label_map_dict[k])
                line[0].set_color(marker_colors[idx % len(marker_colors)])
                line[0].set_linestyle(line_styles[idx % len(line_styles)])

            plt.xlabel("Number of epochs", fontsize=font_size)
            # plt.ylabel(f"Fraction of examples learned{'at any point during training' if iden == 1 else ''} (%)", fontsize=font_size)
            plt.ylabel(f"Fraction of examples learned (%)", fontsize=font_size)
            if include_plot_title:
                plt.title(f"Learning dynamics computed for ResNet-50 (CIFAR-100)", fontsize=font_size)
            plt.legend(prop={'size': font_size})
            plt.ylim(0., 100.)

            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)

            plt.tight_layout()
            output_file = os.path.join(output_dir,
                                       f"{'first_learned' if iden == 1 else 'learning'}_dynamics_{dataset}{'_val' if val_included else ''}.png")
            if main_proc and output_file is not None:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close('all')


def one_more_plot(sorted_losses_all, class_names, label_map_dict, dataset_probe_identity,
                  dataset, main_proc, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(50, 10))
    labels = list(range(1, len(sorted_losses_all) + 1))
    color_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive',
                  'tab:gray']

    handles = []
    legend_label = []
    for i, cls in enumerate(class_names):
        if cls in ["train", "train_noisy"]:
            continue
        print("Class:", cls)
        color = color_list[i % len(color_list)]
        patch = mpatches.Patch(color=color)
        handles.append(patch)
        # legend_label.append(cls.replace("_", " ").title())
        legend_label.append(label_map_dict[cls])

        data = []
        for epoch in range(len(sorted_losses_all)):
            current_losses = [float(sorted_losses_all[epoch][i]) for i in range(len(sorted_losses_all[epoch])) if
                              str(dataset_probe_identity[i]) == cls and sorted_losses_all[epoch][i] is not None]
            data.append(current_losses)

        # parts = ax.boxplot(data, notch=True, patch_artist=True, showfliers=False)
        parts = ax.boxplot(data, notch=True, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color, color=color, alpha=0.3),
                           capprops=dict(color=color),
                           whiskerprops=dict(color=color),
                           flierprops=dict(color=color, markeredgecolor=color),
                           medianprops=dict(color=color))


    ax.legend(handles, legend_label, prop={'size': font_size})
    plt.ylabel("Loss values", fontsize=font_size)
    plt.xlabel("Epochs", fontsize=font_size)
    # plt.ylim(0, 6)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"loss_dist_{dataset}.png")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")