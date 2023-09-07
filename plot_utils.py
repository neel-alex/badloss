import os
import natsort
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_curve, auc

from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def plot(x, y=None, memorization_val=None, class_names=None, output_dir=None, output_file=None, add_mem_scores=False,
         diff_image=False, use_abs_val=True):
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

    if diff_image:
        if use_abs_val:
            input = np.abs(input)

            # Scale the maximum value to 1 for better visibility
            max_vals = input.reshape(len(input), -1).max(axis=1)
            max_vals[max_vals == 0] = 1.
            
            if is_grayscale:
                input = input / max_vals[:, None, None]
            else:
                input = input / max_vals[:, None, None, None]
        else:
            input = (2 * input - 0.5).clip(0.0, 1.0)  # Rescale input range -- 0.5 means no change

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
    for attack_type in attack_types:
        plot(probes[f"{attack_type}"], probes[f"{attack_type}_labels"], None, class_names=train_set.classes,
             output_dir=output_dir, output_file=f"backdoor_{dataset}_{attack_type}_{rank}.png")
        plot(probes[f"{attack_type}_diff"], probes[f"{attack_type}_labels"], None, class_names=train_set.classes,
             output_dir=output_dir, output_file=f"backdoor_{dataset}_{attack_type}_diff_{rank}.png", diff_image=True)

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


def plot_loss_dynamics_and_violin(sorted_losses_all, class_names, label_map_dict, dataset_probe_identity,
                                  dataset, output_dir, main_proc):
    loss_dynamics_output_dir = os.path.join(output_dir, "loss_distribution")
    violin_loss_dynamics_output_dir = os.path.join(output_dir, "loss_distribution_violin")
    if main_proc and not os.path.exists(loss_dynamics_output_dir):
        os.mkdir(loss_dynamics_output_dir)
    if main_proc and not os.path.exists(violin_loss_dynamics_output_dir):
        os.mkdir(violin_loss_dynamics_output_dir)

    for epoch in range(0, len(sorted_losses_all), 5):
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        labels = list(range(1, len(sorted_losses_all) + 1))
        color_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan',
                      'tab:olive', 'tab:gray']
        plot_points = False

        handles = []
        legend_label = []
        data = []
        iterator = 0

        rej_classes = []

        for i, cls in enumerate(class_names):
            if cls in rej_classes:
                print(f"Ignoring class {cls} at index {i}")
                continue
            print("Class:", cls)
            color = color_list[iterator % len(color_list)]
            patch = mpatches.Patch(color=color)
            handles.append(patch)
            # legend_label.append(cls.replace("_", " ").title())
            legend_label.append(label_map_dict[cls])

            data = [[] for _ in range(len(class_names) - len(rej_classes))]
            current_losses = [float(sorted_losses_all[epoch][i]) for i in range(len(sorted_losses_all[epoch])) if
                              str(dataset_probe_identity[i]) == cls and sorted_losses_all[epoch][i] is not None]
            data[iterator] = current_losses

            parts = ax.boxplot(data, notch=True, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=color, color=color, alpha=1.0),
                               capprops=dict(color=color),
                               whiskerprops=dict(color=color),
                               flierprops=dict(color=color, markeredgecolor=color),
                               medianprops=dict(color=color))

            iterator += 1

        # ax.legend(handles, legend_label, prop={'size': font_size})
        plt.ylabel("Loss values", fontsize=font_size)
        ax.set_xticks(range(1, len(legend_label) + 1))
        ax.set_xticklabels(legend_label, fontsize=font_size)
        plt.xticks(rotation=90)
        plt.yticks(fontsize=font_size - 2)
        plt.ylim(0., 14.)

        plt.tight_layout()
        output_file = os.path.join(loss_dynamics_output_dir, f"loss_dist_ep_{epoch}_{dataset}.png")
        if main_proc and output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close('all')

    for epoch in range(0, len(sorted_losses_all), 5):
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        labels = list(range(1, len(sorted_losses_all) + 1))
        color_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan',
                      'tab:olive', 'tab:gray']
        plot_points = False

        handles = []
        legend_label = []
        data = []
        iterator = 0

        print("Rejected classes:", rej_classes)

        for i, cls in enumerate(class_names):
            if cls in rej_classes:
                print(f"Ignoring class {cls} at index {i}")
                continue
            print("Class:", cls)
            color = color_list[iterator % len(color_list)]
            patch = mpatches.Patch(color=color)
            handles.append(patch)
            # legend_label.append(cls.replace("_", " ").title())
            legend_label.append(label_map_dict[cls])

            data = [[float('nan'), float('nan')] for _ in range(len(class_names) - len(rej_classes))]
            current_losses = [float(sorted_losses_all[epoch][i]) for i in range(len(sorted_losses_all[epoch])) if
                              str(dataset_probe_identity[i]) == cls and sorted_losses_all[epoch][i] is not None]
            data[iterator] = current_losses

            parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False, widths=0.8)
            for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
                if part_name in parts:
                    pc = parts[part_name]
                    pc.set_edgecolor(color)
                    pc.set_linewidth(1)
            for pc in parts['bodies']:
                pc.set_facecolor(color)

            # Plot the points
            num_points = 250
            include_points = True
            if include_points:
                ax.scatter([iterator + 1 for _ in range(num_points)], np.random.choice(data[iterator], num_points),
                           alpha=0.1, color=color)

            iterator += 1

        # ax.legend(handles, legend_label, prop={'size': font_size})
        plt.ylabel("Loss values", fontsize=font_size)
        ax.set_xticks(range(1, len(legend_label) + 1))
        ax.set_xticklabels(legend_label, fontsize=font_size)
        plt.xticks(rotation=90)
        plt.yticks(fontsize=font_size - 2)
        plt.ylim(0., 14.)

        plt.tight_layout()
        output_file = os.path.join(violin_loss_dynamics_output_dir, f"loss_dist_violin_ep_{epoch}_{dataset}.png")
        if main_proc and output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close('all')


def visualize_loss_trajectories(class_names, label_map_dict, dataset_probe_identity,
                                sorted_losses_all, output_dir, main_proc, dataset,
                                val_included=False, clf=None, output_file=None):
    current_class_names = [x for x in class_names if x not in ["train", "train_noisy", "train_non_noisy"]]
    if not val_included:
        current_class_names = [x for x in current_class_names if not x.endswith("_val")]
    print("Selected class names:", current_class_names)

    num_colors = len(current_class_names)
    if num_colors > 9:
        cm = plt.get_cmap('hsv')
        color_list = [cm(1. * i / len(current_class_names)) for i in range(len(current_class_names))]
    elif num_colors > 4:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive",
                      "tab:brown", "tab:cyan"]
    else:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]
    assert num_colors <= len(color_list), f"{num_colors} <= {len(color_list)}"
    num_trajectories = 250
    font_size = 18

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    handles = []
    legend_label = []

    iterator = 0
    traj_list = []
    for i, cls in enumerate(current_class_names):
        # if "val" in cls or "train" in cls:
        #     continue
        color = color_list[iterator]
        patch = mpatches.Patch(color=color)
        handles.append(patch)
        # legend_label.append(cls.title().replace("_", " "))
        legend_label.append(label_map_dict[cls])

        relevant_idx = [i for i in range(len(dataset_probe_identity)) if dataset_probe_identity[i] == cls]
        print(f"Class: {cls} / # relevant idx: {len(relevant_idx)}")

        x_axis = list(range(len(sorted_losses_all)))
        all_trajs = []
        for j in range(num_trajectories):
            if j > len(relevant_idx) - 1:
                break
            trajectory = [float(sorted_losses_all[epoch][relevant_idx[j]]) for epoch in range(len(sorted_losses_all))]
            plt.plot(x_axis, trajectory, color=color_list[iterator], alpha=0.05)
            all_trajs.append(trajectory)
        traj_list += all_trajs

        if clf is None:
            # Plot the trajectory mean
            mean_traj = np.array(all_trajs).mean(axis=0)
            plt.plot(x_axis, mean_traj, color=color_list[iterator], alpha=0.9, linewidth=5.)
        iterator += 1

    if clf is not None:
        num_clusters = len(clf.cluster_centers_)
        cm = plt.get_cmap('viridis')
        new_color_list = [cm(1. * i / num_clusters) for i in range(num_clusters)]

        for i in range(num_clusters):
            # Plot the cluster center
            color = new_color_list[i]
            cluster_center = clf.cluster_centers_[i]
            plt.plot(x_axis, cluster_center, color=color, alpha=0.9, linewidth=5.)

            # Add the color to the legend
            patch = mpatches.Patch(color=color)
            handles.append(patch)
            legend_label.append(f"Cluster # {i + 1}")

    ax.legend(handles, legend_label, prop={'size': font_size})
    plt.ylabel("Loss values", fontsize=font_size)
    plt.xlabel("Epochs", fontsize=font_size)
    max_val = np.percentile(traj_list, 99)
    plt.ylim(0., max_val)
    plt.xlim(0., len(x_axis) - 1)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    if output_file is None:
        output_file = os.path.join(output_dir,
                                   f"loss_trajectories_{dataset}{'_val' if val_included else ''}.png")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close('all')


def visualize_loss_trajectories_specific(class_names, label_map_dict, dataset_probe_identity,
                                         sorted_losses_all, output_dir, main_proc, dataset, output_file=None):
    current_class_names = [x for x in class_names if x not in ["train", "train_noisy", "train_non_noisy"]]
    current_class_names = [x for x in current_class_names if "_val" not in x or x.replace("_val", "") not in class_names]
    print("Selected class names:", current_class_names)
    
    num_colors = len(current_class_names)
    if num_colors > 9:
        cm = plt.get_cmap('hsv')
        color_list = [cm(1.*i/len(current_class_names)) for i in range(len(current_class_names))]
    elif num_colors > 4:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive", "tab:brown", "tab:cyan"]
    else:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]
    assert num_colors <= len(color_list), f"{num_colors} <= {len(color_list)}"
    num_trajectories = 250
    font_size = 18

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    handles = []
    legend_label = []

    iterator = 0
    traj_list = []
    num_checkpoints_to_consider = 20
    for i, cls in enumerate(current_class_names):
        color = color_list[iterator]
        patch = mpatches.Patch(color=color)
        handles.append(patch)
        label = label_map_dict[cls].replace(" [Val]", "")
        legend_label.append(label)
        
        relevant_idx = [i for i in range(len(dataset_probe_identity)) if dataset_probe_identity[i] == cls]
        print(f"Class: {cls} / # relevant idx: {len(relevant_idx)}")

        x_axis = list(range(1, len(sorted_losses_all)+1))
        x_axis = x_axis[:num_checkpoints_to_consider]  # Subsample
        all_trajs = []
        for j in range(num_trajectories):
            if j > len(relevant_idx) - 1:
                break
            trajectory = [float(sorted_losses_all[epoch][relevant_idx[j]]) for epoch in range(len(sorted_losses_all))]
            trajectory = trajectory[:num_checkpoints_to_consider]  # Subsample
            plt.plot(x_axis, trajectory, color=color_list[iterator], alpha=0.02)
            all_trajs.append(trajectory)
        traj_list += all_trajs
        
        # Plot the trajectory mean
        mean_traj = np.array(all_trajs).mean(axis=0)
        plt.plot(x_axis, mean_traj, color=color_list[iterator], alpha=0.9, linewidth=5.)
        iterator += 1
    
    ax.legend(handles, legend_label, prop={'size': font_size-2})
    plt.ylabel("Loss values", fontsize=font_size)
    plt.xlabel("Epochs", fontsize=font_size)
    max_val = np.percentile(traj_list, 99)
    plt.ylim(0., max_val)
    plt.xlim(1, len(x_axis)-1)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    if output_file is None:
        output_file = os.path.join(output_dir, f"loss_trajectories_{dataset}_specific.pdf")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close('all')


def generate_embeddings_from_trajectories(class_names, label_map_dict, dataset_probe_identity,
                                          sorted_losses_all, output_dir, main_proc, dataset,
                                          output_file=None, embedding_type='tsne'):
    assert embedding_type in ["tsne", "pca", "mds"]
    
    current_class_names = [x for x in class_names if x not in ["train", "train_noisy", "train_non_noisy"]]
    current_class_names = [x for x in current_class_names if "_val" not in x or x.replace("_val", "") not in class_names]
    print("Selected class names:", current_class_names)
    
    num_colors = len(current_class_names)
    if num_colors > 9:
        cm = plt.get_cmap('hsv')
        color_list = [cm(1.*i/len(current_class_names)) for i in range(len(current_class_names))]
    elif num_colors > 4:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive", "tab:brown", "tab:cyan"]
    else:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]
    assert num_colors <= len(color_list), f"{num_colors} <= {len(color_list)}"
    num_trajectories = 250
    font_size = 18
    
    losses_np = np.array(sorted_losses_all).transpose().astype(np.float64)  # Should be in format (# ex \times # epochs)
    missing_vals = np.isnan(losses_np).any(axis=1)  # Identify probe examples
    available_ex = np.logical_not(missing_vals)
    losses_np = losses_np[available_ex]  # Remove empty trajectories
    current_dataset_probe_identity = np.array(dataset_probe_identity)[available_ex]
    
    print("All trajectories shape before PCA:", losses_np.shape)
    scale_transformer = StandardScaler()
    all_trajectories_scaled = scale_transformer.fit_transform(losses_np)  # Perform data scaling
    
    num_components = 2
    
    if embedding_type == "tsne":
        print(f"Selected number of tSNE components: {num_components}")
        embedding_func = TSNE(n_components=num_components, learning_rate='auto', init='random', perplexity=3)
    elif embedding_type == "pca":
        print(f"Selected number of PCA components: {num_components}")
        embedding_func = PCA(n_components=num_components)
    elif embedding_type == "mds":
        print(f"Selected number of MDS components: {num_components}")
        embedding_func = MDS(n_components=num_components, max_iter=300, n_init=4, random_state=0)
    else:
        raise RuntimeError(f"Unknown embedding type: {embedding_type}")

    embedded_trajectories = embedding_func.fit_transform(all_trajectories_scaled)
    print(f"{embedding_type.upper()} transform output shape: {embedded_trajectories.shape}")
    
    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.default_rng(20)

    # Project the probe trajectories using the computed tSNE transform
    legend_elements = []
    for i, k in enumerate(current_class_names):
        identifier = current_dataset_probe_identity == k
        selected_trajs = embedded_trajectories[identifier]
        print(f"k: {k} / all trajs: {len(embedded_trajectories)} / seletected trajs: {np.sum(identifier)} (shape: {selected_trajs.shape})")
        
        # Select and plot a small number of points#
        selection_size = 250
        if len(selected_trajs) > selection_size:
            selected_points = rng.choice(len(selected_trajs), size=selection_size, replace=False)
        else:
            selected_points = np.arange(len(selected_trajs))
        alpha = 0.2
        label = label_map_dict[k].replace(" [Val]", "")
        plt.scatter(selected_trajs[selected_points, 0], selected_trajs[selected_points, 1], alpha=alpha, color=color_list[i], label=label)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color_list[i], alpha=1, markersize=10))

    ax.legend(handles=legend_elements, loc='best', prop={'size': 14})
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    if output_file is None:
        output_file = os.path.join(output_dir, f"{dataset}_{embedding_type}_trajs.pdf")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close('all')


def plot_confusion_matrix_from_preds(y_true, y_pred, classes, include_all_val, num_train_probes,
                                     output_dir, normalize=False, title=None, cmap=plt.cm.Blues,
                                     fontsize=15):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cm = confusion_matrix(
        y_true,
        y_pred,
        sample_weight=None,
        labels=None,
        normalize=None,
    )

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 7))

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None:
        plt.title(title)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(len(classes))
    display_labels = [x.title().replace("_", " ") for x in classes]
    plt.xticks(tick_marks, display_labels, fontsize=fontsize, rotation=45, ha="right")
    plt.yticks(tick_marks, display_labels, fontsize=fontsize, rotation=0, ha="right")

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", fontsize=fontsize,
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label', fontsize=fontsize)
        plt.xlabel('Predicted label', fontsize=fontsize)

    plt.tight_layout()
    output_file = os.path.join(output_dir,
                               f"probe_confusion_matrix_trajectories_val_probes{'_all' if include_all_val else ''}_{num_train_probes}{'_norm' if normalize else ''}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


def plot_auc(labels, predictions, key_list, output_file, log_plot=False, adapt_auc=False, title=None):
    assert isinstance(predictions, dict), predictions
    assert isinstance(key_list, list), key_list

    num_colors = len(key_list)
    if num_colors > 9:
        cm = plt.get_cmap('hsv')
        color_list = [cm(1. * i / len(key_list)) for i in range(len(key_list))]
    elif num_colors > 4:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive",
                      "tab:brown", "tab:cyan"]
    else:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]
    assert num_colors <= len(color_list), f"{num_colors} <= {len(color_list)}"

    fontsize = 15
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    import matplotlib
    from cycler import cycler
    matplotlib.rcParams['lines.linewidth'] = 3
    plt.rcParams['axes.prop_cycle'] = cycler(alpha=[0.5])

    if log_plot and adapt_auc:
        # Compute fpr and tpr values for the linear scale
        fpr_dict = {}
        tpr_dict = {}
        for k in key_list:
            fpr_dict[k], tpr_dict[k], _ = roc_curve(labels[k], predictions[k])
            print(auc(fpr_dict[k], tpr_dict[k]))
        # If log_plot is True, transform fpr and tpr values to the log scale
        eps = 1e-10
        if log_plot:
            for k in key_list:
                fpr_dict[k] = np.log10(fpr_dict[k] + eps)
                tpr_dict[k] = np.log10(tpr_dict[k] + eps)

        # Plot the ROC curve
        for i, k in enumerate(key_list):
            # If log_plot is True, set the label to include the log scale
            if log_plot:
                label = f"{k} (log scale)"
            else:
                label = k
            out = RocCurveDisplay(fpr=fpr_dict[k], tpr=tpr_dict[k], roc_auc=auc(fpr_dict[k], tpr_dict[k]),
                                  estimator_name=label).plot(ax=ax, color=color_list[i])

    else:
        # Plot the ROC curve
        for i, k in enumerate(key_list):
            out = RocCurveDisplay.from_predictions(labels[k], predictions[k], ax=ax, name=k)
            out.line_.set_color(color_list[i])

    alpha = 0.8
    for l in plt.gca().lines:
        l.set_alpha(alpha)

    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontsize=fontsize)
    if log_plot:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-5, 1e0)
        plt.ylim(1e-5, 1e0)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", alpha=0.5)

    # Sort legend labels
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, prop={'size': fontsize})

    if title is not None:
        plt.title(title, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")


def plot_attack_success_stats(output_dict, label_map_dict, ref_probe_classes, output_file, title=None):
    fontsize = 15
    fig, ax = plt.subplots(1, 1, figsize=(6, 7 + (1 if title is not None else 0)))

    keys = natsort.natsorted(list(output_dict.keys()))
    accuracies = [output_dict[k]['accuracy'] for k in keys]
    keys = [label_map_dict[f"{k}_val" if k not in ref_probe_classes else k].replace(" [Val]", "") for k in
            keys]
    plt.bar(keys, accuracies)

    plt.xlabel("Attack type", fontsize=fontsize)
    plt.ylabel("Accuracy (%)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=45, ha="right")
    plt.yticks(fontsize=fontsize)
    plt.ylim(0, 100)

    if title is not None:
        plt.title(title, fontsize=fontsize - 4)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
