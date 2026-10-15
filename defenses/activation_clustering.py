"""Activation Clustering (Chen et al., 2018), adapted to filter the training set and retrain.

Per predicted class, reduce penultimate activations with ICA and split them into two clusters; a class whose
clustering is well separated (silhouette score) or lopsided (relative size) is treated as attacked, and its
smaller cluster is removed. With --ac_oracle, in attack target classes the cluster containing more poisons is
removed instead.
"""
import numpy as np
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
import torch


def get_indices_for_class_clusters(detected_classes, clusterings, activation_by_predicted_class, oracle=None):
    """oracle: optional (target classes, is_poison array); in those classes, flag the cluster with more poisons."""
    identified_indices = []
    for cls in detected_classes:
        clustering = clusterings[cls]
        # Flag the smaller cluster
        if sum(clustering) * 2 > len(clustering):
            selected_indices = np.where(clustering == 0)
        else:
            selected_indices = np.where(clustering == 1)
        if oracle is not None and cls in oracle[0]:
            poisons = oracle[1][activation_by_predicted_class[cls][1].cpu().numpy()]
            in_cluster_1, in_cluster_0 = poisons[clustering == 1].sum(), poisons[clustering == 0].sum()
            if in_cluster_1 != in_cluster_0:
                selected_indices = np.where(clustering == (1 if in_cluster_1 > in_cluster_0 else 0))
        identified_indices.append(activation_by_predicted_class[cls][1][selected_indices])
    return torch.hstack(identified_indices) if identified_indices else np.array([], dtype=int)


def run(exp):
    args = exp.args
    exp.pretrain()
    exp.report_attacked_model()
    exp.wandb_prefix = "retraining_"

    activations, indices, classes, predictions = exp.last_layer_activations(exp.model, exp.new_idx_loader_wo_aug)
    activation_by_predicted_class = {}
    for i in range(exp.num_classes):
        indices_to_pick = np.where(predictions.cpu() == i)[0]
        activation_by_predicted_class[i] = (activations[indices_to_pick], indices[indices_to_pick])

    dim_reducer = sklearn.decomposition.FastICA(n_components=args.ac_ica_components, random_state=exp.seed)
    clusterer = sklearn.cluster.KMeans(n_clusters=2, random_state=exp.seed)

    clusterings, rsc_scores, sil_scores = [], [], []
    for cls in range(exp.num_classes):
        data = activation_by_predicted_class[cls][0].cpu()
        data = data[:, data.sum(dim=0).bool()]  # remove zero columns, otherwise dim reduction outputs all 0s

        fit = dim_reducer.fit_transform(data)
        clustering = clusterer.fit_predict(fit)
        rsc_score = sum(clustering) / len(data)  # Relative size of the clusters
        if rsc_score > 0.5:
            rsc_score = 1 - rsc_score
        clusterings.append(clustering)
        rsc_scores.append(rsc_score)
        sil_scores.append(sklearn.metrics.silhouette_score(fit, clustering))

    def detected_classes(thresh):
        if args.ac_mode == "sil":
            return [i for i in range(exp.num_classes) if sil_scores[i] > thresh]
        return [i for i in range(exp.num_classes) if rsc_scores[i] < thresh]

    auc_threshes = np.geomspace(0.01, 0.5 if args.ac_mode == "sil" else 0.4, num=50)
    oracle = (exp.target_classes(), exp.is_poison()) if args.ac_oracle else None
    auc_idx = [get_indices_for_class_clusters(detected_classes(t), clusterings, activation_by_predicted_class, oracle)
               for t in auc_threshes]
    exp.report_detection_auc("AC", auc_idx)

    identified_indices = get_indices_for_class_clusters(detected_classes(args.ac_threshold), clusterings,
                                                        activation_by_predicted_class, oracle)
    exp.report_detection(identified_indices)
    exp.retrain(identified_indices)
