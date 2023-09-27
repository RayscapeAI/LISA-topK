import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from utils.utils import flatten_dict
from utils.config import classes


def write_stats(domain_stats, writer, domain, epoch):
    for score_name, value in domain_stats.items():
        if "average" in score_name:
            score_name = domain + "/" + score_name
        else:
            score_name = "Extra/" + domain + "/" + score_name
        writer.add_scalar(score_name, value, epoch)


def stats_handler(
    epoch, pred_all, target_all, domain_all, loss, writer, thresholds=None
):
    """
    Compute statistics and write to tensorboard
    """
    domain_stats = compute_xray_stats(target_all, pred_all, thresholds=thresholds)
    stats = {"All": domain_stats}
    write_stats(domain_stats, writer, "All", epoch)

    for domain in np.unique(domain_all):
        domain_stats = compute_xray_stats(
            target_all[domain_all == domain],
            pred_all[domain_all == domain],
            thresholds=thresholds,
        )
        stats[domain] = domain_stats
        write_stats(domain_stats, writer, domain, epoch)

    stats["loss"] = loss
    writer.add_scalar("loss", loss, epoch)

    return stats


def compute_xray_stats(y_true, y_pred, thresholds=None):
    assert y_true.min() >= 0 and y_true.max() <= 1
    assert y_pred.min() >= 0 and y_pred.max() <= 1

    results = {"average": {}}
    for i, class_name in enumerate(classes):
        results[class_name] = {}
        results[class_name]["F1"] = F1(y_true[:, i], y_pred[:, i])
        results[class_name]["roc_auc_score"] = AUROC(y_true[:, i], y_pred[:, i])

        if thresholds is not None:
            results[class_name]["F1_val_th"] = F1(
                y_true[:, i], y_pred[:, i], threshold=thresholds[class_name]
            )

    results["average"]["F1"] = np.mean(
        [results[class_name]["F1"]["f1"] for class_name in classes]
    )
    results["average"]["roc_auc_score"] = np.mean(
        [results[class_name]["roc_auc_score"] for class_name in classes]
    )

    if thresholds is not None:
        results["average"]["F1_val_th"] = np.mean(
            [results[class_name]["F1_val_th"]["f1"] for class_name in classes]
        )
    else:
        results["average"]["F1_val_th"] = results["average"]["F1"]

    return flatten_dict(results)


def F1(y_true, y_pred, threshold=None):
    f1_max = 0
    threshold_max = 0
    precision = 0
    recall = 0

    if y_true.sum() == 0:
        return {"f1": 0, "threshold": 0}  #'precision' : 0, 'recall': 0}

    thresholds = np.arange(0, 1, 0.05) if threshold is None else [threshold]

    for t in thresholds:
        x = y_pred >= t
        f1 = f1_score(y_true, x)

        if f1 > f1_max:
            f1_max = f1
            threshold_max = t
            precision = (y_true * x).sum() / x.sum() if x.sum() else 0
            recall = (y_true * x).sum() / y_true.sum()

    return {
        "f1": f1_max,
        "threshold": threshold_max,
    }  # , 'precision' : precision, 'recall': recall}


def AUROC(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return 0.5

    return roc_auc_score(y_true, y_pred)
