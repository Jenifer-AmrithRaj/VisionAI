# utils/graph_utils.py
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def plot_roc_pr(y_true, y_score, out_dir="static/graphs", prefix="eval"):
    """
    y_true: array-like shape (n_samples,) with integer class labels 0..K-1 OR binary 0/1
    y_score: if multiclass: shape (n_samples, n_classes) with probs; if binary: (n_samples,) with prob for class 1
    This will save roc and pr for multiclass by one-vs-rest average.
    """
    out_dir = ensure_dir(out_dir)
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # if binary
    if y_score.ndim == 1 or (y_score.ndim==2 and y_score.shape[1]==1):
        # binary
        if y_score.ndim==2:
            s = y_score.ravel()
        else:
            s = y_score
        fpr, tpr, _ = roc_curve(y_true, s)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
        roc_path = Path(out_dir)/f"{prefix}_roc.png"
        plt.savefig(str(roc_path)); plt.close()

        precision, recall, _ = precision_recall_curve(y_true, s)
        ap = average_precision_score(y_true, s)
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f'AP = {ap:.3f}')
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall"); plt.legend()
        pr_path = Path(out_dir)/f"{prefix}_pr.png"
        plt.savefig(str(pr_path)); plt.close()
        return str(roc_path), str(pr_path)

    # multiclass: one-vs-rest AUC per class and micro-average
    n_classes = y_score.shape[1]
    # binarize y_true
    from sklearn.preprocessing import label_binarize
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    roc_paths = []
    pr_paths = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:,i], y_score[:,i])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,4))
        plt.plot(fpr,tpr,label=f'class {i} AUC={roc_auc:.3f}')
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC class {i}"); plt.legend()
        p = Path(out_dir)/f"{prefix}_roc_class_{i}.png"
        plt.savefig(str(p)); plt.close()
        roc_paths.append(str(p))

        precision, recall, _ = precision_recall_curve(Y[:,i], y_score[:,i])
        ap = average_precision_score(Y[:,i], y_score[:,i])
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f'AP={ap:.3f}')
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR class {i}"); plt.legend()
        p2 = Path(out_dir)/f"{prefix}_pr_class_{i}.png"
        plt.savefig(str(p2)); plt.close()
        pr_paths.append(str(p2))

    return roc_paths, pr_paths

def plot_confusion_matrix(y_true, y_pred, labels, out_path="static/graphs/confusion.png"):
    Path = __import__("pathlib").Path
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_f1_bar(y_true, y_pred, labels, out_path="static/graphs/f1_bar.png"):
    from sklearn.metrics import f1_score
    scores = []
    for i in range(len(labels)):
        s = f1_score(y_true, y_pred, labels=[i], average='binary', zero_division=0)
        scores.append(s)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(labels, [s*100 for s in scores])
    ax.set_ylabel("F1 (%)"); ax.set_title("F1 per class")
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path
