from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import itertools as itertool
import torch, torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def test(test_loader, model):
    model.cuda()
    model.eval()
    targets = torch.autograd.Variable().cuda()
    predictions = torch.autograd.Variable().cuda()
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for (input, target) in pbar:
            input, target = input.cuda(), target.float().cuda()
            targets = torch.cat((targets, target), 0)
            output = model(input)
            predictions = torch.cat((predictions, output.float()), 0)
            # plotImages(input.cpu(), n_images=(4, 4), title='real images')
            # plotImages(output.cpu(), n_images=(4, 4), title='reconstruction')
    targets = targets.cpu().numpy()
    predictions = predictions.cpu()
    bar_text = 'Testing:'
    pbar.set_description(bar_text)
    return targets, predictions

def get_roc_curve(pred_classes, true_classes, class_labels):
    # calculate ROC curve per class
    n_classes = np.shape(pred_classes)[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_classes[:, i], pred_classes[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_classes.ravel(), pred_classes.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    lw = 2
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='black', linestyle=':', linewidth=4)

    colors = itertool.cycle(['crimson', 'green', 'purple', 'yellow', 'blue', 'fuchsia', 'gray'])
    for i, color, emotion in zip(range(n_classes), colors, class_labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC for class {2} (area = {1:0.2f})'.format(i, roc_auc[i], emotion))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    legend = np.insert(class_labels, 0, ['micro', 'macro'])
    plt.legend(loc="lower right")
    plt.show()

def get_pr_curve(pred_classes, true_classes, class_labels):
    # For each class
    n_classes = np.shape(pred_classes)[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(true_classes[:, i],
                                                            pred_classes[:, i])
        average_precision[i] = average_precision_score(true_classes[:, i], pred_classes[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(true_classes.ravel(),
                                                                    pred_classes.ravel())
    average_precision["micro"] = average_precision_score(true_classes, pred_classes,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    colors = itertool.cycle(['crimson', 'green', 'purple', 'yellow', 'blue', 'fuchsia', 'gray'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='black', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class %s (area = %0.2f)' % (class_labels[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    n_classes = np.shape(gt)[-1]
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_classes):
        AUROCs.append(roc_auc_score(gt_np, pred_np))
    return AUROCs

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertool.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




