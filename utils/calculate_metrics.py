import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, jaccard_score, average_precision_score)

def metrics(true, pred) :
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = (pred >= 0.5).astype(np.int_)

    true = np.asarray(true.flatten(), dtype=np.int)
    pred = np.asarray(pred.flatten(), dtype=np.int)

    acc = accuracy_score(true, pred)
    pre = precision_score(true, pred, average='macro')
    rec = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    iou = jaccard_score(true, pred, average='macro')
    ap = average_precision_score(true, pred, average='macro')

    return acc, pre, rec, f1, iou, ap