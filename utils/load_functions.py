import os
import copy

import torch

import numpy as np

from .get_functions import get_save_path

def load_model(save_path, data_type, image_size,
               batch_size, model_name, lr, epochs, c_fold=-1, **kwargs) :
    kwargs['save_path'] = save_path
    kwargs['data_type'] = data_type
    kwargs['image_size'] = image_size
    kwargs['batch_size'] = batch_size
    kwargs['model_name'] = model_name
    kwargs['lr'] = lr
    kwargs['epochs'] = epochs
    kwargs['fold'] = c_fold

    model_dirs, load_model_path = get_save_path(**kwargs)

    load_path = os.path.join(model_dirs.format(c_fold), '{}.pth'.format(load_model_path))

    print("Your model is loaded from {}.".format(load_path))
    checkpoint = torch.load(load_path)
    print(".pth keys() =  {}.".format(checkpoint.keys()))

    model = checkpoint['model']
    model.load_state_dict(copy.deepcopy(checkpoint['model_state_dict']))

    return model

def load_total_metric_results(**kwargs) :
    model_dirs, save_model_path = get_save_path(**kwargs)
    save_path = os.path.join(model_dirs, 'Total results {}.txt'.format(save_model_path))
    loss, acc, pre, rec, f1, iou, ap = [], [], [], [], [], [], []

    for fold in range(1, kwargs['k_fold'] + 1):
        load_path = os.path.join(model_dirs, 'fold {}'.format(fold),
                                 'test report {}.txt'.format(save_model_path))
        f = open(load_path, 'r')
        while True:
            line = f.readline()
            if not line: break
            line_split = line.split()
            if 'loss' in line_split         : loss.append(float(line_split[-1]))
            if 'accuracy' in line_split     : acc.append(float(line_split[-1]))
            if 'precision' in line_split    : pre.append(float(line_split[-1]))
            if 'recall' in line_split       : rec.append(float(line_split[-1]))
            if 'f1_score' in line_split     : f1.append(float(line_split[-1]))
            if 'iou' in line_split          : iou.append(float(line_split[-1]))
            if 'ap' in line_split           : ap.append(float(line_split[-1]))
        f.close()

    loss = np.array(loss)
    acc = np.array(acc)
    pre = np.array(pre)
    rec = np.array(rec)
    f1 = np.array(f1)
    iou = np.array(iou)
    ap = np.array(ap)

    return loss, acc, pre, rec, f1, iou, ap, save_path