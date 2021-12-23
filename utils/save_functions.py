import os

import torch

import numpy as np
import pandas as pd

from utils.get_functions import get_save_path
from utils.plot_functions import plot_loss
from utils.load_functions import load_total_metric_results

def save_result(model, optimizer, history, **kwargs) :
    model_dirs, save_model_path = get_save_path(**kwargs)
    save_model(model, optimizer, model_dirs, save_model_path, kwargs['model_name'])
    plot_loss(history, os.path.join(model_dirs, '{}.png'.format(save_model_path)))
    save_loss(history, model_dirs, save_model_path)

    if 'test_result' in kwargs.keys() :
        save_metrics(kwargs['test_result'], model_dirs, save_model_path)

def save_model(model, optimizer, model_dirs, save_path, model_name) :
    check_point = {
        'model': model.module if torch.cuda.device_count() > 1 else model,
        'model_name': model_name,
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(check_point,
               os.path.join(model_dirs, '{}.pth'.format(save_path)))

def save_loss(history, model_dirs, save_model_path) :
    loss_tendency_df = pd.DataFrame(history)
    loss_tendency_df.to_csv(os.path.join(model_dirs, 'loss {}.csv'.format(save_model_path)), index=False)

def save_metrics(test_results, model_dirs, save_model_path) :
    test_loss, accuracy, precision, recall, f1_score, iou, ap = test_results

    print("###################### TEST REPORT ######################")
    print("FINAL TEST loss : {} | accuracy : {} | precision : {} | recall : {} | f1_score : {} | iou : {} | ap : {}".format(
        test_loss, accuracy, precision, recall, f1_score, iou, ap
    ))
    print("###################### TEST REPORT ######################")

    f = open(os.path.join(model_dirs, 'test report {}.txt'.format(save_model_path)), 'w')

    f.write("###################### TEST REPORT ######################\n")
    f.write("test loss : {}\n".format(test_loss))
    f.write("test accuracy : {}\n".format(accuracy))
    f.write("test precision : {}\n".format(precision))
    f.write("test recall : {}\n".format(recall))
    f.write("test f1_score : {}\n".format(f1_score))
    f.write("test iou : {}\n".format(iou))
    f.write("test ap : {}\n".format(ap))
    f.write("###################### TEST REPORT ######################")

    f.close()

def save_total_metrics(**kwargs) :
    loss, acc, pre, rec, f1, iou, ap, save_path = load_total_metric_results(**kwargs)

    print("###################### TOTAL TEST REPORT!!! ######################")
    print("TOTAL test loss      : {}".format(loss))
    print("TOTAL test accuracy  : {}".format(acc))
    print("TOTAL test precision : {}".format(pre))
    print("TOTAL test recall    : {}".format(rec))
    print("TOTAL test f1_score  : {}".format(f1))
    print("TOTAL test iou       : {}".format(iou))
    print("TOTAL test ap       : {}".format(ap))

    print("\nAverage test loss        : {}({})".format(np.round(np.mean(loss), 4), np.round(np.std(loss), 4)))
    print("Average test accuracy    : {}({})".format(np.round(np.mean(acc), 4), np.round(np.std(acc), 4)))
    print("Average test precision   : {}({})".format(np.round(np.mean(pre), 4), np.round(np.std(pre), 4)))
    print("Average test recall      : {}({})".format(np.round(np.mean(rec), 4), np.round(np.std(rec), 4)))
    print("Average test f1_score    : {}({})".format(np.round(np.mean(f1), 4), np.round(np.std(f1), 4)))
    print("Average test iou         : {}({})".format(np.round(np.mean(iou), 4), np.round(np.std(iou), 4)))
    print("Average test ap          : {}({})".format(np.round(np.mean(ap), 4), np.round(np.std(ap), 4)))
    print("###################### TOTAL TEST REPORT!!! ######################")

    f = open(save_path, 'w')

    f.write("###################### TOTAL TEST REPORT!!! ######################\n")
    f.write("TOTAL test loss      : {}\n".format(loss))
    f.write("TOTAL test accuracy  : {}\n".format(acc))
    f.write("TOTAL test precision : {}\n".format(pre))
    f.write("TOTAL test recall    : {}\n".format(rec))
    f.write("TOTAL test f1_score  : {}\n".format(f1))
    f.write("TOTAL test iou       : {}\n".format(iou))
    f.write("TOTAL test ap        : {}\n".format(ap))

    f.write("\nAverage test loss        : {}({})\n".format(np.round(np.mean(loss), 4), np.round(np.std(loss), 4)))
    f.write("Average test accuracy    : {}({})\n".format(np.round(np.mean(acc), 4), np.round(np.std(acc), 4)))
    f.write("Average test precision   : {}({})\n".format(np.round(np.mean(pre), 4), np.round(np.std(pre), 4)))
    f.write("Average test recall      : {}({})\n".format(np.round(np.mean(rec), 4), np.round(np.std(rec), 4)))
    f.write("Average test f1_score    : {}({})\n".format(np.round(np.mean(f1), 4), np.round(np.std(f1), 4)))
    f.write("Average test iou         : {}({})\n".format(np.round(np.mean(iou), 4), np.round(np.std(iou), 4)))
    f.write("Average test ap          : {}({})\n".format(np.round(np.mean(ap), 4), np.round(np.std(ap), 4)))
    f.write("###################### TOTAL TEST REPORT!!! ######################")

    f.close()