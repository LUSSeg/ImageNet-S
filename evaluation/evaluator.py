import torch
import numpy as np
import argparse
from tqdm import tqdm
from metric import FMeasureGPU, IoUDifferentSizeGPU, IoUGPU
from tools import get_loader, get_param
from prettytable import PrettyTable
import os
import json


def evaluator(loader, num_classes):

    log = {}

    T = torch.zeros(size=(num_classes + 1,)).cuda()
    P = torch.zeros(size=(num_classes + 1,)).cuda()
    TP = torch.zeros(size=(num_classes + 1,)).cuda()
    IoU = torch.zeros(size=(num_classes + 1,)).cuda()
    FMeasure = 0.
    ACC = 0.

    # mIoU with different object sizes
    Ts = [torch.zeros(size=(num_classes + 1,)).cuda() for _ in range(4)]
    Ps = [torch.zeros(size=(num_classes + 1,)).cuda() for _ in range(4)]
    TPs = [torch.zeros(size=(num_classes + 1,)).cuda() for _ in range(4)]
    mIoUs = [torch.zeros(size=(num_classes + 1,)).cuda() for _ in range(4)]

    for gt, predict in tqdm(loader):
        gt = gt.cuda()
        predict = predict.cuda()

        area_intersection, area_output, area_target = IoUGPU(predict.view(-1), gt.view(-1), num_classes + 1)
        IoUDifferentSizeGPU(predict.view(-1), gt.view(-1), num_classes + 1, Ts, Ps, TPs)
        f_score = FMeasureGPU(predict, gt)

        T += area_output
        P += area_target
        TP += area_intersection
        FMeasure += f_score

        # image-level accuracy
        img_label = torch.argmax(area_output[1:]) + 1
        ACC += (area_target[img_label] > 0) * (area_output[img_label] > 0)

    IoU = TP / (T + P - TP + 1e-10) * 100
    FMeasure = FMeasure / len(loader.dataset)
    ACC = ACC / len(loader.dataset)

    mIoU = torch.mean(IoU).item()
    FMeasure = FMeasure.item() * 100
    ACC = ACC.item() * 100

    for i in range(4):
        mIoUs[i] = torch.mean(TPs[i] / (Ts[i] + Ps[i] - TPs[i] + 1e-10)).item() * 100

    log['Acc'] = ACC
    log['mIoU'] = mIoU
    log['S'] = mIoUs[0]
    log['MS'] = mIoUs[1]
    log['ML'] = mIoUs[2]
    log['L'] = mIoUs[3]
    log['IoUs'] = IoU.tolist()
    log['FMeasure'] = FMeasure

    return log


def display(log):
    class_table_data = PrettyTable()

    IoUs = [np.round(value, 2) for value in log['IoUs']]
    class_table_data.add_column('classes', log['classes'])
    class_table_data.add_column('IoUs', IoUs)

    summary_table_data = PrettyTable()
    for key, val in log.items():
        if key == 'IoUs' or key == 'classes':
            continue
        summary_table_data.add_column(key, [np.round(val, 2)])

    print('per class IoUs:\n')
    print(class_table_data.get_string())
    print('Summary:\n')
    print(summary_table_data.get_string())


def evaludation(args):

    params = get_param(args.mode)

    val_loader = get_loader(args.predict_dir, args.gt_dir,
                            name_list=os.path.join('../data', 'names', params['names']),
                            workers=args.workers,
                            match=args.match)

    log = evaluator(val_loader, num_classes=params['num_classes'])

    # display
    classes = getattr(val_loader.dataset, params['classes'])
    log['classes'] = classes
    display(log)

    return log


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-dir", default='imagenet50', type=str)
    parser.add_argument("--gt-dir", default='imagenet50', type=str)
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--match', default=None, type=str)
    parser.add_argument('--mode', type=str, default='50', choices=['50', '300', '919'])
    args = parser.parse_args()

    evaludation(args)
