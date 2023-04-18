from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from dataloader import PatientData, TestData
import se_resnet
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics
import xlwt
import timm

class_dict = {0: 'I', 1: 'II', 2: 'IIF', 3: 'III', 4: 'IV'}


def patient_confusion_matrix(pred, label, conf_matrix):
    conf_matrix[pred, label] += 1
    return conf_matrix


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def auc_matrixs(prob, labels, prob_matrix, label_matrix):
    for p, t in zip(prob, labels.cpu().numpy()):
        label_matrix.append(t)
        prob_matrix.append(p.cpu().numpy())
    return prob_matrix,label_matrix


def plot_confusion_matrix(cm, classes, accuracy, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=0))
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        if cm[i, j] > thresh:
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white")
        else:
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    if normalize:
        plt.savefig('./confusion_matrix/normalized_{:.2f}'.format(accuracy) + '.png', dpi=800)
    else:
        plt.savefig('./confusion_matrix/{:.2f}'.format(accuracy) + '.png', dpi=800)
    plt.close('all')

def plot_roc_curve(prob_matrix, label_matrix, classes):
    Workbook = xlwt.Workbook()
    worksheet = Workbook.add_sheet('Roc Data')
    for col, (l, p) in enumerate(zip(label_matrix, prob_matrix)):
        worksheet.write(col,0,col + 1)
        worksheet.write(col,1,classes[l])
        worksheet.write(col,2,str(p[0]))
        worksheet.write(col,3,str(p[1]))
        worksheet.write(col,4,str(p[2]))
        worksheet.write(col,5,str(p[3]))
        worksheet.write(col,6,str(p[4]))
    Workbook.save('Data for ROC.xls')

def survey(cm, classes):
    print(cm)
    for true_class in range(cm.shape[1]):
        print(classes[true_class])
        total_label = cm.sum(axis = 1)[true_class]
        total_pred = cm.sum(axis = 0)[true_class]
        TP = cm[true_class][true_class]
        FN = total_label - TP
        FP = total_pred - TP
        TN = cm.sum() - total_label - FP
        Precision = TP / (TP + FP + 1e-6)
        Sensitivity = TP/(TP+FN+1e-6)
        print('Precision:', Precision)
        print('Sensitivity:', Sensitivity)
        print('Specificity', TN/(TN+FP+1e-6))
        print('F1-score:', 2 * Precision * Sensitivity / (Precision + Sensitivity + 1e-6))


def validation_model(args, model1, model2, dataloders, dataset_sizes, visualization=True):
    # Each epoch has a training and validation phase
    conf_matrix = torch.zeros(5, 5)
    prob_matrix = []
    label_matrix = []
    model1.train(False)  # Set model to evaluate mode
    model2.train(False)
    running_corrects = 0
    # Iterate over data.
    for i, (inputs, labels, ex_label, file) in enumerate(dataloders):
        outputs = torch.zeros((len(inputs), args.num_class))
        images = inputs * 255
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = Variable(outputs.cuda())
        else:
            inputs, labels, outputs = Variable(inputs), Variable(labels), Variable(outputs)

        # step1
        outputs_step1 = model1(inputs)[:, :3]
        _, preds = torch.max(outputs_step1.data, 1)
        outputs_step1 = torch.softmax(outputs_step1, dim=1)
        outputs[:, 0] = outputs_step1[:, 0]
        outputs[:, 4] = outputs_step1[:, 2]
        preds[preds == 2] = 4

        # step2
        outputs_step2 = model2(inputs)
        outputs_step2 = outputs_step2[:, :3]
        _, preds_step2 = torch.max(outputs_step2.data, 1)
        outputs_step2 = torch.softmax(outputs_step2, dim=1)
        for i in range(3):
            outputs[:, i+1] = outputs_step1[:, 1] * outputs_step2[:, i]
        for i in range(len(preds)):
            if preds[i] == 1:
                preds[i] = preds_step2[i] + 1

        if visualization:
            save_path = './results/gt3/'
            os.makedirs(save_path, exist_ok=True)
            for i in range(len(images)):
                plt.figure()
                plt.imshow(images[i].squeeze())
                save_name = save_path + file[i] + '_' + class_dict[preds[i].cpu().numpy().tolist()] + '.png'
                plt.savefig(save_name)
                plt.close('all')

        # statistics
        running_corrects += torch.sum(preds == labels.data)

        conf_matrix = confusion_matrix(preds, labels=labels.data, conf_matrix=conf_matrix)
        prob_matrix, label_matrix = auc_matrixs(outputs.data, labels.data, prob_matrix,
                                                label_matrix)

    acc = running_corrects.cpu().numpy() / dataset_sizes
    survey(conf_matrix.numpy(), classes=['I', 'II', 'IIF', 'III', 'IV'])
    plot_confusion_matrix(conf_matrix.numpy(), accuracy=acc, classes=['I', 'II', 'IIF', 'III', 'IV'],
                          normalize=True, title='Normalized confusion matrix')
    plot_confusion_matrix(conf_matrix.numpy(), accuracy=acc, classes=['I', 'II', 'IIF', 'III', 'IV'],
                          normalize=False, title='Confusion_matrix')
    plot_roc_curve(prob_matrix, label_matrix, classes=['I', 'II', 'IIF', 'III', 'IV'])
    print('Acc: {:.4f}'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=18)
    parser.add_argument('--num-class', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output2")
    parser.add_argument('--resume1', type=str, default="F:\\renal_cyst\\src\\Detector\\Classifier\\output\\step1\\multilabel3\\latest.pth.tar", help="For training from one checkpoint")
    parser.add_argument('--resume2', type=str, default="F:\\renal_cyst\\src\\Detector\\Classifier\\output\\step2\\multilabel3\\best.pth.tar", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network1', type=str, default="se_resnet_18", help="")
    parser.add_argument('--network2', type=str, default="se_resnet_18", help="")
    args = parser.parse_args()

    # read data
    dataloders, dataset_sizes = TestData(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    script_name1 = '_'.join([args.network1.strip().split('_')[0], args.network1.strip().split('_')[1]])
    script_name2 = '_'.join([args.network2.strip().split('_')[0], args.network2.strip().split('_')[1]])

    model1 = timm.create_model('resnet18', pretrained=False, in_chans=1, num_classes=3+15)
    model2 = timm.create_model('resnet18', pretrained=False, in_chans=1, num_classes=3+15)
    # if script_name1 == "se_resnet":
    #     model1 = getattr(se_resnet, args.network1)(num_classes = 4)
    # else:
    #     raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")
    #
    # if script_name2 == "se_resnet":
    #     model2 = getattr(se_resnet, args.network2)(num_classes = 2)
    # else:
    #     raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")

    if args.resume1:
        if os.path.isfile(args.resume1):
            print(("=> loading checkpoint '{}'".format(args.resume1)))
            checkpoint1 = torch.load(args.resume1)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint1.state_dict().items())}
            model1.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume1)))

    if args.resume2:
        if os.path.isfile(args.resume2):
            print(("=> loading checkpoint '{}'".format(args.resume2)))
            checkpoint2 = torch.load(args.resume2)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint2.state_dict().items())}
            model2.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume2)))

    if use_gpu:
        model1 = model1.cuda()
        model2 = model2.cuda()
        torch.backends.cudnn.enabled = False
        #model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    model = validation_model(args=args, model1=model1, model2=model2, dataloders=dataloders, dataset_sizes=dataset_sizes)