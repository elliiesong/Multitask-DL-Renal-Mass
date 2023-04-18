from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from dataloader import PatientData, TestData, KfoldData
import se_resnet
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics
import xlwt
import timm

# class_dict = {0: 'I', 1: 'II', 2: 'IIF', 3: 'III', 4: 'IV'}
class_dict = {0: 'II', 1: 'IIF', 2: 'III'}

# classes = ['I', 'II', 'IIF', 'III', 'IV']
# classes = ['I', 'II/IIF/III', 'IV']
classes = ['II', 'IIF', 'III']


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
    return prob_matrix, label_matrix


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
        worksheet.write(col, 0, col + 1)
        worksheet.write(col, 1, classes[l])
        for i in range(len(classes)):
            worksheet.write(col, i+2, str(p[i]))
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


def validation_model(args, model, dataloders, dataset_sizes, visualization=False):
    # Each epoch has a training and validation phase
    conf_matrix = torch.zeros(args.num_class, args.num_class)
    prob_matrix = []
    label_matrix = []
    model.train(False)  # Set model to evaluate mode
    running_corrects = 0
    ex_running_corrects = {
        'septa': 0,
        'septa_tn': 0,
        'wall_tn': 0,
        'nodule': 0,
        'calcification': 0
    }
    # Iterate over data.

    for i, (inputs, labels, ex_labels, file) in enumerate(dataloders):
        images = inputs * 255
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        ex_preds = dict()
        # forward
        # outputs = model(inputs)
        # _, preds = torch.max(outputs.data, 1)
        outputs = model(inputs)
        _, preds = torch.max(outputs[:, :3].data, 1)
        _, ex_preds['septa'] = torch.max(outputs[:, 3:6].data, 1)
        _, ex_preds['septa_tn'] = torch.max(outputs[:, 6:10].data, 1)
        _, ex_preds['wall_tn'] = torch.max(outputs[:, 10:14].data, 1)
        _, ex_preds['nodule'] = torch.max(outputs[:, 14:16].data, 1)
        _, ex_preds['calcification'] = torch.max(outputs[:, 16:].data, 1)

        if visualization:
            save_path = './results/foreign/'
            os.makedirs(save_path, exist_ok=True)
            for i in range(len(images)):
                plt.figure()
                plt.imshow(images[i].squeeze())
                save_name = save_path + file[i] + '_' +  class_dict[preds[i].cpu().numpy().tolist()] + '.png'
                plt.savefig(save_name)
                plt.close('all')

        # statistics
        running_corrects += torch.sum(preds == labels.data)

        ex_running_corrects['septa'] += torch.sum(ex_preds['septa'].cpu() == ex_labels[0].data)
        ex_running_corrects['septa_tn'] += torch.sum(ex_preds['septa_tn'].cpu() == ex_labels[1].data)
        ex_running_corrects['wall_tn'] += torch.sum(ex_preds['wall_tn'].cpu() == ex_labels[2].data)
        ex_running_corrects['nodule'] += torch.sum(ex_preds['nodule'].cpu() == ex_labels[3].data)
        ex_running_corrects['calcification'] += torch.sum(ex_preds['calcification'].cpu() == ex_labels[4].data)

        conf_matrix = confusion_matrix(preds, labels=labels.data, conf_matrix=conf_matrix)
        prob_matrix, label_matrix = auc_matrixs(torch.softmax(outputs[:, :3].data, dim=1), labels.data, prob_matrix,
                                                label_matrix)

    acc = running_corrects.cpu().numpy() / dataset_sizes
    ex_acc = dict()
    ex_acc['septa'] = ex_running_corrects['septa'].numpy() / dataset_sizes
    ex_acc['septa_tn'] = ex_running_corrects['septa_tn'].numpy() / dataset_sizes
    ex_acc['wall_tn'] = ex_running_corrects['wall_tn'].numpy() / dataset_sizes
    ex_acc['nodule'] = ex_running_corrects['nodule'].numpy() / dataset_sizes
    ex_acc['calcification'] = ex_running_corrects['calcification'].numpy() / dataset_sizes

    survey(conf_matrix.numpy(), classes=classes)
    plot_confusion_matrix(conf_matrix.numpy(), accuracy=acc, classes=classes,
                          normalize=True, title='Normalized confusion matrix')
    plot_confusion_matrix(conf_matrix.numpy(), accuracy=acc, classes=classes,
                          normalize=False, title='Confusion_matrix')
    plot_roc_curve(prob_matrix, label_matrix, classes=classes)
    print('Acc: {:.4f}'.format(acc))
    print('septa acc: {} septa_tn acc: {} wall_tn acc: {} nodule acc: {} calcification acc: {}'
          .format(ex_acc['septa'], ex_acc['septa_tn'], ex_acc['wall_tn'], ex_acc['nodule'], ex_acc['calcification']))

# def patient_validation_model(args, model, patient_list, dataloders, dataset_sizes):
#     since = time.time()
#
#     # Each epoch has a training and validation phase
#     model.train(False)  # Set model to evaluate mode
#
#     running_corrects = 0
#
#     conf_matrix = torch.zeros(5, 5)
#     prob_matrix = []
#     label_matrix = []
#     for patient in patient_list:
#         print('Patient: ', patient)
#         dataloder = dataloders[patient]
#         pred_list = []
#         label_list = []
#         for i, (inputs, labels) in enumerate(dataloder):
#             if use_gpu:
#                 inputs = Variable(inputs.cuda())
#                 labels = Variable(labels.cuda())
#             else:
#                 inputs, labels = Variable(inputs), Variable(labels)
#
#             # forward
#             outputs = model(inputs)
#             _, preds = torch.max(outputs.data, 1)
#             pred_list.extend(preds)
#             label_list.extend(labels)
#         # pred = max(pred_list, key=pred_list.count)
#         pred = max(pred_list)
#         label = max(label_list)
#         conf_matrix = patient_confusion_matrix(pred, label, conf_matrix=conf_matrix)
#         # prob_matrix, label_matrix = auc_matrixs(torch.softmax(outputs.data, dim=1), labels.data, prob_matrix, label_matrix)
#         if pred == label:
#             running_corrects += 1
#
#     acc = running_corrects / len(patient_list)
#     time_elapsed = time.time() - since
#     survey(conf_matrix.numpy(), classes=classes)
#     plot_confusion_matrix(conf_matrix.numpy(), accuracy=acc, classes=classes,
#                           normalize=True, title='Normalized Patient confusion matrix')
#     plot_confusion_matrix(conf_matrix.numpy(), accuracy=acc, classes=classes,
#                           normalize=False, title='Patient confusion matrix')
#     plot_roc_curve(prob_matrix, label_matrix, classes=classes)
#     print('Validation complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=18)
    parser.add_argument('--num-class', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='slice')
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output2")
    parser.add_argument('--resume', type=str, default="F:\\renal_cyst\\src\\Detector\\Classifier\\output\\step2\\fold2\\latest.pth.tar", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="se_resnet_18", help="")
    args = parser.parse_args()

    # read data
    if args.mode == 'patient':
        patient_list, dataloders, dataset_sizes = PatientData(args)
    elif args.mode == 'slice':
        dataloders, dataset_sizes, w1, w2 = KfoldData(args, fold=2)
    else:
        raise Exception("mode Error")
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])
    model = timm.create_model('resnet18', pretrained=False, in_chans=1, num_classes=args.num_class+15)

    # if script_name == "se_resnet":
    #     model = getattr(se_resnet, args.network)(num_classes=args.num_class)
    # else:
    #     raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        torch.backends.cudnn.enabled = False
        #model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    if args.mode == 'patient':
        model = patient_validation_model(args=args, model=model, patient_list=patient_list, dataloders=dataloders,
                                         dataset_sizes=dataset_sizes)
    elif args.mode == 'slice':
        model = validation_model(args=args, model=model, dataloders=dataloders['val'], dataset_sizes=dataset_sizes['val'])
    else:
        raise Exception("mode Error")