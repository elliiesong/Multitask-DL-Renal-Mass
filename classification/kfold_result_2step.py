import torch
import argparse
import os
import timm
from torch.autograd import Variable
from dataloader import KfoldData, TestData
import numpy as np
from matplotlib import pyplot as plt
import xlwt
import itertools


classes = ['I', 'II', 'IIF', 'III', 'IV']
# classes = ['I', 'II/IIF/III', 'IV']
# classes = ['II', 'IIF', 'III']


def validation_model(model, dataloders):
    # Each epoch has a training and validation phase
    model[0].train(False)  # Set model to evaluate mode
    model[1].train(False)
    label_gt = []

    # Iterate over data.
    for j, (inputs, labels, ex_labels, file) in enumerate(dataloders):
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        outputs = np.zeros((len(inputs), args.num_class))
        # forward
        outputs_step1 = softmax(model[0](inputs).detach().cpu().numpy()[:, :3])
        outputs_step2 = softmax(model[1](inputs).detach().cpu().numpy()[:, :3])
        # ex_label
        s = np.argmax(model[1](inputs).detach().cpu().numpy()[:, 3:6], 1)
        s_t = np.argmax(model[1](inputs).detach().cpu().numpy()[:, 6:10], 1)
        w_t = np.argmax(model[1](inputs).detach().cpu().numpy()[:, 10:14], 1)
        n = np.argmax(model[1](inputs).detach().cpu().numpy()[:, 14:16], 1)
        c = np.argmax(model[1](inputs).detach().cpu().numpy()[:, 16:18], 1)

        outputs[:, 0] = outputs_step1[:, 0]
        outputs[:, 4] = outputs_step2[:, 2]
        for i in range(3):
            outputs[:, i + 1] = outputs_step1[:, 1] * outputs_step2[:, i]
        preds = np.argmax(outputs_step1, 1)
        preds_step2 = np.argmax(outputs_step2, 1)
        preds[preds == 2] = 4
        for i in range(len(preds)):
            if preds[i] == 1:
                preds[i] = preds_step2[i] + 1

        for i in range(len(inputs)):
            preds_csv.write(file[i] + ',' + classes[labels[i]] + ',' + classes[preds[i]] + ',' + str(outputs[i, 0]) + ',' +
                            str(outputs[i, 1]) + ',' + str(outputs[i, 2]) + ',' + str(outputs[i, 3]) + ',' +str(outputs[i, 4]) + ','
                            + str(ex_labels[0][i].item()) + ' ' + str(ex_labels[1][i].item()) + ' ' + str(ex_labels[2][i].item())
                            + ' ' + str(ex_labels[3][i].item()) + ' ' + str(ex_labels[4][i].item())
                            + ',' + str(s[i]) + ' ' + str(s_t[i]) + ' ' + str(w_t[i]) + ' '
                            + str(n[i]) + ' ' + str(c[i]) + '\n')
        label_gt.extend(labels.numpy())
        if j == 0:
            output_all = outputs
            preds_all = preds
        else:
            output_all = np.concatenate((output_all, outputs), axis=0)
            preds_all = np.concatenate((preds_all, preds), axis=0)
    return output_all, label_gt, preds_all


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def auc_matrixs(prob, labels, prob_matrix, label_matrix):
    for p, t in zip(prob, labels):
        label_matrix.append(t)
        prob_matrix.append(p)
    return prob_matrix, label_matrix


def plot_confusion_matrix(cm, classes, savename, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    print('Confusion matrix, without normalization')
    print(cm)
    savepath = f'./confusion_matrix/{savename}'
    os.makedirs(savepath, exist_ok=True)
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
    plt.savefig(os.path.join(savepath, 'cm.png'), dpi=800)
    plt.close('all')


def plot_roc_curve(prob_matrix, label_matrix, classes):
    Workbook = xlwt.Workbook()
    worksheet = Workbook.add_sheet('Roc Data')
    for col, (l, p) in enumerate(zip(label_matrix, prob_matrix)):
        worksheet.write(col, 0, col + 1)
        worksheet.write(col, 1, classes[l])
        for i in range(len(classes)):
            worksheet.write(col, i + 2, str(p[i]))
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


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
parser.add_argument('--data-dir', type=str, default="/ImageNet")
parser.add_argument('--batch-size', type=int, default=18)
parser.add_argument('--num-class', type=int, default=5)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--gpus', type=str, default=1)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-epoch-freq', type=int, default=10)
parser.add_argument('--save-path', type=str, default=r"output\step2\multilabel_all")
parser.add_argument('--resume1', type=str, default=r"None", help="For training from one checkpoint")
parser.add_argument('--resume2', type=str, default=r"None", help="For training from one checkpoint")
parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
parser.add_argument('--network', type=str, default="se_resnet_18", help="")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# use gpu or not
use_gpu = torch.cuda.is_available()
print("use_gpu:{}".format(use_gpu))

preds_csv = open(r'F:\renal_cyst\src\Detector\Classifier\preds\preds.csv', 'w')
preds_csv.write('Sample' + ',' + 'truth label' + ',' + 'pred label' + ',' + 'prob I' +
                ',' + 'prob II' + ',' + 'prob IIF' + ',' + 'prob III' + ',' + 'prob IV' + ','
                + 'Auxiliary truth' + ',' + 'Auxiliary pred' + '\n')

# dataloders, dataset_sizes = TestData(args)
label_kfold = []
for i in range(5):
    print(f'fold{i} start')
    # if i in [0, 2]:
    args = parser.parse_args(
            ['--resume1', rf'F:\renal_cyst\src\Detector\Classifier\output\step1\fold{i}\best.pth.tar',
             '--resume2', rf'F:\renal_cyst\src\Detector\Classifier\output\step2\fold{i}\best.pth.tar'])
    # else:
    # args = parser.parse_args(
    #         ['--resume1', rf'F:\renal_cyst\src\Detector\Classifier\output\step1\fold{i}\latest.pth.tar',
    #          '--resume2', rf'F:\renal_cyst\src\Detector\Classifier\output\step2\fold{i}\latest.pth.tar'])
    # read data
    dataloders, dataset_sizes, w1, w2 = KfoldData(args, i)

    model = dict()
    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])
    model[0] = timm.create_model('resnet18', pretrained=False, in_chans=1, num_classes=3+15)
    model[1] = timm.create_model('resnet18', pretrained=False, in_chans=1, num_classes=3+15)

    if args.resume1:
        if os.path.isfile(args.resume1):
            print(("=> loading checkpoint '{}'".format(args.resume1)))
            checkpoint = torch.load(args.resume1)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            model[0].load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume1)))

    if args.resume2:
        if os.path.isfile(args.resume2):
            print(("=> loading checkpoint '{}'".format(args.resume2)))
            checkpoint = torch.load(args.resume2)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            model[1].load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume2)))

    if use_gpu:
        model[0] = model[0].cuda()
        model[1] = model[1].cuda()
        torch.backends.cudnn.enabled = False

    out, label, preds = validation_model(model=model, dataloders=dataloders['val'])

    if i == 0:
        out_kfold = out
        preds_kfold = preds
    else:
        out_kfold = np.concatenate((out_kfold, out), axis=0)
        preds_kfold = np.concatenate((preds_kfold, preds), axis=0)
    label_kfold.extend(label)

conf_matrix = np.zeros((args.num_class, args.num_class))
prob_matrix = []
label_matrix = []
conf_matrix = confusion_matrix(preds_kfold, labels=label_kfold, conf_matrix=conf_matrix)
prob_matrix, label_matrix = auc_matrixs(out_kfold, label_kfold, prob_matrix, label_matrix)
survey(conf_matrix, classes=classes)
plot_confusion_matrix(conf_matrix, savename='5fold', classes=classes,
                      normalize=False, title='Confusion_matrix')
plot_roc_curve(prob_matrix, label_matrix, classes=classes)