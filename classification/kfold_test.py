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


def validation_model(model, dataloders):
    # Each epoch has a training and validation phase
    for i in range(len(model)):
        model[i].train(False)  # Set model to evaluate mode
    label_gt = []

    preds_csv = open(rf'F:\renal_cyst\src\Detector\Classifier\preds\step{task}\{args.testdata}_preds.csv', 'w')
    preds_csv.write('Sample' + ',' + 'truth label' + ',' + 'pred label' + '\n')
    # Iterate over data.
    for j, (inputs, labels, ex_labels, file) in enumerate(dataloders):
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        outputs = 0
        # forward
        for i in range(len(model)):
            outputs += softmax(model[i](inputs).detach().cpu().numpy()[:, :3])
        outputs = outputs / len(model)
        preds = np.argmax(outputs, 1)

        for i in range(len(inputs)):
            preds_csv.write(file[i] + ',' + classes[labels[i]] + ',' + classes[preds[i]] + '\n')

        label_gt.extend(labels.numpy())
        if j == 0:
            output_all = outputs
        else:
            output_all = np.concatenate((output_all, outputs), axis=0)
    return output_all, label_gt


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
parser.add_argument('--num-class', type=int, default=3)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--gpus', type=str, default=1)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-epoch-freq', type=int, default=10)
parser.add_argument('--save-path', type=str, default=r"output\step2\multilabel_all")
parser.add_argument('--resume', type=str, default=r"None", help="For training from one checkpoint")
parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
parser.add_argument('--network', type=str, default="se_resnet_18", help="")
parser.add_argument('--task', type=str, default='1')
parser.add_argument('--testdata', type=str, default='NF 11.23')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# use gpu or not
use_gpu = torch.cuda.is_available()
print("use_gpu:{}".format(use_gpu))

task = args.task
print(f'task {task}')

model = dict()
dataloders, dataset_sizes = TestData(args)

for i in range(5):
    print(f'fold{i} start')
    # if i in [0, 1, 2]:
    args = parser.parse_args(
            ['--resume', rf'F:\renal_cyst\src\Detector\Classifier\output\step{task}\fold{i}\best.pth.tar'])
    # else:
    # args = parser.parse_args(
    #         ['--resume', rf'F:\renal_cyst\src\Detector\Classifier\output\step2\fold{i}\latest.pth.tar'])

    # get model
    script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])
    model[i] = timm.create_model('resnet18', pretrained=False, in_chans=1, num_classes=args.num_class+15)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {k.replace('module.',''): v for k, v in list(checkpoint.state_dict().items())}
            model[i].load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model[i] = model[i].cuda()
        torch.backends.cudnn.enabled = False

if task == '1':
    classes = ['I', 'II/IIF/III', 'IV']
elif task == '2':
    classes = ['II', 'IIF', 'III']
else:
    raise ValueError

out, label = validation_model(model=model, dataloders=dataloders)

conf_matrix = np.zeros((args.num_class, args.num_class))
prob_matrix = []
label_matrix = []
preds = np.argmax(out, 1)
conf_matrix = confusion_matrix(preds, labels=label, conf_matrix=conf_matrix)
prob_matrix, label_matrix = auc_matrixs(out, label, prob_matrix, label_matrix)
survey(conf_matrix, classes=classes)
plot_confusion_matrix(conf_matrix, savename=args.testdata, classes=classes,
                      normalize=False, title='Confusion_matrix')
plot_roc_curve(prob_matrix, label_matrix, classes=classes)