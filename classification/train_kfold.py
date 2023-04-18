from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
from dataloader import KfoldData
import se_resnet
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics
import xlwt
from itertools import cycle
from sklearn.metrics import precision_recall_fscore_support
import timm
from loss_fn import loss_fn

# classes = ['I', 'II', 'IIF', 'III', 'IV']
reclasses_dict = {0: 'I', 1: 'II', 2: 'IIF', 3: 'III', 4: 'IV'}


# class weightce(nn.Module):
#     def __init__(self, alpha=1, num_classes=5):
#         super(weightce, self).__init__()
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.ce = F.cross_entropy()
#
#     def forward(self, inputs, labels):
#         alpha = self.alpha.cuda()
#         preds_softmax = F.softmax(inputs, dim=1)
#         preds_logsoft = torch.log(preds_softmax)
#         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
#         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
#         alpha = alpha.gather(0, labels.view(-1))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, (float, int)):
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, labels):
        alpha = self.alpha.cuda()
        preds_softmax = F.softmax(inputs, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = - (torch.pow((1 - preds_softmax), self.gamma) * preds_logsoft)
        loss = alpha * loss.t()
        loss = loss.mean()
        return loss


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


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
            worksheet.write(col, i + 2, str(p[i]))
    Workbook.save('Data for ROC.xls')


# def plot_precision(precision):
#     lw = 2
#     colors = ['aqua', 'darkorange', 'cornflowerblue', 'gold', 'lime']
#     for i in range(len(precision[0])):
#         plt.figure()
#         true_class = reclass_dict[i]
#         plt.plot(precision[:, i], color=colors[i], lw=lw, label='class {0}'.format(true_class))
#         plt.ylim([0.0, 1.0])
#         plt.xlabel('epochs')
#         plt.ylabel('Precison')
#         plt.title('Class Precision')
#         plt.legend(loc='lower right')
#         plt.savefig(f'./precision_{true_class}.png')
#         plt.close()


def survey(cm, classes):
    # print(cm)
    Precisions = np.zeros(cm.shape[1])
    for true_class in range(cm.shape[1]):
        # print(classes[true_class])
        total_label = cm.sum(axis=1)[true_class]
        total_pred = cm.sum(axis=0)[true_class]
        TP = cm[true_class][true_class]
        FN = total_label - TP
        FP = total_pred - TP
        TN = cm.sum() - total_label - FP
        Precision = TP / (TP + FP + 1e-6)
        Sensitivity = TP / (TP + FN + 1e-6)
        Precisions[true_class] = Precision
        # print('Precision:', Precision)
        # print('Sensitivity:', Sensitivity)
        # print('Specificity', TN/(TN+FP+1e-6))
        # print('F1-score:', 2 * Precision * Sensitivity / (Precision + Sensitivity + 1e-6))
    return Precisions


def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False
    loss_train = []
    loss_val = []
    best_model_wts = model.state_dict()
    best_accuracy = 0
    Precisions = []
    for epoch in range(args.start_epoch + 1, num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            conf_matrix = torch.zeros(args.num_class, args.num_class)
            prob_matrix = []
            label_matrix = []
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch + 1)
                    resumed = True
                else:
                    scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_l_loss = 0.0
            running_s_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels, ex_labels, file) in enumerate(dataloders[phase]):
                [septa, septa_tn, wall_tn, nodule, calcification] = ex_labels
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    septa = Variable(septa.cuda())
                    septa_tn = Variable(septa_tn.cuda())
                    wall_tn = Variable(wall_tn.cuda())
                    nodule = Variable(nodule.cuda())
                    calcification = Variable(calcification.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    septa, septa_tn, wall_tn, nodule, calcification = \
                        Variable(septa), Variable(septa_tn), Variable(wall_tn), Variable(nodule), Variable(
                            calcification)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                label_loss = criterion.get_label_loss(outputs[:, :3], labels)
                # septa_loss = criterion.get_ex_loss(outputs[:, 3:6], septa,
                #                                    weight=[1.596, 4.522, 6.565])  #[2.426, 3.945, 2.992]
                # septa_tn_loss = criterion.get_ex_loss(outputs[:, 6:10], septa_tn,
                #                                       weight=[1.596, 6.565, 6.783, 13.567])  #[3.663, 6.527, 4.174, 2.992]
                # wall_tn_loss = criterion.get_ex_loss(outputs[:, 10:14], wall_tn,
                #                                      weight=[1.26, 27.133, 8.66, 18.5])  #[6.904, 7.18, 13.296, 1.561]
                # nodule_loss = criterion.get_ex_loss(outputs[:, 14:16], nodule,
                #                                     weight=[1.015, 67.833])  #[1.023, 44.875]
                # calcification_loss = criterion.get_ex_loss(outputs[:, 16:18], calcification,
                #                                            weight=[1.762, 2.313])

                # loss = label_loss + septa_loss + septa_tn_loss + wall_tn_loss + nodule_loss + calcification_loss
                loss = label_loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs[:, :3].data, 1)

                # statistics
                running_loss += loss.data.item()
                # running_l_loss += label_loss.data.item()
                # running_s_loss += septa_loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i + 1) * args.batch_size)
                batch_acc = running_corrects.cpu().numpy() / ((i + 1) * args.batch_size)

                if phase == 'train' and i % args.print_freq == 0:
                    print(
                        '[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                            epoch, num_epochs - 1, i, round(dataset_sizes[phase] / args.batch_size) - 1,
                            scheduler.get_lr()[0], phase, batch_loss, batch_acc, \
                                   args.print_freq / (time.time() - tic_batch)))
                    tic_batch = time.time()
                    # conf_matrix_train = confusion_matrix(preds, labels=labels.data, conf_matrix=conf_matrix)

                if phase == 'val':
                    conf_matrix = confusion_matrix(preds, labels=labels.data, conf_matrix=conf_matrix)
                    prob_matrix, label_matrix = auc_matrixs(torch.softmax(outputs[:, :3].data, dim=1), labels.data,
                                                            prob_matrix, label_matrix)

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_l_loss = running_l_loss / dataset_sizes[phase]
            # epoch_s_loss = running_s_loss / dataset_sizes[phase]
            if phase == 'train':
                loss_train.append(epoch_loss)
            elif phase == 'val':
                loss_val.append(epoch_loss)
            # if phase == 'train':
            #     loss_list.append(epoch_loss)
            #     Precision = survey(conf_matrix_train.numpy(), classes=['I', 'II', 'IIF', 'III', 'IV'])
            #     Precisions.append(Precision)
            epoch_acc = running_corrects.cpu().numpy() / dataset_sizes[phase]
            if phase == 'val':
                Precisions = survey(conf_matrix.numpy(), classes=classes)
                mean_acc = Precisions.mean()
                # mean_acc = epoch_acc
                print(f'val mean_acc: {mean_acc}')
                if epoch > 30:
                    if mean_acc > 0.5:
                        os.makedirs(args.save_path, exist_ok=True)
                        if mean_acc <= best_accuracy:
                            latest_pth = os.path.join(args.save_path, 'latest.pth.tar')
                            torch.save(model, latest_pth)
                        else:
                            best_accuracy = mean_acc
                            best_pth = os.path.join(args.save_path, "best.pth.tar")
                            torch.save(model, best_pth)
                            # plot_confusion_matrix(conf_matrix.numpy(), accuracy = epoch_acc, classes=['I', 'II', 'IIF', 'III', 'IV'], normalize=True, title='Normalized confusion matrix')
                            plot_confusion_matrix(conf_matrix.numpy(), accuracy=epoch_acc, classes=classes,
                                                  normalize=False, title='Confusion_matrix')
                            plot_roc_curve(prob_matrix, label_matrix, classes=classes)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    # Precisions = np.array(Precisions)
    # plot_precision(Precisions)
    plt.figure()
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.savefig('./loss.png')
    plt.close()
    # plt.figure()
    # plt.plot(l_loss_list)
    # plt.savefig('./l_loss.png')
    # plt.close()
    # plt.figure()
    # plt.plot(s_loss_list)
    # plt.savefig('./s_loss.png')
    # plt.close()

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/ImageNet")
    parser.add_argument('--batch-size', type=int, default=18)
    parser.add_argument('--num-class', type=int, default=3)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=10)
    parser.add_argument('--save-path', type=str, default=r"output\step2\multilabel_all")
    parser.add_argument('--resume', type=str, default=r"None", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    parser.add_argument('--network', type=str, default="se_resnet_18", help="")
    parser.add_argument('--task', type=str, default='1')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    for i in range(5):
        print(f'fold{i} start')
        args = parser.parse_args(['--save-path', rf'output\step2\fold{i}'])

        if args.task == '1':
            classes = ['I', 'II/IIF/III', 'IV']
        elif args.task == '2':
            classes = ['II', 'IIF', 'III']

        # read data
        dataloders, dataset_sizes, w1, w2 = KfoldData(args, i)
        # get model
        script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])

        # if script_name == "se_resnet":
        #     model = getattr(se_resnet, args.network)(num_classes=args.num_class)
        # else:
        #     raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")

        model = timm.create_model('resnet18', pretrained=True, in_chans=1, num_classes=args.num_class + 15)

        if args.resume:
            if os.path.isfile(args.resume):
                print(("=> loading checkpoint '{}'".format(args.resume)))
                checkpoint = torch.load(args.resume)
                base_dict = {k.replace('module.', ''): v for k, v in list(checkpoint.state_dict().items())}
                model.load_state_dict(base_dict)
            else:
                print(("=> no checkpoint found at '{}'".format(args.resume)))

        if use_gpu:
            model = model.cuda()
            torch.backends.cudnn.enabled = False
            # model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

        # define loss function
        weight = w1 if args.task == '1' else w2
        criterion = loss_fn(label_weight=weight)  # [2.821, 3.437, 2.821]
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001)
        # Decay LR by a factor of 0.1 every 7 epochs
        # scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.98)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.num_epochs, eta_min=0)
        model = train_model(args=args,
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer_ft,
                            scheduler=scheduler,
                            num_epochs=args.num_epochs,
                            dataset_sizes=dataset_sizes)