import torch
import torch.nn as nn
import torch.nn.functional as F


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


class loss_fn(nn.Module):
    def __init__(self, label_weight):
        super(loss_fn, self).__init__()
        self.label_weight = label_weight
        self.loss_focal = FocalLoss(alpha=label_weight)

    def get_ex_loss(self, y_pred, y_true, weight):
        if type(weight) is not 'Tensor':
            weight = torch.Tensor(weight)
        ls = nn.CrossEntropyLoss(weight=weight.cuda())
        loss = ls(y_pred, y_true)
        return loss

    def get_label_loss(self, y_pred, y_true):
        loss = self.loss_focal(y_pred, y_true)
        return loss

    def get_total_loss(self, loss_array, weight):
        return loss_array * weight