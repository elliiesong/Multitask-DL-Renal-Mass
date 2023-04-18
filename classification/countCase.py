import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools
import csv


# 5分类
labelDict = {'I': 0, 'II': 1, 'IIF': 2, 'III': 3, 'IV': 4}
labelDictRe = {0: 'I', 1: 'II', 2: 'IIF', 3: 'III', 4: 'IV'}
classes = ['I', 'II', 'IIF', 'III', 'IV']

# 3分类
# labelDict = {'I': 0, 'II/IIF/III': 1, 'IV': 2}
# labelDictRe = {0: 'I', 1: 'II/IIF/III', 2: 'IV'}
# classes = ['I', 'II/IIF/III', 'IV']


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


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


def plot_confusion_matrix(cm, classes, savename, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    print('Confusion matrix, without normalization')
    print(cm)
    savepath = f'./confusion_matrix/cases'
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
    plt.savefig(os.path.join(savepath, f'{savename}.png'), dpi=800)
    plt.close('all')


def getCases(root, dataset):
    csvPath = os.path.join(root, f'{dataset}.csv')
    df = pd.read_csv(csvPath)
    names = list(df['Sample'])
    IDs = []
    for name in names:
        id = name.split('_')[0]
        if id not in IDs:
            IDs.append(id)

    preds = []
    labels = []
    for id in IDs:
        print(id)
        maxPred = 0
        maxTruth = 0
        for _, slice in df.iterrows():
            name = slice['Sample']
            if id == name.split('_')[0]:
                maxPred = max(labelDict[slice['pred label']], maxPred)
                maxTruth = max(labelDict[slice['truth label']], maxTruth)
        preds.append(maxPred)
        labels.append(maxTruth)

    csvOutDir = os.path.join(root, 'case')
    os.makedirs(csvOutDir, exist_ok=True)
    f = open(rf'{csvOutDir}/{dataset}.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['case', 'truth', 'pred'])
    for i in range(len(IDs)):
        csv_writer.writerow([IDs[i], labelDictRe[labels[i]], labelDictRe[preds[i]]])
    f.close()
    return preds, labels


def plotCM(preds, labels, savename):
    class_num = len(classes)
    conf_matrix = np.zeros((class_num, class_num))
    conf_matrix = confusion_matrix(preds, labels=labels, conf_matrix=conf_matrix)
    survey(conf_matrix, classes=classes)
    plot_confusion_matrix(conf_matrix, savename=savename, classes=classes,
                          normalize=False, title='Confusion_matrix')


if __name__ == '__main__':
    root = r'F:\renal_cyst\src\Detector\Classifier\preds'
    # preds1, labels1 = getCases(root, 'HAINAN_preds')
    # preds2, labels2 = getCases(root, 'Testset2_preds')
    preds3, labels3 = getCases(root, 'Testset3_preds')
    preds = preds3
    labels = labels3
    plotCM(preds, labels, 'Testset3')
