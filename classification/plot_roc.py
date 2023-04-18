import xlrd
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from itertools import cycle
from bootstrap import bootstrap

class_dict = {'I': 0, 'II': 1, 'IIF': 2, 'III': 3, 'IV': 4}
# class_dict = {'I': 0, 'II/IIF/III': 1, 'IV': 2}
# class_dict = {'II': 0, 'IIF': 1, 'III': 2}
reclass_dict = {v: k for k, v in class_dict.items()}

n_classes = len(class_dict)
data = xlrd.open_workbook('Data for ROC.xls')
table = data.sheets()[0]
rows = table.nrows
cols = table.ncols
L = np.zeros((rows, cols - 2))
prob = np.zeros((rows, cols - 2))
for row in range(rows):
    info = table.row_values(row)
    label = class_dict[info[1]]
    prob[row] = info[2:cols + 1]
    L[row][label] = 1

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(L[:, i], prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    interval = bootstrap(L[:, i], prob[:, i], B=1000, func1=roc_curve, func2=auc)
    print(f'{reclass_dict[i]}, interval: {interval}')

fpr['micro'], tpr['micro'], _ = roc_curve(L.ravel(), prob.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += zoom(tpr[i], len(all_fpr) / len(tpr[i]))
    if zoom(tpr[i], len(all_fpr) / len(tpr[i]))[-1] == 0:
        mean_tpr[-1] += 1
mean_tpr /= n_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

lw = 2
plt.figure()
plt.plot(fpr['micro'], tpr['micro'],
         label='micro-average (area = {0:0.3f})'.format(roc_auc['micro']),
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr['macro'], tpr['macro'],
         label='macro-average (area = {0:0.3f})'.format(roc_auc['macro']),
         color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'gold', 'lime'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='class {0} (area = {1:0.3f})'.format(reclass_dict[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()
