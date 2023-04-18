import os
import pandas as pd
import numpy as np
from collections import Counter

class_dict = {'I': 0, 'II': 1, 'IIF': 2, 'III': 3, 'IV': 4}
class_redict = {0: 'I', 1: 'II', 2: 'IIF', 3: 'III', 4: 'IV'}

path = 'F:\\renal_cyst\\dataset'
train_path = os.path.join(path, 'train_labels.csv')
val_path = os.path.join(path, 'val_labels.csv')
# test_path = 'F:\\renal_cyst\\testdata\\test_labels.csv'
multi1 = r'F:\renal_cyst\multilabel.csv'
multi_df1 = pd.read_csv(multi1)
multi2 = r'F:\NF 7.20\补充数据辅助标签.csv'
multi_df2 = pd.read_csv(multi2)

multi_df = pd.concat([multi_df1, multi_df2])

train_df = pd.read_csv(train_path, header=None, names=['file', 'x_0', 'y_0', 'x_1', 'y_1', 'label'])
val_df = pd.read_csv(val_path, header=None, names=['file', 'x_0', 'y_0', 'x_1', 'y_1', 'label'])
# test_df = pd.read_csv(test_path, header=None, names=['file', 'x_0', 'y_0', 'x_1', 'y_1', 'label'])
# test_df = test_df.drop_duplicates(subset=['file'])
t = np.zeros(5)
for fname in train_df['file']:
    info = fname.split('_')
    if info[2] == 'MUMC':
        t[class_dict[info[1]]] += 1

print('MUMC:', t)
print('traindata:', Counter(train_df['label']))
print('valdata:', Counter(val_df['label']))
# print('testdata:', Counter(test_df['label']))

print('septa:', Counter(multi_df['septa']))
print('septa tn:', Counter(multi_df['septa thickness']))
print('wall tn:', Counter(multi_df['wall thickness']))
print('nodule:', Counter(multi_df['nodule']))
print('calcification:', Counter(multi_df['calcification']))
