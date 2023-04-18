import os
import pandas as pd
import random
from collections import Counter


def getnpy(dir):
    files = []
    for file in os.listdir(dir):
        if file.endswith('.npy'):
            files.append(file)
    return files


def getinfo(files):
    info = []
    for file in files:
        info.append(file.split('.')[0].split('_'))
    return pd.DataFrame(info, columns=['case', 'label', 'slice'])


def getfold_bydict(fold=5):
    folds = {
        '0': ['100', '103', '104', '105', '106', '109', '10', '110',
       '113', '114', '115', '117', '118', '119', '11', '122',
       '123', '125', '128', '129', '12', '130', '131', '132', '134',
       '137', '138', '139', '141', '142', '143', '144', '145', '146',
       '148', '149', '150', '152', '153', '155', '156', '157', '158',
       '159', '160', '161', '162', '163', '164', '165', '166', '167',
       '168', '169', '170', '171', '173', '175', '176', '177', '178',
       '179', '261', '264', '290', '309', '332', '338', '339'],
        '1': ['17', '180', '182', '183', '185', '186', '187', '188', '189',
       '190', '192', '193', '195', '197', '199', '1', '200', '201',
       '202', '203', '204', '205', '206', '208', '20', '210', '211',
       '212', '215', '217', '218', '21', '220', '221', '222', '224',
       '225', '226', '227', '228', '229', '22', '231', '232', '233',
       '234', '235', '236', '237', '238', '23', '240', '241', '243',
       '244', '245', '247', '24', '250', '251', '252', '253', '254', '322', '325'],
        '2': ['194', '255', '256', '257', '25', '260', '262', '263',
       '265', '266', '267', '268', '269', '26', '270', '271', '272',
       '273', '274', '275', '276', '278', '279', '281', '282', '283',
       '286', '287', '288', '289', '291', '292', '294', '296',
       '297', '298', '299', '2', '300', '302', '303', '304', '305', '306',
       '307', '308', '30', '310', '311', '312', '313', '314',
       '315', '316', '318', '31', '320', '321', '323', '324', '70', '74'],
        '3': ['107', '112', '326', '327', '328', '329', '331', '333', '334', '335',
       '336', '337', '33', '340', '341', '342', '343',
       '345', '346', '347', '348', '349', '351', '353', '355', '356',
       '358', '359', '35', '361', '363', '364', '365', '367', '369', '36',
       '370', '371', '372', '373', '374', '375', '376', '377', '378',
       '379', '37', '380', '381', '382', '384', '386', '387', '388',
       '389', '390', '391', '392', '393', '394', '395', '396', '397'],
        '4': ['398', '399', '39', '400', '401', '402', '403', '404', '405',
       '406', '408', '409', '410', '411', '412', '416', '417', '418',
       '419', '41', '421', '422', '42', '45', '46', '48', '50', '51',
       '52', '53', '54', '55', '56', '58', '59', '5', '61', '62', '63',
       '64', '65', '68', '69', '6', '72', '73', '75', '76',
       '78', '79', '7', '81', '82', '83', '84', '86', '90', '91', '92',
       '95', '97', '99', '9'],
    }
    new_list = []
    for i in range(fold):
        new_list.append(folds[f'{i}'])
    return new_list


def getfold_case(list, fold=5):
    num = len(list)
    new_list = []
    for i in range(fold):
        front = int(num * i / fold)
        behind = int(num * (i + 1) / fold)
        new_list.append(list[front:behind])
    return new_list


def statistics(case_fold, info):
    labels = {}
    for i in range(len(case_fold)):
        labels.update({i: []})
        for j in case_fold[i]:
            labels[i].extend(info[info['case'] == j]['label'])


def Kfold(dir, k=5):
    files = getnpy(dir)
    info = getinfo(files)
    case_ = info['case'].unique()
    # case_fold = getfold_case(case_, fold=k)
    case_fold = getfold_bydict(fold=k)
    folds = []
    labels_num = []
    for i in range(len(case_fold)):
        items = []
        labels = []
        for j in case_fold[i]:
            df = info[info['case'] == j]
            labels.extend(df['label'])
            for k in range(len(df)):
                file = df.iloc[k]['case'] + '_' + df.iloc[k]['label'] + '_' + df.iloc[k]['slice'] + '.npy'
                items.append(file)
        labels_num.append(Counter(labels))
        folds.append(items)
    return folds, labels_num


def getRandom(dir, rate=0.1):
    files = getnpy(dir)
    info = getinfo(files)
    case_ = info['case'].unique()
    case_num = len(case_)
    val_num = int(case_num * rate)
    L = [case_[random.randint(0, case_num - 1)] for _ in range(val_num)]
    files = {
        'train': [],
        'val': []
    }
    labels_train = []
    for i in case_:
        df = info[info['case'] == i]
        if i not in L:
            labels_train.extend(df['label'])
            for k in range(len(df)):
                file = df.iloc[k]['case'] + '_' + df.iloc[k]['label'] + '_' + df.iloc[k]['slice'] + '.npy'
                files['train'].append(file)
        else:
            for k in range(len(df)):
                file = df.iloc[k]['case'] + '_' + df.iloc[k]['label'] + '_' + df.iloc[k]['slice'] + '.npy'
                files['val'].append(file)
    return files, Counter(labels_train)
