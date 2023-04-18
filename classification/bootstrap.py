import numpy as np
import scipy.stats as st

def average(data):
    return sum(data) / len(data)


def getInterval(x, alpha=0.95):
    # x_mean = np.mean(x)
    # x_std = np.std(x)
    # x_n = len(x)
    # x_se = x_std / np.sqrt(x_n)
    # x_ci = st.t.interval(alpha, x_n - 1, loc=x_mean, scale=x_se)
    return np.percentile(x, (1-alpha)*100), np.percentile(x, alpha*100)


def bootstrap(L, P, B, func1, func2, c=0.95):
    """
    计算bootstrap置信区间
    :param data: array 保存样本数据
    :param B: 抽样次数 通常B>=1000
    :param c: 置信水平
    :param func: 样本估计量
    :return: bootstrap置信区间上下限
    """
    L = np.array(L)
    P = np.array(P)
    n = len(L)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        index_arr = np.unique(index_arr)
        # index_arr = sorted(index_arr)
        L_sample = L[index_arr]
        P_sample = P[index_arr]
        if L_sample.sum() == 0:
            continue
        fpr, tpr, _ = func1(L_sample, P_sample)
        sample_result = func2(fpr, tpr)
        sample_result_arr.append(sample_result)
    # a = 1 - c
    # k1 = int(B * a / 2)
    # k2 = int(B * (1 - a / 2))
    # auc_sample_arr_sorted = sorted(sample_result_arr)
    # lower = auc_sample_arr_sorted[k1]
    # higher = auc_sample_arr_sorted[k2]
    lower, higher = getInterval(np.array(sample_result_arr), c)
    return lower, higher