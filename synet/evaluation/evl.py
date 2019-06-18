import numpy as np


def class_F1_R_P_for_th(gru, pre, th):
    best = np.zeros(4)
    for t in th:
        tidx = gru == 1
        vpred = pre.copy()
        vpred[vpred > t] = 1
        vpred[vpred <= t] = 0

        TP = vpred[tidx].sum()

        if TP == 0:
            continue

        P = TP / float(vpred.sum())
        R = TP / float(gru.sum())
        F1 = 2*(P*R)/(R+P)

        if F1 > best[1]:
            best = np.array([t, F1, R, P])

    return best


def multi_evl_for_th(i, gru, pre, th):
    evl_metrics = np.zeros(4)

    th = np.arange(0, 1, 0.01)
    evl_metrics[:4] = class_F1_R_P_for_th(gru[:, i], pre[:, i], th)

    return evl_metrics


def evl_for_th(gru, pre, va_th=[]):
    evl_metrics = np.zeros((pre.shape[-1], 4))

    for i in np.arange(pre.shape[-1]):

        evl_metrics[i] = multi_evl_for_th(i, gru=gru, pre=pre, th=va_th)

    va_th = evl_metrics[:, 0].copy()
    evl_metrics = evl_metrics[:, 1:]
    acc = evl_metrics.mean(axis=0) * 100

    out = '[%s] F1-CB:%.1f%% R-CB:%.1f%% P-CB:%.1f%%'\
        % ('TR', acc[0], acc[1], acc[2])
    print(out)

    return va_th, evl_metrics, out


def class_f1_R_P(gru, pre, th):
    tidx = gru == 1
    vpred = pre.copy()
    vpred[vpred > th] = 1
    vpred[vpred <= th] = 0

    TP = vpred[tidx].sum()
    P = TP / float(vpred.sum())
    R = TP / float(gru.sum())
    F1 = 2 * (P * R) / (R + P)

    result = np.array([F1, R, P])

    return result


def multi_evl(i, gru, pre, th):
    evl_metrics = np.zeros(3)
    evl_metrics[:3] = class_f1_R_P(gru[:, i], pre[:, i], th[i])

    return evl_metrics


def evl(gru, pre, th):
    evl_metrics = np.zeros((pre.shape[-1], 3))

    for i in np.arange(pre.shape[-1]):
        evl_metrics[i] = multi_evl(i, gru=gru, pre=pre, th=th)

    acc = evl_metrics.mean(axis=0) * 100
    out = '[%s] F1-CB:%.1f%% R-CB:%.1f%% P-CB:%.1f%%'\
        % ('VA', acc[0], acc[1], acc[2])
    print(out)

    return evl_metrics, out
