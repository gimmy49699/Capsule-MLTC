import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score


def micro_p_r_f(y, preds):
    return precision_recall_fscore_support(y, preds, average='micro')


def macro_p_r_f(y, preds):
    y = np.reshape(y, [-1])
    preds = np.reshape(preds, [-1])
    return precision_recall_fscore_support(y, preds, average='macro')


def accuracy(y, preds):
    return accuracy_score(np.array(y), np.array(preds))


def hm_loss(y, preds):
    return hamming_loss(np.array(y), np.array(preds))
