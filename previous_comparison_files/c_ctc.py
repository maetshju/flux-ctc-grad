import chainer
from chainer.functions import connectionist_temporal_classification as ctc
import numpy as np

def c_ctc(yhat, y):

    yhat = [np.expand_dims(yhat[:,i], 0) for i in range(yhat.shape[1])]
    yhat = [chainer.Variable(yI) for yI in yhat]
    loss = ctc(yhat, y, 61)
    g = chainer.grad([loss], yhat)
    g = np.vstack([gI.data for gI in g]).T
    return loss.data, g