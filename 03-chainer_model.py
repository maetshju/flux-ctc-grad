import chainer
import os
from tqdm import tqdm
from chainer.functions import connectionist_temporal_classification as ctc
import statistics
import chainer.links as L
import numpy as np
from textdistance import levenshtein as lev
import sys
import random

DATA_DIR = 'train_txt'
EPOCHS = 300

N_FILES = 400
SHUFFLED_FNAMES_FILES = 'shuffled_names.txt'

random.seed(2)
np.random.seed(4)

class LSTMforSpeech(chainer.Chain):

    def __init__(self):
	
        super(LSTMforSpeech, self).__init__()
        with self.init_scope():
	
            self.l1 = L.LSTM(39, 200)
            self.l2 = L.Linear(200, 62)
		
        for i, param in enumerate(self.params()):
        
            param.array[...] = np.random.uniform(-0.1, 0.1, param.shape)
			
    def reset_state(self):
	
        self.l1.reset_state()
		
    def forward(self, x):
	
        h0 =  [self.l1(xI) for xI in x]
        y = [self.l2(h0I) for h0I in h0]
        return y
        
def collapse(y):

    y = np.argmax(y, 1).tolist()
    s = [y[0]]
    for c in y[1:]:
    
        if c != s[-1] and c != 61:
        
            s.append(c)
            
    if s == [61]:
        s = []
    return s

def read_data(mydir):

    with open(SHUFFLED_FNAMES_FILES, 'r') as f:

        fnames = [x.strip() for x in f.readlines()]

    data = list()

    for fname in tqdm(fnames[:(N_FILES)]):
            
        y = np.loadtxt(os.path.join(mydir, fname + '_labs.txt'), dtype='int32').T
    
        y = collapse(y)
        y = np.array(y, dtype='int32')
        y = np.expand_dims(y, 0)
            
        x = np.loadtxt(os.path.join(mydir, fname + '.txt'), dtype='float32')
        x = [chainer.Variable(np.expand_dims(x[i,:], 0)) for i in range(x.shape[0])]
            
        data.append((x, y))

    return data
		
def per(x, y):

    model.reset_state()
    chainer.using_config('train', False)
    yhat = model(x)
    yhat = np.vstack([yI.data for yI in yhat])
    
    yhat = collapse(yhat)
    y = y.squeeze().tolist()
    p = lev(yhat, y) / len(y)
    return p

model = LSTMforSpeech()

optimizer = chainer.optimizers.MomentumSGD(1e-4)
optimizer.setup(model)

data = read_data(DATA_DIR)

for e in range(EPOCHS):

    print('BEGINNING EPOCH {}/{}'.format(e+1, EPOCHS))

    losses = []
    chainer.using_config('train', True)
    
    for x, y in tqdm(data):
    
        model.reset_state()
        
        yhat = model(x)
        loss = ctc(yhat, y, 61)
        
        optimizer.update(lossfun=lambda: loss)
        losses.append(loss.data.tolist())
        
    p = statistics.mean([per(x, y) for x, y in data])
    print('PER:\t{}'.format(p * 100))
    print('Mean loss:\t{}'.format(statistics.mean(losses)))
    if p < 0.35:
        sys.exit()
