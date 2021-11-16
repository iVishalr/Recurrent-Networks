import torch
import numpy as np
import sys
from model.Stacked_RNN import RNN

torch.set_num_threads(8)

data = open(sys.argv[1],"r").read()
vocab = list(set(data))
data_size,vocab_size = len(data),len(vocab)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_idx = {ch:i for i,ch in enumerate(vocab)}
idx_to_char = {i:ch for i,ch in enumerate(vocab)}

hidden_size = 100
seq_length = 25
learning_rate = 1e-3
layers = 3
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Uncomment this line to make torch use GPU

rnn = RNN(layers=layers, hidden_size=hidden_size,vocab_size=vocab_size,seq_length=seq_length)

n,p = 0,0


smooth_loss = -np.log(1.0/vocab_size)*seq_length

while True:
    if p+seq_length+1 >= data_size or n==0:
        hprev = {}
        for layer in range(layers):
            hprev[layer] = torch.zeros(hidden_size,1,device=device)
        p = 0

    inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

    if n % 1000 == 0:
        sample_ix = rnn.sampler(hprev, inputs[0], 200)
        txt = ''.join(idx_to_char[ix] for ix in sample_ix)
        print ('----\n %s \n----' % (txt, ))

    loss, hprev = rnn.forward(inputs,targets,hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n % 1000 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress

    grads = rnn.backward()

    rnn.optimize(grads=grads,learning_rate=learning_rate)

    p += seq_length
    n += 1 
