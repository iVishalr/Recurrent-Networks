import numpy as np
import sys
#read the data from the file
data = open(sys.argv[1],'r').read()
characters = list(set(data))
data_size, vocab_size = len(data),len(characters)

print("Data has %d characters, %d unique characters."%(data_size,vocab_size))

#char to idx mapping
char_to_idx = {ch:i for i,ch in enumerate(characters)}
idx_to_char = {i:ch for i,ch in enumerate(characters)}

#define some hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

#model parameters
Wxh = np.random.randn(hidden_size,vocab_size)*0.01
# in the above statement, we init with (hidden_size,vocab_size) because we are multiplying
# the weights with inputs that have one hot representation of the char
Whh = np.random.randn(hidden_size,hidden_size)*0.01
Why = np.random.randn(vocab_size,hidden_size)*0.01

bh = np.zeros((hidden_size,1))
by = np.zeros((vocab_size,1))

def rnn(inputs,targets,hprev):
    """
    inputs and targets are both lists of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model params and last hidden state
    """

    xs, hs, ys, ps = {}, {}, {}, {}
    #copy the hprev to last element of hs dict
    hs[-1] = np.copy(hprev)

    loss = 0

    #forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) #encode in 1-of-k representation
        xs[t][inputs[t]]=1
        hs[t] = np.tanh(np.dot(Wxh,xs[t]) + np.dot(Whh,hs[t-1]) + bh)
        ys[t] = np.dot(Why,hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        loss += -np.log(ps[t][targets[t],0])

    #backward pass 
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy,hs[t].T)
        dby += dy

        dh = np.dot(Why.T,dy) + dhnext
        dhraw = (1-hs[t]*hs[t])*dh
        dbh += dhraw

        dWxh += np.dot(dhraw,xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T,dhraw)

    for grads in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(grads,-5,5,out=grads)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h,seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for the first time step
    """ 

    x = np.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        h = np.tanh(np.dot(Wxh,x) + np.dot(Whh,h) + bh)
        y = np.dot(Why,h) + by
        p = np.exp(y)/np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size),p=p.ravel())
        x = np.zeros((vocab_size,1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

n, p = 0,0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length
while True:
    if p+seq_length+1 >= len(data) or n==0:
        hprev = np.zeros((hidden_size,1))
        p = 0
    inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

    if n%1000 == 0:
        sample_idx = sample(hprev,inputs[0],200)
        txt = ''.join(idx_to_char[idx] for idx in sample_idx)
        print('----\n %s \n----'%(txt,))

    loss, dWxh, dWhh, dWhy, dbh, dby, hrev = rnn(inputs,targets,hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n%1000==0 : 
        print("iter %d, loss: %f"%(n,smooth_loss))
    
    for param, grads, cache in zip([Wxh,Whh,Why,bh,by],
                                   [dWxh, dWhh, dWhy,dbh,dby],
                                   [mWxh,mWhh,mWhy,mbh,mby]):
        cache += grads * grads
        param += -learning_rate * grads/np.sqrt(cache+1e-8)

    p += seq_length
    n += 1