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
learning_rate = 1e-2

#model parameters
Wxi = np.random.randn(hidden_size,vocab_size)*0.01
Whi = np.random.randn(hidden_size,hidden_size)*0.01
bi = np.zeros((hidden_size,1))

Wxr = np.random.randn(hidden_size,vocab_size)*0.01
Whr = np.random.randn(hidden_size,hidden_size)*0.01
br = np.zeros((hidden_size,1))

Wxh = np.random.randn(hidden_size,vocab_size)*0.01
Whh = np.random.randn(hidden_size,hidden_size)*0.01
bh = np.zeros((hidden_size,1))

Why = np.random.randn(vocab_size,hidden_size)*0.01
by = np.zeros((vocab_size,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(input):
    # Subtraction of max value improves numerical stability.
    e_input = np.exp(input - np.max(input))
    return e_input / e_input.sum()

def gru(inputs,targets,hprev):
    """
    inputs and targets are both lists of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model params and last hidden state
    """

    x,r,i,h,h_hat,y,p = {},{},{},{},{},{},{}
    #copy the hprev to last element of hs dict
    h[-1] = np.copy(hprev)

    loss = 0

    #forward pass
    for t in range(len(inputs)):
        x[t] = np.zeros((vocab_size,1)) #encode in 1-of-k representation
        x[t][inputs[t]]=1
        
        r[t] = sigmoid(np.dot(Whr,h[t-1]) + np.dot(Wxr,x[t]) + br)
        i[t] = sigmoid(np.dot(Whi,h[t-1]) + np.dot(Wxi,x[t]) + bi)

        h_hat[t] = np.tanh(np.dot(Whh,np.multiply(r[t],h[t-1])) + np.dot(Wxh,x[t]) + bh)
        h[t] = np.multiply(i[t],h[t-1]) + np.multiply((1-i[t]), h_hat[t])

        y[t] = np.dot(Why,h[t]) + by
        p[t] = softmax(y[t])

        loss += -np.log(p[t][targets[t],0])

    #backward pass 
    dWhy,dWhi,dWhr,dWhh,dWxi,dWxr,dWxh = np.zeros_like(Why),np.zeros_like(Whi),np.zeros_like(Whr),np.zeros_like(Whh),np.zeros_like(Wxi),np.zeros_like(Wxr),np.zeros_like(Wxh)
    dby,dbi,dbr,dbh = np.zeros_like(by),np.zeros_like(bi),np.zeros_like(br),np.zeros_like(bh)
    dhnext = np.zeros_like(h[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(p[t])
        dy[targets[t]] -= 1

        dWhy += np.dot(dy,h[t].T)
        dby += dy

        dh = np.dot(Why.T,dy) + dhnext
        di = np.multiply(dh,(h[t]-h_hat[t]))
        dh_hat = np.multiply(dh,(1-i[t])) 

        dh_raw = dh_hat * (1-h_hat[t]*h_hat[t])
        dWhh += np.dot(dh_raw, np.multiply(r[t], h[t-1]).T)
        dWxh += np.dot(dh_raw, x[t].T)
        dbh += dh_raw

        dr = np.multiply(np.dot(Whh.T, dh_raw),h[t-1])
        dr_raw = dr * (r[t]*(1-r[t]))
        dWhr += np.dot(dr_raw,h[t-1].T)
        dWxr += np.dot(dr_raw,x[t].T)
        dbr += dr_raw

        di_raw = di * (i[t] * (1-i[t]))
        dWhi += np.dot(di_raw,h[t-1].T)
        dWxi += np.dot(di_raw,x[t].T)
        dbi += di_raw

        dhnext = np.multiply(dh,i[t]) + np.multiply(np.dot(Whh.T, dh_raw),r[t]) + np.dot(Whr.T,dr_raw) + np.dot(Whi.T,di_raw)

    for grads in [dWhy,dWhi,dWhr,dWhh,dWxi,dWxr,dWxh,dby,dbi,dbr,dbh]:
        np.clip(grads,-5,5,out=grads)

    return loss, dWhy,dWhi,dWhr,dWhh,dWxi,dWxr,dWxh,dby,dbi,dbr,dbh, h[len(inputs)-1]

def sample(h,seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for the first time step
    """ 

    x = np.zeros((vocab_size,1))
    x[seed_ix] = 1
    ixes = []

    #forward pass
    for t in range(n):        
        i = sigmoid(np.dot(Whi,h) + np.dot(Wxi,x) + bi)
        r = sigmoid(np.dot(Whr,h) + np.dot(Wxr,x) + br)

        h_hat = np.tanh(np.dot(Whh,np.multiply(r,h)) + np.dot(Wxh,x) + bh)
        h = np.multiply(i,h) + np.multiply((1-i), h_hat)

        y = np.dot(Why,h) + by
        p = softmax(y)
        ix = np.random.choice(range(vocab_size),p=p.ravel())
        x = np.zeros((vocab_size,1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

n, p = 0,0
mWhy,mWhi,mWhr,mWhh,mWxi,mWxr,mWxh = np.zeros_like(Why),np.zeros_like(Whi),np.zeros_like(Whr),np.zeros_like(Whh),np.zeros_like(Wxi),np.zeros_like(Wxr),np.zeros_like(Wxh)
mby,mbi,mbr,mbh = np.zeros_like(by),np.zeros_like(bi),np.zeros_like(br),np.zeros_like(bh)

smooth_loss = -np.log(1.0/vocab_size)*seq_length
while True:
    if p+seq_length+1 >= len(data) or n==0:
        hprev = np.zeros((hidden_size,1))
        p = 0
    inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

    if n%1000 == 0:
        sample_idx = sample(hprev,inputs[0],1000)
        txt = ''.join(idx_to_char[idx] for idx in sample_idx)
        print('----\n %s \n----'%(txt,))

    loss, dWhy,dWhi,dWhr,dWhh,dWxi,dWxr,dWxh,dby,dbi,dbr,dbh, hrev = gru(inputs,targets,hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n%1000==0 : 
        print("iter %d, loss: %f"%(n,smooth_loss))
    
    for param, grads, cache in zip([Why,Whi,Whr,Whh,Wxi,Wxr,Wxh,by,bi,br,bh],
                                   [dWhy,dWhi,dWhr,dWhh,dWxi,dWxr,dWxh,dby,dbi,dbr,dbh],
                                   [mWhy,mWhi,mWhr,mWhh,mWxi,mWxr,mWxh,mby,mbi,mbr,mbh]):
        cache += grads * grads
        param += -learning_rate * grads/np.sqrt(cache+1e-8)

    p += seq_length
    n += 1