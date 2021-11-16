import os
os.environ["OMP_NUM_THREADS"] = "12" # export OMP_NUM_THREADS=4


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
Wg =  np.random.randn(hidden_size,hidden_size+vocab_size) *0.01
Wi =  np.random.randn(hidden_size,hidden_size+vocab_size)*0.01
Wf =  np.random.randn(hidden_size,hidden_size+vocab_size)*0.01
Wo =  np.random.randn(hidden_size,hidden_size+vocab_size)*0.01
Why =  np.random.randn(vocab_size,hidden_size)*0.01

bg =  np.zeros((hidden_size,1))
bi =  np.zeros((hidden_size,1))
bf =  np.zeros((hidden_size,1))
bo =  np.zeros((hidden_size,1))
by =  np.zeros((vocab_size,1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(input):
    # Subtraction of max value improves numerical stability.
    e_input = np.exp(input - np.max(input))
    return e_input / e_input.sum()

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

def rnn(inputs,targets,h_prev,c_prev):
    """
    inputs and targets are both lists of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model params and last hidden state
    """

    i = {}
    f = {}
    o = {}
    g = {}
    c = {}
    h = {}
    y = {}
    p = {}
    x = {}
    z = {}

    h[-1] = np.copy(h_prev)
    c[-1] = np.copy(c_prev)
    loss = 0

    for t in range(len(inputs)):
        x[t] = np.zeros((vocab_size,1))
        x[t][inputs[t]] = 1
        z[t] = np.row_stack((h[t-1],x[t]))
        f[t] = sigmoid(np.dot(Wf,z[t]) + bf)
        i[t] = sigmoid(np.dot(Wi,z[t]) + bi)
        g[t] = np.tanh(np.dot(Wg,z[t]) + bg)
        
        c[t] = np.multiply(i[t],g[t]) + np.multiply(f[t],c[t-1])
        
        o[t] = sigmoid(np.dot(Wo,z[t]) + bo)
        h[t] = np.multiply(o[t],np.tanh(c[t]))
        
        y[t] = np.dot(Why,h[t]) + by
        p[t] = softmax(y[t])
        loss += -np.log(p[t][targets[t],0])

    #backward pass 
    dWi,dWf,dWg,dWo,dWhy = np.zeros_like(Wi),np.zeros_like(Wf),np.zeros_like(Wg),np.zeros_like(Wo),np.zeros_like(Why)
    dbi,dbf,dbg,dbo,dby = np.zeros_like(bi),np.zeros_like(bf),np.zeros_like(bg),np.zeros_like(bo),np.zeros_like(by)

    dh_next = np.zeros_like(h[0]) #dh from the next character
    dc_next = np.zeros_like(c[0]) #dh from the next character

    for t in reversed(range(len(inputs))):
        #backprop into predictor of LSTM
        dy = np.copy(p[t])
        dy[targets[t]] -= 1
        
        dWhy += np.dot(dy,h[t].T)
        dby += dy  

        #backprop into the hidden cell of LSTM

        dh = np.copy(dh_next)
        dh += np.dot(Why.T,dy)

        inter = np.tanh(c[t])
        do = np.multiply(dh, inter)
        
        do_raw = do * (o[t]*(1-o[t]))
        dWo += np.dot(do_raw, z[t].T)
        dbo += do_raw

        dc = np.copy(dc_next)
        dc += np.multiply( dh, np.multiply(o[t],1-inter*inter))

        di = np.multiply(dc, g[t])
        df = np.multiply(dc, c[t-1])
        dg = np.multiply(dc, i[t])

        dg_raw = dg * (1-g[t]*g[t])
        dWg += np.dot(dg_raw, z[t].T)
        dbg += dg_raw

        df_raw = df * (f[t]*(1-f[t]))
        dWf += np.dot(df_raw, z[t].T)
        dbf += df_raw

        di_raw = di * (i[t]*(1-i[t]))
        dWi += np.dot(di_raw, z[t].T)
        dbi += di_raw

        dh_next = np.dot(Wi.T,di_raw) + np.dot(Wf.T,df_raw) + np.dot(Wo.T,do_raw) + np.dot(Wg.T,dg_raw)
        dh_next = dh_next[:hidden_size]
        dc_next = np.multiply(f[t],dc)
        

    for grad in [dWi,dWf,dWg,dWo,dWhy,dbi,dbf,dbg,dbo,dby]:
        np.clip(grad,-1,1,out=grad)

    return loss,dWi,dWf,dWg,dWo,dWhy,dbi,dbf,dbg,dbo,dby, h[len(inputs) - 1], c[len(inputs) - 1]

def sample(h,c,seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for the first time step
    """ 

    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        z = np.row_stack((h,x))
        i = sigmoid(np.dot(Wi,z) + bi)
        f = sigmoid(np.dot(Wf,z) + bf)
        g = np.tanh(np.dot(Wg,z) + bg)
        
        c = np.multiply(i,g) + np.multiply(f,c)
        
        o = sigmoid(np.dot(Wo,z) + bo)
        h = np.multiply(o,np.tanh(c))
        
        y = np.dot(Why,h) + by
        p = softmax(y)

        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

n, p = 0,0
mWi,mWf,mWg,mWo,mWhy = np.zeros_like(Wi),np.zeros_like(Wf),np.zeros_like(Wg),np.zeros_like(Wo),np.zeros_like(Why)
mbi,mbf,mbg,mbo,mby = np.zeros_like(bi),np.zeros_like(bf),np.zeros_like(bg),np.zeros_like(bo),np.zeros_like(by)

smooth_loss = -np.log(1.0/vocab_size)*seq_length
while True:
    if p+seq_length+1 >= len(data) or n==0:
        h_prev = np.zeros((hidden_size,1))
        c_prev = np.zeros((hidden_size,1))
        p = 0
    inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

    if n%1000 == 0:
        sample_idx = sample(h_prev,c_prev,inputs[0],1000)
        txt = ''.join(idx_to_char[idx] for idx in sample_idx)
        print('----\n %s \n----'%(txt,))

    loss, dWi,dWf,dWg,dWo,dWhy,dbi,dbf,dbg,dbo,dby, h_prev,c_prev = rnn(inputs,targets,h_prev,c_prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n%1000==0 : 
        print("iter %d, loss: %f"%(n,smooth_loss))
    
    for param, dparam, mem in zip([Wi,Wf,Wg,Wo,Why,bi,bf,bg,bo,by],
                                    [dWi,dWf,dWg,dWo,dWhy,dbi,dbf,dbg,dbo,dby],
                                    [mWi,mWf,mWg,mWo,mWhy,mbi,mbf,mbg,mbo,mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1