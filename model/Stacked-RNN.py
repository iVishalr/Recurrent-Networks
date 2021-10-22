import torch

torch.set_num_threads(2)

class RNN:
    
    def __init__(self, layers, hidden_size, vocab_size, seq_length):

        self.layers = layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.inputs = None
        self.targets = None
        self.len_inputs = 0

        self.make_rnn_layers()
        self.mWxh, self.mWhh, self.mbh, self.mWhy, self.mby = {}, {}, {}, {}, {}
        for layer in range(self.layers):
            self.mWxh[layer], self.mWhh[layer], self.mbh[layer] = torch.zeros_like(self.layer_params[layer]["Wxh"],device=self.device), \
                                        torch.zeros_like(self.layer_params[layer]["Whh"],device=self.device), \
                                        torch.zeros_like(self.layer_params[layer]["bh"],device=self.device),

        self.mWhy, self.mby = torch.zeros_like(self.pred_params["Why"],device=self.device), torch.zeros_like(self.pred_params["by"],device=self.device)

    def make_rnn_layers(self):

        self.layer_params = {}

        for layer in range(self.layers):
            
            Wxh_shape = (self.hidden_size,self.vocab_size) if layer == 0 else (self.hidden_size,self.hidden_size)
            self.layer_params[layer] = {"Wxh": torch.randn(Wxh_shape, device=self.device) * 0.01,
                                    	"Whh": torch.randn(self.hidden_size,self.hidden_size, device=self.device) * 0.01,
                                    	"bh": torch.zeros(self.hidden_size,1, device=self.device)}

        self.pred_params = {"Why": torch.randn(self.vocab_size,self.hidden_size,device=self.device) * 0.01,
                            "by": torch.zeros(self.vocab_size,1,device=self.device)}


    def forward(self,inputs,targets,hprev):

        self.inputs = inputs
        self.targets = targets

        self.xs, self.hs, self.ys, self.ps = {}, {}, {}, {}

        for layer in range(self.layers):
            self.xs[layer] = {}
            self.hs[layer] = {}
        
        self.loss = 0

        self.len_inputs = len(inputs)

        for layer in range(self.layers):
            self.hs[layer][-1] = hprev[layer].detach().clone()

        for time_step in range(self.len_inputs):
            for layer in range(self.layers):
                if layer==0:
                    val = torch.zeros(self.vocab_size,1, device=self.device)
                    val[inputs[time_step]] = 1
                else:
                    val = self.hs[layer-1][time_step]
                
                self.xs[layer][time_step] = val
                self.hs[layer][time_step] = torch.tanh(torch.matmul(self.layer_params[layer]["Wxh"],self.xs[layer][time_step]) \
                                                        + torch.matmul(self.layer_params[layer]["Whh"],self.hs[layer][time_step-1]) + self.layer_params[layer]["bh"])

            y = torch.matmul(self.pred_params["Why"], self.hs[layer][time_step]) + self.pred_params["by"]
            self.ys[time_step] = y
            
            #compute the softmax probs
            exps = torch.exp(y)
            self.ps[time_step] = exps/torch.sum(exps)
            self.loss += 0 - torch.log(self.ps[time_step][targets[time_step],0])

        hprev = {}
        for layer in range(self.layers):
            hprev[layer] = self.hs[layer][time_step]
            
        return self.loss, hprev

    def backward(self):

        self.dWxh, self.dWhh, self.dbh, self.dhnext = {},{},{},{}

        for layer in range(self.layers):
            self.dWxh[layer] = torch.zeros_like(self.layer_params[layer]["Wxh"],device=self.device)
            self.dWhh[layer] = torch.zeros_like(self.layer_params[layer]["Whh"],device=self.device)
            self.dbh[layer] = torch.zeros_like(self.layer_params[layer]["bh"],device=self.device)
            self.dhnext[layer] = torch.zeros(self.hidden_size,1,device=self.device)

        self.dWhy = torch.zeros_like(self.pred_params["Why"],device=self.device)
        self.dby = torch.zeros_like(self.pred_params["by"],device=self.device)
        
        for time_step in reversed(range(self.len_inputs)):
            
            dy = self.ps[time_step].detach().clone()
            dy[self.targets[time_step]] -= 1
            self.dWhy += torch.matmul(dy,self.hs[self.layers-1][time_step].T)
            self.dby += dy

            dh = torch.matmul(self.pred_params["Why"].T,dy) + self.dhnext[self.layers-1]
            for layer in reversed(range(self.layers)):
                dhraw = (1-self.hs[layer][time_step]**2) * dh
                self.dWxh[layer] += torch.matmul(dhraw,self.xs[layer][time_step].T)
                self.dWhh[layer] += torch.matmul(dhraw,self.hs[layer][time_step-1].T)
                self.dbh[layer] += dhraw
                self.dhnext[layer] = torch.matmul(self.layer_params[layer]["Whh"].T,dhraw)
                if layer > 0:
                    dh = torch.matmul(self.layer_params[layer]["Wxh"].T,dhraw) + self.dhnext[layer-1] 
        
        for layer in range(self.layers):
            for grads in [self.dWxh[layer],self.dWhh[layer],self.dbh[layer]]:
                torch.clip(grads,-5,5,out=grads)
        for grads in [self.dWhy,self.dby]:
            torch.clip(grads,-5,5,out=grads)
        
        return (self.dWxh,self.dWhh,self.dbh,self.dWhy,self.dby)

    def sampler(self,h,seed,n):
        x = {}
        for layer in range(self.layers):
            x[layer] = torch.zeros(self.vocab_size,1,device=self.device)
        x[0][seed] = 1
        preds = []

        for t in range(n):
            for layer in range(self.layers):
                if layer==0:
                    val = x[layer]
                else:
                    val = h[layer-1]
                
                in_ = val

                h[layer] = torch.tanh( torch.matmul(self.layer_params[layer]["Wxh"],in_) + torch.matmul(self.layer_params[layer]["Whh"],h[layer]) + self.layer_params[layer]["bh"])
            y = torch.matmul(self.pred_params["Why"],h[layer]) + self.pred_params["by"]
            exps = torch.exp(y)
            p = exps/torch.sum(exps)
            probs = p.ravel()
            idx = probs.multinomial(num_samples=1,replacement=True)
            ix = range(self.vocab_size)[idx]
            for layer in range(self.layers):
                x[layer] = torch.zeros(self.vocab_size,1,device=self.device)
            x[0][ix] = 1
            preds.append(ix)

        return preds

    def unpack_params(self):
        parameters = [self.layer_params,self.pred_params]
        
        params = []
        for key in ["Wxh","Whh","bh"]:
            for layer in range(self.layers):
                params.append(parameters[0][layer][key])

        params.append(parameters[1]["Why"])
        params.append(parameters[1]["by"])

        return params

    def unpack_grads(self,grads):
        (dWxh,dWhh,dbh,dWhy,dby) = grads

        keys = ["dWxh","dWhh","dbh"]
        data = [dWxh,dWhh,dbh]
        dictionary = dict(zip(keys, data))

        grad = []
        for key in keys:
            for layer in range(self.layers):
                grad.append(dictionary[key][layer])

        grad.append(dWhy)
        grad.append(dby)

        return grad
    
    def optimize(self,grads,learning_rate):
        parameters = self.unpack_params()
        gradients = self.unpack_grads(grads)

        caches = []
        keys = ["mWxh","mWhh","mbh"]
        data = [self.mWxh,self.mWhh,self.mbh]
        dictionary = dict(zip(keys, data))

        for key in keys:
            for layer in range(self.layers):
                caches.append(dictionary[key][layer])

        caches.append(self.mWhy)
        caches.append(self.mby)

        for param, grad, cache in zip(parameters,gradients,caches):
            cache += grad * grad
            param += -learning_rate * grad/torch.sqrt(cache+1e-8)
            
        count = 0
        for key in ["Wxh","Whh","bh"]:
            for layer in range(self.layers):
                self.layer_params[layer][key] = parameters[count]
                count+=1
        
        self.pred_params["Why"] = parameters[count]
        self.pred_params["by"] = parameters[count+1]

        count = 0
        for layer in range(self.layers):
            self.mWxh[layer] = caches[count]
            count+=1

        for layer in range(self.layers):
            self.mWhh[layer] = caches[count]
            count+=1

        for layer in range(self.layers):
            self.mbh[layer] = caches[count]
            count+=1
        
        self.mWhy = caches[count]
        self.mby = caches[count+1]