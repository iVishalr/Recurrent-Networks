# Recurrent Neural Networks

Implements simple recurrent network and a stacked recurrent network in `numpy` and `torch` respectively. Both flavours implements a forward and backward function API that is resposible for handling the model behaviour in forward pass and backward pass. Backward pass has been implemented using native numpy/torch tensors and no autograd engines have been used to perform the backward pass.

`main.py` is a thin wrapper that calls the appropriate model class and trains a recurrent network on [tinyshakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). After training for a while, we can autoregressively sample random poems from the model.

*Note : As of now, only Stacked-RNN and RNN have been implemented. Look forward to implementations of LSTM, GRU and Transformers in numpy/torch*

## Requirements

1. numpy
2. pytorch

## Training

Edit the `main.py` file to configure a RNN model by specifying number of hidden layers, sequence_length and so on. Exceute the following command in terminal.

```bash
$ python3 main.py
```

## License 

MIT
