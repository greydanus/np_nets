# uloss
Testing an ansatz for layer-wise training

The idea was that you might be able to train a multilayer neural network by asking each layer to maximize the **strong** activations, minimize the **medium** activations and sort of be ambivalent towards **low** activations of the _next_ layer's neurons. The reward, as a function of activation, looks sort of like a "U". Of course the topmost layer would still be trained with backprop.

![uloss-function](../static/uloss-function.png?raw=true)

I went ahead and implemented this idea and compared it to 1) baseline backprop and 2) training the very topmost layer alone with backprop. Turns out, this doesn't work too well (at least the way I implemented it). That's ok, most of my projects are failures :'(

## Contains

1. Vanilla 3-layer neural net I wrote in numpy for MNIST
  * [Jupyter notebook](https://github.com/greydanus/np_nets/blob/master/uloss/np-mnist-backprop.ipynb)
2. Vanilla 3-layer neural net I wrote in numpy for MNIST **but I only train the topmost layer**
  * [Jupyter notebook](https://github.com/greydanus/np_nets/blob/master/uloss/np-mnist-backprop%20(train%20top%20layer%20only).ipynb)
3. 3-layer neural net I wrote in numpy for MNIST that I attempt to train using the uloss ansatz
  * [Jupyter notebook](https://github.com/greydanus/np_nets/blob/master/uloss/np-mnist%20(U%20loss%20function).ipynb)
