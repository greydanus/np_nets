# uloss
Testing an ansatz for layer-wise training

The idea was that you might be able to train a multilayer neural network by asking each layer to maximize the **strong** activations, minimize the **medium** activations and sort of be ambivalent towards **low** activations of the _next_ layer's neurons. The reward, as a function of activation, looks sort of like a "U". Of course the topmost layer would still be trained with backprop.

I went ahead and implemented this idea and compared it to 1) baseline backprop and 2) training the very topmost layer alone with backprop. Turns out, this doesn't work too well (at least the way I implemented it). That's ok. In science, most of my projects are failures.

## Contains

1. Vanilla 3-layer neural net I wrote in numpy for MNIST
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/uloss/mnist_nn.ipynb)
2. Vanilla 3-layer neural net I wrote in numpy for MNIST **but I only train the topmost layer**
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/synthetic_gradients.ipynb)
3. 3-layer neural net I wrote in numpy for MNIST that I attempt to train using the uloss ansatz
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/hebb-pset5.ipynb)
  * For Human Memory (PSYC 051.09) taught by [Jeremy Manning](http://www.context-lab.com/)

<i>**EDIT:** These ideas came from a conversation with Kenneth Norman. I don't know if this is _at all_ what he was talking about but I got the idea in my head after the conversation and had to test it out. I'm guessing he meant something else...having finished this experiment, I realize that it has some serious flaws.
