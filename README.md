# np_nets
Neural network experiments written purely in numpy

1. Learning backprop with the MNIST classification task
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/mnist_nn.ipynb)
2. Synthetic gradients with the MNIST classification task
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/synthetic_gradients.ipynb)
  * also check out the minimalist [145-line Gist](https://gist.github.com/greydanus/1cb90875f24015660ae91fa637f167a9) for this project
  * inspired by [this Google DeepMind paper](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/)
3. Hebbian learning (for a Dartmouth class)
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/hebb-pset5.ipynb)
  * For Human Memory (PSYC 051.09) taught by [Jeremy Manning](http://www.context-lab.com/)
4. "U loss" learning
  * I test an ansatz for layer-wise training of neural networks. **It didn't work**. That's how research goes.
  * Folder is [here](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/hebb-pset5.ipynb)

<i>**EDIT:** I made a mistake in my implementation of synthetic gradients. Runtime is not the issue I imagined it was because the targets of the synthetic gradient models should be the output activations of each layer rather than the actual gradients on the weights. I'm in the process of fixing this in my code. For now, just know that there's a bug in the second jupyter notebook.</i>
>>>>>>> origin/master
