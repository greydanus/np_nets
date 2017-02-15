# np_nets
Neural network experiments written purely in numpy

1. Learning backprop with the MNIST classification task
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/mnist_nn.ipynb)
2. Synthetic gradients with the MNIST classification task
  * [Jupyter notebook](https://nbviewer.jupyter.org/github/greydanus/np_nets/blob/master/synthetic_gradients.ipynb)
  * also check out the minimalist [145-line Gist](https://gist.github.com/greydanus/1cb90875f24015660ae91fa637f167a9) for this project
  * inspired by [this Google DeepMind paper](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/)

<i>**EDIT:** I made a mistake in my implementation of synthetic gradients. Runtime is not the issue I imagined it was because the targets of the synthetic gradient models should be the output activations of each layer rather than the actual gradients on the weights. I'm in the process of fixing this in my code. For now, just know that there's a bug in the second jupyter notebook.</i>
>>>>>>> origin/master
