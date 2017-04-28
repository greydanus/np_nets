# uloss
Testing an ansatz for layer-wise training

The idea was that you might be able to train a multilayer neural network by asking each layer to maximize the **strong** activations, minimize the **medium** activations and sort of be ambivalent towards **low** activations of the _next_ layer's neurons. The reward, as a function of activation, looks sort of like a "U". Of course the topmost layer would still be trained with backprop.

I went ahead and implemented this idea and compared it to 1) baseline backprop and 2) training the very topmost layer alone with backprop. Turns out, this doesn't work too well (at least the way I implemented it). That's ok, most of my projects are failures :'(
