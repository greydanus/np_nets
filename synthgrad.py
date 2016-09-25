""" Trains an MNIST classifier using Synthetic Gradients. See Google DeepMind paper @ arxiv.org/abs/1608.05343. """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.examples.tutorials.mnist import input_data # just use tensorflow's mnist api
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# hyperparameters
global_step = 0
batch_size = 10
lr, slr = 2e-2, 1e-4         # learning rates
D = 28*28                    # dimensionality
h1_size, h2_size = 100, 10   # first and second hidden layer
synth_step = 10              # ratio of model updates to synthetic gradient updates

# initialize mnist model
model = {}
model['W1'] = np.random.randn(D,h1_size) / np.sqrt(h1_size) # Xavier initialization: paper @ goo.gl/HYyVa7
model['b1'] = np.random.randn(1,h1_size) / np.sqrt(h1_size)
model['W2'] = np.random.randn(h1_size,h2_size) / np.sqrt(h2_size)
model['b2'] = np.random.randn(1,h2_size) / np.sqrt(h2_size)
# model for predicting synthetic gradients
smodel = { k : np.random.randn(v.shape[1], np.prod(v.shape)) * 1e-2 / np.sqrt(np.prod(v.shape)) \
          for k,v in model.iteritems() } # the 1e-2 scale factor helps...synthetic grads are weird

def softmax(x):
    maxes = np.amax(x, axis=1, keepdims=True)
    e = np.exp(x - maxes) # improves numerics
    return e / np.sum(e, axis=1, keepdims=True)

def forward(X, model):
    # evaluate class scores, [N x K]
    hs = [] # we'll need the h's for computing gradients
    h1 = np.dot(X, model['W1']) + model['b1'] ; hs.append(h1)
    h1[h1<0] = 0 # relu activation
    h2 = np.dot(h1, model['W2']) + model['b2']; hs.append(h2)
    h2[h2<0] = 0 # relu activation
    probs = softmax(h2)
    return probs, hs

def backward(y, probs, hs, model):
    grads = { k : np.zeros_like(v) for k,v in model.iteritems() } # dict to hold gradients
    probs[range(batch_size),y] -= 1 # backprop through softmax
    dh2 = probs / batch_size
    # second hidden layer
    grads['W2'] = np.dot(hs[0].T, dh2)
    grads['b2'] = np.sum(dh2, axis=0, keepdims=True)
    # first hidden layer
    dh1 = np.dot(dh2, model['W2'].T)
    dh1[hs[0] <= 0] = 0 # backprop through relu
    grads['W1'] = np.dot(X.T, dh1)
    grads['b1'] = np.sum(dh1, axis=0, keepdims=True)
    return grads

# forward propagate synthetic gradient model
def sforward(hs, smodel, model):
    synthetic_grads = { k : np.zeros_like(v) for k,v in smodel.iteritems() }
    synthetic_grads['W1'] = np.dot(hs[0], smodel['W1'])
    synthetic_grads['b1'] = np.dot(hs[0], smodel['b1'])
    synthetic_grads['W2'] = np.dot(hs[1], smodel['W2'])
    synthetic_grads['b2'] = np.dot(hs[1], smodel['b2'])
    return synthetic_grads

# backward propagate synthetic gradient model
def sbackward(hs, ds, smodel):
    sgrads = { k : np.zeros_like(v) for k,v in smodel.iteritems() }
    sgrads['W2'] = np.dot(hs[1].T, ds['W2']) ; sgrads['b2'] = np.dot(hs[1].T, ds['b2'])
    sgrads['W1'] = np.dot(hs[0].T, ds['W1'])
    sgrads['b1'] = np.dot(hs[0].T, ds['b1'])
    return sgrads

# evaluate training set accuracy
def eval_model(model):
    X = mnist.test.images
    y = mnist.test.labels
    hidden_layer = np.maximum(0, np.dot(X, model['W1']) + model['b1'])
    scores = np.dot(hidden_layer, model['W2']) + model['b2']
    predicted_class = np.argmax(scores, axis=1)
    return np.mean(predicted_class == y), predicted_class

loss_history = []
smoothing_factor = 0.95
for i in xrange(global_step, 1000):
    X, y = mnist.train.next_batch(batch_size)
    probs, hs = forward(X, model)
    synthetic_grads = sforward(hs, smodel, model) # compute synthetic gradients
    
    # synthetic gradient model updates
    if i % synth_step == 0:
        # compute the mnist model loss
        y_logprobs = -np.log(probs[range(batch_size),y])
        loss = np.sum(y_logprobs)/batch_size
        if i is 0 : smooth_loss = loss
        
        grads = backward(y, probs, hs, model) # data gradients

        # compute the synthetic gradient loss
        ds = {k : - v.ravel() + synthetic_grads[k] for (k, v) in grads.iteritems()}
        sloss = np.sum([np.sum(slr*v*v) for v in ds.itervalues()])/batch_size
        sgrads = sbackward(hs, ds, smodel)
        smodel = {k : smodel[k] - slr*sgrads[k] for (k,v) in sgrads.iteritems()} # update smodel parameters
        
        smooth_loss = (smoothing_factor*smooth_loss + (1-smoothing_factor)*loss)
        loss_history.append((i,smooth_loss))
        
    # estimate gradients using synthetic gradient model
    est_grad = {k : np.reshape(np.sum(v, axis=0), model[k].shape)/batch_size for k,v in synthetic_grads.iteritems()}
    model = {k : model[k] - lr*v for (k,v) in est_grad.iteritems()} # update model using estimated gradient
        
    # boring book-keeping
    if (i+1) % 500 == 0: print "iteration {}: test accuracy {:3f}".format(i, eval_model(model)[0])
    if (i) % 100 == 0: print "\titeration {}: smooth_loss {:3f}, synth_loss {:3f}".format(i, smooth_loss, sloss)
    global_step += 1

# plot W2 gradients
plt.figure(0, figsize=(12,16))
plt.subplot(121) ; plt.title("Actual gradient W2")
plt.imshow(grads['W2'].T[:,:50], cmap=cm.gray)
plt.subplot(122) ; plt.title("Synthetic gradient W2")
plt.imshow(grads['W2'].T[:,:50], cmap=cm.gray)
plt.show()

# plot loss over time
plt.figure(1)
plt.title("Smoothed loss") ; plt.xlabel("training steps") ; plt.ylabel("loss")
train_steps, smoothed_losses = zip(*loss_history)
plt.plot(train_steps, smoothed_losses)
plt.show()

# sample from test set with model
acc, predicted_class = eval_model(model)
X = mnist.test.images ; y = mnist.test.labels
r = np.random.randint(1000)
for i in range(0 + r,6 + r,3):
    img1 = np.reshape(X[i+0,:], (28,28))
    img2 = np.reshape(X[i+1,:], (28,28))
    img3 = np.reshape(X[i+2,:], (28,28))
    plt.figure(2+i, figsize=(5,2))
    plt.subplot(131) ; plt.title("predicted: " + str(predicted_class[i]))
    plt.imshow(img1, cmap=cm.gray)
    plt.subplot(132) ; plt.title("predicted: " + str(predicted_class[i+1]))
    plt.imshow(img2, cmap=cm.gray)
    plt.subplot(133) ; plt.title("predicted: " + str(predicted_class[i+2]))
    plt.imshow(img3, cmap=cm.gray)
    plt.show()