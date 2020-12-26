from typing import List, cast
from pyClarion import nd
from pprint import pprint
from random import gauss


# A basic form of reverse-mode automatic differentiation is supported in 
# pyClarion with an API similar to that of Tensorflow 2. 


###################
### Basic Usage ###
###################

# The basic tool for automatic differentiation is the GradientTape, which is a 
# context manager that tracks operations on numdicts and provides methods for 
# computing gradients over recorded tapes. 

# In the simplest case, the tape may be used in 'eager' mode. In this mode, the 
# tape will record all registered ops carried out in the tape context and flush
# its data when a gradients are retrieved. 

tape = nd.GradientTape()

print("Example 1")
print()

with tape:
    d1 = nd.NumDict(default=1.0)
    d2 = nd.NumDict(default=2.0)
    d3 = d1 * d2

# We can inspect the contents of the gradient tape through the data property.

pprint(tape.data)
print()

# To get the gradients of d1 and d2 with respect to d3, it is sufficient to 
# call the tape.gradients() method.

d3, grads = tape.gradients(d3, (d1, d2))

print("d3:", d3)
print("grads:", grads)
print()

print("Example 2")
print()

with tape:
    d1 = nd.NumDict({1: 1.0, 2: 2.0})
    d2 = nd.NumDict({1: 3.0, 2: 4.0})
    d3 = nd.sum_by(d1 * d2, keyfunc=lambda x: 3)

pprint(tape._tape)
print()

d3, grads = tape.gradients(d3, (d1, d2))

print("d3:", d3)
print("grads:", grads)
print()


########################
### Gradient Descent ###
########################

# For this example, we will create a linear-regression-like setup, with an 
# input x, an output y, and parameters w. 

# We will be learning the relationship:
# y = m * x + b

m, b = 3.0, -1.0

batch_size = 20
epochs = 10
lr = 0.5

batches = [[gauss(0, 1) for _ in range(batch_size)] for _ in range(epochs)]

x = nd.MutableNumDict({1: 0.0, 2: 1.0}) 
y = nd.MutableNumDict({3: 0.0})

w = nd.MutableNumDict({1: gauss(0, 1), 2: gauss(0, 1)})

# To get a persistent gradient tape, we simply pass in True as the value for 
# the `persistent` parameter to the tape constructor. 

with nd.GradientTape(persistent=True) as tape:
    y_hat = nd.sum_by(x * w, keyfunc=lambda x: 3)
    cost = (y - y_hat) ** 2 / 2

mse: List[float] = []
for i in range(epochs):
    
    grad_w = nd.MutableNumDict(default=0.0)
    
    _mse = []
    for x1 in batches[i]:

        x[1] = x1
        y[3] = m * x1 + b

        # When the tape is in persistent mode, calling tape.gradients() will, 
        # by default, execute a forward pass to ensure that the gradients are 
        # fresh. The forward pass can be blocked, if needed. 

        # When the forward pass is performed, the new output is returned as the 
        # first return value of tape.gradients(). This new output must be 
        # passed to tape.gradients() on the next call, as the tape will forget 
        # about the old output after the forward pass.

        cost, grad = tape.gradients(cost, w)
        grad_w += grad / batch_size
        _mse.append(cost[3]) 

    w -= lr * grad_w  
    mse.append(sum(_mse) / batch_size)

print("Example 3: Gradient Descent")
print()

print("Target relationship is y = m * x + b")
print("m = {}, b = {}".format(m, b))
print()

print("Mean Squared Error by Epoch:")
pprint(mse)
print()

print("Learned Weights (1 is m, 2 is b):")
print(w)


##################
### CONCLUSION ###
##################

# The pyClarion library supports automatic differentiation for building and 
# training basic neural networks as required by Clarion theory. However, the 
# provided autodiff tools are not optimized and run natively in python. These 
# tools are best used for learning and small-scale experimentation.

# If the need arises for more sophisticated neural network models (e.g., 
# convolutional networks) or faster performance, it is best to integrate 
# pyClarion with a dedicated deep-learning/autodiff library by, e.g., writing 
# Propagator classes that wrap deep neural networks implemented using such 
# libraries.
