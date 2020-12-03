"""A demo for using pyClarion autodiff tools."""


from pyClarion import (
    diffable, GradientTape, DiffableNumDict, nd, pprint
)


# A basic form of reverse-mode automatic differentiation is supported in 
# pyClarion through the classes diffable and GradientTape. The diffable class 
# wraps float values with metadata supporting reverse-mode differentiation. The 
# GradientTape is a context manager that tracks operations on diffables and 
# provides methods for computing gradients over recorded tapes. A 
# DiffableNumDict class is also provided for working with diffable values in 
# bulk.

# The diffable instances can behave like variables or constants, depending on 
# the context. Numerical values associated with diffable instances are stored 
# in the `val` attribute. This allows for implemeneting gradient descent over 
# persistent computational graphs. Diffables additionally keep track of tape 
# indices for themselves, and any operands that may have produced them, as well 
# as the symbolic name of the op that produced them.

# Overall, the GradientTape functionality is similar to the GradientTape 
# provided in tensorflow 2, but it is less powerful. For instance, pyClarion 
# GradientTapes cannot be stacked to compute higher-order derivatives. 


###################
### Basic Usage ###
###################

# To calculate gradients, we must initialize diffables in the context of a 
# GradientTape instance. On initialization, a diffable value will automatically 
# register itself to the GradientTape in the current context, if such a tape 
# exists. To use a pre-initialized diffable in a new gradient tape, assuming 
# the diffable does not already belong to another tape, we use the context-aware 
# diffable.register() method. If a diffable belongs to a preexisting tape, it 
# must be released before being used with another tape. Diffables get released 
# when a preexisting tape is explicitly reset, when the tape gets deleted, or 
# when a call is made to tape.gradients() on a non-persistent tape. 

with GradientTape() as tape:

    a = diffable(2)
    b = diffable(5)
    c = a * b

# To get the gradients of a and b with respect to c, it is sufficient to call 
# the tape.gradients() method.

grads = tape.gradients(c, [a, b])

# The return value of gradients() mimics the structure of the variable 
# containers that are passed to it, assuming these are either lists or tuples 
# of diffables or DiffableNumDict instances. It is possible to pass multiple 
# containers to a single call to tape.gradients(), the gradients are returned 
# in an order that respects the original arguments.

print("grad(c, [a, b]) = {}".format(repr(grads)))
print()


##############################################
### Gradient Descent with DiffableNumDicts ###
##############################################

# DiffableNumDicts differ from regular pyClarion NumDicts in that they are 
# dtype-locked to diffables and provide some specialized functionality for 
# working with diffables and implementing gradient-descent algorithms.

# DiffableNumDicts, by default, do not have default values. More importantly, 
# setting values on DiffableNumDicts has specialized behavior: if the key 
# already exists, the associated diffable will be modified inplace (i.e., its 
# `val` attribute will be updated). Similar behavior is followed for 
# adjustments to default values. Finally, the convenience method 
# DiffableNumDict.register() makes it easy to register all values in a 
# pre-initialized DiffableNumDict with a gradient tape. 

# For this example, we will create a linear-regression-like setup, with an 
# input x, an output y, and parameters w. 

x = DiffableNumDict({"x1": 2.0, "x2": .7, "b": 1.0})
y = DiffableNumDict({"y": 6.3})

w = DiffableNumDict({"x1": 3.0, "x2": -0.4, "b": .4})

# To get a persistent gradient tape, we simply pass in True as the value for 
# the `persistent` parameter to the tape constructor. Note that we must 
# register all diffables that paricipate in the computation, and not only those 
# for which we would like gradients. Furthermore, any constants must also be 
# instantiated as diffables and be registered.

with GradientTape(persistent=True) as tape:
    x.register()
    y.register()
    w.register()
    est = DiffableNumDict({"y": nd.val_sum(x * w)})
    loss = nd.val_sum((est - y) ** diffable(2.0)) / diffable(len(y))

# When the tape is in persistent mode, calling tape.gradients() will, by 
# default, execute a forward pass to ensure that the gradients are fresh. The 
# forward pass can be blocked, if needed.

print("A simple case of gradient descent (with persistent tape):\n")
for i in range(10):
    grads = tape.gradients(loss, w)
    print("Loss on iteration {}:".format(i+1), float(loss))
    w -= diffable(0.01) * grads


##################
### CONCLUSION ###
##################

# The automatic differentiation tools offered by pyClarion are relatively 
# simple and minimalistic. These tools are provided to support training 
# basic neural networks as required by Clarion theory while keeping the library 
# self-contained.

# For more sophisticated neural network models, it is best to integrate 
# pyClarion with a dedicated deep-learning/autodiff library by, e.g., writing 
# Propagator classes that wrap deep neural networks implemented using such 
# libraries.
