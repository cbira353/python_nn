# Practice coding for creating a Neural Net
# using Numpy simulating the exclusive OR function
# with two inputs and one output

# import Numpy
import numpy as np

# a function definition of the sigmoid function, activation function

def nonlin(x, deriv=False):
    if (deriv == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

# input data
X = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])


# output data
y = np.array([[0],
             [1],
             [1],
             [0]])


# seed for random generator so that results
# will be deterministic

np.random.seed(1)

# initialize weights to the random values. syn0 are
# weights between the input layer and hidden layer. It's
# 3x4 matrix (two input weights plus bias term) and
# four nodes in the hidden layer. It's a 4x1 matrix because
# there are 4 nodes and one output. Weights are initiated
# randomly

# synapses

syn0 = 2*np.random.random((3,4)) -1  # 3x4 matrix of weights

syn1 = 2*np.random.random((4,1))

# training step

for j in range(90000):

    # calculate forward through the network
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = y - l2
    if (j % 10000) == 0: #print error every 10000 steps
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error*nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)


    # update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print ("Output after training")
print (l2)


# final output should closely approximate the true
# output [0,1,1,0].  By increasing num of iterations (currently 60000), final output will be even closer
