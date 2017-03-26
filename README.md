Practice coding for creating a Neural Net
using Numpy simulating the exclusive OR function
with two inputs and one output. This one is a neural net explained by Siraj Raval in a tutorial. Main import is Numpy. 
```
import numpy as np
```


Define a sigmoid function as an activation function. Will see in the future the other types of non-linearity to choose. 

A neat function definition where one outcome will occur. If the ```deriv=True``` flag is passed in as an argument, the function sintead calculates the derivative of the function, which is used in the error backpropagation step.

```
def nonlin(x, deriv=False):
    if (deriv == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))
```


Now create input matrix, with the third column for accomodating a bias term.
```
X = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])
```

The output of the exclusive OR functio follows. Output data:
```
y = np.array([[0],
             [1],
             [1],
             [0]])
```

Seed for random generator so that results will be deterministic, ie, will retrurn the same random numbers each time, so that results can be compared with each run, especially in debugging.
```
np.random.seed(1)
```

Iinitialize weights to the random values. syn0 are weights between the input layer and hidden layer. It's 3x4 matrix (two input weights plus bias term) and
four nodes in the hidden layer. It's a 4x1 matrix because
there are 4 nodes and one output. Weights are initiated
randomly

Synapses
```
#a 3x4 matrix of weights; two inputs and 1 bias x 4 nodes in the hidden layer
syn0 = 2*np.random.random((3,4)) -1  # 3x4 matrix of weights

# a 4x1 matrix of weights (4 nodes x 1 output)
syn1 = 2*np.random.random((4,1))
```

The training loop. Output will show the error between the model and expected. The error decreases. 

```
for j in range(60000):

    # calculate forward through the network
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1, syn1))

#back propagation of errors using the chain rule.
    l2_error = y - l2
    if (j % 10000) == 0: #print error every 10000 steps
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error*nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)
```

```
    # update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
```


Prints results. The final output should closely approximate the true
output [0,1,1,0].  By increasing num of iterations (currently 60000), final output will be even closer.

```
print ("Output after training")
print (l2)
```


Output (Error rate decreases per epoch)
```
python_nn$ python3 PythonNNExampleMBI.py 

Error: 0.500247347159
Error: 0.00963923396637
Error: 0.00657590888548
Error: 0.00528330150352
Error: 0.0045305461704
Error: 0.00402433390416
Error: 0.00365454280081
Error: 0.00336941787381
Error: 0.00314103957269
Output after training
[[ 0.0015793 ]
 [ 0.99738176]
 [ 0.99656787]
 [ 0.00418183]]
```



