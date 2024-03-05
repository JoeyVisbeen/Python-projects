import numpy as np

def sigmoid(numb):
    return 1/ (1 + np.exp(1) ** -numb)

def sigmoid_vec(vec):
    arr = np.zeros(vec)
    x_wb = np.dot(w,x) + b
    for i in range(len(vec)):
        arr[i] = 1/ (1 + np.exp(1) ** -x_wb)
    return arr

# a_in/x = np.array([200, 17])
# W = np.array([[1, -1, 3],
#               [2, 4, -1]])
# b = np.array([-1, 2, 3])
def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for i in range(units):
        w = W[:,i]
        z = np.dot(w, a_in) + b[i]
        a_out[i] = sigmoid(z)
    return a_out

# A_in/X = np.array([200, 17])
# W = np.array([[1, -1, 3],
#               [2, 4, -1]])
# B = np.array([-1, 2, 3])
def dense_vec(A_in, W, B):
    Z = np.matmul(A_in,W) + B
    A_out = sigmoid_vec(Z)
    return A_out

# vec multiplication
# | 1 , 2 | 
# | 3 , 1 |  .  | 1 , 3 , 5 , 5 |
# | 4 , 2 |     | 2 , 5 , -3, -2|
#
# the calculation above only work because:
# the first matrix is 3 X 2 and the second is 2 X 4
# Because of the 2's

# common activation functions for the output layer
# Sigmoid hoogdoorlaat filter g(z) = 1/1+e^-z   : use for logistic NN (binary class)
# Linear g(z) = z                               : use for stockprices for instance (+ and -)
# ReLU: g(z) = max(0,z)                         : Predict a price, because price cant be -

# common activation functions for the hidden layers
# ReLU = most common (Faster, flat only below 0)
# sigmoid hardly ever
# Don't use lineair in hidden layers 

# A lineair func of a lineair func is a lineair func

# softmax
# a1 = e^z1 / e^z1  + e^z2 + e^z3
# a2 = e^z2 / e^z1  + e^z2 + e^z3
# a3 = e^z3 / e^z1  + e^z2 + e^z3
#
# general
# zj = Wj . X + bj
# aj = e^zj / summation of e^zk (k =1 til n)
#
# cost of sogftmax
# if y = j loss = -log(aj)

def softmax(z):
    ez = np.exp(z)
    sm = ez/np.sum(ez)
    return sm

# adam algorithm
# multiple alpha paramters for gradient descent. 
# If the directions seems similar increase the alpha

# Implement adam for tensorflow
# model.compile(optimizer=tf.keras.optimizer.Adam(learning_rate=1e-3),
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Convolutional layers
# don't give the entire set of input data to every neuron