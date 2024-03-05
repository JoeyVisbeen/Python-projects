import copy, math
import numpy as np
import matplotlib.pyplot as plt

# training columns
x_train = np.array([[2104, 5, 1, 45], 
                    [1416, 3, 2, 40],
                    [852, 2, 1, 35]])
# training column we want to predict
y_train = np.array([460, 232, 178])

# bias
b_init = 785.1811367994083
# weight vector
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

# slower version
def predict_single_loop(x, w, b):
    # get the columns of x
    n = x.shape[0]
    # init return value
    p = 0
    for i in range(n):
        # for each item in the array of x and w multiply
        p_i = x[i] * w[i]
        p += p_i
    p += b # add the bias
    return p

x_vec = x_train[0,:]

# remember this is not optimal because gradient decent hasnt optimized weight and bias
#f_wb = predict_single_loop(x_vec, w_init, b_init)
#print(f_wb)

# faster version
def predict_numpy(x,w,b):
    p = np.dot(x,w) + b
    return p

#f_wb = predict_numpy(x_vec, w_init, b_init);
#print(f_wb)

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    cost /= (2 * m)
    return cost


def compute_cost_lin_reg(X, y, w, b, lambda_ = 1):
    m = X.shape[0]
    n = len(w)
    cost = 0.0
    reg = 0.0
    
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    for j in range(n):
        reg +=  w[j]**2 

    cost /= (2 * m)
    reg *= (lambda_/(2*m)) 

    return cost + reg

def compute_gradient_lin_reg(X, y, w, b, lambda_):
    m, n = X.shape[0]
    dj_dw = np.zeros((n,))
    dj_db = 0.
    reg = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*X[i,j]
        dj_db += err
    
    dj_dw /= m
    dj_db /= m

    for j in range(n):
        dj_dw[j] += lambda_ / m * w[j]

    return dj_dw, dj_db

# UNQ_C2
# GRADED FUNCTION: compute_cost


def compute_cost_logi_reg(X, y, w, b, lambda_ = 1):
    m = X.shape[0]
    n = len(w)
    cost = 0.
    reg = 0.

    for i in range(m):
        f_wb_i = 1/ (1 + np.exp(1) ** multi_negative(np.dot(w*X[i]) + b))
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost /= m
    
    for j in range(n):
        reg += w[j]**2
    reg *= (lambda_/(2*m))

    return cost + reg 

#cost = compute_cost(x_train, y_train, w_init, b_init)
#print(cost)

def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i,j]
        dj_db += err
    
    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db

#tmp_dj_db, tmp_dj_dw = compute_gradient(x_train, y_train, w_init, b_init)
#print(tmp_dj_db)
#print(tmp_dj_dw)

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)

        w -= (alpha * dj_dw)
        b -= (alpha * dj_db)

        if i<100000:
            J_history.append(cost_function(X, y, w, b))
        
        if i% math.ceil(num_iters/10) == 0:
            print("iteration " + str(i))
            print("cost " + str(J_history[-1])) 
    return w, b, J_history

initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
#print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
#m,_ = x_train.shape
#for i in range(m):
#    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

def multi_negative(x):
    neg = x * (-1)
    return neg

# cost function for logistic regression or classification
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        g_z_i = 1/ (1 + np.exp(1) ** multi_negative(z_i))
        cost +=  multi_negative(y[i]) * np.log(g_z_i) - (1-y[i]) * log(1- g_z_i)
    return cost / m

def gradient_logi_function(X, y, w, b):
    m, n = X.shape[0]
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        z_i = np.dot(w*X[i]) + b
        err = (1/ (1 + np.exp(1) ** multi_negative(z_i))) - y[i]
        for j in range(n):
            dj_dw[j] += (err * X[i,j])
        dj_db += err
    
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db
    

def gradient_desc_logistic(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = cost_function(X, y, w, b)

        w -= alpha*dj_dw
        b -= alpha*dj_db
    
        if i<100000:
            J_history.append(cost_function(X, y, w, b))

    return w, b, J_history