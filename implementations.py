#Script with all functions implemented 
import  numpy as np 

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    
    #MSE 
    loss = 1/(2*len(y))*np.sum((y - np.dot(tx, w))**2)
    
    #MAE
    #loss = 1/len(y) * np.sum(abs(y - np.dot(tx, w)))
    return loss

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    #compute gradient vector
    err = y - np.dot(tx,w)
    grad = -1/(len(y))*np.dot(np.transpose(tx),err )
    
    return grad


def mean_squares_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    weights = [initial_w]
    losses = []
    w = initial_w
    prev_loss=0
    for n_iter in range(max_iters):
        
        #compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)  
        
        #break the loop when convergence is reached
        if abs(prev_loss-loss)<0.0000001: 
            print(f"Convergence reached at iteration {iter_}s and gamma {gamma}")
            break 
    
        #update w by gradient
        w = w - gamma*gradient
        
        # store w and loss
        weights.append(w)
        losses.append(loss)
        
        if n_iter >=1 : 
            prev_loss = loss
        
    loss = np.min(losses)
    iter_ = losses.index(loss)
    w = weights[iter_]

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Performs stochastic gradient descent (SGD) to minimize the mean squared error (MSE) loss.

    Args:
        y (np.array): Target variable of shape (N,).
        tx (np.array): Feature matrix of shape (N, D), where N is the number of samples, and D is the number of features.
        initial_w (np.array): Initial weights of shape (D,).
        max_iters (int): Maximum number of epochs (iterations over the entire dataset).
        gamma (float): Learning rate (step size) for SGD.

    Returns:
        tuple: Updated weights (np.array) of shape (D,) after SGD optimization and final MSE loss (float).
    """
    seed = 42 
    np.random.seed(seed)
   
    # Initialize weights and track previous loss
    w = initial_w # Ensure w is (D, 1)
    temp_loss = float('inf')  # Start with an infinitely high loss for comparison
    tolerance=1e-6

    for iter_ in range(max_iters):
        # Shuffle the data at the beginning of each iteration for better SGD performance
        indices = np.random.permutation(len(y))
        y_shuffled = y[indices]
        tx_shuffled = tx[indices]

        # Update weights for each data point in the shuffled dataset
        for i in range(len(y)):
            e = y_shuffled[i] - np.dot(tx_shuffled[i, :], w)
            gradient = -tx_shuffled[i, :] * e  # Reshape to ensure correct dimensions
            w = w - gamma * gradient  # Update the weights

        # Compute the loss after one full pass (epoch) through the dataset
        #gamma = gamma / (1 + iter_ * decay_rate)
        loss = compute_loss(y, tx, w)
        
        if np.isnan(loss):
            print("NaN encountered in loss, stopping training.")
            break
        
        # Check for convergence based on the change in loss
        if abs(temp_loss - loss) < tolerance:
            print(f"Convergence reached at iteration {iter_} with tolerance {tolerance} and gamma {gamma}")
            break
        
        # Update previous_loss for the next epoch
        temp_loss = loss
    
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    # returns mse, and optimal weights
    w = np.linalg.solve(tx.T @tx, tx.T @y)    
    loss= compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    N, D = tx.shape
    lambda_prime = lambda_*N*2
    I = np.eye(D)
    w = np.linalg.solve(tx.T@tx + lambda_prime*I, tx.T@y)
    loss = compute_loss(y, tx, w)
    return w, loss


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    sigma = (1+np.exp(-t))**(-1)
    return sigma


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        max_iters : scalar number
        gamma : scalar number

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)
    """
    losses = []
    weight = []
    w = initial_w
    threshold = 1e-8
    
    for iter in range(max_iters):
        
        loss = -1/len(y) * np.sum(y * np.log(sigmoid(tx@w)) + (1-y)* np.log(1 - sigmoid(tx@w)))
        gradient = 1/len(y) * tx.T @ (sigmoid(tx@w)- y)
        w = w - gamma * gradient

        # converge criterion
        losses.append(loss)
        weight.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    loss = np.min(losses)
    w = weight[losses.index(loss)]
    w= w.ravel()
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Performs regularized logistic regression using gradient descent.

    Args:
        y (np.array): Target variable of shape (N,), with binary labels (0 or 1).
        tx (np.array): Feature matrix of shape (N, D), where N is the number of samples, and D is the number of features.
        lambda_ (float): Regularization parameter to control overfitting.
        initial_w (np.array): Initial weights of shape (D,).
        max_iters (int): Maximum number of iterations for the gradient descent.
        gamma (float): Learning rate (step size) for gradient descent.

    Returns:
        tuple: Optimal weights (np.array) of shape (D,) after training, and minimum loss (float).
    """
    threshold = 1e-8
    losses = []
    weights = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss 
        loss = -1/len(y) * np.sum(y * np.log(sigmoid(tx@w)) + (1-y)* np.log(1 - sigmoid(tx@w)))
        loss = loss + (lambda_ / 2) * np.squeeze(w.T @ w)
        
        grad =  1/len(y) * tx.T @ (sigmoid(tx@w)- y) +  lambda_* w
        
        
        w = w - gamma * grad
        losses.append(loss)
        weights.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    loss = np.min(losses)
    iter_ = losses.index(loss)
    w = weights[iter_].ravel()
    return w, loss

def f1_score (y_true, y_pred) : 
    """
    Calculates the F1 score, a measure of a model's accuracy in binary classification, 
    based on precision and recall.

    Args:
        y_true (np.array): True binary labels of shape (N,), where each entry is either 0 or 1.
        y_pred (np.array): Predicted probabilities or labels of shape (N,). Values are 
                           thresholded at 0.5 to classify as either 0 or 1.

    Returns:
        float: The F1 score, ranging from 0 to 1, where 1 indicates perfect precision and recall.
    """
    y_pred_bin = (y_pred >= 0.5).astype(int)
    
    # Calcul des True Positives, False Positives et False Negatives
    TP = np.sum((y_true == 1) & (y_pred_bin == 1))
    FP = np.sum((y_true == 0) & (y_pred_bin == 1))
    FN = np.sum((y_true == 1) & (y_pred_bin == 0))
    
    # Calcul de la précision et du rappel
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calcul du F1 score
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
