### utils.py ###

import sys
import numpy as np
from matplotlib import pyplot

def plotData(X, y):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * in red for the positive examples and o in blue for the negative examples.
    
    Parameters
    ----------
    X : array_like
        An M x 2 matrix representing the dataset. 
    
    y : array_like
        Label values for the dataset. A vector of size (M, ).
    
    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.    
    """

    # Create New Figure
    fig = pyplot.figure()
    # ====================== YOUR CODE HERE ======================
    # NOTE: The instructions seems to contradict each other. 
    # The first says to use red * for positive and blue o for negative.
    # But the second says to use the option 'k*' and 'ko', which makes the points
    # grey in color, not red nor blue. 
    # I opted for the options 'r*' and 'bo' to use red stars and blue dots 
    # for a better visualization.

    first_exam_scores = X[:, 0] # All rows, first column
    second_exam_scores = X[:, 1] # All rows, second column

    # Boolean mask where y is equal to 1 for the admitted scores
    # This create an array of booleans like [True, True, False, ...]
    admitted_mask = (y == 1)
    # Get the boolean mask for the rejected scores
    rejected_mask = (y != 1)

    # Use the mask to select the items using numpy
    # Get the admitted scores for exam 1 and 2
    admitted_first_exam_scores = first_exam_scores[admitted_mask]
    admitted_second_exam_scores = second_exam_scores[admitted_mask]
    rejected_first_exam_scores = first_exam_scores[rejected_mask]
    rejected_second_exam_scores = second_exam_scores[rejected_mask]

    # Plot the data. Use the x-axis for the first scores, and y-axis for second scores
    pyplot.plot(admitted_first_exam_scores, admitted_second_exam_scores, 'r*')
    pyplot.plot(rejected_first_exam_scores, rejected_second_exam_scores, 'bo')
    # =============================================================


def sigmoid(z):
    """
    Compute sigmoid function given the input z.
    
    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector 
        or a 2-D matrix. 
    
    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.
        
    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    # convert input to a numpy array
    z = np.array(z)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    # Sigmoid function in vector form
    def sigmoid_fn(z):
        return 1 / (1 + np.exp(-z))
    
    # Apply the sigmoid function to each element
    # numpy vectorizes the operations
    g = sigmoid_fn(z)
    # =============================================================
    return g

def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression. 
    
    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector
        of shape (n+1, ).
    
    X : array_like
        The input dataset of shape (m x n+1) where m is the total number
        of data points and n is the number of features. We assume the 
        intercept has already been added to the input.
    
    y : arra_like
        Labels for the input. This is a vector of shape (m, ).
    
    Returns
    -------
    J : float
        The computed value for the cost function. 
    
    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
        
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to 
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================  
    # 1. Multiply inputs for each weight and apply sigmoid. h = sigmoid(X * theta)
    h = sigmoid(X @ theta)
    # 2. Compute the cost. Cost = y * log(h) + (1-y)*log(1-h)
    J = -(1.0/m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    # 3. Compute gradient. Using formula the w_new = w - n * (1/m) * X^T * (h-y), 
    # we compute the last term without the learning rate.
    grad = (1.0/m) * X.T @ (h-y)
    # =============================================================
    return J, grad

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5 
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)
    
    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).
    
    X : array_like
        The data to use for computing predictions. The rows is the number 
        of points to compute predictions, and columns is the number of
        features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X. 
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned 
    logistic regression parameters.You should set p to a vector of 0's and 1's    
    """
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Apply sigmoid to every input in X with the values of theta
    h = sigmoid(X @ theta)

    # Get a boolean mask for all the values equal or greater to 0.5
    # This result is an array with boolean values like [True, True, False, False, ...]
    mask = (h >= 0.5)

    # Convert boolean values to numeric values
    p = mask.astype(int) 
    # ============================================================
    return p

def plotDecisionBoundary(plotData, theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.

    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, ).
    """
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        pyplot.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        pyplot.xlim([30, 100])
        pyplot.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        pyplot.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        pyplot.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)

    pyplot.show()