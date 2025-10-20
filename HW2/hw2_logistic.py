## hw2_logistic.py ###

import os
import numpy as np
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

from utils import *
import argparse

def main(args):
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X, y = data[:, 0:2], data[:, 2]
    
    plotData(X, y)
    # add axes labels
    pyplot.xlabel('Exam 1 score')
    pyplot.ylabel('Exam 2 score')
    pyplot.legend(['Admitted', 'Not admitted'])
    ## plot should be displayed here ##
    pyplot.show()

    # Test the implementation of sigmoid function here
    z = 0
    g = sigmoid(z)
    print('g(', z, ') = ', g)

    # Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add bias/intercept term to X
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros(n+1)

    cost, grad = costFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))

    print('Gradient at initial theta (zeros):')
    print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost, grad = costFunction(test_theta, X, y)

    print('Cost at test theta: {:.3f}'.format(cost))

    print('Gradient at test theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))

    # set options for optimize.minimize
    options= {'maxiter': 400}

    # see documention for scipy's optimize.minimize  for description about
    # the different parameters
    # The function returns an object `OptimizeResult`
    # We use truncated Newton algorithm for optimization which is 
    # equivalent to MATLAB's fminunc
    # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
    res = optimize.minimize(costFunction,
                            initial_theta,
                            (X, y),
                            jac=True,
                            method='TNC',
                            options=options)

    # the fun property of `OptimizeResult` object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property
    theta = res.x

    # Print theta to screen
    print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))

    print('theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))

    # Plot Boundary
    plotDecisionBoundary(plotData, theta, X, y)

    #  Predict probability for a student with score 45 on exam 1 
    #  and score 85 on exam 2 
    prob = sigmoid(np.dot([1, 45, 85], theta))
    print('For a student with scores 45 and 85,'
        'we predict an admission probability of {:.3f}'.format(prob))

    # Compute accuracy on our training set
    p = predict(theta, X)
    print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))



args = { }
main(args)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--file', action = 'store', type = str, help = 'input file', default = None)

#     args = parser.parse_args()
#     if args.file is None or os.path.exists(args.file) == False:
#         print("Invalid input: %s"%(args.file))
#     else:
#         main(args)


# OUTPUT
