from matplotlib import pyplot
import numpy as np
import math

data = np.loadtxt('ex2data1.txt', delimiter=',')
X, y = data[:, 0:2], data[:, 2]

m, n = X.shape

# PLOT DATA
print(X[0, 0], X[0, 1])
print(y)

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

pyplot.plot(admitted_first_exam_scores, admitted_second_exam_scores, 'r*')
pyplot.plot(rejected_first_exam_scores, rejected_second_exam_scores, 'bo')
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not admitted'])
pyplot.show()


# def sigmoid(z):
#     return 1 / (1 + math.exp(z * -1))

# def sigmoid_fn(z):
#     return 1 / (1 + np.exp(-z))

# z_inputs = [-45, -10, -1, 0, 1, 4, 5, 7, 10, 30, 50]
# z_np = np.array(z_inputs)

# g = sigmoid_fn(z_np)

# print(g)
# print(sigmoid_fn(0))
# print(z_np.shape)
# print(z_np)

# m = z_np.shape[0]

# # Reshape z_np into a column vector
# z_np = z_np.reshape(m, 1)

# z_with_ones = np.concatenate([np.ones((m, 1)), z_np], axis=1)
# print(z_with_ones)

# # Add 1s 
# X = np.concatenate([np.ones((m, 1)), X], axis=1)
# # Initialize fitting parameters
# initial_theta = np.zeros(n+1)
# print('X=',X)
# print('theta = ', initial_theta)
