import numpy as np

class Node:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, threshold=None, leaf_probability=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.threshold = threshold
        self.leaf_probability = leaf_probability
        self.left = left
        self.right = right
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(prediction={self.prediction}, prob={self.leaf_probability:.2f})"
        else:
            return f"Node(feature={self.feature_index}, threshold={self.threshold:.2f}) Left[{self.left}] Right[{self.right}]"
        
    def get_feature_name(self):
        if self.feature_index is None:
            return "Unknown feature"
        
        # We hard code the values in this simple case, 
        # where we know the first feature is the exam 1 score and the second feature is the exam score 2
        if self.feature_index == 0:
            return "Exam 1"
        elif self.feature_index == 1:
            return "Exam 2"
        else:
            return "Unknown feature"


def load_data(filename='ex2data1.txt'):
    """
    Reads and parses the data from the data text file.
    Returns
        X - (m, n) matrix with input features
        y - (n,) target vector
        m - total number of samples (100 in this case)
        n - total number of dimensions (2 in this case)
    """
    print('Loading data from ', filename)
    data = np.loadtxt(filename, delimiter=',')

    # Split data into features and target
    X, y = data[:, 0:2], data[:, 2]

    # Get dimensions
    m, n = X.shape
    return X, y, m, n

def calculate_threshold_candidates(feature):
    """
    Calculates potential threshold candidates for 
    the given data in a feature. This will return the midpoint
    of every consecutive item in order. 
    For example, using the sorted data like 30.28, 35.84, 60.18, 79.03,
    we calculate three 3 midpoints. 
    - 1st Midpoint: (30.28+35.84)/2 
    - 2nd midpoint: (35.84+60.18)/2 
    - 3rd midpoint: (60.18+79.03)/2

    Inputs:
        feature - (n,) vector with the data points for one feature (e.g., exam 1 scores)

    Output:
        midpoints - (n-1,) vector with the candidate threshold for testing each value 
        on splitting the data. 
    """
    print('Calculating midpoints for feature input with dimention', feature.shape)

    # Sort
    feature_sorted = np.sort(feature)

    # Calculate the midpoints x_i = (x_i + x_i+1) / 2 in vector form
    # Get all the elements except the last one
    feature_firsts = feature_sorted[:-1]

    # Get all the elements
    feature_seconds = feature_sorted[1:]

    # Calculate all midpoints in vector form
    midpoints = (feature_firsts + feature_seconds) / 2
    return midpoints

def calculate_entropy(targets):
    """
    Calculates entropy.
    For each possible value of the target vector y, with are
    1-Admitted and 0-Not Admitted, we need to calculate the proportion
    (number of instances/total sample size).
    p_1 - proportion of 1-Admitted - (# of 1s/n)
    p_0 - proportion of 0-Not Admitted - (# of 0s/n)
    The total entropy is equal to 
    h = -1 * (p_1 * log_2(p_1) + p_0 * log_2(p_0))

    Inputs:
        targets - a vector with the target values for a given data set, the data set
        could the original data as a whole or split values of it.

    Outputs:
        entropy - scalar value that determines the entropy of the given data
    """
    # Get the size of the data
    m = len(targets)

    # Make sure to return 0 for an empty array and not NaN
    if m == 0:
        return 0.0
    
    # The sum of the target equals the number of 1s
    positive_count = np.sum(targets)
    negative_count = m - positive_count

    # Get proportions
    p1 = positive_count / m
    p0 = negative_count / m

    # Calculate entropy
    # entropy = -1 * (p1 * np.log2(p1) + p0 * np.log2(p0)), 
    # but skip any proportion term that is 0, since 0*log(0)=0
    # This is to avoid getting nan value
    entropy = 0
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    return entropy

def calculate_conditional_entropy(X, y, threshold, feature_index):
    """
    For a given threshold and feature (using the column index), 
    calculate the conditional entropy when spliting the data

    The conditional entry is calculated as follows:
    H(Y|X split by theta) = (n_top/n)*H(Y_top) + (n_bottom/n)*H(Y_bottom).

    Inputs: 
        X - (m x n) matrix with features
        y - (m,) target vector
        threshold - scalar to split the data into two
        feature_index - column index of the feature to take into account (base 0)

    Outputs:
        con_entropy - scalar for the conditional entropy H(Y|X split by theta)
        X_top - the top section of the divided data in X, where feature values >= theta
        X_bottom - the bottom section of the divided data in X, where feature values < theta
        y_top - the top section of the divided data in y, where feature values >= theta
        y_bottom - the bottom section of the divided data in y, where feature values < theta
    """

    n = len(y)

    # We need to split the data into top and bottom
    # First, Select the column from the input matrix
    feature_values = X[:, feature_index] # All rows, only the selected column

    # Create a mask for splitting the data into the upper side (top)
    # and bottom side using the threshold theta. These are boolean arrays [True, True, False,...]
    mask_top = (feature_values >= threshold)
    mask_bottom = (feature_values < threshold)

    # Split X into top and bottom based on the threshold
    X_top = X[mask_top]
    X_bottom = X[mask_bottom]

    # Split the target vector y into top and bottom based on the threshold
    y_top = y[mask_top]
    y_bottom = y[mask_bottom]

    # Calculate entropy for y_top and y_bottom
    y_top_entropy = calculate_entropy(y_top)
    y_bottom_entropy = calculate_entropy(y_bottom)

    # Count data points in each set
    y_top_count = len(y_top)
    y_bottom_count = len(y_bottom)

    # Canculate conditional entropy
    con_entropy = (y_top_count/n)*y_top_entropy + (y_bottom_count/n)*y_bottom_entropy
    return con_entropy, X_top, X_bottom, y_top, y_bottom

def calculate_best_information_gain(X, y, x1_midpoints, x2_midpoints):
    """
    Calculates all the conditional entropy for each threshold for each feature
    and finds the value with the lowest entropy. This makes the information gain higher.

    So, for each midpoint (theta) for each feature, calculate the information gain IG:
    IG(Y|theta) = H(Y) - H(Y|X split at theta)
    """
    # Calculate original entropy
    base_entropy = calculate_entropy(y)
    print('Base entropy:', base_entropy)

    # Variables to keep track of the best split
    max_info_gain = float('-inf')
    best_feature_index = None
    best_threshold = None
    min_entropy = float('inf')
    min_X_top = X
    min_X_bottom = X
    min_y_top = y
    min_y_bottom = y

    # Find the best threshold for the first feature (Exam 1 scores)
    for theta in x1_midpoints:
        conditional_entropy, X_top, X_bottom, y_top, y_bottom = calculate_conditional_entropy(X, y, theta, 0)
        info_gain = base_entropy - conditional_entropy
        if info_gain > max_info_gain:
            best_feature_index = 0
            best_threshold = theta
            max_info_gain = info_gain
            min_entropy = conditional_entropy
            min_X_top = X_top
            min_X_bottom = X_bottom
            min_y_top = y_top
            min_y_bottom = y_bottom
    
    # Find the best threhold for the second feature (Exam 2 scores)
    for theta in x2_midpoints:
        conditional_entropy, X_top, X_bottom, y_top, y_bottom = calculate_conditional_entropy(X, y, theta, 1)
        info_gain = base_entropy - conditional_entropy
        if info_gain > max_info_gain:
            best_feature_index = 1
            best_threshold = theta
            max_info_gain = info_gain
            min_entropy = conditional_entropy
            min_X_top = X_top
            min_X_bottom = X_bottom
            min_y_top = y_top
            min_y_bottom = y_bottom

    return best_threshold, best_feature_index, max_info_gain, min_entropy, min_X_top, min_X_bottom, min_y_top, min_y_bottom

def majority(targets):
    """
    Returns the majority class label from the target vector.
    This works for binary classification (0s and 1s) only.

    Inputs:
    - targets: A numpy array of shape (n_samples,) containing the target labels (0s and 1s).

    Outputs:
    - The majority class label (0 or 1).
    """
    total = len(targets)
    midpoint = total / 2
    positive_count = np.sum(targets)
    if positive_count >= midpoint:
        return 1
    else:
        return 0
    
def target_value_probability(targets, value):
    """
    Calculates the proportion of a specific target value in the given target vector (m,).
    Inputs:
    - targets: A numpy array of shape (m,) containing the target labels (0s and 1s).
    - value: The specific target value (0 or 1) for which to calculate the proportion.
    """
    total = len(targets)
    positive_count = np.sum(targets)
    negative_count = total - positive_count
    if value == 1:
        return positive_count / total
    else:
        return negative_count / total


def build_decision_tree(X, y):
    """
    Builds a decision tree classifier.
    It recursively splits the data based on feature thresholds to create a tree structure.
    The thresholds are calculated using the training data: it will find the midpoints for the consecutive feature values
    and test them as potential splits.

    Inputs:
    - X: A numpy matrix (m x n) containing the feature values.
    - y: A numpy vector (m,) containing the target labels (0s and 1s).

    Output:
    - root: The root node of the decision tree.
    """
    max_depth = 100

    def build_decision_tree_helper(X_current, y_current, level):

        # Stop and make a leaf node 
        # if the current node has all the labels in y identical (entropy=0)
        node_entropy = calculate_entropy(y_current)
        if node_entropy <= 0:
            prediction = majority(y_current)
            return Node(is_leaf=True, prediction=prediction, leaf_probability=target_value_probability(y_current, prediction))

        exam1_feature = X_current[:, 0] # Select all rows, first column
        exam2_feature = X_current[:, 1] # Select all rows, second column

        # Calculate midpoints for each feature
        exam1_midpoints = calculate_threshold_candidates(exam1_feature)
        exam2_midpoints = calculate_threshold_candidates(exam2_feature)

        best_threshold, best_feature_index, max_info_gain, conditional_entropy, X_top, X_bottom, y_top, y_bottom = calculate_best_information_gain(X_current, y_current, exam1_midpoints, exam2_midpoints)
        print('For level', level+1,', the best threshold is ', best_threshold, 'for feature', best_feature_index, '. IG:', max_info_gain)


        # Stop and make a leaf if Reached max level
        if level >= max_depth:
            return Node(is_leaf=True, prediction=majority(y_current), leaf_probability=np.mean(y_current))

        # Stop and make a leaf if No candiates, meaning all features are identical (max IG <= 0)
        if max_info_gain <= 0:
            return Node(is_leaf=True, prediction=majority(y_current), leaf_probability=np.mean(y_current))
        
        # Add a condtion node
        root = Node(is_leaf=False, feature_index=best_feature_index, threshold=best_threshold)
        
        # Process the right (selected feature >= best_threshold)
        root.right = build_decision_tree_helper(X_top, y_top, level + 1)

        # Process the left (selected feature < best_threshold)
        root.left = build_decision_tree_helper(X_bottom, y_bottom, level + 1)
        
        return root

    # Build tree starting with all the data as a whole        
    return build_decision_tree_helper(X, y, 1)

def print_tree(root):
    """
    Prints the decision tree in an easy to read tree structure.
    Inputs:
    - root: The root node of the decision tree.
    """
    print('Tree structure:')
    if root is None:
        print('No tree to print')

    def print_tree_helper(node, indent):
        if node is None:
            return
        
        if node.is_leaf:
            print(f"{indent}Leaf: Prediction={node.prediction}, Proportion={node.leaf_probability:.2f}")
            return
        
        # Print decision node
        feature_name = node.get_feature_name()
        print(f"{indent}|{feature_name}")
        print(f"{indent}|- < {node.threshold:.2f}:")
        print_tree_helper(node.left, indent + "|  ")
        print(f"{indent}|- >= {node.threshold:.2f}:")
        print_tree_helper(node.right, indent + "|  ")

    print_tree_helper(root, "")

def predict(tree, x):
    """
    Predicts the class label for a single sample using the trained decision tree.
    Returns the predicted label and the proportion of the predicated label inside the leaf node.
    For example, if the predicted label 1 is and the leaf node has samples with only 1s, then the proportion would be 1.0.
    For example, if the predicted label is 0 and the leaf node has only 3 samples with (0, 0, 1), then the proportion would be 0.6667.

    Inputs:
    - tree: The root node of the decision tree.
    - x: A numpy array of shape (n_features,) containing the feature values for the sample.

    Outputs:
    - prediction: The predicted class label (0 or 1).
    - proportion: The proportion of the predicted class inside the leaf node.
    """

    # Traverse the tree
    current_node = tree

    # Loop until we reach a leaf node
    while not current_node.is_leaf:

        # Check the feature value and compare it to the threshold
        if x[current_node.feature_index] >= current_node.threshold:

            # Move to the right node
            current_node = current_node.right
        else:

            # Move to the left node
            current_node = current_node.left

    # We reached a leaf node. Return the prediction
    return current_node.prediction, current_node.leaf_probability

# Get features matrix (X), target vector (y) and dimensions
X, y, m, n = load_data()

root = build_decision_tree(X, y)
print('Tree built.')
print_tree(root)

print('Predicting label for student with scores (Exam 1: 45, Exam 2: 85):')
prediction, proportion = predict(root, np.array([45, 85]))
print('Prediction:', prediction)
