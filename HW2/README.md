
# Part 1

# Part 2 - Decision Trees

Using a decision tree model to predict the class (1-Admitted, 0-Not Admitted) for a student with 45 for the first exam and 85 for the second exam. 

The model is located at `decision_trees.py`. Run by simply calling `python decision_trees.py`. The program will do the following steps:

1. Load data from local file `ex2data1.txt` and create a (100x2) numpy matrix `X` and a target numpy vector (100,) `y`.
2. Build a descion tree by doing following:
    - Calculate the entropy at the given node (or root).
    - If the node entropy is 0, meaning that all the target values are the same, then create a leaf node for that target label.
    - Calculate the midpoints of the consecutive feature values for the Exam 1 scores. Midpoint: (y[i] + y[i+n])/2. These values will be used for testing as the threshold to split the data.
    - Do the same as above to calculate the midpoints for Exam 2.
    - For each midpoint for each feature, 
        - Split the data by the current feature threshold that we are testing.
        - Calculate information gain as `IG(Y|theta) = H(Y) - H(Y|X split at theta)`, where `H(Y)` is the entropy at the node and `H(Y|theta)` is the conditional entropy based on the split value theta. The conditional entropy is calculated as `H(Y|X split by theta) = (n_top/n)*H(Y_top) + (n_bottom/n)*H(Y_bottom)`
        - Keep track of the maximum information gain
    - Based on the maximum information gain, split the data by the selected threshold into two datasets: left and right.
    - Recursively start again to build a subtree for the dataset left.
    - Recursively start again to build a subtree for the dataset right.
3. Print the build decision tree in a human readable format.
4. Predict the target of a new data point (45, 85) by iterating the tree accordingly.

### Output
```
Loading data from  ex2data1.txt
Calculating midpoints for feature input with dimention (100,)
Calculating midpoints for feature input with dimention (100,)
Base entropy: 0.9709505944546686
For level 2 , the best threshold is  56.746261906407426 for feature 0 . IG: 0.2697300237806306
Calculating midpoints for feature input with dimention (65,)
Calculating midpoints for feature input with dimention (65,)
Base entropy: 0.6900703653284017
For level 3 , the best threshold is  43.11452006339603 for feature 1 . IG: 0.4304210723866662
Calculating midpoints for feature input with dimention (56,)
Calculating midpoints for feature input with dimention (56,)
Base entropy: 0.3013786435930858
For level 4 , the best threshold is  52.400730839267226 for feature 1 . IG: 0.1353275667565358
Calculating midpoints for feature input with dimention (11,)
Calculating midpoints for feature input with dimention (11,)
Base entropy: 0.8453509366224364
For level 5 , the best threshold is  71.48580052225253 for feature 0 . IG: 0.8453509366224364
Calculating midpoints for feature input with dimention (35,)
Calculating midpoints for feature input with dimention (35,)
Base entropy: 0.7219280948873623
For level 3 , the best threshold is  64.02978088788288 for feature 1 . IG: 0.24718229780225043
Calculating midpoints for feature input with dimention (17,)
Calculating midpoints for feature input with dimention (17,)
Base entropy: 0.9774178175281716
For level 4 , the best threshold is  40.347222359601375 for feature 0 . IG: 0.572838961141255
Calculating midpoints for feature input with dimention (9,)
Calculating midpoints for feature input with dimention (9,)
Base entropy: 0.7642045065086203
For level 5 , the best threshold is  45.966265416645335 for feature 0 . IG: 0.31976006206417584
Calculating midpoints for feature input with dimention (4,)
Calculating midpoints for feature input with dimention (4,)
Base entropy: 1.0
For level 6 , the best threshold is  82.9743184708675 for feature 1 . IG: 1.0
Tree built.
Tree structure:
|Exam 1
|- < 56.75:
|  |Exam 2
|  |- < 64.03:
|  |  Leaf: Prediction=0, Proportion=1.00
|  |- >= 64.03:
|  |  |Exam 1
|  |  |- < 40.35:
|  |  |  Leaf: Prediction=0, Proportion=1.00
|  |  |- >= 40.35:
|  |  |  |Exam 1
|  |  |  |- < 45.97:
|  |  |  |  |Exam 2
|  |  |  |  |- < 82.97:
|  |  |  |  |  Leaf: Prediction=0, Proportion=1.00
|  |  |  |  |- >= 82.97:
|  |  |  |  |  Leaf: Prediction=1, Proportion=1.00
|  |  |  |- >= 45.97:
|  |  |  |  Leaf: Prediction=1, Proportion=1.00
|- >= 56.75:
|  |Exam 2
|  |- < 43.11:
|  |  Leaf: Prediction=0, Proportion=1.00
|  |- >= 43.11:
|  |  |Exam 2
|  |  |- < 52.40:
|  |  |  |Exam 1
|  |  |  |- < 71.49:
|  |  |  |  Leaf: Prediction=0, Proportion=1.00
|  |  |  |- >= 71.49:
|  |  |  |  Leaf: Prediction=1, Proportion=1.00
|  |  |- >= 52.40:
|  |  |  Leaf: Prediction=1, Proportion=1.00
Predicting label for student with scores (Exam 1: 45, Exam 2: 85):
Prediction: 1
```

### Compare to logistical regression

Compare with the other model, when we plot the data we can see the tree predicts similarly to the linear regression model.
[See image](/HW2/decision_tree_prediction.png)