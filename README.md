# Likert Prediction using Decision Trees and Random Forests

The problem statement is predicting Likert scores (between 1-9) given to Microsuturing performed by trainees by senior doctors. I had already used linear regression for this task, and now have implemented decision trees and random forests for this purpose.

I first implemented a decision tree for the binary classification problem from scratch. We then made use of the scikit-learn library in Python to experiment and find the best set of hyper-parameters to maximize the accuracy of our model. We experiment with various boosting techniques such as Gradient Boosting and XGBoost.

A decision tree is a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. Decision tree learning employs a divide-and-conquer strategy by conducting a greedy search to identify the optimal split points within a tree. This process of splitting is then repeated in a top-down, recursive manner until all, or the majority of records have been classified under specific class labels.
