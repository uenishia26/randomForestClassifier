# randomForestClassifier

This is a simple RandomForestClassifier utilizing plotting tools, adjusting hyperparameters, and understanding RFC/Bagging/Bootstrapping/Ensemble methods.

## Strengths and Weaknesses of RandomForestClassifier

### Strengths:
- RandomForestClassifier is an ensemble learning method.
    - Ensemble learning methods use multiple models to make a prediction, thus minimizing overfitting.
- Can be used for both Regression and Classification.

### Weaknesses:
- Not very good for imbalanced datasets (However, we will discuss ways around this).

## Ways Around Imbalanced Datasets with Random Forest

First, we must cover some vocabulary. Random Forest Classifier (RFC) is a type of ensemble method: it uses multiple trees to make a final prediction. RFC is also a bagging algorithm/model: it trains multiple classifiers on bootstrapped samples (randomly selected data from the dataset with replacement).

### What's the difference between Bagging and RFC?
- Bagging uses bootstrapped samples and then uses these samples to train multiple ML models before making a prediction.
- RFC uses bootstrapped samples to train the trees but then further randomly splits the sample when splitting the node.

### Do you see the problem?
If RFC uses bootstrapped samples to train the ML model, and there is a class that dominates the dataset, the model will become very bad at predicting the minority class and very good at predicting the majority class.

### How to Overcome This Issue?
- **OverBagging**: Oversample the minority class.
- **UnderBagging**: Undersample the majority class.
- **Weights**: Add weights to the minority or majority class.
- **BalancedRFC**: Ensures each bootstrapped sample has the same amount of data as the minority class (could cause underfitting).

## Parameters/Hyperparameters of Random Forest Classifier

- **n_estimators**: Number of trees.
- **max_depth**: How much the tree can grow.
- **min_samples_split**: The number of samples a node must have in order to split (usually between 2 and 6).
- **min_samples_leaf**: Minimum number of samples a node must hold after getting split.
- **max_features**: Number of features to take into account for the best split.
- **max_leaf_nodes**: Prevents the splitting of the node (controls depth of the tree).

This README should give you a good understanding of RandomForest and how it works. The .py file will show all the different experiments and ways 
randomForest is used given the parameters and etc
    























  
  
  
