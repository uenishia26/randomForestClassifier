# randomForestClassifier
This is a simple randomForestClassifier utilizing plotting tools + adjusting hyperparameters + understanding RFC/Bagging/Bootstrapping/Ensemble methods

#Strengths and weakness of randomForestClassifier 
Strengths: 
randomForestClassifer is an ensemble learning method 
    - Ensemble learning methods: Uses multiple models to make a prediction, thus minimizes overfitting
Can be used for both Regression/Classification   

Weakness: 
  - Not very good for imbalanced datasets (However, we will talk about ways around this)

#Ways around imabalanced dataset w/h Random Forest
  First, we must cover some vocab. Random Forest Classifier (rfc) is a type of ensemble method: uses multile trees to make a final prediction.
  rfc is also a bagging algorithm/model: Trains multiple classifiers on boostrapped samples: (Randomly selected data from dataset w/h replacement)
  #Whats the diffrence between Bagging/rfc? 
    Bagging uses bootstrapped samples and then uses these samples to train multiple ML models before making a prediction
    On the other hand, rfc uses boostrapped samples to train the trees but then further randomly splits the sample when splitting the node 

#Do you see the problem?
If rfc uses bootstrapped samples to make train ML model, and there is a class that dominates the dataset, the model is going to become very bad
at predicting the minority class and very good at predicting the minority class. 

How to overcome this issue?  
  OverBagging: OverSample the minority class 
  UnderBagging: Undersample the majority class
  Weights: Add weights to minority or majority class
  Balancedrfc: Ensures each bootstrapped sample has the same amount of data as the minority class (could cause underfitting)
  
  
  
