# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Liu Zongyuan

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
There was no negative prediction when I checked the predictions with describe() funcion. So I could submit the predictions successfully.

### What was the top ranked model that performed?
WeightedEnsemble_L3

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The number of total rentals for workingdays and non-holidays is more than that for non-workingdays and holidays. In addition, one would expect a greater demand for rental bikes during office hours such as mornings or evenings. So one can split datetime into date and time to feed the model with the corresponding features. Below are the additional features I added:

```
year,
month,
week,
day,
hour,
saturday,
sunday,
```

### How much better did your model preform after adding additional features and why do you think that is?
Before adding new features, the initial value of rmse with autogluon is 1.80690. I tried adding different features to see if kaggle's results improved:

1. After adding the new features of `year, month, week, day, hour`, the rmse is 0.66850;
2. After adding new features such as `year, month, week, day, dayofweek, hour, saturday, sunday`, the rmse is 0.69380;
3. After adding new features such as `year, month, day, hour, perform map processing on year - {2011:0, 2012:1}`, rmse is 0.67641;

Obviously, the first feature combination has the best improvement with an increase of 63% dropping from 1.8069 to 0.6685 of rmse.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?

For Autogluon's hyperparamter optimization, I choose the snippet script on the official website, and then make a comparison on this basis

```
import autogluon.core as ag

nn_options = {  # specifies non-default hyperparameter values for neural network models
    'num_epochs': 10,  # number of training epochs (controls training time of NN models)
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
}

gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {  # hyperparameters of each model type
                   'GBM': gbm_options,
                   'NN_TORCH': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                  }  # When these keys are missing from hyperparameters dict, no models of that type are trained

time_limit = 10*60  # train various models for ~2 min
num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
    'num_trials': num_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}

```

hpo1: set `time_limit = 10*60`, rmse is 0.46777;
hpo2: Set `num_epochs=100` instead of 10 for nn_options, the others are the same as hpo1, rmse is 0.46521;
hpo3: set `num_trials=10` instead of 5, others are consistent with hpo1, rmse is 0.46876;
hpo4: set `time_limit = 10*60` and `auto_stack=True`, rmse is 0.47746;

The rmse of hpo2 is the best, which drops from 0.6685 to 0.4652, a 30.41% improvement.


### If you were given more time with this dataset, where do you think you would spend more time?

I want to do more feature engineering like adjusting for the seasons or working hours;
Also, I would like to try models other than Autogluon and do hyperparameter tuning.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|score|
|--|--|
|initial|1.80690|
|add_features|0.6685|
|hpo|0.4652|


### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](model_test_score.png)

## Summary
Autogluon is an easy-to-use, super-powerful package that can implement the entire ML workflow with just a few lines of code, including input data such as tabular, text, and images.
Nevertheless, feature engineering is still very necessary to improve its performance. For example, in our project, we split the year, month, day, hour from datetime column. Compared with the initial ones without feature extraction, it improves the performance of the model by up to 63%.

There is no complete official documentation for Autogluon's hyperparameter optimization, and we look forward to corresponding updates in the future.
