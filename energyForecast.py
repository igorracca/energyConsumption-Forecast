#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


# In[2]:


DEBUG = False
MAX_DEPTH_TUNNING_PLOT = False
K_FOLD_CROSS_VALIDATION = False


# In[3]:


# Load the features 
X = pd.read_csv("features.csv")
X
#X.describe()


# In[4]:


# Load the labels
y = pd.read_csv("consumption.csv", header=None)
y
y.describe()


# In[5]:


# Split the dataset - train test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[6]:


# Instantiate and fit Decision Tree Regressor
regressor = DecisionTreeRegressor(max_depth=18,
                                  min_samples_split=40)
regressor.fit(X_train, y_train)
#regressor.score(X_train.head(240), y_train.head(240))


# In[7]:


# Predict on test data
y_pred = regressor.predict(X_test)


# In[8]:


# List Actual and Predicted values
def cmp_prediction():
    import itertools
    y_test_arr = list(itertools.chain(*y_test.values)) #convert dataframe to array

    # Compare y and y^ 
    df_cmp = pd.DataFrame({'Actual':y_test_arr, 'Predicted':y_pred})
    
    return df_cmp
    
if DEBUG:
    cmp_prediction()


# In[9]:


# Calculate and print the Mean Absolute Error and Mean Squared Error
def calc_MSE_MAE(y_test, y_pred):
    print('MSE:', mse(y_test, y_pred))
    print('MAE:', mae(y_test, y_pred))

calc_MSE_MAE(y_test, y_pred)


# In[10]:


if MAX_DEPTH_TUNNING_PLOT:
    import matplotlib.pyplot as plt

    max_depths = range(5, 20)
    training_error = []
    for max_depth in max_depths:
        model_1 = DecisionTreeRegressor(max_depth=max_depth)
        model_1.fit(X, y)
        e = mse(y, model_1.predict(X))
        training_error.append(e)

    testing_error = []
    for max_depth in max_depths:
        model_2 = DecisionTreeRegressor(max_depth=max_depth)
        model_2.fit(X_train, y_train)
        testing_error.append(mse(y_test, model_2.predict(X_test)))

    plt.figure(figsize=(10,5)) 
    plt.plot(max_depths, training_error, color='blue', label='Training error')
    plt.plot(max_depths, testing_error, color='green', label='Testing error')
    plt.xlabel('Tree depth')
    plt.axvline(x=13, color='orange', linestyle='--')
    plt.annotate('optimum = 13', xy=(14, 6e+15), color='red')
    plt.ylabel('Mean squared error')
    plt.title('Hyperparameter Tuning - Max_Depth', pad=15, size=15)
    plt.legend()

    plt.savefig('error.png')


# In[11]:


if K_FOLD_CROSS_VALIDATION:
    from sklearn.model_selection import GridSearchCV

    model = DecisionTreeRegressor()

    gs = GridSearchCV(model,
                      param_grid = {'max_depth': range(10, 20),
                                    'min_samples_split': range(10, 60, 10)},
                      cv=5,
                      n_jobs=1,
                      scoring='neg_mean_squared_error')

    gs.fit(X_train, y_train)

    print(gs.best_params_)
    print(-gs.best_score_)

