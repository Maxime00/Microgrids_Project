"Regression"

# Import function to create training and test set splits
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

from scipy.optimize import curve_fit
from scipy.optimize import least_squares



def regression_for_tile_weights(valueFunction, training_vf):

    prev_vf = valueFunction

    test_set_fraction = 0.33

    x = np.arange(len(training_vf.weights)).reshape(-1,1)
    y = np.array(training_vf.weights)

    # Alpha (regularization strength) of LASSO regression
    lasso_eps = 0.0001
    lasso_nalpha=20
    lasso_iter=5000
    
    # Min and max degree of polynomials features to consider
    #degree_min = 2
    #degree_max = 8
    degree = 6 #only run one regression to avoid wasting time
    
    # Test/train split
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=test_set_fraction)
    
    # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, 
    # run it for increasing degree of polynomial (complexity of the model)
    
    #for degree in range(degree_min,degree_max+1):   
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), 
        LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,
        normalize=True,cv=5))
    
    model.fit(X_train,y_train)
    test_pred = np.array(model.predict(X_test))
    #RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
    test_score = model.score(X_test,y_test)
    
    
    "Update Value Function"
    k = 0
    for index in X_test :                 
        valueFunction.weights[index] = test_pred[k]
        k +=1


    "Print"
    #print('RSE : ', RMSE)
    print('Test Scores : ', test_score)

    x= np.arange(len(training_vf.weights))
  

    return valueFunction





            