# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:05:41 2018

@author: Andrius Vabalas

Prerequisites: Numpy, Sklearn, Opearator, Scipy libraries. Using Anaconda is recommended as all required libraries are included: https://www.anaconda.com/

Select validation method, feature selector, classifier and data type (discriminative or non-discriminative) below. Refer to paper for description of these methods. 

Simulation results will be saved to the file "results.csv" in a current folder. 

"""
# SELECT VALUES:
VALIDATION = 1 # 1 - NESTED, 2 - N-FOLD, 3 - feature selection NESTED, paremeter tunning N-FOLD, 4 - feature selection N-FOLD, parameter tuning NESTED, 5 - Train/Test split
FEATURE_SELECTOR = 2 # 1 - SVM-RFE, 2 - T-test, 3 no-feature selecion
CLASSIFIER = 2 # 1 - SVM. 2 - logistic regression
DISCRIMINATIVE = 1 # 1 - Non-discriminative, 2 - Discriminative

# Import NECESARY LIBRARIES
import numpy as np
from sklearn.svm import SVC # SVM
from sklearn import linear_model # For logistic regression
import operator # for finding highest value in dictionary
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split  # for parameter optimisation using grid_search, for crossvalidation, for Kfold split 
par_grid={'C':[2,4,8,16,32,64,128],'gamma':[0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125]} # grid for SVM parameter optimisation C = 2^j , where j = 1, 2, ... 7 and  gamma=2^i, where i = -1, -2, ... -7.
par_grid2={'C':np.logspace(0, 4, 10),'penalty':['l1', 'l2']} # grid for LOGISTIC REGRESSION parameter optimisation
from sklearn.metrics import accuracy_score # for accuracy estimates
from scipy import stats


# SVM RFE FUNCTION

def SVM_RFE(features,labels):

    #prealocation
    Best_features = np.arange(0,features.shape[1]) # to retain feature order in the original feature space
    features_inner = features
    #loop starts here to select best features up to 20
    while features_inner.shape[1] > 20:
        model = SVC(kernel='linear', C=1)
        model.fit(features_inner,np.ravel(labels)) # fit linear model to inner features
        w = np.abs(model.coef_) # find feature weight vector
        worst_feature = w.argmin(axis=1)[0] # find lowest ranked feature
        features_inner = np.delete(features_inner,worst_feature,1) # remove worst feature from featurearray
        Best_features = np.delete(Best_features,worst_feature) # to retain labels of best features after the loop

    best_feature_recrd = {}
    accuracies= {}
    for n in list(range(12)):
        model = SVC(kernel='linear', C=1)
        model.fit(features_inner,np.ravel(labels)) # fit linear model to inner features
        w = np.abs(model.coef_) # find feature weight vector
        worst_feature = w.argmin(axis=1)[0] # find lowest ranked feature
        features_inner = np.delete(features_inner,worst_feature,1) # remove worst feature from featurearray
        Best_features = np.delete(Best_features,worst_feature) # to retain labels of best features after the loop
        
        # find cross-validated accuracy
        model2 = SVC(kernel='linear', C=1)
        scores = cross_val_score(model2, features_inner, np.ravel(labels), cv=int(labels.shape[0]/2))
        acc = scores.mean() # crossvalidation accuracy
    
        # record inner loop results into dictionary
        best_feature_recrd[n]=Best_features
        accuracies[n]=acc

    dict_key = max(accuracies.items(), key=operator.itemgetter(1))[0] # find dictionary key (loop) in which highest accuracy was achieved
    feature_set = best_feature_recrd[dict_key] # feature set is selected from SVM_RFE loops 20 t0 8 last loops where highest accuracy was achieved
    #accuracy = accuracies[dict_key] # accuracy atchieved by selected features in the training set
    # END of SVM-RFE
    return feature_set

# T-TEST feature selection selecting 10 highest features

def T_Test(features,labels):
    t_test = [] #prealocate
    for y in range(0,np.shape(features)[1]):
        t_test.append(stats.ttest_ind(features[np.where(labels==1),y][0],features[np.where(labels==0),y][0])[0]) # perform t-test for each feature and record result
    t_test = np.absolute(np.array(t_test)) #take absoolute values of t-test statistic
    feature_set = t_test.argsort()[::-1][:10] # find rank idexes from lowest to highest, reverse an array and take 10 first elements - indecies of 10 highest t statistics
    return feature_set

# PARAMETER TUNNING FUNCTION
if CLASSIFIER == 1: # SVM RBF
    def Tunned_params(features,feature_set,labels): #takes full feature array, selected best features, and labels
        # SVM RBF PARAMETER OPTIMISATION
        inner_best_features = features[:,feature_set]
        cross_val = StratifiedKFold(n_splits=10, shuffle=True)
        clf = GridSearchCV(estimator=SVC(kernel="rbf"), param_grid=par_grid, cv=cross_val,iid='False') # define a model with parameter tunning
        clf.fit(inner_best_features,np.ravel(labels)) # fit a model
        best_params = clf.best_params_ # gives optimised C and Gamma parameters
        C = best_params['C']
        gamma = best_params['gamma']
        accuracy_p_tunned = clf.best_score_
        return  C, gamma, accuracy_p_tunned
elif CLASSIFIER == 2: # Logistic regression
    def Tunned_params(features,feature_set,labels): #takes full feature array, selected best features, and labels
        # LOGISTIC REGREESION PARAMETER OPTIMISATION
        inner_best_features = features[:,feature_set]
        cross_val = StratifiedKFold(n_splits=10, shuffle=True)
        logistic = linear_model.LogisticRegression(solver='liblinear')
        clf = GridSearchCV(estimator=logistic, param_grid=par_grid2, cv=cross_val,iid='False') # define a model with parameter tunning
        clf.fit(inner_best_features,np.ravel(labels)) # fit a model
        penalty = clf.best_estimator_.get_params()['penalty']
        C = clf.best_estimator_.get_params()['C']
        accuracy_p_tunned = clf.score(inner_best_features, np.ravel(labels))
        return  C, penalty, accuracy_p_tunned
    
itera = 0
results = np.zeros([1,5]) # to record results
for n in [i for i in range(40,1020,20) for _ in range(50)]: # main loop

    if DISCRIMINATIVE == 1:  # non discriminative features  
        features = np.random.normal(loc = 0.0,scale=1.0,size=(n,50)) # create features from random normaly distributed numbers sample size -n, features 50
    elif DISCRIMINATIVE == 2: # discriminative features
        features = np.random.normal(loc = 0.0,scale=1.0,size=(n,50)) # create features from random normaly distributed numbers sample size -n, features 50
        features[int(float(n)/2):n,40:50] = np.random.normal(loc = 0.5,scale=1.0,size=(int(float(n)/2),10)) # change distribution of one class to mean = 0.5 for 10 features
        
    labels = np.concatenate((np.ones(shape=(int(n/2),1)),np.zeros(shape=(int(n/2),1)))) # create equally balanced labels vector n/2 of ones and n/2 of zeros
    itera = itera+1
    print('iteration:',itera,'sample:',n) # prit where in the loop execution is
    
    # VALIDATION PART 
    
        
    if VALIDATION == 1: # NESTED
        kf = StratifiedKFold(n_splits=10, shuffle=True) # split into 10 different training and testing samples with suffling
        kf.get_n_splits(X=features,y=labels) 
        inner_loop_acc = [] #prealocate
        for train_index, test_index in kf.split(features,labels): # start ten-fold nested validation pouter loop 
            train_data, validation_data = features[train_index], features[test_index] # get inner loop training data and outer loop valdiaiton data
            train_labels, validation_labels = labels[train_index], labels[test_index] # get inner loop training labels and outer loop valdiaiton labels
            
            if FEATURE_SELECTOR == 1: #SVM-RFE
                feature_set = SVM_RFE(features=train_data,labels=train_labels)
            elif FEATURE_SELECTOR == 2: # T-test
                feature_set = T_Test(features=train_data,labels=train_labels)
            elif FEATURE_SELECTOR == 3: # none
                feature_set = np.array(list(range(0,(features.shape[1]),1)))
                
            if CLASSIFIER == 1:            
                C, gamma,_ = Tunned_params(train_data,feature_set,train_labels) # fild optimal C and Gamma parameters
                model = SVC(C=C,kernel='rbf',gamma = gamma) # create SVM model with optimal parameters
                model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
                predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
                acc = accuracy_score(validation_labels,predictions)
                inner_loop_acc.append(acc)
            elif CLASSIFIER == 2:
                C, penalty,_ = Tunned_params(train_data,feature_set,train_labels)
                model = linear_model.LogisticRegression(penalty = penalty,C=C,solver='liblinear') # create LINEAR REGRESSION model with optimal parameters
                model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
                predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
                acc = accuracy_score(validation_labels,predictions)
                inner_loop_acc.append(acc)                
                
        accuracy = np.mean(inner_loop_acc)
            
    elif VALIDATION == 2: # N-FOLD validation
        if FEATURE_SELECTOR == 1: #SVM-RFE
            feature_set = SVM_RFE(features=features,labels=labels)
        elif FEATURE_SELECTOR == 2: # T-test
            feature_set = T_Test(features=features,labels=labels)
        elif FEATURE_SELECTOR == 3: # none
            feature_set = np.array(list(range(0,(features.shape[1]),1)))
        
        if CLASSIFIER == 1:     # SVM
            C, gamma,accuracy = Tunned_params(features,feature_set,labels) # fild optimal C and Gamma parameters and crossvalidated accuracy
        elif CLASSIFIER == 2:     # LOGISTIC REGRESSION
            C, penalty,accuracy = Tunned_params(features,feature_set,labels) # fild optimal C and Gamma parameters and crossvalidated accuracy
    
        
        # RUN THESE LINES FOR DOUBLECKING
        #model = SVC(C=C,kernel='rbf',gamma = gamma) # create a model with optimal parameters
        #scores = cross_val_score(model, features[:,feature_set], np.ravel(labels), cv=10) # 10 fold crossvalidated accuracies from each fold
        #accuracy2 = scores.mean()
        
    elif VALIDATION == 3: # feature selection NESTED, paremeter tunning N-FOLD
        kf = StratifiedKFold(n_splits=10, shuffle=True) # split into 10 different training and testing samples with suffling
        kf.get_n_splits(X=features,y=labels) 
        inner_loop_acc = [] #prealocate
        for train_index, test_index in kf.split(features,labels): # start ten-fold nested validation pouter loop 
            train_data, validation_data = features[train_index], features[test_index] # get inner loop training data and outer loop valdiaiton data
            train_labels, validation_labels = labels[train_index], labels[test_index] # get inner loop training labels and outer loop valdiaiton labels
            
            if FEATURE_SELECTOR == 1: #SVM-RFE
                feature_set = SVM_RFE(features=train_data,labels=train_labels)
            elif FEATURE_SELECTOR == 2: # T-test
                feature_set = T_Test(features=train_data,labels=train_labels)
            elif FEATURE_SELECTOR == 3: # none
                feature_set = np.array(list(range(0,(features.shape[1]),1)))
            
            if CLASSIFIER == 1:                
                C, gamma,_ = Tunned_params(features,feature_set,labels) # fild optimal C and Gamma parameters THE ONLY DIFFERENCE FROM NESTED
                model = SVC(C=C,kernel='rbf',gamma = gamma) # create a model with optimal parameters
                model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
                predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
                acc = accuracy_score(validation_labels,predictions)
                inner_loop_acc.append(acc)
            elif CLASSIFIER == 2:
                C, penalty,_ = Tunned_params(features,feature_set,labels) # fild optimal C and penalty parameters THE ONLY DIFFERENCE FROM NESTED
                model = linear_model.LogisticRegression(penalty = penalty,C=C,solver='liblinear') # create LINEAR REGRESSION model with optimal parameters
                model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
                predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
                acc = accuracy_score(validation_labels,predictions)
                inner_loop_acc.append(acc)
        accuracy = np.mean(inner_loop_acc)
        
    elif VALIDATION == 4: # feature selection N-FOLD, parameter tuning NESTED
        kf = StratifiedKFold(n_splits=10, shuffle=True) # split into 10 different training and testing samples with suffling
        kf.get_n_splits(X=features,y=labels) 
        inner_loop_acc = [] #prealocate
        a = 0 # for running feature selection only once (instea dof rununing same thing 10 times)
        for train_index, test_index in kf.split(features,labels): # start ten-fold nested validation pouter loop 
            a = a+1
            train_data, validation_data = features[train_index], features[test_index] # get inner loop training data and outer loop valdiaiton data
            train_labels, validation_labels = labels[train_index], labels[test_index] # get inner loop training labels and outer loop valdiaiton labels
            
            if FEATURE_SELECTOR == 1: #SVM-RFE
                if a==1:
                    feature_set = SVM_RFE(features=features,labels=labels) # select features using SVM_RFE THE ONLY DIFFERENCE FROM NESTED
            elif FEATURE_SELECTOR == 2: # T-test
                if a==1:
                    feature_set = T_Test(features=features,labels=labels) # select features using T-test THE ONLY DIFFERENCE FROM NESTED
            elif FEATURE_SELECTOR == 3: # none
                feature_set = np.array(list(range(0,(features.shape[1]),1)))
            
            if CLASSIFIER == 1:     
                C, gamma,_ = Tunned_params(train_data,feature_set,train_labels) # fild optimal C and Gamma parameters THE ONLY DIFFERENCE FROM NESTED
                model = SVC(C=C,kernel='rbf',gamma = gamma) # create a model with optimal parameters
                model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
                predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
                acc = accuracy_score(validation_labels,predictions)
                inner_loop_acc.append(acc)
            elif CLASSIFIER == 2:
                C, penalty,_ = Tunned_params(train_data,feature_set,train_labels)
                model = linear_model.LogisticRegression(penalty = penalty,C=C,solver='liblinear') # create LINEAR REGRESSION model with optimal parameters
                model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
                predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
                acc = accuracy_score(validation_labels,predictions)
                inner_loop_acc.append(acc)                
                
        accuracy = np.mean(inner_loop_acc)
        
        
    elif VALIDATION == 5: # train test split
        train_data, validation_data, train_labels, validation_labels = train_test_split(features, np.ravel(labels), test_size=0.10,stratify=np.ravel(labels))
    
        if FEATURE_SELECTOR == 1: #SVM-RFE
            feature_set = SVM_RFE(features=train_data,labels=train_labels)
        elif FEATURE_SELECTOR == 2: # T-test
            feature_set = T_Test(features=train_data,labels=train_labels)
        elif FEATURE_SELECTOR == 3: # Feature selection nenone
                feature_set = np.array(list(range(0,(features.shape[1]),1)))
                
        if CLASSIFIER == 1:            
            C, gamma,_ = Tunned_params(train_data,feature_set,train_labels) # fild optimal C and Gamma parameters
            model = SVC(C=C,kernel='rbf',gamma = gamma) # create SVM model with optimal parameters
            model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
            predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
            accuracy = accuracy_score(validation_labels,predictions)
        elif CLASSIFIER == 2:
            C, penalty,_ = Tunned_params(train_data,feature_set,train_labels)
            model = linear_model.LogisticRegression(penalty = penalty,C=C,solver='liblinear') # create LINEAR REGRESSION model with optimal parameters
            model.fit(train_data[:,feature_set],np.ravel(train_labels)) # fit model
            predictions = model.predict(validation_data[:,feature_set]) # predicted labels with a model on a validation data
            accuracy = accuracy_score(validation_labels,predictions)   

        
    results = np.concatenate([results,np.concatenate([[[VALIDATION]],[[FEATURE_SELECTOR]],[[CLASSIFIER]],[[n]],[[accuracy]]], axis = 1)]) # record results
        
    #write results to file every 50 itterations
    if (itera/50).is_integer():
        with open("results.csv", "a") as myfile:
            np.savetxt(myfile, results[(np.shape(results)[0]-50):np.shape(results)[0],:], delimiter=',', newline='\n')