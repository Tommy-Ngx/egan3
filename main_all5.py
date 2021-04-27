'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np

from data_loader import data_loader, data_loader2, data_loader3
from gain_ori import gain
from egain import egain
from utils import rmse_loss, normalization
from missingpy import MissForest
from sklearn import metrics
from math import sqrt
from impyute.imputation.cs import mice
import pandas as pd
from autoimpute.imputations import MiceImputer, SingleImputer, MultipleImputer
from autoimpute.analysis import MiLinearRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.linear_model import BayesianRidge
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from warnings import filterwarnings
from IPython.display import clear_output
from tqdm import tqdm_notebook as tq
from google.colab import output
filterwarnings('ignore')

def auc_dt(impute,data):
    df1 = pd.read_csv("/content/tommy/data/{}.csv".format(data))
    df1 = df1.rename(columns={'target': 'Class'})
    df1['Class'] = pd.factorize(df1['Class'])[0] + 1
    col_Names=list(df1.columns)
    df = pd.read_csv("/content/tommy/data/{}.csv".format(impute), names=col_Names)
    X = df.drop(['Class'], axis=1)
    targets = df1['Class'].values

    X_train, test_x, y_train, test_lab = train_test_split(X,targets,test_size = 0.3,random_state = 42)
    clf = DecisionTreeClassifier( random_state = 42) # max_depth =3,
    clf.fit(X_train, y_train)
    test_pred_decision_tree = clf.predict(test_x)
    return  metrics.accuracy_score(test_lab, test_pred_decision_tree)

    #scf = StratifiedShuffleSplit(n_splits=5)
    #score_dct = cross_val_score(DecisionTreeClassifier(max_depth=5), X, targets, cv=scf, scoring='accuracy')
    #metrics.accuracy_score(test_lab, test_pred_decision_tree)

    #print('Method: {}'.format(impute))
    #print('Mean Validation AUC AUC: {}'.format(round(np.mean(score_dct),6)))

def auc_mlp(impute,data):
    df1 = pd.read_csv("/content/tommy/data/{}.csv".format(data))
    df1 = df1.rename(columns={'target': 'Class'})
    df1['Class'] = pd.factorize(df1['Class'])[0] + 1
    col_Names=list(df1.columns)
    df = pd.read_csv("/content/tommy/data/{}.csv".format(impute), names=col_Names)
    X = df.drop(['Class'], axis=1)
    targets = df1['Class'].values

    X_train, test_x, y_train, test_lab = train_test_split(X,targets,test_size = 0.3,random_state = 42)
    #clf = DecisionTreeClassifier( random_state = 42) # max_depth =3,
    #clf = MLPClassifier(hidden_layer_sizes= X_train.shape[1]//2,  early_stopping=True) #max_iter=500,
    clf = MLPClassifier(hidden_layer_sizes=X_train.shape[1]//2, max_iter=500, early_stopping=True, 
      #learning_rate_init=0.01,
      learning_rate='constant')
    clf.fit(X_train, y_train)
    test_pred_decision_tree = clf.predict(test_x)
    return  metrics.accuracy_score(test_lab, test_pred_decision_tree)


def auc_lr(impute,data):
    df1 = pd.read_csv("/content/tommy/data/{}.csv".format(data))
    df1 = df1.rename(columns={'target': 'Class'})
    df1['Class'] = pd.factorize(df1['Class'])[0] + 1
    col_Names=list(df1.columns)
    df = pd.read_csv("/content/tommy/data/{}.csv".format(impute), names=col_Names)
    X = df.drop(['Class'], axis=1)
    targets = df1['Class'].values
    X_train, test_x, y_train, test_lab = train_test_split(X,targets,test_size = 0.3,random_state = 42)
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    test_pred_decision_tree = clf.predict(test_x)
    return  metrics.accuracy_score(test_lab, test_pred_decision_tree)

def clf_MLP(imputed_data_x, y, train_idx, test_idx):
    clf = MLPClassifier(hidden_layer_sizes=len(imputed_data_x[1])//2, max_iter=500,
                                 learning_rate='constant', learning_rate_init=0.1)
    clf.fit(imputed_data_x[train_idx], y[train_idx])
    score = clf.score(imputed_data_x[test_idx], y[test_idx])
    return np.round(score*100,4)

def clf_DT(imputed_data_x, y, train_idx, test_idx):
    clf = DecisionTreeClassifier()
    clf.fit(imputed_data_x[train_idx], y[train_idx])
    score = clf.score(imputed_data_x[test_idx], y[test_idx])
    return np.round(score*100,4)

def clf_SGD(imputed_data_x, y, train_idx, test_idx):
    clf = SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(imputed_data_x[train_idx], y[train_idx])
    score = clf.score(imputed_data_x[test_idx], y[test_idx])
    return np.round(score*100,4)

def clf_SVC(imputed_data_x, y, train_idx, test_idx):
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(imputed_data_x[train_idx], y[train_idx])
    score = clf.score(imputed_data_x[test_idx], y[test_idx])
    return np.round(score*100,4)

def clf_GAU(imputed_data_x, y, train_idx, test_idx):
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
    clf.fit(imputed_data_x[train_idx], y[train_idx])
    score = clf.score(imputed_data_x[test_idx], y[test_idx])
    return np.round(score*100,4)

def clf_LR(imputed_data_x, y, train_idx, test_idx):
    clf = LogisticRegression(random_state=0)
    clf.fit(imputed_data_x[train_idx], y[train_idx])
    score = clf.score(imputed_data_x[test_idx], y[test_idx])
    return np.round(score*100,4)

def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  random    = args.seed
  time      = args.time

  x1 = data_name
  x = x1.split("+")

  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'time': args.time}
   # Load data and introduce missingness
    
  #ori_data_x, miss_data_x, data_m = data_loader2(data_name, miss_rate,random)
  miss_rate_caption = "{}% Missing".format(miss_rate)
  col1 = [miss_rate_caption,'RMSE','RMSE','RMSE','RMSE','RMSE','RMSE','MLP','MLP','D.Tree','D.Tree','LogisticR','LogisticR','LogisticR','LogisticR','LogisticR','LogisticR','SVC','SVC','SVC','SVC','SVC','SVC','SGD','SGD','SGD','SGD','SGD','SGD']
  col2 = ['Method', 'EGAIN', 'GAIN', 'MEAN','KNN','MICE','M.FORE','EGAIN','GAIN','EGAIN','GAIN','EGAIN', 'GAIN', 'MEAN','KNN','MICE','M.FORE','EGAIN', 'GAIN', 'MEAN','KNN','MICE','M.FORE','EGAIN', 'GAIN', 'MEAN','KNN','MICE','M.FORE']
  result = [col1,col2]

  for data_train in x:
      data_name = data_train
      dataset = ['obesity', 'hepatitisC', 'audit','letter','spam', 'breast', 'credit', 'news','blood','vowel','ecoli','ionosphere','parkinsons','seedst','vehicle','vertebral','wine','banknote','balance','yeast','bean','shill','phishing','firewall','iBeacon','steel']
      if (data_name not in dataset):
        print("Wrong name: {} Dataset. Skip this datasets".format(data_train))
        break
      col3=[]
      col3.append(data_name)

      print("****** {} Dataset ******".format(data_train))
      gan_rs, egain_rs, mice_rs,miss_rs, gan_mlp, gan_dt, egan_mlp, egan_dt = [],[],[],[],[],[],[],[];
      gan_svc, egan_svc, gan_lr, egan_lr, gan_sgd, egan_sgd, gan_gau, egan_gau = [],[],[],[],[],[],[],[];
      knn_rmse , mean_rmse, miss_rmse, mice_rmse =  [],[],[],[];
      knn_lr, knn_svc, knn_sgd, mean_lr, mean_svc, mean_sgd =    [],[],[],[],[],[];
      miss_lr, miss_svc, miss_sgd, mice_lr, mice_svc, mice_sgd = [],[],[],[],[],[];

      for i in range(time):
        # Load data and introduce missingness
        # Fix loader i=42
        ori_data_x, miss_data_x, data_m, y  = data_loader3(data_name, miss_rate, 42)# 7) #i) block i 
        train_idx, test_idx = train_test_split(range(len(y)), test_size=0.2, stratify=y, random_state= i)#7) #i) block i
        miss_data_x2 = miss_data_x #* 10000
        if i % 5 == 0:
            print('=== Working on {}/{} ==='.format(i, time))

        # Impute missing data
        imputed_data_x1   = gain(miss_data_x2, gain_parameters)
        imputed_data_x_e1 = egain(miss_data_x2, gain_parameters)
        imputed_data_x    = imputed_data_x1 #* 1/10000
        imputed_data_x_e  = imputed_data_x_e1 #* 1/10000


        imp_MEAN = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_data_x_mean = imp_MEAN.fit_transform(miss_data_x)
        imputed_data_x_mean = imputed_data_x_mean.round()
        #imputed_data_x_mean = imp_MEAN.fit_transform(miss_data_x2)  *1/10000

        imp_KNN = KNNImputer(missing_values=np.nan, n_neighbors=3)
        imputed_data_x_knn = imp_KNN.fit_transform(miss_data_x)# *1/10000
        imputed_data_x_knn = imputed_data_x_knn.round()

        # ExtraTreesRegressor: similar to missForest in R; DecisionTreeRegressor()
        imp_mf   = IterativeImputer(estimator = ExtraTreesRegressor(), max_iter = 1, initial_strategy= "constant", n_nearest_features = 1, imputation_order = 'descending') #20
        imputed_data_mf = imp_mf.fit_transform(miss_data_x) #*1/10000
        imputed_data_mf = imputed_data_mf.round()
        #imp_mf = MissForest(max_iter=1)
        #imputed_data_mf = imp_mf.fit_transform(miss_data_x)
        
        imp_mice = IterativeImputer(estimator = BayesianRidge(), max_iter = 1, initial_strategy= 'constant', n_nearest_features = 1, imputation_order = 'descending')# 'mean') #20
        imputed_data_mice = imp_mice.fit_transform(miss_data_x) #*1/10000
        imputed_data_mice = imputed_data_mice.round()
        
        # Report the RMSE performance
        rmse      = rmse_loss (ori_data_x, imputed_data_x, data_m)
        rmse_e    = rmse_loss (ori_data_x, imputed_data_x_e, data_m)
        rmse_mean = rmse_loss (ori_data_x, imputed_data_x_mean, data_m)
        rmse_knn  = rmse_loss (ori_data_x, imputed_data_x_knn, data_m)
        rmse_mf   = rmse_loss (ori_data_x, imputed_data_mf, data_m)
        rmse_mice = rmse_loss (ori_data_x, imputed_data_mice, data_m)

        gan_rs.append(rmse)
        egain_rs.append(rmse_e)

        mean_rmse.append(rmse_mean)
        knn_rmse.append(rmse_knn)
        mice_rmse.append(rmse_mice)
        miss_rmse.append(rmse_mf)


        mi_data = miss_data_x.astype(float)
        no, dim = imputed_data_mice.shape
        miss_data = np.reshape(mi_data,(no,dim))
        np.savetxt("data/{}missing_data.csv".format(i),mi_data,delimiter=',',fmt='%1.2f')
        np.savetxt("data/{}imputed_data_gain.csv".format(i),imputed_data_x, delimiter=',',  fmt='%d')
        np.savetxt("data/{}imputed_data_egain.csv".format(i),imputed_data_x_e, delimiter=',',  fmt='%d')

        imputed_data_x, _     = normalization(imputed_data_x)
        imputed_data_x_e, _   = normalization(imputed_data_x_e)

        imputed_data_x_mean,_ = normalization(imputed_data_x_mean)
        imputed_data_x_knn,_  = normalization(imputed_data_x_knn)
        imputed_data_mf, _    = normalization(imputed_data_mf)
        imputed_data_mice, _  = normalization(imputed_data_mice)

        gan_score_mlp  = clf_MLP(imputed_data_x  , y, train_idx, test_idx)
        egan_score_mlp = clf_MLP(imputed_data_x_e, y, train_idx, test_idx)
        gan_mlp.append(gan_score_mlp)
        egan_mlp.append(egan_score_mlp)

        gan_score_dt   = clf_DT(imputed_data_x    , y, train_idx, test_idx)
        egan_score_dt  = clf_DT(imputed_data_x_e  , y, train_idx, test_idx)
        gan_dt.append(gan_score_dt)
        egan_dt.append(egan_score_dt)

        gan_score_lr   = clf_LR(imputed_data_x    , y, train_idx, test_idx)
        egan_score_lr  = clf_LR(imputed_data_x_e  , y, train_idx, test_idx)

        mean_score_lr  = clf_LR(imputed_data_x_mean, y, train_idx, test_idx)
        knn_score_lr   = clf_LR(imputed_data_x_knn , y, train_idx, test_idx)
        miss_score_lr  = clf_LR(imputed_data_mf, y, train_idx, test_idx)
        mice_score_lr  = clf_LR(imputed_data_mice, y, train_idx, test_idx)
        
        gan_lr.append(gan_score_lr)
        egan_lr.append(egan_score_lr)
        mean_lr.append(mean_score_lr)
        knn_lr.append(knn_score_lr)
        miss_lr.append(miss_score_lr)
        mice_lr.append(mice_score_lr)

        mean_score_svc  = clf_SVC(imputed_data_x_mean, y, train_idx, test_idx)
        knn_score_svc   = clf_SVC(imputed_data_x_knn , y, train_idx, test_idx)
        miss_score_svc  = clf_SVC(imputed_data_mf, y, train_idx, test_idx)
        mice_score_svc  = clf_SVC(imputed_data_mice, y, train_idx, test_idx)
        mean_svc.append(mean_score_svc)
        knn_svc.append(knn_score_svc)
        miss_svc.append(miss_score_svc)
        mice_svc.append(mice_score_svc)

        gan_score_svc   = clf_SVC(imputed_data_x    , y, train_idx, test_idx)
        egan_score_svc  = clf_SVC(imputed_data_x_e  , y, train_idx, test_idx)
        gan_svc.append(gan_score_svc)
        egan_svc.append(egan_score_svc)

        mean_score_sgd  = clf_SGD(imputed_data_x_mean, y, train_idx, test_idx)
        knn_score_sgd   = clf_SGD(imputed_data_x_knn , y, train_idx, test_idx)
        miss_score_sgd  = clf_SGD(imputed_data_mf, y, train_idx, test_idx)
        mice_score_sgd  = clf_SGD(imputed_data_mice, y, train_idx, test_idx)
        mean_sgd.append(mean_score_sgd)
        knn_sgd.append(knn_score_sgd)
        miss_sgd.append(miss_score_sgd)
        mice_sgd.append(mice_score_sgd)

        gan_score_sgd   = clf_SGD(imputed_data_x    , y, train_idx, test_idx)
        egan_score_sgd  = clf_SGD(imputed_data_x_e  , y, train_idx, test_idx)
        gan_sgd.append(gan_score_sgd)
        egan_sgd.append(egan_score_sgd)

        #gan_score_gau   = clf_GAU(imputed_data_x    , y, train_idx, test_idx)
        #egan_score_gau  = clf_GAU(imputed_data_x_e  , y, train_idx, test_idx)
        #gan_gau.append(gan_score_gau)
        #egan_gau.append(egan_score_gau)

      print()
      print("Datasets: ",data_name)
      #print(gan_rs,egain_rs, mice_rs,miss_rs)
      col3.append( round(np.mean(egain_rs)*1,2)  + ' ± ' + round(np.std(egain_rs),4) )
      col3.append( round(np.mean(gan_rs)*1,2)    + ' ± ' + round(np.std(gan_rs),4) )
      col3.append( round(np.mean(mean_rmse)*1,2) + ' ± ' + round(np.std(mean_rmse),4) )
      col3.append( round(np.mean(knn_rmse)*1,2)  + ' ± ' + round(np.std(knn_rmse),4) )
      col3.append( round(np.mean(mice_rmse)*1,2) + ' ± ' + round(np.std(mice_rmse),4) )
      col3.append( round(np.mean(miss_rmse)*1,2) + ' ± ' + round(np.std(miss_rmse),4) )

      col3.append( round(np.mean(egan_mlp)*1,2) + ' ± ' + round(np.std(egan_mlp),4) )
      col3.append( round(np.mean(gan_mlp)*1,2)  + ' ± ' + round(np.std(gan_mlp),4) )
      col3.append( round(np.mean(egan_dt)*1,2)  + ' ± ' + round(np.std(egan_dt),4) )
      col3.append( round(np.mean(gan_dt)*1,2)   + ' ± ' + round(np.std(gan_dt),4) )

      col3.append( round(np.mean(egan_lr)*1,2) + ' ± ' + round(np.std(egan_lr),4) )
      col3.append( round(np.mean(gan_lr)*1,2)  + ' ± ' + round(np.std(gan_lr),4) )
      col3.append( round(np.mean(mean_lr)*1,2) + ' ± ' + round(np.std(mean_lr),4) )
      col3.append( round(np.mean(knn_lr)*1,2)  + ' ± ' + round(np.std(knn_lr),4) )
      col3.append( round(np.mean(mice_lr)*1,2) + ' ± ' + round(np.std(mice_lr),4) )
      col3.append( round(np.mean(miss_lr)*1,2) + ' ± ' + round(np.std(miss_lr),4) )

      col3.append( round(np.mean(egan_svc)*1,2) + ' ± ' + round(np.std(egan_svc),4) )
      col3.append( round(np.mean(gan_svc)*1,2)  + ' ± ' + round(np.std(gan_svc),4) )
      col3.append( round(np.mean(mean_svc)*1,2) + ' ± ' + round(np.std(mean_svc),4) )
      col3.append( round(np.mean(knn_svc)*1,2)  + ' ± ' + round(np.std(knn_svc),4) )
      col3.append( round(np.mean(mice_svc)*1,2) + ' ± ' + round(np.std(mice_svc),4) )
      col3.append( round(np.mean(miss_svc)*1,2) + ' ± ' + round(np.std(miss_svc),4) )       

      col3.append( round(np.mean(egan_sgd)*1,2) + ' ± ' + round(np.std(egan_sgd),4) )
      col3.append( round(np.mean(gan_sgd)*1,2)  + ' ± ' + round(np.std(gan_sgd),4) )
      col3.append( round(np.mean(mean_sgd)*1,2) + ' ± ' + round(np.std(mean_sgd),4) )
      col3.append( round(np.mean(knn_sgd)*1,2)  + ' ± ' + round(np.std(knn_sgd),4) )
      col3.append( round(np.mean(mice_sgd)*1,2) + ' ± ' + round(np.std(mice_sgd),4) )
      col3.append( round(np.mean(miss_sgd)*1,2) + ' ± ' + round(np.std(miss_sgd),4) )

      '''
      print('RMSE  GAIN: {} ± {}'.format(round(np.mean(gan_rs)*1,2), round(np.std(gan_rs),4)))
      #print(gan_rs)
      print('RMSE EGAIN: {} ± {}'.format(round(np.mean(egain_rs)*1,2), round(np.std(egain_rs),4)))
      #print(egain_rs)
      print('RMSE  MEAN: {} ± {}'.format(round(np.mean(mean_rmse)*1,2), round(np.std(mean_rmse),4)))
      #print(knn_rmse)
      print('RMSE   KNN: {} ± {}'.format(round(np.mean(knn_rmse)*1,2), round(np.std(knn_rmse),4)))
      #print(mice_rmse)
      print('RMSE  MICE: {} ± {}'.format(round(np.mean(mice_rmse)*1,2), round(np.std(mice_rmse),4)))
      #print(miss_rmse)
      print('RMSE MFORE: {} ± {}'.format(round(np.mean(miss_rmse)*1,2), round(np.std(miss_rmse),4)))
      #print(miss_rmse)
      print()
      print('MLP   GAIN: {} ± {}'.format(round(np.mean(gan_mlp)*1,2), round(np.std(gan_mlp),4)))
      print('MLP  EGAIN: {} ± {}'.format(round(np.mean(egan_mlp)*1,2), round(np.std(egan_mlp),4)))
      print()
      print('DT    GAIN: {} ± {}'.format(round(np.mean(gan_dt)*1,2), round(np.std(gan_dt),4)))
      print('DT   EGAIN: {} ± {}'.format(round(np.mean(egan_dt)*1,2), round(np.std(egan_dt),4)))
      print()

      print('LR    GAIN: {} ± {}'.format(round(np.mean(gan_lr)*1,2), round(np.std(gan_lr),4)))
      #print(gan_lr)
      print('LR   EGAIN: {} ± {}'.format(round(np.mean(egan_lr)*1,2), round(np.std(egan_lr),4)))
      #print(egan_lr)
      print('LR    MEAN: {} ± {}'.format(round(np.mean(mean_lr)*1,2), round(np.std(mean_lr),4)))
      #print(mean_lr)
      print('LR     KNN: {} ± {}'.format(round(np.mean(knn_lr)*1,2), round(np.std(knn_lr),4)))
      #print(knn_lr)
      print('LR    MICE: {} ± {}'.format(round(np.mean(mice_lr)*1,2), round(np.std(mice_lr),4)))
      #print(mice_lr)
      print('LR MISSFOR: {} ± {}'.format(round(np.mean(miss_lr)*1,2), round(np.std(miss_lr),4)))
      #print(miss_lr)
      print()
      print('SVC   GAIN: {} ± {}'.format(round(np.mean(gan_svc)*1,2), round(np.std(gan_svc),4)))
      #print(gan_svc)
      print('SVC  EGAIN: {} ± {}'.format(round(np.mean(egan_svc)*1,2), round(np.std(egan_svc),4)))
      #print(egan_svc)
      print('SVC   MEAN: {} ± {}'.format(round(np.mean(mean_svc)*1,2), round(np.std(mean_svc),4)))
      #print(mean_svc)
      print('SVC    KNN: {} ± {}'.format(round(np.mean(knn_svc)*1,2), round(np.std(knn_svc),4)))
      #print(knn_svc)
      print('SVC   MICE: {} ± {}'.format(round(np.mean(mice_svc)*1,2), round(np.std(mice_svc),4)))
      #print(mice_svc)
      print('SVC   MISS: {} ± {}'.format(round(np.mean(miss_svc)*1,2), round(np.std(miss_svc),4)))
      #print(miss_svc)
      print()
      print('SGD   GAIN: {} ± {}'.format(round(np.mean(gan_sgd)*1,2), round(np.std(gan_sgd),4)))
      #print(gan_sgd)
      print('SGD  EGAIN: {} ± {}'.format(round(np.mean(egan_sgd)*1,2), round(np.std(egan_sgd),4)))
      #print(egan_sgd)
      print('SGD   MEAN: {} ± {}'.format(round(np.mean(mean_sgd)*1,2), round(np.std(mean_sgd),4)))
      #print(mean_sgd)
      print('SGD    KNN: {} ± {}'.format(round(np.mean(knn_sgd)*1,2), round(np.std(knn_sgd),4)))
      #print(knn_sgd)
      print('SGD   MICE: {} ± {}'.format(round(np.mean(mice_sgd)*1,2), round(np.std(mice_sgd),4)))
      #print(mice_sgd)
      print('SGD   MISS: {} ± {}'.format(round(np.mean(miss_sgd)*1,2), round(np.std(miss_sgd),4)))
      
      '''
      result.append(col3)
      df_result = pd.DataFrame(result)
      print(df_result)


      #print(miss_sgd)
      #print()
      #print('GAU   GAIN: {} ± {}'.format(round(np.mean(gan_gau)*1,2), round(np.std(gan_dt),4)))
      #print('GAU  EGAIN: {} ± {}'.format(round(np.mean(egan_gau)*1,2), round(np.std(egan_dt),4)))
      
      # MissForest

      #print()
      #print('=== MissForest RMSE ===')
      #data = miss_data_x
      #imp_mean = MissForest(max_iter = 1)
      #miss_f = imp_mean.fit_transform(data)
      #miss_f = pd.DataFrame(imputed_train_df)
      #rmse_MF = rmse_loss (ori_data_x, miss_f, data_m)
      #print('RMSE Performance: ' + str(np.round(rmse_MF, 6)))
      #np.savetxt("data/imputed_data_MF.csv",miss_f, delimiter=',',  fmt='%d')
      #print( 'Save results in Imputed_data_MF.csv')

      # MICE From Auto Impute
      #print()
      #print('=== MICE of Auto Impute RMSE ===')
      #data_mice = pd.DataFrame(miss_data_x)
      #mi = MiceImputer(k=1, imp_kwgs=None, n=1, predictors='all', return_list=True,
      #      seed=None, strategy='interpolate', visit='default')
      #mice_out = mi.fit_transform(data_mice)
      #c = [list(x) for x in mice_out]
      #c1= c[0]
      #c2=c1[1]
      #c3=np.asarray(c2)
      #mice_x=c3
      #print('here :', mice_x, miss_f, miss_f.shape)
      #rmse_MICE = rmse_loss (ori_data_x, mice_x, data_m)
      #print('=== MICE of Auto Impute RMSE ===')
      #print('RMSE Performance: ' + str(np.round(rmse_MICE, 6)))
      #np.savetxt("data/imputed_data_MICE.csv",mice_x, delimiter=',',  fmt='%d')
      #print( 'Save results in Imputed_data_MICE.csv')


      return imputed_data_mf, rmse_mf


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      #choices=['obesity', 'hepatitisC', 'audit','letter','spam', 'breast', 'credit', 'news','blood','vowel','ecoli','ionosphere','parkinsons','seedst','vehicle','vertebral','wine','banknote','balance','yeast','bean','shill','phishing','firewall','iBeacon','steel'],
      #default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  parser.add_argument(
      '--time',
      help='number of repeated process',
      default=1,
      type=int)
  parser.add_argument(
      '--seed',
      help='number of repeated process',
      default=42,
      type=int)
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)
