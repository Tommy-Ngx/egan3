'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from utils import binary_sampler, binary_sampler2
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder


def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if data_name in ['obesity', 'hepatitisC', 'audit','letter', 'spam', 'breast', 'credit', 'news','blood','vowel','ecoli','ionosphere','parkinsons','parkinsons2','seedst','vehicle','vertebral','wine','banknote','balance','yeast']:
    file_name = 'data/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  elif data_name == 'mnist':
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m

def data_loader2 (data_name, miss_rate, random):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if data_name in ['obesity', 'hepatitisC', 'audit','letter', 'spam', 'breast', 'credit', 'news','blood','vowel','ecoli','ionosphere','parkinsons','parkinsons2','seedst','vehicle','vertebral','wine','banknote','balance','yeast']:
    file_name = 'data/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  elif data_name == 'mnist':
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data
  data_m = binary_sampler2(1-miss_rate, no, dim,random)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m

def data_loader3 (data_name, miss_rate, random):
  # Load data
  if data_name in ['obesity', 'hepatitisC', 'audit','letter', 'spam', 'breast', 'credit', 'news','blood','vowel','ecoli','ionosphere','parkinsons','parkinsons2','seedst','vehicle','vertebral','wine','banknote','balance','yeast']:
    file_name = 'data/'+data_name+'_full.csv'
    df = pd.read_csv(file_name)
    df = df.rename(columns={'target': 'Class'})
    x = df.drop(['Class'],axis = 1)
    x = x.values
    y = df['Class'].values
    if data_name == 'letter':
      le = LabelEncoder()
      y = le.fit_transform(y)
    elif data_name == 'balance':
      le = LabelEncoder()
      y = le.fit_transform(y)
    elif data_name == 'ecoli':
      le = LabelEncoder()
      y = le.fit_transform(y)
    elif data_name == 'obesity':
      le = LabelEncoder()
      y = le.fit_transform(y)

    x = x.astype(np.float32)
    data_x = x #np.loadtxt(file_name, delimiter=",", skiprows=1)
  elif data_name == 'mnist':
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data
  data_m = binary_sampler2(1-miss_rate, no, dim,random)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m, y


