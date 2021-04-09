'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from utils import normalization, renormalization, rounding
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from sklearn.model_selection import train_test_split
from data_loader import data_loader, make_missing_data
from gain import gain
from tqdm import tqdm
import xlwt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
def train_LR(x_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    return clf
def train_DCT(x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf
def train_MLP(x_train, y_train):
    hidden_units = x_train.shape[1]//2
    clf = MLPClassifier(hidden_layer_sizes=hidden_units, max_iter=500,
                        early_stopping=True, learning_rate='constant', learning_rate_init=0.1)
    clf.fit(x_train,y_train)
    return clf
def predict(clf, x_test):
    return clf.predict(x_test)

def main ():
    # data_names = ['letter', 'spam','credit','breast']
    # data_names = ['spam', 'letter','breast','banknote']
    data_names = ['parkinsons']
    # data_names = ['breast','banknote','connectionistvowel']
    # data_names = ['connectionistvowel']
    # data_names = ['parkinsons','seedst']
    # data_names = ['breasttissue','glass', 'thyroid']
    # data_names = ['credit', 'breast', 'balance','banknote','blood','climate','connectionistvowel',
    #               'ecoli','glass','hillvalley','ionosphere', 'parkinsons','planning','seedst',
    #               'thyroid','vehicle','vertebral','wine','yeast']
    # data_names = ['parkinsons']
    # data_names = ['balance','banknote','blood','connectionistvowel','vehicle','yeast']
    miss_rate = 0.1
    batch_size = 128
    alpha = 100
    iterations = 10000
    n_times = 30
    gain_parameters = {'batch_size': batch_size,
                       'alpha': alpha,
                       'iterations': iterations}
    wb = xlwt.Workbook()
    sh_dct_mgain = wb.add_sheet("DCT_mgain")
    sh_mlp_mgain = wb.add_sheet("MLP_mgain")
    sh_dct_gain = wb.add_sheet("DCT_gain")
    sh_mlp_gain = wb.add_sheet("MLP_gain")

    for k in range(len(data_names)):
        data_name = data_names[k]
        sh_dct_mgain.write(0, k, data_name)
        sh_mlp_mgain.write(0, k, data_name)

        sh_dct_gain.write(0, k, data_name)
        sh_mlp_gain.write(0, k, data_name)

        print("Dataset: ", data_name)
        ori_data_x, y = data_loader(data_name)
        train_idx, test_idx = train_test_split(range(len(y)), test_size=0.3, stratify=y, random_state=42)
        for i in tqdm(range(n_times)):
            miss_data_x, m = make_missing_data(ori_data_x, miss_rate, seed=i)

            # Impute missing data
            # 20 imputed data for MGAIN
            # 1 imputed data for GAIN
            imputed_data_xs = gain(miss_data_x, gain_parameters, n_times = 21)
            imputed_data_xs = np.array(imputed_data_xs)

            # Normalize data
            imputed_data_xs = np.array([normalization(data)[0] for data in imputed_data_xs])

            #classify
            num_cores = multiprocessing.cpu_count()

            # Training with MLP
            # 20 for esemble learning MGAIN
            # 1 for GAIN
            clfs = Parallel(n_jobs=num_cores)(delayed(train_MLP)(imputed_data_xs[i][train_idx], y[train_idx])
                                              for i in range(imputed_data_xs.shape[0]))

            # Testing
            # MGAIN
            x_test = np.mean(imputed_data_xs[1:, test_idx, :], axis=0)
            ys = Parallel(n_jobs=num_cores)(delayed(predict)(clf, x_test)
                                            for clf in clfs[1:])
            ys = np.array(ys)
            # voting
            y_test = [max(set(list(ys[:, i])), key=list(ys[:, i]).count) for i in range(ys.shape[1])]
            score = accuracy_score(y[test_idx], y_test)
            sh_mlp_mgain.write(i + 1, k, np.round(score, 4))

            score = accuracy_score(y[test_idx], predict(clfs[0], imputed_data_xs[0][test_idx]))
            sh_mlp_gain.write(i + 1, k, np.round(score, 4))

            # # Training with LR
            # # 20 for esemble learning MGAIN
            # # 1 for GAIN
            # clfs = Parallel(n_jobs=num_cores)(delayed(train_LR)(imputed_data_xs[i][train_idx], y[train_idx])
            #                                   for i in range(imputed_data_xs.shape[0]))
            #
            # # Testing
            # # MGAIN
            # x_test = np.mean(imputed_data_xs[1:,test_idx,:],axis=0)
            # ys = Parallel(n_jobs=num_cores)(delayed(predict)(clf, x_test)
            #                                 for clf in clfs[1:])
            # ys = np.array(ys)
            # # voting
            # y_test = [max(set(list(ys[:,i])),key=list(ys[:,i]).count) for i in range(ys.shape[1])]
            # score = accuracy_score(y[test_idx],y_test)
            # sh_lr_mgain.write(i+1,k,np.round(score,4))
            #
            # score = accuracy_score(y[test_idx], predict(clfs[0],imputed_data_xs[0][test_idx]))
            # sh_lr_gain.write(i + 1, k, np.round(score, 4))

            # Training with DCT
            # 20 for esemble learning MGAIN
            # 1 for GAIN
            clfs = Parallel(n_jobs=num_cores)(delayed(train_DCT)(imputed_data_xs[i][train_idx], y[train_idx])
                                              for i in range(imputed_data_xs.shape[0]))

            # Testing
            # MGAIN
            x_test = np.mean(imputed_data_xs[1:, test_idx, :], axis=0)
            ys = Parallel(n_jobs=num_cores)(delayed(predict)(clf, x_test)
                                            for clf in clfs[1:])
            ys = np.array(ys)
            # voting
            y_test = [max(set(list(ys[:, i])), key=list(ys[:, i]).count) for i in range(ys.shape[1])]
            score = accuracy_score(y[test_idx], y_test)
            sh_dct_mgain.write(i + 1, k, np.round(score, 4))

            score = accuracy_score(y[test_idx], predict(clfs[0], imputed_data_xs[0][test_idx]))
            sh_dct_gain.write(i + 1, k, np.round(score, 4))

    wb.save("./final_results/Mgain_10_parkinsons.xls")
if __name__ == '__main__':
    main()
