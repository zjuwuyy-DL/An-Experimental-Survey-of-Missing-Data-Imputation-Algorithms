'''
==============================================================================
To install this framework use pip. pip install git + https: // github.com / spindro / GINN.git
==============================================================================
'''

import csv
import numpy as np
from sklearn import model_selection, preprocessing

from ginn import GINN
from ginn.utils import degrade_dataset, data2onehot
import time
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#data_list = ['Abalone', 'Wine', 'Anuran_Calls', 'Yeast', 'Car_evaluation', 'Website_Phishing', 'Turkiye', 'Balance_Scale', 'Heart', 'Wireless', 'EEG', 'Chess', 'Letter', 'Connect']
data_list = ['Wine']
missing_rate = 0.2
train_missing_rate = 0
train_rate = 0.8
epoch_number = 5


'''
==============================================================================
Data--System Parameters
==============================================================================
'''
#
#mechanism_list = ['MCAR', 'MNAR']
mechanism_list = ['MCAR']
for mechanism in mechanism_list:
    for dataset in data_list:
        MAE_D = []
        RMSE_D = []
        time_all = []
        for epoch in range(epoch_number):

            epoch_num = epoch + 1
            test_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_test.csv', dtype=int, delimiter=",", skiprows=0)
            train_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_train.csv', dtype=int, delimiter=",", skiprows=0)
            miss_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_miss.csv', delimiter=",", skiprows=0)
            Top_feature = [int(i) for i in list(set(miss_list[:, 1] - 1))]

            data = np.loadtxt('../Dataset/' + dataset + '_normalization.csv', delimiter=",", skiprows=0)
            X = data[:, :-1]
            y = data[:, -1]
            cat_cols = []
            y = np.reshape(y, -1)
            num_classes = len(np.unique(y))
            Number = len(y)
            Dimension = len(X[0])
            num_cols = [num for num in range(Dimension)]

            test_all_dic = {}
            test_dim_dic = {}
            train_all_dic = {}
            train_dim_dic = {}
            num_ = 0
            for id in test_list:
                test_all_dic[id] = num_
                num_ += 1
            num_ = 0
            for id in train_list:
                train_all_dic[id] = num_
                num_ += 1
            num_ = 0
            for id in Top_feature:
                test_dim_dic[id] = num_
                num_ += 1
            train_dim_dic = test_dim_dic
            '''
            ==============================================================================
            We divide the dataset in train and test set to show what our framework can do when new data arrives.
            We induce missing_rate with a completely at random mechanism and remove 20 % of elements from the data matrix of both sets.
            We store also the matrices indicating wether an element is missing or not.
            ==============================================================================
            '''

            seed = 42

            # x_train, x_test, y_train, y_test = model_selection.train_test_split(
            #     X, y, test_size=1-train_rate, stratify=y
            # )
            x_train = X[train_list, :]
            y_train = y[train_list]
            x_test = X[test_list, :]
            y_test = y[test_list]

            relationship = np.corrcoef(np.c_[X, y], rowvar=False) * 0.5 + 0.5
            relationship = relationship[-1, :]
            rank = np.argsort(-relationship, axis=0)[1:]

            # cx_train, cx_train_mask = degrade_dataset(x_train, train_missing_rate, seed, rank, np.nan)
            # cx_test, cx_test_mask = degrade_dataset(x_test, missing_rate, seed, rank, np.nan)
            cx_train_mask = np.ones(x_train.shape)
            cx_train = x_train
            cx_test_mask = np.ones(x_test.shape)
            cx_test = x_test
            for list_ in miss_list:
                dim = int(list_[1] - 1)
                number = int(list_[0])
                if int(list_[0]) in test_list:
                    cx_test_mask[test_all_dic[number]][dim] = 0
                    cx_test[test_all_dic[number]][dim] = np.nan
                if int(list_[0]) in train_list:
                    cx_train_mask[train_all_dic[number]][dim] = 0
                    cx_train[train_all_dic[number]][dim] = np.nan

            cx_tr = np.c_[cx_train, y_train]
            cx_te = np.c_[cx_test, y_test]

            mask_tr = np.c_[cx_train_mask, np.ones(y_train.shape)]
            mask_te = np.c_[cx_test_mask, np.ones(y_test.shape)]

            '''
            ==============================================================================
            Here we proprecess the data applying a one - hot encoding for the categorical variables.
            We get the encoded dataset three different masks that indicates the missing features and if these features are categorical or numerical, 
            plus the new columns for the categorical variables with their one-hot range.
            ==============================================================================
            '''

            [oh_x, oh_mask, oh_num_mask, oh_cat_mask, oh_cat_cols] = data2onehot(
                np.r_[cx_tr, cx_te], np.r_[mask_tr, mask_te], num_cols, cat_cols
            )

            '''
            ==============================================================================
            We scale the features with a min max scaler that will preserve the one-hot encoding
            ==============================================================================
            '''

            oh_x_tr = oh_x[:x_train.shape[0], :]
            oh_x_te = oh_x[x_train.shape[0]:, :]

            oh_mask_tr = oh_mask[:x_train.shape[0], :]
            oh_num_mask_tr = oh_mask[:x_train.shape[0], :]
            oh_cat_mask_tr = oh_mask[:x_train.shape[0], :]

            oh_mask_te = oh_mask[x_train.shape[0]:, :]
            oh_num_mask_te = oh_mask[x_train.shape[0]:, :]
            oh_cat_mask_te = oh_mask[x_train.shape[0]:, :]

            scaler_tr = preprocessing.MinMaxScaler()          # Data preprocessing with minmaxscaler on training set
            oh_x_tr = scaler_tr.fit_transform(oh_x_tr)        #

            scaler_te = preprocessing.MinMaxScaler()          # Data preprocessing with minmaxscaler on test set
            oh_x_te = scaler_te.fit_transform(oh_x_te)


            '''
            ==============================================================================
            Now we are ready to impute the missing values on the training set!
            ==============================================================================
            '''
            start = time.clock()
            imputer = GINN(oh_x_tr,
                           oh_mask_tr,
                           oh_num_mask_tr,
                           oh_cat_mask_tr,
                           oh_cat_cols,
                           num_cols,
                           cat_cols
                           )

            imputer.fit()
            imputed_tr = scaler_tr.inverse_transform(imputer.transform())

            '''
            ==============================================================================
            ### OR ###
            # imputed_ginn = scaler_tr.inverse_transform(imputer.fit_transorm())
            # for the one-liners
            # %% md
            In case arrives new data, you can just reuse the model...
            *Add the new data *Impute!
            ==============================================================================
            '''
            imputer.add_data(oh_x_te, oh_mask_te, oh_num_mask_te, oh_cat_mask_te)

            imputed_te = imputer.transform()
            imputed_te = scaler_te.inverse_transform(imputed_te[x_train.shape[0]:])

            end = time.clock()
            time_ = end - start

            MSE_final = np.mean(((1-oh_mask_te) * oh_x_te - (1-oh_mask_te)*imputed_te)**2) / np.mean(1-oh_mask_te)
            MAE_final = np.mean(np.sqrt(((1-oh_mask_te) * oh_x_te - (1-oh_mask_te)*imputed_te)**2)) / np.mean(1-oh_mask_te)
            RMSE = np.sqrt(MSE_final)
            print(dataset + 'Final RMSE: ' + str(RMSE))
            print(dataset + 'Final MAE: ' + str(MAE_final))
            RMSE_D.append(RMSE)
            MAE_D.append(MAE_final)
            time_all.append(time_)

