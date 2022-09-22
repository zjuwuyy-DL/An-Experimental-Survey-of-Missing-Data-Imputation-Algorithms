"""
Main file to run VAE for missing data imputation.
Presented at IFAC MMM2018 by JT McCoy, RS Kroon and L Auret.

Based on implementations
of VAEs from:
    https://github.com/twolffpiggott/autoencoders
    https://jmetzen.github.io/2015-11-27/vae.html
    https://github.com/lazyprogrammer/machine_learning_examples/blob/master/unsupervised_class3/vae_tf.py
    https://github.com/deep-learning-indaba/practicals2017/blob/master/practical5.ipynb

VAE is designed to handle real-valued data, not binary data, so the source code
has been adapted to work only with Gaussians as the output of the generative
model (p(x|z)).
"""

import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencoders import TFVariationalAutoencoder
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
'''
==============================================================================
# DEFINE HYPERPARAMETERS
==============================================================================
'''


#dataset_list = ['EEG', 'Abalone', 'Wireless', 'Yeast', 'Balance_Scale', 'Wine', 'Connect', 'Chess', 'Letter', 'Turkiye', 'Car_evaluation', 'Website_Phishing', 'Anuran_Calls', 'Heart']
dataset_list = ['Wine']
Epoch_number = 1

mechanism = 'MCAR'
missing_rate_list = [0.2]
for missing_rate in missing_rate_list:
    for dataset in dataset_list:
        MAE_D = []
        RMSE_D = []
        Norm_D = []
        Norm1_D = []
        time_all = []
        for epoch in range(Epoch_number):

            epoch_num = epoch + 1
            test_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_test.csv', dtype=int, delimiter=",", skiprows=0)
            train_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_train.csv', dtype=int, delimiter=",", skiprows=0)
            miss_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_miss.csv', delimiter=",", skiprows=0)
            Top_feature = [int(i) for i in list(set(miss_list[:, 1] - 1))]

            training_rate = 0.8
            epoch_number = 5

            Xdata_df = pd.read_csv('../Dataset/'+dataset + '_normalization.csv')
            Xdata_df = Xdata_df.drop(Xdata_df.columns[len(Xdata_df.columns) - 1], axis=1)
            Data_number = len(Xdata_df)
            dimension = Xdata_df.shape[1]


            # Xdata_df, test = train_test_split(Xdata_df, test_size = 1-training_rate)

            test = Xdata_df.loc[test_list, :]
            Train = Xdata_df.loc[train_list, :]

            Each_Missing_number = int(missing_rate * len(test))
            Test_original = copy.deepcopy(test)
            Train_number = len(Xdata_df)
            Test_number = len(test)
            Missing_number = int(missing_rate * Test_number)


            def spike_in_generation(All_data, miss_list):
                spike_in_ = pd.DataFrame(np.zeros_like(All_data), columns=All_data.columns)
                for list_ in miss_list:
                    node_num = list_[0]
                    feature_num = All_data.columns[list_[1] - 1]
                    spike_in_.loc[node_num, feature_num] = 1
                # spike_in = spike_in_.loc[test_list, :]
                return spike_in_

            # Missing_number = Each_Missing_number * len(Top_feature)
            # spike_in = spike_in_generation(test_list, Xdata_df, miss_list)
            all_spike = spike_in_generation(Xdata_df, miss_list)
            spike_in = all_spike.loc[test_list, :]
            # train_spike_in = all_spike[train_list, :]
            start = time.clock()
            # VAE network size: In VAE, there are two encoder layers and two decoder layers.
            Decoder_hidden1 = 20
            Decoder_hidden2 = 20
            Encoder_hidden1 = 20
            Encoder_hidden2 = 20
            # dimensionality of latent space:
            latent_size = 5
            # training parameters:
            training_epochs = 50  # Iterations number
            batch_size = 10      # The number of training tuples in each batch.
            learning_rate = 0.001

            # specify number of imputation iterations:
            ImputeIter = 25
            '''
            ==============================================================================
            # LOAD DATA AND PRECESSING
            ==============================================================================
            '''
            test[spike_in == 1] = np.nan
            test = copy.deepcopy(test.values)
            Xdata_df[all_spike == 1] = np.nan
            Xdata_df = copy.deepcopy(Xdata_df.values)
            # Xdata_df = sc.transform(Xdata_df)

            '''
            ==============================================================================
            INITIALISE AND TRAIN VAE
            ==============================================================================
            '''
            # define dict for network structure:
            network_architecture = \
                dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
                     n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
                     n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
                     n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
                     n_input=dimension, # data input size
                     n_z=latent_size)  # dimensionality of latent space

            # initialise VAE:
            vae = TFVariationalAutoencoder(network_architecture,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size)

            # train VAE on corrupted data:
            vae = vae.train(XData=Xdata_df,
                            training_epochs=training_epochs)

            '''
            ==============================================================================
            IMPUTE MISSING VALUES
            ==============================================================================
            '''
            # impute missing values:
            X_impute = vae.impute(X_corrupt=test, max_iter=ImputeIter)

            end = time.clock()
            time_ = end - start
            X_impute[spike_in == 0] = 0
            Test_original[spike_in == 0] = 0

#            predict_name = '1.Result/'+mechanism + '/prediction/' + dataset+'_'+mechanism+'_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch+1) + '_GINN_prediction.csv'
            test_all_dic = {}
            num_ = 0
            for id in test_list:
                test_all_dic[id] = num_
                num_ += 1
            predict_list = []
            for list_ in miss_list:
                num_id = int(list_[0])
                dim_id = int(list_[1])
                if num_id in test_list:
                    predict_list.append([num_id, dim_id, X_impute[test_all_dic[num_id], dim_id-1]])

            # df = pd.DataFrame(predict_list)
            # df.to_csv(predict_name, index=False, header=False)

            missing_number = np.sum(np.sum(spike_in.values))

            output = np.nan_to_num(X_impute-Test_original)
            RMSE = np.sqrt(np.sum((np.sum(output ** 2))) / missing_number)
            MAE = np.sum((np.sum(np.nan_to_num(np.abs(output)))))/missing_number
            Norm = (np.sum((np.sum(np.nan_to_num(np.abs(output)))))**2)/missing_number
            Norm1 = (np.sum((np.sum(np.nan_to_num(np.abs(output)))))/missing_number)**2


            print(dataset+' == RMSE: ', RMSE)
            print(dataset+' == MAE: ', MAE)

