from midas import Midas
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import os
import time
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#dataset_list = ['Abalone', 'Wine', 'Anuran_Calls', 'EEG', 'Yeast', 'Car_evaluation', 'Website_Phishing', 'Turkiye', 'Balance_Scale', 'Heart', 'Wireless', 'Chess', 'Letter', 'Connect']
dataset_list = ['Wine']
Epoch_number = 1
mechanism = 'MCAR'

for dataset_input in dataset_list:
    MAE_D = []
    RMSE_D = []
    time_all = []
    for epoch in range(Epoch_number):

        epoch_num = epoch + 1
        missing_rate = 0.2
        test_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate*100)) + '_Epoch' + str(epoch_num) + '_test.csv', dtype=int, delimiter=",", skiprows=0)
        train_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate*100)) + '_Epoch' + str(epoch_num) + '_train.csv', dtype=int, delimiter=",", skiprows=0)
        miss_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate*100)) + '_Epoch' + str(epoch_num) + '_miss.csv', delimiter=",", skiprows=0)
        Top_feature = [int(i) for i in list(set(miss_list[:, 1] - 1))]


        data_0 = pd.read_csv('../Dataset/'+dataset_input+'_normalization.csv')
        data_0 = data_0.drop(data_0.columns[len(data_0.columns)-1], axis=1)



        #.drop(['Unnamed: 0', 'Num_13'], axis=1)
        Number = len(data_0)
        Dimension = data_0.shape[1]

        training_rate = 0.8
        training_number = int(training_rate*Number)

        Missing_number = int(missing_rate * (Number-training_number))
        Each_Missing_number = int(missing_rate * len(data_0))
        np.random.seed(441)


        def spike_in_generation(All_data, miss_list):
            spike_in_ = pd.DataFrame(np.zeros_like(All_data), columns=All_data.columns)
            for list_ in miss_list:
                node_num = list_[0]
                feature_num = All_data.columns[list_[1] - 1]
                spike_in_.loc[node_num, feature_num] = 1
            # spike_in = spike_in_.loc[test_list, :]
            return spike_in_


        Original_data = copy.deepcopy(pd.DataFrame(data_0, columns=data_0.columns))
        # spike_in = spike_in_generation(test,  Missing_number)
        spike_in = spike_in_generation(data_0, miss_list)
        data_0[spike_in == 1] = np.nan

        columns_list = []
        # scaler = MinMaxScaler()
        # na_loc = data_0.isnull()
        # data_0.fillna(data_0.median(), inplace=True)
        # data_0 = pd.DataFrame(scaler.fit_transform(data_0), columns=data_0.columns)

        # data_0[na_loc] = np.nan
        start = time.clock()

        imputer = Midas(layer_structure=[32], vae_layer=False, seed=42)
        imputer.build_model(data_0, softmax_columns=columns_list)
        imputer.overimpute(training_epochs=100, report_ival=1,
                           report_samples=5, plot_all=False)
        imputer.train_model(training_epochs=200, verbosity_ival=1)

        imputer.batch_generate_samples(m=1)
        imputed_vals = []
        Original_data[spike_in == 0] = 0

        for dataset in imputer.output_list:
            imputed_vals.append(pd.DataFrame(dataset,
                                             columns=dataset.columns))
            imputed = pd.DataFrame(dataset, columns=dataset.columns)
            imputed[spike_in == 0] = 0
            Output_value = imputed.values
            original_value = Original_data.values
            output = Output_value[test_list, :]-original_value[test_list, :]
            missing_number = np.sum(np.sum(spike_in.values[test_list, :]))
            RMSE=np.sqrt(np.sum((np.sum(output**2)))/(missing_number))
            MAE = np.sum((np.sum(np.sqrt(output ** 2)))) / (missing_number)
            end = time.clock()
            time_ = end - start
            print('RMSE: ', RMSE)
            print('MAE: ', MAE)

