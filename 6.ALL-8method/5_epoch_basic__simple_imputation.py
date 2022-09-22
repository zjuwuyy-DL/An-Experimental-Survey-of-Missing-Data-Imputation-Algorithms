from fancyimpute import KNN,  MatrixFactorization, SoftImpute
from ycimpute.imputer.mice import MICE
from ycimpute.imputer.iterforest import IterImput
from missingpy import MissForest
from MissingImputer import MissingImputer
import numpy as np
import math
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
import time

'''
====================
Data precessing
====================
'''
#data_list = ['Abalone', 'Wine', 'Anuran_Calls', 'Yeast', 'Car_evaluation', 'Website_Phishing', 'Turkiye', 'Balance_Scale', 'Heart', 'Wireless', 'EEG', 'Chess', 'Letter']
data_list = ['Wine']
missing_rate = 0.2
training_rate = 0.8
epoch_number = 5
mechanism = 'MCAR'
Method_list = ['Mean', 'KNN', 'MICE', 'XGBoost', 'RF', 'MissF', 'MC', 'MatrixF']

for dataset in data_list:
    each_rmse = {}
    each_mae = {}
    time_all = {}
    for i in range(8):
        each_rmse[i] = []
        each_mae[i] = []
        time_all[i] = []
    for epoch in range(epoch_number):
        epoch_num = epoch + 1
        test_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_test.csv', dtype=int, delimiter=",", skiprows=0)
        train_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_train.csv', dtype=int, delimiter=",", skiprows=0)
        miss_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch_num) + '_miss.csv', delimiter=",", skiprows=0)
        Top_feature = [int(i) for i in list(set(miss_list[:, 1] - 1))]

        All_data = pd.read_csv('../Dataset/' + dataset + '_normalization.csv')
        All_data = All_data.drop(All_data.columns[len(All_data.columns)-1], axis=1)
        Data_number = len(All_data)
        dimension = All_data.shape[1]

        Top_columns = All_data.columns
        Top_columns = Top_columns[Top_feature]

        # Train, test = train_test_split(All_data, test_size=1-training_rate)
        test = All_data.loc[test_list, :]
        Train = All_data.loc[train_list, :]

        Test_original = test.values
        Train_number = len(Train)
        Test_number = len(test)
        Each_Missing_number = int(missing_rate * Test_number)

        def spike_in_generation(All_data, miss_list):
            spike_in_ = pd.DataFrame(np.zeros_like(All_data), columns=All_data.columns)
            for list_ in miss_list:
                node_num = list_[0]
                feature_num = All_data.columns[list_[1]-1]
                spike_in_.loc[node_num, feature_num] = 1
            # spike_in = spike_in_.loc[test_list, :]
            return spike_in_
        Missing_number = Each_Missing_number * len(Top_feature)
        all_spike = spike_in_generation(All_data, miss_list)

        all_complete = copy.deepcopy(All_data).values
        All_data[all_spike == 1] = np.nan
        all = All_data.values



        # for epoch in range(epoch_number):
        '''
        ====================
        Training KNN and MatrixFactorization
        ====================
        '''
        # Mean
        Mean_start = time.clock()
        Mean_attribute = np.zeros((dimension))
        Mean_number = np.zeros((dimension))
        all_complete_spike = np.copy(all_complete)
        all_complete_spike[all_spike == 1] = 10000
        for tuple in all_complete_spike:
            attribute_number = 0
            for attribute in tuple:
                if attribute != 10000:
                    Mean_attribute[attribute_number] += float(attribute)
                    Mean_number[attribute_number] += 1
                attribute_number += 1
        Mean_each_attribute = []
        for i in range(dimension):
            Mean_each_attribute.append(Mean_attribute[i] / float(Mean_number[i]))
        Mean_end = time.clock()
        Mean_time = Mean_end - Mean_start
        time_all[0].append(Mean_time)
        # KNN
        KNN_start = time.clock()
        X_filled_knn = KNN(k=5).fit_transform(all)
        X_filled_knn[all_spike == 0] = 0
        KNN_end = time.clock()
        KNN_time = KNN_end - KNN_start
        time_all[1].append(KNN_time)

        # MICE
        MICE_start = time.clock()
        X_filled_MICE = MICE().complete(all)
        X_filled_MICE[all_spike == 0] = 0
        MICE_end = time.clock()
        MICE_time = MICE_end - MICE_start
        time_all[2].append(MICE_time)

        # XGBoost
        XGBoost_start = time.clock()
        X_XGBoost_Impute = MissingImputer(ini_fill=True, model_reg="xgboost", model_clf="xgboost")
        X_XGBoost = X_XGBoost_Impute.fit(all).transform(all.copy())
        X_XGBoost[all_spike == 0] = 0
        XGBoost_end = time.clock()
        XGBoost_time = XGBoost_end - XGBoost_start
        time_all[3].append(XGBoost_time)

        # Random Forest
        RF_start = time.clock()
        X_IterImput = IterImput().complete(all)
        X_IterImput[all_spike == 0] = 0
        RF_end = time.clock()
        RF_time = RF_end - RF_start
        time_all[4].append(RF_time)

        # MissForest
        MissForest_start = time.clock()
        X_MissForest = MissForest().fit_transform(all)
        X_MissForest[all_spike == 0] = 0
        MissForest_end = time.clock()
        MissForest_time = MissForest_end - MissForest_start
        time_all[5].append(MissForest_time)

        # Matrix Completion
        MC_start = time.clock()
        clf = SoftImpute(J=2, lambda_=0.0)
        clf.fit(all)
        Soft_Impute = clf.predict(all)
        Soft_Impute[all_spike == 0] = 0
        MC_end = time.clock()
        MC_time = MC_end - MC_start
        time_all[6].append(MC_time)

        # MatrixFactorization
        MF_start = time.clock()
        X_filled_M_F = MatrixFactorization().fit_transform(all)
        X_filled_M_F[all_spike == 0] = 0
        MF_end = time.clock()
        MF_time = MF_end - MF_start
        time_all[7].append(MF_time)

        '''
        ====================
        Output the results
        ====================
        '''
        all_complete[all_spike == 0] = 0
        test_spike = all_spike.loc[test_list, :]
        Test_original[test_spike == 0] = 0
        spike_in = test_spike.values
        Missing_number = np.sum(np.sum(test_spike))

        missing_num = np.sum(np.sum(spike_in))

        XMean_mse = 0
        XMean_mae = 0
        predict_list = []
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+Method_list[0]+ '_prediction.csv'
        for tup in range(len(Test_original)):
            for attr in range(dimension):
                if spike_in[tup][attr] != 0:
                    XMean_mae += (np.sqrt((Test_original[tup][attr]-Mean_each_attribute[attr]) ** 2))
                    XMean_mse += ((Test_original[tup][attr]-Mean_each_attribute[attr]) ** 2)
        for id in miss_list:
            num_point = int(id[0])
            if int(id[0]) in test_list:
                dim = int(id[1])
                predict_list.append([num_point, dim, Mean_each_attribute[dim-1]])
        # print("1. Mean RMSE: %f" % np.sqrt(XMean_mse/Missing_number), " =====  MAE: %f" % round(XMean_mae/Missing_number, 4))
        each_rmse[0].append(round(np.sqrt(XMean_mse/Missing_number), 4))
        each_mae[0].append(round(XMean_mae/Missing_number, 4))
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


        X_filled_knn = X_filled_knn[test_list, :]
        knn_mse = np.sum((X_filled_knn - Test_original) ** 2)
        knn_mae = np.sum(np.sqrt((X_filled_knn - Test_original) ** 2))
        each_rmse[1].append(np.sqrt(knn_mse / Missing_number))
        each_mae[1].append(round(knn_mae / Missing_number, 4))
        test_all_dic = {}
        num_ = 0
        for id in test_list:
            test_all_dic[id] = num_
            num_ += 1
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+ Method_list[1]+ '_prediction.csv'
        predict_list = []
        for id in miss_list:
            num_id = int(id[0])
            dim_id = int(id[1])
            if num_id in test_list:
                predict_list.append([num_id, dim_id, X_filled_knn[test_all_dic[num_id], dim_id-1]])
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


        X_filled_MICE = X_filled_MICE[test_list, :]
        MICE_mse = np.sum((X_filled_MICE - Test_original) ** 2)
        MICE_mae = np.sum(np.sqrt((X_filled_MICE - Test_original) ** 2))
        # print("3. MICE RMSE: %f" % np.sqrt(MICE_mse/Missing_number), " =====  MAE: %f" % round(MICE_mae/Missing_number, 4))
        each_rmse[2].append(np.sqrt(MICE_mse / Missing_number))
        each_mae[2].append(round(MICE_mae / Missing_number, 4))
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+ Method_list[2]+ '_prediction.csv'
        predict_list = []
        for id in miss_list:
            num_id = int(id[0])
            dim_id = int(id[1])
            if num_id in test_list:
                predict_list.append([num_id, dim_id, X_filled_MICE[test_all_dic[num_id], dim_id-1]])
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


        X_XGBoost = X_XGBoost[test_list, :]
        XGBoost_mse = np.sum((X_XGBoost - Test_original) ** 2)
        XGBoost_mae = np.sum(np.sqrt((X_XGBoost - Test_original) ** 2))
        # print("5. XGBoost RMSE: %f" % np.sqrt(XGBoost_mse/Missing_number), " =====  MAE: %f" % round(XGBoost_mae/Missing_number, 4))
        each_rmse[3].append(np.sqrt(XGBoost_mse / Missing_number))
        each_mae[3].append(round(XGBoost_mae / Missing_number, 4))
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+ Method_list[3]+ '_prediction.csv'
        predict_list = []
        for id in miss_list:
            num_id = int(id[0])
            dim_id = int(id[1])
            if num_id in test_list:
                predict_list.append([num_id, dim_id, X_XGBoost[test_all_dic[num_id], dim_id-1]])
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


        X_IterImput = X_IterImput[test_list, :]
        IterImput_mse = np.sum((X_IterImput - Test_original) ** 2)
        IterImput_mae = np.sum(np.sqrt((X_IterImput - Test_original) ** 2))
        # print("6. Random Forest RMSE: %f" % np.sqrt(IterImput_mse/Missing_number), " =====  MAE: %f" % round(IterImput_mae/Missing_number, 4))
        each_rmse[4].append(np.sqrt(IterImput_mse / Missing_number))
        each_mae[4].append(round(IterImput_mae / Missing_number, 4))
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+ Method_list[4]+ '_prediction.csv'
        predict_list = []
        for id in miss_list:
            num_id = int(id[0])
            dim_id = int(id[1])
            if num_id in test_list:
                predict_list.append([num_id, dim_id, X_IterImput[test_all_dic[num_id], dim_id-1]])
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


        X_MissForest = X_MissForest[test_list, :]
        MissForest_mse = np.sum((X_MissForest - Test_original) ** 2)
        MissForest_mae = np.sum(np.sqrt((X_MissForest - Test_original) ** 2))
        # print("7. MissForest RMSE: %f" % np.sqrt(MissForest_mse/Missing_number), " =====  MAE: %f" % round(MissForest_mae/Missing_number, 4))
        each_rmse[5].append(np.sqrt(MissForest_mse / Missing_number))
        each_mae[5].append(round(MissForest_mae / Missing_number, 4))
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+ Method_list[5]+ '_prediction.csv'
        predict_list = []
        for id in miss_list:
            num_id = int(id[0])
            dim_id = int(id[1])
            if num_id in test_list:
                predict_list.append([num_id, dim_id, X_MissForest[test_all_dic[num_id], dim_id-1]])
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


        Soft_Impute = Soft_Impute[test_list, :]
        Soft_Impute_mse = np.sum((Soft_Impute - Test_original) ** 2)
        Soft_Impute_mae = np.sum(np.sqrt((Soft_Impute - Test_original) ** 2))
        # print("8. Matrix Completion RMSE: %f" % np.sqrt(Soft_Impute_mse/Missing_number), " =====  MAE: %f" % round(Soft_Impute_mae/Missing_number, 4))
        each_rmse[6].append(np.sqrt(Soft_Impute_mse / Missing_number))
        each_mae[6].append(round(Soft_Impute_mae / Missing_number, 4))
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+ Method_list[6]+ '_prediction.csv'
        predict_list = []
        for id in miss_list:
            num_id = int(id[0])
            dim_id = int(id[1])
            if num_id in test_list:
                predict_list.append([num_id, dim_id, Soft_Impute[test_all_dic[num_id], dim_id-1]])
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


        X_filled_M_F = X_filled_M_F[test_list, :]
        M_F_mse = np.sum((X_filled_M_F - Test_original) ** 2)
        M_F_mae = np.sum(np.sqrt((X_filled_M_F - Test_original) ** 2))
        # print("9. MatrixFactorization RMSE: %f" % np.sqrt(M_F_mse/Missing_number), " =====  MAE: %f" % round(M_F_mae/Missing_number, 4))
        each_rmse[7].append(round(np.sqrt(M_F_mse / Missing_number), 4))
        each_mae[7].append(round(M_F_mae / Missing_number, 4))
        predict_name = '1.Result/' + mechanism + '/prediction/' + dataset + '_' + mechanism + '_missing_rate_' + str(int(missing_rate * 100)) + '_Epoch' + str(epoch + 1) + '_'+ Method_list[7]+ '_prediction.csv'
        predict_list = []
        for id in miss_list:
            num_id = int(id[0])
            dim_id = int(id[1])
            if num_id in test_list:
                predict_list.append([num_id, dim_id, X_filled_M_F[test_all_dic[num_id], dim_id-1]])
        df = pd.DataFrame(predict_list)
        df.to_csv(predict_name, index=False, header=False)


    for i in range(8):
        All_name = ['', '1', '2', '3', '4', '5', 'Average', 'Bias']
        RMSE_D = each_rmse[i]
        MAE_D = each_mae[i]
        each_time = time_all[i]
        Average_RMSE = round(np.average(RMSE_D), 4)
        Average_MAE = round(np.average(MAE_D), 4)
        Average_time = round(np.average(each_time), 10)
        Max_RMSE = round(np.max(RMSE_D), 4)
        Min_RMSE = round(np.min(RMSE_D), 4)
        Max_MAE = round(np.max(MAE_D), 4)
        Min_MAE = round(np.min(MAE_D), 4)
        Max_time = round(np.max(each_time), 4)
        Min_time = round(np.min(each_time), 4)
        print(dataset + '-' + Method_list[i] + '-Average_RMSE: ', Average_RMSE, ', -Average_MAE: ', Average_MAE, ', -Average_time: ', Average_time)
        print(dataset + '-' + Method_list[i] + '-Max_RMSE: ', Max_RMSE, ', -Min_MAE: ', Min_RMSE, ', -Bias1: ', Max_RMSE - Average_RMSE, ', -Bias2: ', Average_RMSE - Min_RMSE)
        RMSE_D.append(Average_RMSE)
        RMSE_D.append(round((Max_RMSE - Min_RMSE) / 2, 4))
        MAE_D.append(Average_MAE)
        MAE_D.append(round((Max_MAE - Min_MAE) / 2, 4))
        each_time.append(Average_time)
        each_time.append(round((Max_time - Min_time) / 2, 4))
        ALL = []
        ALL.append(All_name)
        ALL.append(['RMSE'] + RMSE_D)
        ALL.append(['MAE'] + MAE_D)
        ALL.append(['Time'] + each_time)
        df = pd.DataFrame(ALL)
        df.to_csv('1.Result/' + mechanism + '/' + str(int(missing_rate * 100)) + 'Result-' + Method_list[i] + '-' + dataset + '.csv', index=False, header=False)

