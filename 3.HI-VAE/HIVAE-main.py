#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import tensorflow as tf
import graph_new
import time
import numpy as np
#import plot_functions
import read_functions
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)


def parse_args(input):
    [model, m_perc, batch_size, epoch, train, dataset, missing_name] = input
    parser = argparse.ArgumentParser(description='Default parameters of the models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Size of the batches')
    parser.add_argument('--data_file', type=str, default='../Dataset/'+dataset+'_normalization.csv', help='File with the data')
    parser.add_argument('--types_file', type=str, default='../Dataset/data_types.csv',
                        help='File with the types of the data')
    parser.add_argument('--miss_file', type=str, default=missing_name, help='File with the missing indexes mask')
    parser.add_argument('--true_miss_file', type=int, default=0,
                        help='File with the missing indexes when there are NaN in the data')
    parser.add_argument('--model_name', type=str, default=model, help='File of the training model')
    parser.add_argument('--dim_latent_s', type=int, default=4, help='Dimension of the categorical space')
    parser.add_argument('--dim_latent_z', type=int, default=2, help='Dimension of the Z latent space')
    parser.add_argument('--dim_latent_y', type=int, default=3, help='Dimension of the Y latent space')
    parser.add_argument('--epochs', type=int, default=epoch, help='Number of epochs of the simulations')
    parser.add_argument('--train', type=int, default=train, help='Training model flag')
    parser.add_argument('--restore', type=int, default=0, help='To restore session, to keep training or evaluation')
    parser.add_argument('--save_file', type=str, default=model+'_dataset_Missing'+str(m_perc)+'_1_z2_y3_s4_batch' + str(batch_size),
                        help='Save file name')

    parser.add_argument('--miss_percentage_train', type=float, default=0.0,
                        help='Percentage of missing data in training')
    parser.add_argument('--miss_percentage_test', type=float, default=0.0, help='Percentage of missing data in test')


    parser.add_argument('--perp', type=int, default=10, help='Perplexity for the t-SNE')
    parser.add_argument('--display', type=int, default=1, help='Display option flag')
    parser.add_argument('--save', type=int, default=1000, help='Save variables every save iterations')
    parser.add_argument('--plot', type=int, default=1, help='Plot results flag')
    parser.add_argument('--dim_latent_y_partition', type=int, nargs='+', help='Partition of the Y latent space')

    return parser.parse_args()


def print_loss(epoch, start_time, avg_loss, avg_test_loglik, avg_KL_s, avg_KL_z):
    print("Epoch: [%2d]  time: %4.4f, train_loglik: %.8f, KL_z: %.8f, KL_s: %.8f, ELBO: %.8f, Test_loglik: %.8f"
          % (epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss-avg_KL_z-avg_KL_s, avg_test_loglik))

def main(args):
    #Create a directoy for the save file
    if not os.path.exists('./Saved_Networks/' + args.save_file):
        os.makedirs('./Saved_Networks/' + args.save_file)

    train_data, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
    # Check batch size
    if args.batch_size > n_samples:
        args.batch_size = n_samples
    # Get an integer number of batches
    n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
    # Compute the real miss_mask
    miss_mask = np.multiply(miss_mask, true_miss_mask)

    #Creating graph
    sess_HVAE = tf.Graph()

    with sess_HVAE.as_default():

        tf_nodes = graph_new.HVAE_graph(args.model_name, args.types_file, args.batch_size,
                                    learning_rate=1e-3, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, y_dim_partition=args.dim_latent_y_partition)

    ################### Running the VAE Training #################################

    with tf.Session(graph=sess_HVAE, config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()

        print('Training the HVAE ...')
        if(args.train == 1):

            start_time = time.time()
            # Training cycle

            loglik_epoch = []
            testloglik_epoch = []
            error_train_mode_global = []
            error_test_mode_global = []
            KL_s_epoch = []
            KL_z_epoch = []
            for epoch in range(args.epochs):
                avg_loss = 0.
                avg_KL_s = 0.
                avg_KL_z = 0.
                samples_list = []
                p_params_list = []
                q_params_list = []
                log_p_x_total = []
                log_p_x_missing_total = []

                # Annealing of Gumbel-Softmax parameter
                tau = np.max([1.0 - 0.01*epoch,1e-3])
    #            tau = 1e-3
                tau2 = np.min([0.001*epoch,1.0])

                #Randomize the data in the mini-batches
                random_perm = np.random.permutation(range(np.shape(train_data)[0]))
                train_data_aux = train_data[random_perm,:]
                miss_mask_aux = miss_mask[random_perm,:]
                true_miss_mask_aux = true_miss_mask[random_perm,:]

                for i in range(n_batches):

                    #Create inputs for the feed_dict
                    data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, args.batch_size, index_batch=i)

                    #Delete not known data (input zeros)
                    data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]

                    #Create feed dictionary
                    feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                    feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                    feedDict[tf_nodes['miss_list']] = miss_list
                    feedDict[tf_nodes['tau_GS']] = tau
                    feedDict[tf_nodes['tau_var']] = tau2

                    #Running VAE
                    _, loss, KL_z, KL_s, samples, log_p_x, log_p_x_missing, p_params, q_params = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                                                             tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'], tf_nodes['p_params'], tf_nodes['q_params']],
                                                             feed_dict=feedDict)

                    samples_test, log_p_x_test, log_p_x_missing_test, test_params = session.run([tf_nodes['samples_test'], tf_nodes['log_p_x_test'], tf_nodes['log_p_x_missing_test'], tf_nodes['test_params']],
                                                                 feed_dict=feedDict)


                    #Evaluate results on the imputation with mode, not on the samlpes!
                    samples_list.append(samples_test)
                    p_params_list.append(test_params)
        #                        p_params_list.append(p_params)
                    q_params_list.append(q_params)
                    log_p_x_total.append(log_p_x_test)
                    log_p_x_missing_total.append(log_p_x_missing_test)

                    # Compute average loss
                    avg_loss += np.mean(loss)
                    avg_KL_s += np.mean(KL_s)
                    avg_KL_z += np.mean(KL_z)

                #Concatenate samples in arrays
                s_total, z_total, y_total, est_data = read_functions.samples_concatenation(samples_list)

                #Transform discrete variables back to the original values
                train_data_transformed = read_functions.discrete_variables_transformation(train_data_aux[:n_batches*args.batch_size,:], types_dict)
                est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
                est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask_aux[:n_batches*args.batch_size,:], types_dict)

    #            est_data_transformed[np.isinf(est_data_transformed)] = 1e20

                #Create global dictionary of the distribution parameters
                p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, args.dim_latent_z, args.dim_latent_s)
                q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  args.dim_latent_z, args.dim_latent_s)

                #Number of clusters created
                cluster_index = np.argmax(q_params_complete['s'],1)
                cluster = np.unique(cluster_index)
                # print('Clusters: ' + str(len(cluster)))

                #Compute mean and mode of our loglik models
                loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)
    #            loglik_mean[np.isinf(loglik_mean)] = 1e20

                #Try this for the errors
                error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
                error_train_mode, error_test_mode = read_functions.error_computation(train_data_transformed, loglik_mode, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
                error_train_samples, error_test_samples = read_functions.error_computation(train_data_transformed, est_data_transformed, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
                error_train_imputed, error_test_imputed = read_functions.error_computation(train_data_transformed, est_data_imputed, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])

                #Compute test-loglik from log_p_x_missing
                log_p_x_total = np.transpose(np.concatenate(log_p_x_total,1))
                log_p_x_missing_total = np.transpose(np.concatenate(log_p_x_missing_total,1))
                if args.true_miss_file:
                    log_p_x_missing_total = np.multiply(log_p_x_missing_total,true_miss_mask_aux[:n_batches*args.batch_size,:])
                avg_test_loglik = np.sum(log_p_x_missing_total)/np.sum(1.0-miss_mask_aux)

                # Display logs per epoch step
                if epoch % args.display == 0:
                    print_loss(epoch, start_time, avg_loss/n_batches, avg_test_loglik, avg_KL_s/n_batches, avg_KL_z/n_batches)
                    print('Test error mode: ' + str(np.round(np.mean(error_test_mode),5)))
                    print("")

                #Compute train and test loglik per variables
                loglik_per_variable = np.sum(log_p_x_total,0)/np.sum(miss_mask_aux,0)
                loglik_per_variable_missing = np.sum(log_p_x_missing_total,0)/np.sum(1.0-miss_mask_aux,0)

                #Store evolution of all the terms in the ELBO
                loglik_epoch.append(loglik_per_variable)
                testloglik_epoch.append(loglik_per_variable_missing)
                KL_s_epoch.append(avg_KL_s/n_batches)
                KL_z_epoch.append(avg_KL_z/n_batches)
                error_train_mode_global.append(error_train_mode)
                error_test_mode_global.append(error_test_mode)

            # print('error_train_mode_global: ', error_train_mode_global)
            print('Training Finished ...')
            RMSE = 0
            MAE  = 0
            Value = 0
        #Test phase
        else:

            start_time = time.time()
            # Training cycle

    #        f_toy2, ax_toy2 = plt.subplots(4,4,figsize=(8, 8))
    #         loglik_epoch = []
    #         testloglik_epoch = []
    #         error_train_mode_global = []
            error_test_mode_global = []
            # error_imputed_global = []
            # est_data_transformed_total = []

            # Only one epoch needed, since we are doing mode imputation
            for epoch in range(args.epochs):
                avg_loss = 0.
                avg_KL_s = 0.
                avg_KL_y = 0.
                avg_KL_z = 0.
                samples_list = []
                p_params_list = []
                q_params_list = []
                log_p_x_total = []
                log_p_x_missing_total = []

                label_ind = 2

                # Constant Gumbel-Softmax parameter (where we have finished the annealing)
                tau = 1e-3
    #            tau = 1.0

                # Randomize the data in the mini-batches
        #        random_perm = np.random.permutation(range(np.shape(data)[0]))
                random_perm = range(np.shape(train_data)[0])
                train_data_aux = train_data[random_perm,:]
                miss_mask_aux = miss_mask[random_perm,:]
                true_miss_mask_aux = true_miss_mask[random_perm,:]

                for i in range(n_batches):

                    #Create train minibatch
                    data_list, miss_list = read_functions.next_batch(train_data_aux, types_dict, miss_mask_aux, args.batch_size,
                                                                     index_batch=i)
    #                print(np.mean(data_list[0],0))

                    #Delete not known data
                    data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]


                    #Create feed dictionary
                    feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                    feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                    feedDict[tf_nodes['miss_list']] = miss_list
                    feedDict[tf_nodes['tau_GS']] = tau

                    #Get samples from the model
                    loss, KL_z, KL_s, samples, log_p_x, log_p_x_missing, p_params, q_params  = session.run([tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'],
                                                                 tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'],tf_nodes['q_params']],
                                                                 feed_dict=feedDict)

                    samples_test, log_p_x_test, log_p_x_missing_test, test_params  = session.run([tf_nodes['samples_test'],tf_nodes['log_p_x_test'],tf_nodes['log_p_x_missing_test'],tf_nodes['test_params']],
                                                                 feed_dict=feedDict)


                    samples_list.append(samples_test)
                    p_params_list.append(test_params)
        #                        p_params_list.append(p_params)
                    q_params_list.append(q_params)
                    log_p_x_total.append(log_p_x_test)
                    log_p_x_missing_total.append(log_p_x_missing_test)

                #Separate the samples from the batch list
                s_aux, z_aux, y_total, est_data = read_functions.samples_concatenation(samples_list)

                # Transform discrete variables to original values
                train_data_transformed = read_functions.discrete_variables_transformation(train_data_aux[:n_batches*args.batch_size,:], types_dict)
                est_data_transformed = read_functions.discrete_variables_transformation(est_data, types_dict)
                est_data_imputed = read_functions.mean_imputation(train_data_transformed, miss_mask_aux[:n_batches*args.batch_size,:], types_dict)

                #Create global dictionary of the distribution parameters
                p_params_complete = read_functions.p_distribution_params_concatenation(p_params_list, types_dict, args.dim_latent_z, args.dim_latent_s)
                q_params_complete = read_functions.q_distribution_params_concatenation(q_params_list,  args.dim_latent_z, args.dim_latent_s)

                #Number of clusters created
                cluster_index = np.argmax(q_params_complete['s'],1)
                cluster = np.unique(cluster_index)
                # print('Clusters: ' + str(len(cluster)))

                #Compute mean and mode of our loglik models
                loglik_mean, loglik_mode = read_functions.statistics(p_params_complete['x'],types_dict)

                #Try this for the errors
                error_train_mean, error_test_mean = read_functions.error_computation(train_data_transformed, loglik_mean, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
                error_mae, error_test_mode = read_functions.error_computation_test(train_data_transformed, loglik_mode, types_dict, miss_mask_aux[:n_batches*args.batch_size,:])
                RMSE = np.round(np.mean(error_test_mode), 4)
                MAE = np.round(np.mean(error_mae), 4)
                print('RMSE: ' + str(RMSE))
                print('MAE: ' + str(MAE))
    return RMSE, MAE, loglik_mode



if __name__ == "__main__":
    model = "model_HIVAE_inputDropout"
    # Initialization settings
    m_perc_list = [20, 40, 60, 80]
    m_perc = 80
    training_rate = 0.8
    dataset_list = ['Abalone', 'Wine', 'Anuran_Calls', 'EEG', 'Yeast', 'Car_evaluation', 'Website_Phishing', 'Turkiye', 'Balance_Scale', 'Heart', 'Wireless', 'Chess', 'Letter', 'Connect']
    # dataset_list = ['Abalone']
    Epoch_number = 5
    mechanism_list = ['MCAR']
    # for m_perc in m_perc_list:
    for mechanism in mechanism_list:
        for dataset in dataset_list:
            MAE_D = []
            RMSE_D = []
            time_all = []
            for epoch in range(Epoch_number):

                epoch_num = epoch + 1
                test_list = np.loadtxt('../Dataset/' + dataset + '/' + mechanism + '/' + mechanism + '_missing_rate_' + str(int(m_perc)) + '_Epoch' + str(epoch_num) + '_test.csv', dtype=int, delimiter=",", skiprows=0)
                miss_list = np.loadtxt('../Dataset/' + dataset + '/' + mechanism + '/' + mechanism + '_missing_rate_' + str(int(m_perc)) + '_Epoch' + str(epoch_num) + '_miss.csv', delimiter=",", skiprows=0)
                Top_feature = [int(i) for i in list(set(miss_list[:, 1] - 1))]
                missing_name = '../Dataset/' + dataset + '/' + mechanism + '/' + mechanism + '_missing_rate_' + str(int(m_perc)) + '_Epoch' + str(epoch_num) + '_miss.csv'
                Imput = [model, m_perc, 1000, 100, 1, dataset, missing_name]
                args = parse_args(Imput)
                start = time.clock()
                RMSE, MAE, non = main(args)
                Imput = [model, m_perc, 10000000, 1, 0, dataset, missing_name]
                args = parse_args(Imput)
                RMSE, MAE, Value = main(args)

                predict_name = '1.Result/' + mechanism + '/prediction/' + dataset+'_' + mechanism + '_missing_rate_' + str(int(m_perc)) + '_Epoch' + str(epoch + 1) + '_GINN_prediction.csv'
                predict_list = []
                for list_ in miss_list:
                    num_id = int(list_[0])
                    dim_id = int(list_[1])
                    if num_id in test_list:
                        predict_list.append([num_id, dim_id, Value[num_id, dim_id-1]])
                df = pd.DataFrame(predict_list)
                df.to_csv(predict_name, index=False, header=False)

                end = time.clock()
                time_ = end - start
                time_all.append(time_)
                RMSE_D.append(MAE)
                MAE_D.append(RMSE)
            Average_RMSE = round(np.average(RMSE_D), 4)
            Average_MAE = round(np.average(MAE_D), 4)
            Average_time = round(np.average(time_all), 10)
            Max_RMSE = round(np.max(RMSE_D), 4)
            Min_RMSE = round(np.min(RMSE_D), 4)
            Max_MAE = round(np.max(MAE_D), 4)
            Min_MAE = round(np.min(MAE_D), 4)
            Max_time = round(np.max(time_all), 10)
            Min_time = round(np.min(time_all), 10)
            print(dataset + ' == Average_RMSE: ', Average_RMSE, '  Average_MAE: ', Average_MAE, '  Average_time: ', Average_time)

            All_name = ['', '1', '2', '3', '4', '5', 'Average', 'Bias']
            RMSE_D.append(Average_RMSE)
            RMSE_D.append(round((Max_RMSE - Min_RMSE) / 2, 4))
            MAE_D.append(Average_MAE)
            MAE_D.append(round((Max_MAE - Min_MAE) / 2, 4))
            time_all.append(Average_time)
            time_all.append(round((Max_time - Min_time) / 2, 4))
            ALL = []
            ALL.append(All_name)
            ALL.append(['RMSE'] + RMSE_D)
            ALL.append(['MAE'] + MAE_D)
            ALL.append(['Time'] + time_all)
            df = pd.DataFrame(ALL)
            df.to_csv('1.Result/' + mechanism + '/' + str(int(m_perc)) + 'Result-GAIN-' + dataset + '.csv', index=False, header=False)
