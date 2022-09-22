'''
Written by Jinsung Yoon
Date: Jan 29th 2019
Generative Adversarial Imputation Networks (GAIN) Implementation on Spam Dataset
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN.pdf
Appendix Link: http://medianetlab.ee.ucla.edu/papers/ICML_GAIN_Supp.pdf
Contact: jsyoon0823@g.ucla.edu
'''

#%% Packages
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)

#%% System Parameters
# 1. Mini batch size
mb_size = 8
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8
# 6. Epoch Number
Epoch_number = 5
#%% Data

#dataset_list = ['Abalone', 'Wine', 'Anuran_Calls', 'EEG', 'Yeast', 'Car_evaluation', 'Website_Phishing', 'Turkiye', 'Balance_Scale', 'Heart', 'Wireless', 'Chess', 'Letter', 'Connect']
dataset_list = ['Wine']
# mechanism_list = ['MCAR', 'MNAR']
mechanism_list = ['MCAR']
for mechanism in mechanism_list:
    for dataset in dataset_list:
        # Result = open('1.Result/'+mechanism+'/Result-GAIN-'+dataset+'.txt', 'w')
        MAE_D = []
        RMSE_D = []
        time_all = []
        for epoch in range(Epoch_number):
            epoch_num = epoch + 1
            test_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(p_miss * 100)) + '_Epoch' + str(epoch_num) + '_test.csv', dtype=int, delimiter=",", skiprows=0)
            train_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(p_miss * 100)) + '_Epoch' + str(epoch_num) + '_train.csv', dtype=int, delimiter=",", skiprows=0)
            miss_list = np.loadtxt('../Dataset/' + mechanism + '_missing_rate_' + str(int(p_miss * 100)) + '_Epoch' + str(epoch_num) + '_miss.csv', delimiter=",", skiprows=0)
            Top_feature = [int(i) for i in list(set(miss_list[:, 1] - 1))]
            # Data generation
            Data = np.loadtxt('../Dataset/' + dataset + '_normalization.csv', delimiter=",", skiprows=0)
            Data = Data[:, :-1]
            # Parameters
            No = len(Data)             # Number of tuples
            Dim = len(Data[0, :])       # Dimension

            # Hidden state dimensions
            H_Dim1 = Dim
            H_Dim2 = Dim

            relationship = np.corrcoef(Data, rowvar=False) * 0.5 + 0.5
            relationship = relationship[-1, :]
            rank = np.argsort(-relationship, axis=0)[1:]
            Top_K = int((0.5 + 0.5 * p_miss) * Dim)
            # Top_feature = rank[:Top_K]
            Bottom = rank[Top_K:]

            #%% Train Test Division

            idx = np.random.permutation(No)    # Random Index

            Train_No = len(train_list)    # The number of tuples in training set
            Test_No = len(test_list)            # The number of tuples in test set

            trainX = Data[train_list, :]
            testX = Data[test_list, :]

            # Train / Test Missing Indicators
            Missing = np.ones(Data.shape)
            for list_ in miss_list:
                dim = int(list_[1]-1)
                Missing[int(list_[0])][dim] = 0

            trainM = Missing[train_list, :]
            testM = Missing[test_list, :]

            #%% Necessary Functions

            # 1. Xavier Initialization Definition
            def xavier_init(size):
                in_dim = size[0]
                xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
                return tf.random_normal(shape=size, stddev=xavier_stddev)

            # Hint Vector Generation
            def sample_M(m, n, p):
                A = np.random.uniform(0., 1., size=[m, n])
                B = A > p
                C = 1.*B
                return C

            '''
            GAIN Consists of 3 Components
            - Generator
            - Discriminator
            - Hint Mechanism
            '''
            start = time.clock()
            #%% GAIN Architecture

            #%% 1. Input Placeholders
            # 1.1. Data Vector
            X = tf.placeholder(tf.float32, shape=[None, Dim])
            # 1.2. Mask Vector
            M = tf.placeholder(tf.float32, shape=[None, Dim])
            # 1.3. Hint vector
            H = tf.placeholder(tf.float32, shape=[None, Dim])
            # 1.4. X with missing values
            New_X = tf.placeholder(tf.float32, shape=[None, Dim])

            #%% 2. Discriminator
            D_W1 = tf.Variable(xavier_init([Dim*2, H_Dim1]))     # Data + Hint as inputs
            D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

            D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
            D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

            D_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
            D_b3 = tf.Variable(tf.zeros(shape=[Dim]))       # Output is multi-variate

            theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

            #%% 3. Generator
            G_W1 = tf.Variable(xavier_init([Dim*2, H_Dim1]))     # Data + Mask as inputs (Random Noises are in Missing Components)
            G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

            G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
            G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

            G_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
            G_b3 = tf.Variable(tf.zeros(shape=[Dim]))

            theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

            # GAIN Function

            # 1. Generator
            def generator(new_x,m):
                inputs = tf.concat(axis=1, values=[new_x,m])  # Mask + Data Concatenate
                G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
                G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
                G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output

                return G_prob

            # 2. Discriminator
            def discriminator(new_x, h):
                inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate
                D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
                D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
                D_logit = tf.matmul(D_h2, D_W3) + D_b3
                D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

                return D_prob

            # 3. Other functions
            # Random sample generator for Z
            def sample_Z(m, n):
                return np.random.uniform(0., 0.01, size=[m, n])

            # Mini-batch generation
            def sample_idx(m, n):
                A = np.random.permutation(m)
                idx = A[:n]
                return idx

            #%% Structure
            # Generator
            G_sample = generator(New_X, M)                # Definition of generator: input X without missing values and Mask Vector

            # Combine with original data
            Hat_New_X = New_X * M + G_sample * (1-M)     # Generated complete data

            # Discriminator
            D_prob = discriminator(Hat_New_X, H)         # Definition of generator: input generated data and Hint Vector

            #%% Loss
            D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8))
            G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
            MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample)**2) / tf.reduce_mean(M)

            D_loss = D_loss1
            G_loss = G_loss1 + alpha * MSE_train_loss

            #%% MSE Performance metric
            MSE_test_loss = tf.reduce_mean(((1-M) * X - (1-M)*G_sample)**2) / tf.reduce_mean(1-M)
            MAE_test_loss = tf.reduce_mean((((1-M) * X - (1-M)*G_sample)**2)**0.5) / tf.reduce_mean(1-M)

            #%% Solver
            D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)   # Optimizer for D
            G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)   # Optimizer for G

            # Sessions Definition
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess.run(tf.global_variables_initializer())

            #%% Iterations
            class ProgressBar():

                def __init__(self, max_steps):
                    self.max_steps = max_steps
                    self.current_step = 0
                    self.progress_width = 50

                def update(self, step=None):
                    self.current_step = step

                    num_pass = int(self.current_step * self.progress_width / self.max_steps) + 1
                    num_rest = self.progress_width - num_pass
                    percent = (self.current_step+1) * 100.0 / self.max_steps
                    progress_bar = '[' + '■' * (num_pass-1) + '▶' + '-' * num_rest + ']'
                    progress_bar += '%.2f' % percent + '%'
                    if self.current_step < self.max_steps - 1:
                        progress_bar += '\r'
                    else:
                        progress_bar += '\n'
                    sys.stdout.write(progress_bar)
                    sys.stdout.flush()
                    if self.current_step >= self.max_steps:
                        self.current_step = 0
                        print

            #%% Start Iterations
            for it in tqdm(range(10000)):

                #%% Inputs
                mb_idx = sample_idx(Train_No, mb_size)
                X_mb = trainX[mb_idx, :]

                Z_mb = sample_Z(mb_size, Dim)
                M_mb = trainM[mb_idx, :]
                H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
                H_mb = M_mb * H_mb1

                New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

                # Training discriminator
                _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
                # Training generator
                _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                                                                                   feed_dict = {X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})

                # for iteration in range(max_batchs):
                progress_bar = ProgressBar(5000)

                #%% Intermediate Losses
                if it % 100 == 0:
                    progress_bar.update(it)


            #%% Final Loss
            end = time.clock()
            time_ = end - start


            Z_mb = sample_Z(Test_No, Dim)
            M_mb = testM
            X_mb = testX

            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

            ## MSE_final is the output of GAIN,and Sample is the output of generator
            MSE_final, Sample = sess.run([MSE_test_loss, G_sample], feed_dict = {X: testX, M: testM, New_X: New_X_mb})
            MAE_final, Sample_ = sess.run([MAE_test_loss, G_sample], feed_dict = {X: testX, M: testM, New_X: New_X_mb})
            RMSE = np.sqrt(MSE_final)
            print('Final RMSE: ' + str(RMSE))
            print('Final MAE: ' + str(MAE_final))
            RMSE_D.append(RMSE)
            MAE_D.append(MAE_final)
            time_all.append(time_)