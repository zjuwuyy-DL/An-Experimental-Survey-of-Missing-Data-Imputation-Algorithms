
# Algorithm 1

from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
import numpy as np
import math
initial_neighbors = 4
#input l
r = np.array([[0, 5.8, 3], [0.8, 4.6, 5], [1.9, 3.8, 6], [2.9, 3.2, 2], [6.8, 3, 3], [7.5, 4.1, 6], [8.2, 4.8, 4], [9, 5.5, 1]])
#input r
rF=[]
#complete attribute
for idx,item in enumerate(r):
    rF += [[r[idx][0]]]
#rF为complete attribute
nbrs = NearestNeighbors(n_neighbors=initial_neighbors, algorithm='ball_tree').fit(rF)
#n_neighbors 邻居数量
distances, indices = nbrs.kneighbors(rF)

reg = linear_model.LinearRegression()
coef=[]
intercept=[]
predict=[]

for idx, item in enumerate(r):
    X = []
    Y = []
    for i in range(0, initial_neighbors):
        X += [[r[indices][idx][i][0]]]
        Y += [r[indices][idx][i][1]]
    reg.fit(X, Y)
        #前为X后为Y
    coef += [reg.coef_]
    intercept += [reg.intercept_]
    predict += [reg.predict([[5]])]
#predict即为Algorithm2中的candidates
print(predict)

# Algorithm 2: Imputation
k=3
#input k
rF+=[[5]]
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(rF)
#n_neighbors 邻居数量
distances, indices = nbrs.kneighbors(rF)

candidate=[]
for i in range (1,k+1):
    candidate+=[predict[indices[len(indices)-1][i]][0]]


c=[]
for i in range (0,k):
    c_value=0
    for j in range (0,k):
        c_value += abs(candidate[i]-candidate[j])
    c+=[c_value]

w=[]
w_regularization=0
for i in range (0,k):
    w_regularization+=1/c[i]
for i in range (0,k):
    w+=[(1/c[i])/w_regularization]

combined=0
for i in range (0,k):
    combined+=candidate[i]*w[i]

