"isolated forest functions"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random as rn
import os
import warnings

def c_factor(n) :
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))

class iForest(object):
    def __init__(self,X, ntrees,  sample_size, limit=None):
        self.ntrees = ntrees
        self.X = X 	# training data
        self.data_size = len(X)
        self.sample = sample_size #sample size
        self.Trees = []
        self.limit = limit
        if limit is None:
            self.limit = int(np.ceil(np.log2(self.sample)))
        self.c = c_factor(self.sample)        
        for i in range(self.ntrees):
            ix = rn.sample(range(self.data_size), self.sample)
            X_sample = self.X[ix]	# random sampling
            self.Trees.append(iTree(X_sample, 0, self.limit)) # yahan tak ensemble ban gya

    def compute_paths(self, X_in = None):	
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in  range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(X_in[i],self.Trees[j]).path*1.0
            Eh = h_temp/self.ntrees
            S[i] = 2.0**(-Eh/self.c)
        return S

    def compute_paths_single(self, x):
        S = np.zeros(self.ntrees)
        for j in range(self.ntrees):
            path =  PathFactor(x,self.Trees[j]).path*1.0
            S[j] = 2.0**(-1.0*path/self.c)
        return S


class Node(object):
    def __init__(self, X, q, p, e, left, right, node_type = '' ):
        self.e = e
        self.size = len(X)
        self.X = X # to be removed
        self.q = q
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type

class iTree(object):
    """
    Unique entries for X
    """
    def __init__(self,X,e,l):
        self.e = e # depth
        self.X = X #save data for now
        self.size = len(X) #  number of objects
        self.Q = np.arange(np.shape(X)[1], dtype='int') # number of dimensions
        print(type(Q))
        self.l = l # depth limit
        self.p = None
        self.q = None
        self.exnodes = 0
        self.root = self.make_tree(X,e,l)
        
    def make_tree(self,X,e,l):
        self.e = e
        if e >= l or len(X) <= 1:
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.q, self.p, e, left, right, node_type = 'exNode' )
        else:
            self.q = rn.choice(self.Q)
            mini = X[:,self.q].min()
            maxi = X[:,self.q].max()
            if mini==maxi:
                left = None
                right = None
                self.exnodes += 1
                return Node(X, self.q, self.p, e, left, right, node_type = 'exNode' )
            self.p = rn.uniform(mini,maxi)
            w = np.where(X[:,self.q] < self.p,True,False)
            return Node(X, self.q, self.p, e,\
            left=self.make_tree(X[w],e+1,l),\
            right=self.make_tree(X[~w],e+1,l),\
            node_type = 'inNode' )

    def get_node(self, path):
        node = self.root
        for p in path:
            if p == 'L' : node = node.left
            if p == 'R' : node = node.right
        return node
    
class PathFactor(object):
    def __init__(self,x,itree):
        self.path_list=[]        
        self.x = x
        self.e = 0
        self.path = self.find_path(itree.root)

    def find_path(self,T):
        if T.ntype == 'exNode':
            if T.size == 1: return self.e
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:
            a = T.q
            self.e += 1
            if self.x[a] < T.p:
                self.path_list.append('L')
                return self.find_path(T.left)
            else:
                self.path_list.append('R')
                return self.find_path(T.right)


data = pd.read_csv('creditcard.csv')
print(type(data))

#print(data.head())
from sklearn.preprocessing import StandardScaler

np_array = np.array(data.Amount)
np_array_reshaped = np_array.reshape(-1, 1)
data['normAmount'] = StandardScaler().fit_transform(np_array_reshaped)

#dropping time and amount as they do not seem significant 
data = data.drop(['Time','Amount'],axis=1)

data_zero = data[data.Class == 0]
print(len(data_zero.index))

data_one = data[data.Class == 1]
print(len(data_one.index))

data_zero = data_zero.iloc[:10000]
print(len(data_zero.index))
data_zero = data_zero.append(data_one) # equal sampling
X = data_zero.ix[:, data_zero.columns != "Class"]
y = data_zero.ix[:, data_zero.columns == "Class"]

data_zero = data[data.Class == 0]
data_one = data[data.Class == 1]
data_zero = data_zero.iloc[:10000]

iso_x_train = data_zero.ix[:, data_zero.columns != 'Class']
#print(iso_x_train.head())


iso_x_outliers = data_one.ix[:, data_one.columns != 'Class']

classifier = iForest(X = iso_x_train, ntrees = 100, sample_size = 128)

y_pred_train = classifier.compute_paths(iso_x_train)
y_pred_test = classifier.compute_paths(iso_x_test)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    