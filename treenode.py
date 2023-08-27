import numpy as np 
from bisect import bisect_right
import time 

class TreeNode:
    def __init__(self, X: np.ndarray, depth, min_samples_split=7, max_depth=10, stop_crit=0, split_crit=0):
        # Set the data 
        self.X = X 

        # Depth of the node 
        self.depth = depth
        self.isterminal = False

        # Set the hyperparameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.stop_crit = stop_crit
        self.split_crit = split_crit

        # Entropy and Gini Index
        # Set these properly later 
        self.gini = 0.0
        self.entropy = 0.0
        if (split_crit == 0):
            self.gini = self.calc_gini(self.X)
        else:
            self.entropy = self.calc_entropy(self.X)

        # Left and right nodes
        self.left = None
        self.right = None

        # Node Split Parameters 
        self.split_idx = -1
        self.split_val = -1
        self.is_terminal = False
        self.prediction = 0

    # Calculate the entropy given a dataset 
    def calc_entropy(self, arr):
        n_pos = 0 
        n_neg = 0

        for row in arr:
            if (row[-1] == 1):
                n_pos += 1
            else:
                n_neg += 1

        n = n_pos + n_neg

        if (n == 0):
            return 0.0
        
        p1 = n_pos / n
        p2 = n_neg / n

        entropy = (-p1*np.log2(p1 + 1e-9)) + (-p2*np.log2(p2+1e-9))
        return entropy


    # Calculate the gini index given a dataset 
    def calc_gini(self, arr):
        n_pos = 0 
        n_neg = 0

        for row in arr:
            if (row[-1] == 1):
                n_pos += 1
            else:
                n_neg += 1

        n = n_pos + n_neg

        if (n == 0):
            return 0.0
        
        p1 = n_pos / n
        p2 = n_neg / n
        gini = 1 - (p1 **2  + p2 ** 2)
        return gini

    def make_split(self, col, split_value):
        left = []
        right = []

        for i in range(len(self.X)):
            if (self.X[i][col] <= split_value):
                left.append(self.X[i])
            else:
                right.append(self.X[i])
        return (left,right)
    

    def find_best_split(self):
        # Initialize best features 
        best_idx = -1
        best_split_val = -1
        best_score = 0
        best_left = None
        best_right = None 

        # Iterate over columns 
        # start_time = time.time()
        for idx in range(len(self.X[0])-1):

            # print("Doing ", idx)

            # Temporary array to reuse work 
            temp = sorted(self.X, key=lambda x: x[idx])

            keys = [temp[i][idx] for i in range(len(self.X))]
            
            # Iterate over rows 
            for val in range(0,256):
            # for row in self.X:
                ridx = bisect_right(keys, val)
                left = temp[0:ridx]
                right = temp[ridx:]

                # left, right = self.make_split(idx, val)
                gain = 0

                if (self.split_crit == 0):
                    gini_l = self.calc_gini(left)
                    gini_r = self.calc_gini(right)

                    nl = len(left)
                    nr = len(right)

                    # Calculate weights of left and right splits
                    wleft = nl/(nl + nr)
                    wright = nr/(nl + nr)

                    #Calculate weighted gini
                    weighted_gini = (wleft*gini_l) + (wright*gini_r)
                    gain = self.gini - weighted_gini

                else: 
                    entropy_l = self.calc_entropy(left)
                    entropy_r = self.calc_entropy(right)

                    nl = len(left)
                    nr = len(right)

                    # Calculate weights of left and right splits
                    wleft = nl/(nl + nr)
                    wright = nr/(nl + nr)

                    #Calculate weighted gini
                    weighted_entropy = (wleft*entropy_l) + (wright*entropy_r)
                    gain = self.entropy - weighted_entropy

                if (gain > best_score):
                    best_idx = idx
                    best_split_val = val
                    best_score = gain
                    best_left = left
                    best_right = right

        # end_time = time.time()
        # print("TOTAL TIME: ", end_time - start_time)
        return (best_idx, best_split_val, best_left, best_right)
    
    def set_pred(self):
        n_pos = 0
        n_neg = 0
        for row in self.X:
            if (row[-1] == 1):
                n_pos += 1
            else:
                n_neg +=1 
        if (n_neg > n_pos):
            self.prediction = -1
        else:
            self.prediction = 1

    def expand_node(self):
        # If max depth is reached 
        if (self.depth >= self.max_depth):
            self.is_terminal = True
            self.set_pred()
            # Clear memory
            self.X = None
            return 
        # If less than min no of samples
        if (self.X.shape[0] < self.min_samples_split):
            self.is_terminal = True
            self.set_pred()
            # Clear memory
            self.X = None
            return 
        
        # Split the node 
        
        idx, split_val, left, right = self.find_best_split()

        # print("Splitting the tree on idx {}, LEFT: <= {}, RIGHT: > {}".format(idx, split_val, split_val))

        if (idx == -1):
            self.is_terminal = True
            self.set_pred()
            # Clear memory
            self.X = None
            return 

        left = np.array(left)
        right = np.array(right)

        node_left = TreeNode(left, depth=self.depth + 1, min_samples_split=self.min_samples_split, max_depth=self.max_depth, \
                            stop_crit=self.stop_crit, split_crit=self.split_crit)
        
        node_right = TreeNode(right, depth=self.depth + 1, min_samples_split=self.min_samples_split, max_depth=self.max_depth, \
                            stop_crit=self.stop_crit, split_crit=self.split_crit)
        
        # Set the split nodes and parameters
        self.left= node_left
        self.right = node_right
        self.split_idx = idx
        self.split_val = split_val

        # Clear memory
        self.X = None

        # Grow the treenodes on the left and right 
        self.left.expand_node()
        self.right.expand_node()
    
        return     

    def predict(self, x):
        # x is the test sample 

        if (self.is_terminal):
            return self.prediction

        feature_val = x[self.split_idx]
        if (feature_val <= self.split_val):
            return self.left.predict(x)
        else:
            return self.right.predict(x)     