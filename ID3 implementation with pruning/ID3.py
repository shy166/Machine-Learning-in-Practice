### Classifier ###
class ID3():
    '''Initialize ID3 Class'''
    class Node():
        '''Tree Node'''
        def __init__(self, feature = None, threshold= None, samples = None, label = None):
            self.left = None
            self.right = None
            self.feature = feature
            self.threshold = threshold
            self.samples = samples
            self.label = label
            self.isleft = None
            self.parent = None
        def set_left(self, other): 
            '''Set left child'''
            self.left = other
            other.isleft = True
            other.parent = self
        def set_right(self, other):
            '''Set right child'''
            self.right = other
            other.isleft = False
            other.parent = self
        def is_leaf(self):
            '''Determine if the Node is leaf'''
            return self.left == None and self.right == None
        def print_info(self):
            '''Print info of the Node'''
            if self.is_leaf():
                print(self.label)
            else:
                print(self.feature, self.threshold, self.samples)
    def _H(self, S):
        '''Helper function of Getting Entropy of S'''
        p = np.unique(S,return_counts=True)[1]/len(S)
        entropy = np.sum(-p*(np.log2(p)))
        return entropy

    def _IG(self,train,target,threshold):
        '''Helper function of Getting Information Gain of training set over target column with certain threshold'''
        t1 = target.loc[train>=threshold]
        t2 = target.loc[train<threshold]

        p = len(t1)/len(train)
        return self._H(target)-p*self._H(t1)-(1-p)*self._H(t2)
    def __init__(self):
        '''Constructor of ID3'''
        self.tree = None #A decision tree in .tree
    
    def _ID3_func(self,X,y,_mode,_prev_mode):
        '''Helper function to build the decision tree'''
        ### base cases ###:
        labels = y.unique()
        # When the dataset is pure
        if len(labels)==1:
            node = self.Node(label=labels[0])
            return node
        # When the dataset is empty Node as the mode overall distribution
        if len(X) == 0:
            return self.Node(label = _mode)
        if len(X.columns) == 0:
        # When the dataset's feature is empty, label the Node as previous mode
            return self.Node(label = _prev_mode)
        
        ### Calculate the entropy & best feature & best threshold###
        gain = []
        threshold = []
        for feature in X.columns:
            x = X.loc[:,feature]
            temp = np.unique(x)
            candidates = np.diff(temp)/2+temp[:-1] 
            ig = [self._IG(x,y,c) for c in candidates]
            if len(ig) == 0:
                gain.append(-np.inf)
                threshold.append(-np.inf)
                continue
            else:
                gain.append(np.amax(ig))
                threshold.append(candidates[np.argmax(ig)])
        ix = np.argmax(gain)
        best_feature = X.columns[ix]
        best_threshold = threshold[ix]
        ### Construct the Tree Node ###
        tree = self.Node(feature = best_feature,threshold = best_threshold, samples = len(X))
        # Splitting the dataset into two piece
        mask = X[best_feature]<best_threshold
        X_left = X.loc[mask]
        y_left = y.loc[mask]
        X_right = X.loc[~mask]
        y_right = y.loc[~mask]
        ### Create left node and right node by recursion ###
        left_node = self._ID3_func(X_left, y_left, _mode, y.unique()[0])
        right_node = self._ID3_func(X_right, y_right, _mode, y.unique()[0])
        tree.set_left(left_node)
        tree.set_right(right_node)
        return tree
    def fit(self, X, y):
        '''To fit the decision tree to classifier'''
        _mode = y.unique()[0]
        self.tree = self._ID3_func(X, y, _mode, _mode)
    
    def _predict(self, Xi, curr = None):
        '''Helper function for Predict Xi'''
        if curr == None:
            # If it curr is None, start from root
            curr = self.tree
        if curr.is_leaf():
            # Base Case, if current node is leaf, return the label
            to_return = curr.label
            return to_return
        # Get splitting decision
        feature, threshold = curr.feature, curr.threshold
        # Classify the Xi into next node
        if Xi[feature] < threshold:
            curr = curr.left
            return self._predict(Xi, curr)
        else:
            curr = curr.right
            return self._predict(Xi, curr)
    def predict(self, X):
        '''funtion to predict X dataset.'''
        return pd.Series([self._predict(X.iloc[Xi,:]) for Xi in range(X.shape[0])], index = X.index)
    def score(self, X, y):
        '''Get the accuracy of prediction'''
        return (self.predict(X) == y).mean()

            
    def _get_sample(self, X, Node):
        ''' Get the samples in this Node by reverse the tree'''
        if Node == self.tree:
            # Base Case Node is root, return sample
            return X
        # Traverse back to parent Node with parent's decision
        elif Node.isleft == True:
            feature, threshold = Node.parent.feature, Node.parent.threshold
            X = X[X[feature] < threshold]
            return self._get_sample(X, Node.parent)
        elif Node.isleft == False:
            feature, threshold = Node.parent.feature, Node.parent.threshold
            X = X[X[feature] >= threshold]
            return self._get_sample(X, Node.parent)
    def _bfs(self):
        '''traverse the decision tree by bfs'''
        thislevel = [self.tree]
        to_return = thislevel
        while thislevel:
            nextlevel = list()
            for n in thislevel:
                if n.left: 
                    nextlevel.append(n.left)
                if n.right: 
                    nextlevel.append(n.right)

            thislevel = nextlevel
            to_return += nextlevel
        return to_return #return as a list
    
    def prune(self, X, y, N):
        nodes = self._bfs()
        # Looping over the bfs of each node by bfs
        for node in nodes:
            # Get samples in the node
            X_sample = self._get_sample(X, node)
            y_sample = y[y.index.isin(X_sample.index)]
            # Calculating err of classifier
            err = 1 - self.score(X_sample, y_sample)
            # Get the most freq label of the prediction
            mode_i = self.predict(X_sample).value_counts().idxmax()
            # Calculate Prune decision
            prune_pred = pd.Series([mode_i for Xi in range(len(X_sample))], index = X_sample.index)
            err_hat = 1 - (prune_pred == y_sample).mean()
            # Replace Node if err_hat is smaller or equal
            if err_hat <= err:
                new_node = self.Node(label = mode_i)
                if node.isleft == True:
                    node.parent.set_left(new_node)
                elif node.isleft == False:
                    node.parent.set_right(new_node)
                else:
                    node = new_node
                N -= 1
            # Stopping
            if N == 0:
                break


    
