class Perceptron():
    def __init__(self, iterations = 1, kind = 'perceptron'):
        self.w = None
        self.classes = None
        self.iterations = iterations
        self.kind = kind
        self.cm = None
        
    def fit(self, X, y):
        self.classes = sorted(np.unique(y))
        c = self.classes[0]
        yi = y.apply(lambda x: 1 if x == c else -1)
        if self.kind == 'perceptron':
            self.w = np.zeros(X.shape[1])
            for _ in range(self.iterations):
                for i in range(X.shape[0]):
                    if yi.iloc[i] * (self.w.dot(X.iloc[i])) <= 0:
                        self.w = self.w + yi.iloc[i] * X.iloc[i]

        elif self.kind == 'voted' or self.kind == 'average':
            cm = 1
            w = np.zeros_like(X.iloc[0])
            w_list = [(w,cm)]

            for _ in range(self.iterations):
                for t in range(X.shape[0]):
                    if yi.iloc[t]*w.dot(X.iloc[t])<=0:
                        w_list.append((w,cm))

                        w = w+yi.iloc[t]*X.iloc[t]
                        cm = 1
                    else:
                        cm +=1

            self.w = w_list
            
                
                        
                    
    def predict(self, X):
        if self.kind == 'perceptron':
            return pd.Series(self.w.dot(X.T), index = X.index)\
                        .apply(lambda x: self.classes[0] if x >= 0 else self.classes[1])
        elif self.kind == 'voted':
            pred = 0
    
            for i in range(len(self.w)):
                w,c = self.w[i]
                pred += c*np.sign(X.dot(w))

            return pd.Series(np.sign(pred), index = X.index)\
                        .apply(lambda x: self.classes[0] if x >= 0 else self.classes[1])
        
        elif self.kind == 'average':
            pred = 0
    
            for i in range(len(self.w)):
                w,c = self.w[i]
                pred += X.dot(c*w)

            return pd.Series(np.sign(pred), index = X.index)\
                        .apply(lambda x: self.classes[0] if x >= 0 else self.classes[1])
            
                
    def score(self, X, y):
        return (self.predict(X) == y.values).mean()