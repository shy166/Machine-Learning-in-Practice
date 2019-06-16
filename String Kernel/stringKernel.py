class stringKernel():
    def __init__(self, p = 2, t = 1):
        self.passes = t
        self.p = p
    def fit(self, X, y):
        self.phi = self._matrix(X)
        self.ind = [0]
        self.x = X
        self.y = y
        for _ in range(self.passes):
            for i in range(1, X.shape[0]):
                if y.iloc[i] * (y.iloc[self.ind] .dot(self.phi[i,:].dot(self.phi[self.ind].T).T.toarray())) <= 0:
                    self.ind += [i]
    def predict(self, X):
        joint_X = self.x.append(X)
        self.phi_pred = self._matrix(joint_X)
        inner_mat = self.phi_pred[self.ind].dot(self.phi_pred[-len(X):,:].T)
        return pd.Series(self.y.iloc[self.ind].dot(inner_mat.toarray()), index = X.index)\
                            .apply(lambda x: np.sign(x) if x != 0 else np.random.choice([-1,1]))
    def score(self, X, y):
        return (self.predict(X) == y.values).mean()
        
    
    def _matrix(self, X):
        p = self.p
        v = lambda x: ' '.join([x[i:i+p] for i in range(len(x) - p + 1)])
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(X.apply(v))