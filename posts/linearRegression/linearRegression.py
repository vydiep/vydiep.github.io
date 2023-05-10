import numpy as np

# Referenced https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/regression.html

class LinearRegression():
    def __init__(self):
        self.score_history = []
        
    def fit_gradient(self, X, y, max_iter = 1000, alpha = 0.01):
        '''
        Fit the linear regression model using gradient descent
        
        Parameters:
            X: Input feature matrix
            y: Target values
            max_iter: Maximum number of iterations for gradient descent (default at 1000)
            alpha: Learning rate for gradient descent (default at 0.01)
        '''
        
        X_ = self.pad(X)
        
        p = X_.shape[1]
        self.w = np.random.rand(p)
        
        P = X_.T@X_
        q = X_.T@y
        
        for i in np.arange(max_iter):
            
            gradient = (P@self.w) - q
            
            self.w -= alpha * gradient 
            
            score = self.score(X_, y)
            
            self.score_history.append(score)
            

    def fit_analytic(self, X, y):
        '''
        Fit the linear regression model using the analytical formula
        
        Parameters:
            X: Input feature matrix
            y: Target values
        '''
        
        X_ = self.pad(X)
        w_hat = np.linalg.inv(X_.T@X_)@X_.T@y
        
        self.w = w_hat
    
    def score(self, X, y):
        '''
        Calculate the coefficient of determination for linear regression model
        
        Parameters:
            X: Input feature matrix
            y: Target values
            
        Returns:
            c: Coefficient of determination between 0 and 1
        '''
        
        y_hat = self.predict(X)
        
        num = np.sum((y_hat - y) ** 2)
        mean_y = np.mean(y)
        
        den = np.sum((mean_y - y) ** 2)
        c = (1 - (num / den))
        
        return c
        
    def pad(self, X):
        '''
        Pad the input feature matrix with a column of 1s
        
        Parameters:
            X: Input feature matrix
            
        Returns:
            Padded feature matrix
        '''
        
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def predict(self, X):
        '''
        Predict the target values for the given input feature matrix
        
        Parameters:
            X: Input feature matrix
            
        Returns:
            Predicted target values
        '''
        
        return X@self.w