import numpy as np
from scipy.optimize import minimize
import warnings

class LogisticRegression():
    
    def fit(self, X, y, alpha = 0.01, max_epochs = 1000):
        '''
        Trains a binary classifier using logistic regression with gradient descent.
        
        Parameters: 
            X: A set of training examples/The input data
            y: Target labels for the training data. Each value is either 0 or 1
            alpha: The learning rate for the gradient descent algorithm. The default is 0.01
            max_epochs: The maximum number of epochs/iterations the algorithm should run for. Default is 1000.
            
        Returns:
            Has no return value
            
        Notes:
            - Modifies input data in place
            - Trained weight vector can be accessed via 'self.w'
            - Score over time can be accessed via 'self.score_history'
            - Loss over time can be accessed via 'self.loss_history'
        '''
        # Determine the number of data points, n, from X
        n = X.shape[0]
        
        # Modify X into X_ where X_ = [X, 1] and 1 is a column of 1s
        X_ = np.append(X, np.ones((n, 1)), 1)
        
        # Determine the number of features, p, from X_
        p = X_.shape[1]
        
        # Initialize a random weight vector where self.w = (w, -b)
        self.w = np.random.rand(p)
        
        # Initialize a list of loss over time
        self.loss_history = []
        
        # Initialize a list of scores over time
        self.score_history = []
        
        # Set loss to positive infinity
        prev_loss = np.inf
        
        # Main loop
        # Referenced: https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
        for epoch in np.arange(max_epochs):
            
            gradient = self.gradient(X_, y)
            
            # Gradient step
            self.w -= alpha * self.gradient(X_, y) 
            
            # Compute loss
            new_loss = self.loss(X_, y)
            
            # Add values to respective lists
            self.loss_history.append(self.loss(X_, y))
            self.score_history.append(self.score(X_, y))
            
            # Check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):
                break
            else:
                prev_loss = new_loss

    def fit_stochastic(self, X, y, alpha = 0.01, max_epochs = 1000, batch_size = 10, momentum = False):
        '''
        Trains a binary classifier using logistic regression with stochastic gradient descent with or without momentum.
        
        Parameters:
            X: A set of training examples/The input data
            y: Target labels for the input data. Each value is either a 1 or 0.
            alpha: The learning rate for the gradient descent. Default is 0.01
            max_epochs: The maximum number of epochs/iterations to run for the algorithm. Default is 100
            batch_size: The number of samples to use for each mini-batch during stochastic gradient descent. Default is 10
            momentum: Boolean. Whether to use momentum in the optimization process
            
        Returns:
            Has no return value
        
        Notes:
            - Can access scores over time via 'self.score_history'
            - Can access loss over time via 'self.loss_history'
            - Can access trained weight via 'self.w'
        '''
        # Modify X into X_ where X_ = [X, 1] and 1 is a column of 1s
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        # Determine the number of data points, n, from X_
        n = X_.shape[0]
        
        # Determine the number of features, p, from X_
        p = X_.shape[1]
        
        # Initialize a random weight vector where self.w = (w, -b)
        self.w = np.random.rand(p)
        
        prev_w = np.array(self.w)
        
        # Initialize list of scores over time
        self.loss_history = []
        
        # Initialize list of loss over time
        self.score_history = []
        
        # Set momentum
        # Referenced: https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html
        if (momentum == True):
            beta = 0.08
        else:
            beta = 0
            
        # Set loss to positive infinity
        prev_loss = np.inf
        
        # Main loop
        # Referenced: https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html
        for epochs in np.arange(max_epochs):
            
            order = np.arange(n)
            np.random.shuffle(order)
            
            # Gradient on batches
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                gradient = self.gradient(x_batch, y_batch)
                
                # Gradient step
                self.w -= (alpha * gradient) + (beta * (self.w - prev_w))
                prev_w = np.array(self.w)
                
            # Compute loss
            new_loss = self.loss(X_, y)
            
            # If the new loss is close enough to the previous less, terminate early
            if np.isclose(new_loss, prev_loss):
                break
            else:
                prev_loss = new_loss
            
            # Add to respective lists
            self.loss_history.append(self.loss(X_, y))
            self.score_history.append(self.score(X_, y))

    def gradient(self, X, y):
        '''
        Computes the gradient.
        
        Parameter:
            X: A set of training samples/The input data
            y: Labels of the data. Each has a value of 1 or 0
            
        Returns:
            Gradient of the logistic loss function
            
        Notes:
            Referenced: 
                - https://middlebury-csci-0451.github.io/CSCI-0451/lecture-notes/gradient-descent.html
                - https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-optimization.html
        '''
        y_ = self.predict(X)
        
        return np.mean(((self.sigmoid(y_) - y)[:,np.newaxis]) * X, axis = 0)
            
        
    def predict(self, X):
        '''
        Predicts the binary labels for the input data using the learned weight vector.
        
        Parameters:
            X: The input data to predict the binary labels for
        
        Returns:
            A vector of predicted binary labels for the input data
        '''
        return np.dot(X, self.w)
    
    
    def score(self, X, y):
        '''
        Computes the mean accuracy, a number between 0 and 1, of the perceptron on the input data and labels. 
        A perfect classification is represented by 1.
        
        Parameters:
            X: The input data
            y: The binary labels for the input data
            
        Returns:
            Accuracy: The mean accuracy, a number between 0 and 1, of the perceptron on the input data.
        '''
        # Predicted labels for X
        y_pred = self.predict(X)

        # Calculate accuracy
        Accuracy = np.mean(y_pred == y)

        return Accuracy
    
    
    def loss(self, X, y):
        '''
        Computes the overall loss (empirical risk) of the current weights on X and y.
        
        Parameters:
            X: The input data
            y: The labels of the data
            
        Returns:
            The overall loss
        '''
        y_ = self.predict(X)
        
        p1 = -y * np.log(self.sigmoid(y_))
        p2 = (1 - y) * np.log(1 - self.sigmoid(y_))
        
        loss = p1 - p2
        
        return loss.mean()
    
    
    def sigmoid(self, z):
        '''
        Calculates the sigmoid function.
        
        Parameters:
            z: A numpy array or scalar representing the input value(s) to the sigmoid function
            
        Returns:
            A numpy array or scalar representing the output value(s) of the sigmoid function, defined as 1 / (1 + exp(-z))
        '''
        return 1 / (1 + np.exp(-z))