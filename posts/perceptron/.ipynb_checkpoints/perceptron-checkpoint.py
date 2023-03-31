import numpy as np
import random

class Perceptron:
    
    def fit(self, X, y, max_steps = 1000):
        '''
        Trains a binary classifier using the prceptron algorithm.
        
        Parameters:
            X: A set of training examples/The input data
            y: Target labels for the training data. Each value is either a 1 or 0.
            max_steps: The maximum number of iterations to run the perceptron algorithm for. Default is 1000.
            
        Returns:
            Has no return value
            
        Notes:
            - The score of the classifier over time can be accessed via 'self.history'
            - The trained weight vector can be accessed via 'self.w'
            - The input data is modified in place
        '''
        
        # Determine the number of data points, n, from X
        n = X.shape[0]

        # Modify X into X_ where X_ = [X, 1] and 1 is a column of 1s
        X_ = np.append(X, np.ones((n, 1)), 1)
        
        # Determine the number of features, p, from X_
        p = X_.shape[1]

        # Modify y into an array y_ of -1s and 1s
        y_ = (2 * y) - 1

        # Initialize a random weight vector where self.w = (w, -b)
        self.w = np.random.rand(p)

        steps = 0
        
        # Initialize list of scores over time
        self.history = []
        
        # Goes until classification is 100% or max number of steps has been reached
        while ((self.score(X, y) != 1.0) and (steps < max_steps)):
            self.history.append(self.score(X, y))
            i = random.randint(0, n - 1)
            self.w += (((y_[i] * self.w@X_[i]) < 0) * y_[i] * X_[i])
            steps += 1
        self.history.append(self.score(X, y))
          
    def predict(self, X):
        '''
        Predicts the binary labels for the input data using the learned weight vector.
        
        Parameters:
            X: The input data to predict the binary labels for
        
        Returns:
            A vector of predicted binary labels for the input data
        '''
        
        # Modify X into X_ where X_ = [X, 1] and 1 is a column of 1s
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        # Computes a boolean array of 'True' or 'False' values, 
        # 'True' is assigned to elements of X_ that are greater than 0, and 'False' is assigned to elements less than or equal to 0
        # Convert to an interger array where 1 is 'True' and 0 is 'False'
        return (X_@self.w > 0).astype(int)

    
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
