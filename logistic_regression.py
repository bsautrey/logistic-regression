# Implement logistic regression from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes1.pdf. Batch gradient ascent is used to learn the parameters, i.e. maximize the likelihood.

import random
from copy import copy
from math import exp

import numpy as np
import matplotlib.pyplot as plot

# alpha - The learning rate.
# dampen - Factor by which alpha is dampened on each iteration. Default is no dampening, i.e. dampen = 1.0
# tol - The stopping criteria
# theta - The parameters to be learned.

class LogisticRegression():
    
    def __init__(self):
        self.X = None
        self.Y = None
        self.alpha = None
        self.dampen = None
        self.tol = None
        self.theta = None
        self.percents = None
        
    def set_X(self,X):
        self.X = X
    
    def set_Y(self,Y):
        self.Y = Y
        
    def set_alpha(self,alpha=0.001,dampen=1.0):
        self.alpha = alpha
        self.dampen = dampen
        
    def set_tao(self,tao=1.0):
        self.tao = tao
        
    def set_tolerance(self,tol=0.001):
        self.tol = tol
        
    def initialize_theta(self,theta=None):
        if not theta:
            number_of_parameters = self.X.shape[1]
            theta = copy(self.X[0,:])
            theta.resize((1,number_of_parameters))
            
        self.theta = theta
        
    def _initialize_percents(self):
        self.percents = []
        
    def run_BGA(self,max_iterations=5000):
        self._initialize_percents()
        old_theta = copy(self.theta)
        iterations = 0
        number_of_rows = self.X.shape[0]
        number_of_columns = self.X.shape[1]
        while True:
            for i in xrange(number_of_rows):
                x = self.X[i,:]
                y = self.Y[i,:][0]
                x.resize((number_of_columns,1))
                for j in xrange(number_of_columns):
                    theta_j = self.theta[0][j]
                    x_j = x[j][0]
                    dot = np.dot(self.theta,x)[0][0]
                    logistic = 1.0/(1 + exp(-dot))
                    new_theta_j = theta_j + self.alpha*(y - logistic)*x_j
                    self.theta[0][j] = new_theta_j
                
            iterations = iterations + 1
            percent = self._calculate_convergence(old_theta)
            self.percents.append((iterations,percent))
            old_theta = copy(self.theta)
            self.alpha = self.alpha*self.dampen
            print iterations,percent,self.alpha,self.theta
            if percent < self.tol or iterations > max_iterations:
                return
        
    def _calculate_convergence(self,old_theta):
        diff = old_theta - self.theta
        diff = np.dot(diff,diff.T)**0.5
        length = np.dot(old_theta,old_theta.T)**0.5
        percent = 100.0*diff/length
        return percent
        
    def generate_example(self,sample_size_per_class=1000):
        # class_1
        mean = np.array([3,3])
        cov = np.array([[1,-0.6],[-0.6,1]])
        res_1 = np.random.multivariate_normal(mean,cov,sample_size_per_class)

        # class_2
        mean = np.array([5,5])
        cov = np.array([[1,-0.75],[-0.75,1]])
        res_2 = np.random.multivariate_normal(mean,cov,sample_size_per_class)
        
        # assemble data
        X = np.row_stack((res_2,res_1))
        intercept = np.ones((2*sample_size_per_class))
        X = np.column_stack((X,intercept))
    
        Y = []
        for i in xrange(2*sample_size_per_class):
            if i < sample_size_per_class:
                class_1 = 1.0
                Y.append(class_1)
            else:
                class_2 = 0.0
                Y.append(class_2)
            
        Y = np.array(Y)
        Y.resize(2*sample_size_per_class,1)
    
        # initialize
        self.set_X(X)
        self.set_Y(Y)
        self.set_alpha(alpha=0.001,dampen=1.0)
        self.set_tolerance(0.05)
        self.initialize_theta()
        self.run_BGA()
        
        # decision boundry
        theta_0 = self.theta[0][2]
        theta_1 = self.theta[0][1]
        theta_2 = self.theta[0][0]
        x_1_1 = 0.0
        x_1_2 = 8.0
        x_2_1 = -(theta_0 + theta_1*x_1_1)/theta_2
        x_2_2 = -(theta_0 + theta_1*x_1_2)/theta_2
        
        # plot
        plot.scatter(self.X[0:sample_size_per_class,0],self.X[0:sample_size_per_class,1],s=0.5,color='orange')
        plot.scatter(self.X[sample_size_per_class:,0],self.X[sample_size_per_class:,1],s=0.5,color='green')
        plot.plot([x_1_1,x_2_1],[x_1_2,x_2_2])
        plot.show()
        
    def plot_convergence(self,start_index,end_index=None):
        if end_index:
            X,Y = zip(*self.percents[start_index:end_index])
        else:
            X,Y = zip(*self.percents[start_index:])
            
        plot.plot(X,Y)
        plot.show()

        