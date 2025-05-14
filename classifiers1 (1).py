''' Import Libraries'''
import pandas as pd
import numpy as np
from discriminants import MultivariateGaussian


class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError

class Prior(Classifier):
    
    def __init__(self):
        ''' Your code here '''
        self.model_params = {}
        pass
    

    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x (numpy array), y (numpy array)'''
        raise NotImplementedError



''' Create our Discriminant Classifier Class'''    
class DiscriminantClassifier(Classifier):
    ''''''
    def __init__(self):
        ''' Initialize Class Dictionary'''
        self.model_params = {}
        self.classes = {}
        
    def set_classes(self, *discs):
        '''Pass discriminant objects and store them in self.classes
            This class is useful when you have existing discriminant objects'''
        for disc in discs:
            self.classes[disc.name] = disc

            
    def fit(self, dataframe, label_key=['Labels'], default_disc=MultivariateGaussian):
        ''' Calculates model parameters from a dataframe for each discriminant.
            Label_Key specifies the column that contains the class labels. ''' 
        X = dataframe.drop(columns=label_key).values 
        y = dataframe[label_key].values
        uniquecls = np.unique(y)
        for clabel in uniquecls:
            class_data = X[y == clabel]
            if clabel not in self.classes:
                self.classes[clabel] = default_disc(class_data, prior=1/len(uniquecls), name=clabel)
            else:
                self.classes[clabel].fit(class_data)
        self.model_params = {clabel: disc.params for clabel, disc in self.classes.items()}
                
    
    def predict(self, x):
        ''' Returns a Key (class) that corresponds to the highest discriminant value'''
        score = -np.inf
        bestcls = None
        for name, disc in self.classes.items():
            s = disc.calc_discriminant(x)
            if s > score:
                score = s
                bestcls = name
        return bestcls

    def pool_variances(self):
        ''' Calculates a pooled variance and sets the corresponding model params '''   
        ntot = sum(len(disc.params['data']) for disc in self.classes.values())
        totcov = sum((len(disc.params['data']) - 1) * disc.params['sigma'] for disc in self.classes.values())
        poolcov = totcov / (ntot - len(self.classes))
        for disc in self.classes.values():
            disc.params['sigma'] = poolcov

       
        
        
        
