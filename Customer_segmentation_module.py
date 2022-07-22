# -*- coding: utf-8 -*-
"""
"""
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Sequential,Input
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import numpy as np



class EDA:
    def displot_graph(self,con_col,df):
        # Continous Visualization
        for i in con_col: 
            plt.figure()
            sns.displot(df[i])
            plt.show()
            
    def countplot_graph(self,cat_col,df):
        # Categorical Visualization
        for i in cat_col:
            plt.figure(figsize=(10,5))
            sns.countplot(df[i])
            plt.show()
            
    def cramers_corrected_stat(self,cmx):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(cmx)[0]
        n = cmx.sum()
        phi2 = chi2/n
        r,k = cmx.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


class ModelDevelopment:
    def simple_dl_model(self,X_shape,nb_class,nb_node=32,dropout_rate=0.3):
        '''
        

        Parameters
        ----------
        X_shape : TYPE
            DESCRIPTION.
        nb_class : TYPE
            DESCRIPTION.
        nb_node : TYPE, optional
            DESCRIPTION. The default is 32.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model = Sequential()
        model.add(Input(shape=X_shape))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_node,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class,activation='softmax'))
        model.summary()
        
        return model


class ModelEvaluation:
    def plot_loss_grapy(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.legend(['Training loss','Validation loss'])
        plt.show()
        hist.history['loss']

    def plot_acc_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.legend(['Training Acc','Validation Acc'])
        plt.show()
        hist.history['loss']