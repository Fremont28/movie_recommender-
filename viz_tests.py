#6/26/18
#Creating a class for simple data visualization using numpy arrays 
import matplotlib.pyplot as plotlib 
import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

frank_rides.info() 
sample_data=frank_rides
sample_data1=sample_data.values 

class Data_Viz:
    def __init__(self):
        pass

    def simple_histogram(self,variable):
        data=pd.DataFrame(np.array(variable)) #convert np array to pandas dataframe  
        data.plot.hist(orientation="horizontal")
        plt.xlabel('Variable')
        plt.ylabel('Frequency')
        plt.show() 

    def simple_scatter(self,variable1,variable2):
        variable1=pd.DataFrame(np.array(variable1))
        variable2=pd.DataFrame(np.array(variable2))
        df=pd.concat([variable1,variable2],axis=1)
        df.columns=['variable1','variable2']
        sns.regplot(x=df['variable1'],y=df['variable2'])
        plt.show() 

    def simple_boxplot(self,variable1):
        variable1=pd.DataFrame(np.array(variable1))
        df=pd.DataFrame([variable1])
        df.columns=['variable1']
        sns.boxplot(x=df['variable1'])
        plt.show()  

plots=Data_Viz() 
plots.simple_boxplot(sample_data1[:,4])






