
from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

nan = float('nan')

dataset = pd.read_excel('JOE_Test4.xlsx')

dataset.loc[dataset['Normal1_T2']== 'Normal.1','Normal1_T2']= 5
dataset.loc[dataset['Normal2_T2']== 'Normal.2','Normal2_T2']= 6
dataset.loc[dataset['RW7_T2']== 'RW7','RW7_T2']= 7
dataset.loc[dataset['Woerter_T2']== 'Woerter','Woerter_T2']= 8


dataset4 = dataset.loc[:, ['Normal1_T2','mean_steplength_RW7_T2',
                            'mean_steptime_RW7_T2','mean_frequency_RW7_T2','mean_speed_RW7_T2',
                            'mean_stepwidth_RW7_T2','MOCA_CERAD_korr_T2']]

dataset4[['mean_steplength_RW7-RG1_T2','mean_steptime_RW7-RG1_T2',
          'mean_frequency_RW7-RG1_T2','mean_speed_RW7-RG1_T2', 
          'mean_stepwidth_RW7-RG1_T2']] = \
          dataset[['mean_steplength_RW7_T2','mean_steptime_RW7_T2','mean_frequency_RW7_T2',
                   'mean_speed_RW7_T2','mean_stepwidth_RW7_T2']]- \
                   dataset[['mean_steplength_Normal1_T2','mean_steptime_Normal1_T2',
                            'mean_frequency_Normal1_T2','mean_speed_Normal1_T2',
                            'mean_stepwidth_Normal1_T2']].values
dataset4['Exp_RW7-RG1'] = 2

dataset4[['mean_steplength_RW7-RG2_T2','mean_steptime_RW7-RG2_T2',
          'mean_frequency_RW7-RG2_T2','mean_speed_RW7-RG2_T2', 
          'mean_stepwidth_RW7-RG2_T2']] = \
          dataset[['mean_steplength_RW7_T2','mean_steptime_RW7_T2','mean_frequency_RW7_T2',
                   'mean_speed_RW7_T2','mean_stepwidth_RW7_T2']]- \
                   dataset[['mean_steplength_Normal2_T2','mean_steptime_Normal2_T2',
                            'mean_frequency_Normal2_T2','mean_speed_Normal2_T2',
                            'mean_stepwidth_Normal2_T2']].values
dataset4['Exp_RW7-RG2'] = 3                           
dataset4[['mean_steplength_RW7-WSPL_T2','mean_steptime_RW7-WSPL_T2',
          'mean_frequency_RW7-WSPL_T2','mean_speed_RW7-WSPL_T2', 
          'mean_stepwidth_RW7-WSPL_T2']] = \
          dataset[['mean_steplength_RW7_T2','mean_steptime_RW7_T2','mean_frequency_RW7_T2',
                   'mean_speed_RW7_T2','mean_stepwidth_RW7_T2']]- \
                   dataset[['mean_steplength_Woerter_T2','mean_steptime_Woerter_T2',
                            'mean_frequency_Woerter_T2','mean_speed_Woerter_T2',
                            'mean_stepwidth_Woerter_T2']].values

dataset4['Exp_RW7-WSPL'] = 4

dataset5 = dataset4.iloc[:,6:]
dataset_RG1_T2 = dataset4.iloc[:, 7:13]
dataset_RG1_T2[['MOCA_CERAD_korr_T2']] = dataset.loc[:,['MOCA_CERAD_korr_T2']]
dataset_RG2_T2 = dataset4.iloc[:, 13:19]
dataset_RG2_T2[['MOCA_CERAD_korr_T2']] = dataset.loc[:,['MOCA_CERAD_korr_T2']]
dataset_RW7_T2 = dataset4.iloc[:, 19:25]
dataset_RW7_T2[['MOCA_CERAD_korr_T2']] = dataset.loc[:,['MOCA_CERAD_korr_T2']]

# number of columns
n_columns = len(dataset_RG1_T2.columns)

# save final columns names
columns2 = dataset_RG1_T2.columns
# rename both columns to numbers
dataset_RG1_T2.columns = range(n_columns)
dataset_RG2_T2.columns = range(n_columns)
dataset_RW7_T2.columns = range(n_columns)

# concat columns
df_T2 = pd.concat([dataset_RG1_T2, dataset_RG2_T2,dataset_RW7_T2], axis=0, ignore_index=True)
# rename columns in new dataframe
df_T2.columns = columns2

df_T2 = df_T2.dropna(how = 'any')

data2 = df_T2

X2 = data2.iloc[:,0:7].values
y2 = data2.iloc[:,6:7].values.ravel()

sc = StandardScaler()
X2[:,0:5] = sc.fit_transform(X2[:,0:5])

# Encoding categorical data
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder',OneHotEncoder(),[5])], remainder = 'passthrough')
X2 = np.array(ct.fit_transform(X2),dtype =np.float)

filename = 'finalized_model_SVM.sav'
filename2 = 'finalized_model2_KNN.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

loaded_model2 = pickle.load(open(filename2, 'rb'))

ynew_SVM = loaded_model.predict(X2)
ynew_KNN = loaded_model2.predict(X2)

data2[('Predicted Value_SVM')] = ynew_SVM
data2[('Predicted Value_KNN')] = ynew_KNN

writer = pd.ExcelWriter('D:/Ankit_Jaiswal/Codes/Python/Predicted_Results.xlsx', engine='openpyxl') 
wb = writer.book
data2.to_excel(writer, index=False)
wb.save('D:/Ankit_Jaiswal/Codes/Python/Predicted_Results.xlsx')