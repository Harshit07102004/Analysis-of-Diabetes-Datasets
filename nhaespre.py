import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import re
import sklearn
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
df1 = pd.read_csv('labs.csv')
df2 = pd.read_csv('examination.csv')
df3 = pd.read_csv('demographic.csv')
df4 = pd.read_csv('diet.csv')
df5 = pd.read_csv('questionnaire.csv')
df2.drop(['SEQN'], axis = 1, inplace=True)
df3.drop(['SEQN'], axis = 1, inplace=True)
df4.drop(['SEQN'], axis = 1, inplace=True)
df5.drop(['SEQN'], axis = 1, inplace=True)
df = pd.concat([df1, df2], axis=1, join='inner')
df = pd.concat([df, df3], axis=1, join='inner')
df = pd.concat([df, df4], axis=1, join='inner')
df = pd.concat([df, df5], axis=1, join='inner')
df.describe()
df.to_csv('concatinated.csv', index=False)
from sklearn.feature_selection import VarianceThreshold
df.dropna(axis=1, how='all')
df.dropna(axis=0, how='all')
df = df.rename(columns = {'SEQN' : 'ID',
                'RIAGENDR' : 'Gender',  'DMDYRSUS' : 'Years_in_US', # Nan -> american iguess
                'INDFMPIR' : 'Family_income','LBXGH' : 'GlycoHemoglobin',
    'BMXARMC' : 'ArmCircum','BMDAVSAD' : 'SaggitalAbdominal',
    'MGDCGSZ' : 'GripStrength','DRABF' : 'Breast_fed'})
df = df.loc[:, ['ID', 'Gender', 'Years_in_US', 'Family_income','GlycoHemoglobin', 'ArmCircum',
                'SaggitalAbdominal', 'GripStrength', 'Breast_fed']]
df.describe()
df.info()
