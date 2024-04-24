#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.datasets import load_diabetes


# In[2]:


#Loading Data
titanic_data = pd.read_csv('https://raw.githubusercontent.com/amberkakkar01/Titanic-Survival-Prediction/master/train.csv')


# In[3]:


#printing data
titanic_data.head()


# In[4]:


#looking for null values
titanic_data.isnull().sum()


# In[5]:


#Dropping  the “Cabin” column from the data frame as it won’t be of much importance due to a lot of null values

titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[6]:


#Replacing the missing values in the “Age” column with the mean value

titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[7]:


#Finding the mode value of the “Embarked” column as it will have occurred the maximum number of times
print(titanic_data['Embarked'].mode())


# In[8]:


#Replacing the missing values in the “Embarked” column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[9]:


#Checking for missing values again
titanic_data.isnull().sum()


# In[10]:


#Converting the string values in 'Sex' and 'Embarked' columns into integer type values, and transform it into a categorical column:

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[11]:


# Columns which are not of much importance in the  process maybe dropped.

titanic_data= titanic_data.drop(columns = ['PassengerId','Name','Ticket'],axis=1)


# In[12]:


#Printing the processed data
titanic_data.head()


# In[13]:


#Exploratory Data Analysis
titanic_data.info()


# In[14]:


#Univariable tendency and variation
titanic_data.describe()


# In[15]:


#Data visualization
plt.figure(figsize=(10, 8))
sns.heatmap(titanic_data.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[16]:


#remove weak variables
remove = []
for data in titanic_data:
    if data != 'Survived':
        cor_coef, pvalue= pearsonr(titanic_data[data], titanic_data['Survived'])
        if (cor_coef<0.08 and cor_coef>-0.08):                                                                              
            remove.append(data)
titanic_data.drop(remove, axis = 1, inplace = True)
print(f'dataframe which has been subtracted {titanic_data.shape[1]} from the remaining coloumns.')
titanic_data.head(5)


# In[17]:


#ordering the dataset columns
titanic_data = titanic_data[['Pclass', 'Sex' , 'Parch', 'Fare' , 'Embarked' , 'Survived']]
titanic_data.head(5)


# In[18]:


#selecting dependent and independent features
features = titanic_data.iloc[:, :-1].values
labels = titanic_data.iloc[:, -1].values


# In[19]:


#Data Normalization
from sklearn.preprocessing import MinMaxScaler

features = MinMaxScaler().fit_transform(features)


# In[20]:


#Test, Train split
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)


# In[21]:


#Data encoder ZZfeature 
from qiskit.circuit.library import ZZFeatureMap

num_features = features.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
feature_map.decompose().draw(output="mpl", fold=20)


# In[22]:


#Optimizer
from qiskit_algorithms.optimizers import COBYLA

optimizer = COBYLA(maxiter=100)


# In[23]:


#ansatz ReaAmplitudes
from qiskit.circuit.library import RealAmplitudes

ansatz = RealAmplitudes(num_qubits=num_features, reps=5)
ansatz.decompose().draw(output="mpl", fold=20)


# In[24]:


#Training
from qiskit.primitives import Sampler

sampler = Sampler()


from matplotlib import pyplot as plt
from IPython.display import clear_output

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()
    
    
    
    
import time
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

# Clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")


# In[25]:


train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


# In[ ]:




