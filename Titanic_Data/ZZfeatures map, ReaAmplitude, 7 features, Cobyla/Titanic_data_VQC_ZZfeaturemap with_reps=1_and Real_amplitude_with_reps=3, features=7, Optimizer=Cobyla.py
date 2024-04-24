#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


#Loading Data
titanic_data = pd.read_csv('https://raw.githubusercontent.com/amberkakkar01/Titanic-Survival-Prediction/master/train.csv')


# In[5]:


#printing data
titanic_data.head()


# In[6]:


#looking for null values
titanic_data.isnull().sum()


# In[7]:


#Dropping  the “Cabin” column from the data frame as it won’t be of much importance due to a lot of null values

titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[8]:


#Replacing the missing values in the “Age” column with the mean value

titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[9]:


#Finding the mode value of the “Embarked” column as it will have occurred the maximum number of times
print(titanic_data['Embarked'].mode())


# In[10]:


#Replacing the missing values in the “Embarked” column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[11]:


#Checking for missing values again
titanic_data.isnull().sum()


# In[12]:


#Converting the string values in 'Sex' and 'Embarked' columns into integer type values, and transform it into a categorical column:

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[13]:


# Columns which are not of much importance in the  process maybe dropped.

titanic_data= titanic_data.drop(columns = ['PassengerId','Name','Ticket'],axis=1)


# In[14]:


#Printing the processed data
titanic_data.head()


# In[15]:


#Exploratory Data Analysis
titanic_data.info()


# In[16]:


#Univariable tendency and variation
titanic_data.describe()


# In[17]:


#Data visualization
plt.figure(figsize=(10, 8))
sns.heatmap(titanic_data.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[18]:


#ordering the dataset columns
titanic_data = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch' , 'Fare' , 'Embarked' , 'Survived']]
titanic_data.head(5)


# In[19]:


#selecting dependent and independent features
features = titanic_data.iloc[:, :-1].values
labels = titanic_data.iloc[:, -1].values


# In[20]:


#Data Normalization
from sklearn.preprocessing import MinMaxScaler

features = MinMaxScaler().fit_transform(features)


# In[21]:


#Test, Train split
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)


# In[22]:


#Data encoder ZZfeature 
from qiskit.circuit.library import ZZFeatureMap

num_features = features.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", fold=20)


# In[23]:


#ansatz ReaAmplitudes
from qiskit.circuit.library import RealAmplitudes

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)


# In[24]:


#Optimizer
from qiskit_algorithms.optimizers import COBYLA

optimizer = COBYLA(maxiter=100)


# In[25]:


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


# In[26]:


train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


# In[ ]:




