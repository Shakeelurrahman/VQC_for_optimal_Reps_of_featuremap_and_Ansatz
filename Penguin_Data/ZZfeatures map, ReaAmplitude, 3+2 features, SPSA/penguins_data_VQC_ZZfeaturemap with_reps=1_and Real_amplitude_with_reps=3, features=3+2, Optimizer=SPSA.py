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


# In[2]:


data = pd.read_csv('https://raw.githubusercontent.com/michellechandraa/michellechandraa/main/penguins.csv' , delimiter = ';')
penguin_df = pd.DataFrame(data)
penguin_df.head(5)


# In[3]:


##checking for missing values
penguin_df.isnull().sum()


# In[4]:


#dropping missing values
penguin_df.dropna(axis=0, inplace=True)
penguin_df.isnull().sum()


# In[5]:


#check for duplicate values
penguin_df.duplicated().value_counts()


# In[6]:


#label encoding
labelencoder = preprocessing.LabelEncoder()
penguin_df.sex = labelencoder.fit_transform(penguin_df.sex)
penguin_df.island = labelencoder.fit_transform(penguin_df.island)
penguin_df.species = labelencoder.fit_transform(penguin_df.species)
penguin_df.head(5)


# In[7]:


#e=Exploratory Data Analysis
penguin_df.info()


# In[8]:


#Univariable tendency and variation
penguin_df.describe()


# In[9]:


#Data visualization
cp = sns.countplot(data=penguin_df, x='island',hue='species',palette='viridis')
cp.set_xticks([0, 1, 2])
cp.set_xticklabels(['Torgersen', 'Biscoe','Dream'])
plt.legend(labels=['Adelie' , 'Chinstrap' , 'Gentoo'])
plt.title('The number of each species on each island')
plt.xlabel('Island')
plt.ylabel('amount')
sns.despine()


# In[10]:


#check for correlation 
corr_coef = penguin_df.corr()
sns.heatmap(corr_coef, annot = True, cmap = 'coolwarm', fmt = '0.1f' , vmin = -1 , vmax = 1)


# In[11]:


#remove weak variables
remove = []
for data in penguin_df:
    if data != 'species':
        cor_coef, pvalue= pearsonr(penguin_df[data], penguin_df['species'])
        if (cor_coef<0.5 and cor_coef>-0.5):                                                                              
            remove.append(data)
penguin_df.drop(remove, axis = 1, inplace = True)
print(f'dataframe which has been subtracted {penguin_df.shape[1]} from the remaining coloumns.')


# In[12]:


#ordering the dataset columns
penguin_df = penguin_df[['island', 'flipper_length_mm', 'body_mass_g', 'species']]
penguin_df.head(5)


# In[13]:


# selecting dependent and independent features
features = penguin_df.iloc[:, :-1].values
labels = penguin_df.iloc[:, -1].values


# In[14]:


alpha = features[:, :2] * features[:, 1:]          
features = np.append(features, alpha, axis=1)       

print(features.shape) 


# In[15]:


#train test split
from sklearn.preprocessing import MinMaxScaler

features = MinMaxScaler().fit_transform(features)


# In[16]:


from sklearn.model_selection import train_test_split
from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 123
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)


# In[17]:


#Data encoder ZZfeature 
from qiskit.circuit.library import ZZFeatureMap

num_features = features.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_map.decompose().draw(output="mpl", fold=20)


# In[18]:


#ansatz ReaAmplitudes
from qiskit.circuit.library import RealAmplitudes

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
ansatz.decompose().draw(output="mpl", fold=20)


# In[19]:


#Optimizer
from qiskit.algorithms.optimizers import SPSA

optimizer = SPSA(maxiter=100)


# In[20]:


#training
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

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

    


# In[21]:


train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")


# In[ ]:




