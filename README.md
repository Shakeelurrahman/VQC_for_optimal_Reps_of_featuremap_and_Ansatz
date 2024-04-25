This project contains jupiter notebooks for Variational Quantum Classifier (VQC), a variational- classification algorithm in Quantum Machine Learning (QML). Two data sets, 'Penguins data' and 'Titanic data' have been trained with different number of features and various combinations of repetitions of Feature-map and Ansatz.
All the coding is done in Qiskit framework; a library in python for quantum computing.
For data encoding 'ZZfeature-map' has been used as a feature map and 'RealAmplitude' has been used as Ansatz. Both ZZfeature-map and RealAmplitude are built into the Qiskit library.
VQC is a hybrid algorithm; employing both quantum and classical resources. For optimisation, two classical optimisers, Cobyla and SPSA have been used. Seperate graphs are plotted for both optimisers.

  The first data set, Penguins data consists of seven
features in total namely: ’species’, ’island’, ’bill length’, ’bill depth’, ’flipper length’,
’body mass’, and ’sex’. Out of these, one feature ’species’ is made target and rest six of
them are the features based on which prediction is made. 
The other data set 
Titanic data consists of 12 Features namely ’PassengerId’, ’Survived’, ’Pclass Name’,
’Sex’, ’Age’, ’SibSp(passendgers with any siblings onboard)’, ’Parch(passengers with
parents or childern onboard)’, ’Ticket’, ’Fare’, ’Cabin’, ’Embarked’. Classification
is made against, whether a passenger survived or not.

For graphs the values on x-axis(11, 12, 13....) represent the number of reps. First number in each value represents the number of reps of feature map and second number represets the number of reps of Ansatz.

![Fig42](https://github.com/Shakeelurrahman/VQC_for_optimal_Reps_of_featuremap_and_Ansatz/assets/114961796/9f4a598c-1b83-42b1-8b62-c8f0869bda0a)

In the bar graph as shown above, blue bars represent the accuracy of training data and orange ones represent the accuracy of test data.
