# Importing Library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)

# Importing Train Dataset
data = pd.read_csv('train.csv')

#Handling Missing Value
na_features = [feature for feature in data.columns if data[feature].isnull().sum()>1]

dataset = data.copy()
dataset.groupby('Survived')['Age'].mean()
dataset["Age"] = dataset.groupby("Survived")['Age'].transform(lambda x: x.fillna(x.mean()))    

import statistics as stats
dataset["Embarked"] = dataset["Embarked"].fillna(stats.mode(dataset['Embarked']))

dataset = dataset.drop(columns=['Cabin','Ticket','Name'])

cate_col = [feature for feature in dataset.columns if dataset[feature].dtype == 'O']

# Feature Encoding
for fea in cate_col:
    fre_map = dataset[fea].value_counts().to_dict()
    dataset[fea]=dataset[fea].map(fre_map)

# Creating X and y
y = pd.DataFrame(dataset['Survived'])
X = dataset.drop(columns=['Survived','PassengerId'])

# Feature Selection
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# Feature Scaling
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_tr = scaler.fit_transform(X)
X_tr = pd.DataFrame(X_tr, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Test Data (Applying Same Processes as Training)
# Importing Test Data
test_data = pd.read_csv("test.csv")
test_dataset = test_data.copy()

# Handling Missing Values
test_dataset["Age"] = test_dataset.groupby("Survived")['Age'].transform(lambda x: x.fillna(x.mean())) 

import statistics as stats
test_dataset["Embarked"] = test_dataset["Embarked"].fillna(stats.mode(test_dataset['Embarked']))

test_dataset['Fare']= test_dataset['Fare'].fillna(test_dataset['Fare'].mean())
test_dataset = test_dataset.drop(columns=['Cabin','Ticket','Name'])

# Feature Encoding
test_cate_col = [feature for feature in test_dataset.columns if test_dataset[feature].dtype == 'O']

for fea in test_cate_col:
    fre_map = test_dataset[fea].value_counts().to_dict()
    test_dataset[fea]=test_dataset[fea].map(fre_map)

# Creating X_test and y_test    
y_test = pd.DataFrame(test_dataset['Survived'])
X_test = test_dataset.drop(columns=['Survived','PassengerId'])

# Feature Scaling
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_test_tr = scaler.fit_transform(X_test)
X_test_tr = pd.DataFrame(X_test_tr, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Appling KNN
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.metrics import accuracy_score
classifier = KNeighborsClassifier(n_neighbors = 231, metric = 'minkowski', p = 2)
classifier.fit(X_tr, y)
prediction = classifier.predict(X_test_tr)
score = accuracy_score(prediction,y_test)
print(score)

# Dumping Model
import pickle
pickle.dump(classifier,open('model.pkl','wb'))




