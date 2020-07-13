# **Titanic Disaster**
### Famouse Titanic Project that predicts that a person survived the disaster or not.

<img src="https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/README-Resources/Titanic.jpg" width=600>

---

### **Web APP on Heroku**
<img src="https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/README-Resources/Titanic-Disaster.gif" width="600">

### **[The Project on Heroku](https://titanicdisaster.herokuapp.com/)**

---

## The Dataset
![](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/README-Resources/Screenshot%20(108).png)

### **Dataset**
**[Train Dataset](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/train.csv)**

**[Test Dataset](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/test.csv)**

---

## **Overview**
* The Dataset has **"PassengerId"**, **"Survived"**, **"Pclass"**, **"Name"**, **"Sex"**, **"Age"**, **"SibSp"**, **"Parch"**, **"Ticket"**, **"Fare"**, **"Cabin"**, **"Embarked"** columns. It has total around **1300 rows** and **12 columns**.
* From the Dataset, we have to predict the **Survived** column: 
* **ExtraTreesClassifier** has been used for **Feature Selection**.
* I have used **Count/Frequency Encoding** Technique for **Feature Encoding**.
* I have applied many different algorithms but at the end, **KNN** gave better results.
---
## **Machine Learning Pipelines:**
---
### **1 Feature Engineering:**
  
**a> Handling Missing Values:**
* Here, In this data, columns **"Age"**, **"Cabin"**, **"Embarked"** has **Null** values. Column **"Cabin"** has way too many Null Values that it is not prefered to replace them. Hence, I dropped the Column. 
    
**b> Feature Encoding:**   
* In this data, For exploration I have used **Count/Frequency Encoding** Technique.

**c> Feature Scaling & Feature Transformation:**    
* Here, I have used **MinMaxScalar** for scaling the data.
---    
### **2 Feature Selection:**    
* There are various techniques for this but here i have used **ExtraTressClassifier**. Here, ExtraTreesClassifier Showed **"Sex"**, **"Age"**, **"Fare"**, **"Embarked"** as important columns.
* But, I wanted to know How model works, if I include other insignificant columns also, so i have taken **'Pclass'**, **'Sex'**, **'Age'**, **'SibSp'**, **'Parch'**, **'Fare'**, **'Embarked'**.

![Feature Selection](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/README-Resources/Screenshot%20(107).png)

---   
### **3,4&5 Model Selection**, **Model Creation** and **Testing**
    
* Here, Initially, I have tested different Algorithms by making **Train-Test Split with Training Data**, In that **"KNN"** came across as better algorithm as compare to others. So, I have used KNN with **proper Training and Testing Dataset**. 
    
* Initially, I tried many algorithms like **Random Forest**, **Decision Tree** and **K-NN**. 
* Among these, K-NN gave better results.
* For this I have tried all the k values **till 500** and **k=251** gave better results.
---
* Finally, Even if I have used **'Pclass'**, **'Sex'**, **'Age'**, **'SibSp'**, **'Parch'**, **'Fare'**, **'Embarked'** columns for Exploring purpose, I managed to score the Rank on Kaggle.
* For detailed look at Project, go to **[Titanic.ipynb](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/Titanic.ipynb)** and **[model.py](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/model.py)**
