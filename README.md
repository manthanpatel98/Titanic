# **Titanic**
## Famouse Titanic Project that predicts that a person survived the disaster or not.
---
### **Web APP on Heroku**
![Project](https://github.com/manthanpatel98/Alcohol-Quality-Checker/blob/master/README-Resources/AlcoholQuality.gif)
**[The Project on Heroku](https://titanicdisaster.herokuapp.com/)**
---
## The Dataset
![](https://github.com/manthanpatel98/Alcohol-Quality-Checker/blob/master/README-Resources/Screenshot%20(105).png)
### **Dataset**
**[Train Dataset](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/train.csv)**
**[Test Dataset](https://github.com/manthanpatel98/Titanic/blob/master/Titanic%20Dataset/test.csv)**

---
## **Machine Learning Pipelines:**
---
### **1> Feature Engineering:**
  
**a> Handling Missing Values:**
* Here, In this data, columns **"Age"**, **"Cabin"**, **"Embarked"** has **Null** values. Column **"Cabin"** has way too many Null Values that it is not prefered to replace them. Hence, I dropped the Column. 
    
**b> Feature Encoding:**   
* In this data, For exploration I have used **Count/Frequency Encoding** Technique.

**c> Feature Scaling & Feature Transformation:**    
* Here, I have used **MinMaxScalar** for scaling the data.
---    
### **2> Feature Selection:**    
* There are various techniques for this but here i have used **ExtraTressClassifier**. Here, ExtraTreesClassifier Showed **"Sex"**, **"Age"**, **"Fare"**, **"Embarked"** as important columns.
* But, I wanted to know How model works, if I include other insignificant columns also, so i have taken **'Pclass'**, **'Sex'**, **'Age'**, **'SibSp'**, **'Parch'**, **'Fare'**, **'Embarked'**.

![Feature Selection](https://github.com/manthanpatel98/Alcohol-Quality-Checker/blob/master/README-Resources/Screenshot%20(106).png)
---   
### **3,4&5> Model Selection**, **Model Creation**, **Testing**
    
* Here, Initially, I have tested different Algorithms by making **Train-Test Split with Training Data**, In that **"KNN"** came across as better algorithm as compare to others. So, I have used KNN with **proper Training and Testing Dataset**. 
    
* Initially, I tried many algorithms like **Random Forest**, **Decision Tree** and **K-NN**. 
* Among these, K-NN gave better results.
* For this I have tried all the values k values **till 500** and **k=125** gave better results.
    
| Algorithm | Average Accuracy |
| ---- | ----|
| Random Forest | 76.19% |
| Decision Tree | 75.88% |
| K-NN | 79.48% |
| SVM | 79.7% |
| Naive bayes | 78.21% |

---
* Finally, I decided to go with KNN because as we know **SVM generally has higher variance**, whereas in KNN we can fix it by **choosing the right K value**. In my project **k=125** gave better results.
* For detailed look at Project, go to **[Alcohol-Quality.ipynb](https://github.com/manthanpatel98/Alcohol-Quality-Checker/blob/master/Alcohol-Quality.ipynb)** and **[model.py](https://github.com/manthanpatel98/Alcohol-Quality-Checker/blob/master/model.py)**
