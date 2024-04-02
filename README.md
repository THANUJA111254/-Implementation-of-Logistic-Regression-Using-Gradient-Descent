# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

## 1.Import Necessary Libraries:
Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

## 2. Define the Linear Regression Function:
Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

## 3. Load and Preprocess the Data:
Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.

## 4. Perform Linear Regression:
Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

## 5.Make Predictions on New Data:
Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

## 6.Print the Predicted Value:
Display the predicted value for the target variable based on the linear regression model applied to the new data.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Hiruthik Sudhakar
RegisterNumber:  212223240054
*/
```
```python
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
x=(df.iloc[:,:-1]).values
print(x)
y=(df.iloc[:,1]).values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```


## Output:
![image](https://github.com/HIRU-VIRU/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972122/a137c458-5989-4c9a-bb00-9afaa2594468)
![image](https://github.com/HIRU-VIRU/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145972122/101c2d6a-2b40-4c1e-a2f4-4dbe9b80feef)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

