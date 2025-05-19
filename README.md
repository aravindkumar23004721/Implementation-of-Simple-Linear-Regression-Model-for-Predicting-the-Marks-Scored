# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARAVIND KUMAR SS
RegisterNumber:212223110004

*/
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# READ CSV FILES
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())

# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
Head:

![Screenshot 2024-08-30 135158](https://github.com/user-attachments/assets/ebc9b83d-a9dc-4d84-ba7c-ce97a806b7de)

Tail:

![Screenshot 2024-08-30 135231](https://github.com/user-attachments/assets/b5f1303a-4c52-4781-a3c0-73376fb8ce33)

Array value of x:

![Screenshot 2024-08-30 135307](https://github.com/user-attachments/assets/e7814e22-8f6f-4626-a195-6a1c79d069e4)


Array value of y:

![Screenshot 2024-08-30 135353](https://github.com/user-attachments/assets/b700ea1d-ce2a-439a-9a34-9d323486cd5a)


Y prediction:

![Screenshot 2024-08-30 135441](https://github.com/user-attachments/assets/9a1465c9-f315-4cdd-b708-7c7968d9e1a6)


Array value of Y test:

![Screenshot 2024-08-30 135528](https://github.com/user-attachments/assets/288392fb-3271-44ae-8941-dd15071e4e4c)


Training set graph:

![Screenshot 2024-08-30 135607](https://github.com/user-attachments/assets/4dc09351-aa45-4ed9-b996-84dd6515bdf4)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
