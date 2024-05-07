# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LOKESH R
RegisterNumber:  212222240055
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the CSV data
df = pd.read_csv('/content/Book1.csv')

# View the beginning and end of the data
df.head()
df.tail()

# Segregate data into variables
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
# Create a linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict values using the model
y_pred = regressor.predict(x_test)

# Display predicted and actual values
print("Predicted values:", y_pred)
print("Actual values:", y_test)

# Visualize the training data
plt.scatter(x_train, y_train, color="black")
plt.plot(x_train, regressor.predict(x_train), color="red")
plt.title("Hours VS scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Visualize the test data
plt.scatter(x_test, y_test, color="cyan")
plt.plot(x_test, regressor.predict(x_test), color="green")
plt.title("Hours VS scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:

df.head()


![ex2](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/0e663e56-3438-488d-bc97-78f15dd21c81)

df.tail()



![tail](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/6a828b4b-5a25-4a86-af93-47e58ed6426c)

Array values of X



![array values](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/6b681ff0-9760-4803-95be-a31705dfb9d4)

Array values of Y



![y](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/908d314c-9990-4d68-b700-c0ff63399cd5)

Values of Y Prediction



![y pred](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/f6c8d02c-7adc-4c68-90a2-bf373a1e1c46)

Values of Y Test



![y test](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/7a956bd7-ac16-4351-afad-598c29c60d34)

Training Set Graph



![graph](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/6dc2f463-1c14-47a1-8d80-5308efbad6c5)

Test Set Graph



![test graph](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/84423878-0a64-4bce-841c-0286efd46fa9)

Values of MSE



![mse](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/2172426c-76a3-4ffb-8644-39ec7792bfde)

Values of MAE



![mae](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/ea4ef639-e4ff-4477-ada9-df9cf5643eaa)

Values of RMSE



![rmse](https://github.com/LokeshRajamani/intro-to-ml-EX2/assets/120544804/ece1a131-2856-47ad-b534-1b11587762ae)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
