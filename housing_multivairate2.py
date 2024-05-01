import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('Housing.csv')
x = df.iloc[:,1:].values
y = df.iloc[:,0].values

#model
reg = LinearRegression()
reg.fit(x,y)
print(reg.coef_)
print(reg.intercept_)

#training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=40)
predictions = reg.predict(xtest)
comparison = pd.DataFrame({'Predicted Values':predictions,'Actual Values':ytest})
print(comparison.head())
#print(comparison)
plt.scatter(xtest[:,0], ytest, color = 'blue')
plt.scatter(xtest[:,0], predictions, color = 'red')
plt.show()
score = reg.score(xtest, ytest)
score2 = reg.score(xtrain, ytrain)
print(f'the score for train and test dataset is {score2},{score}')
