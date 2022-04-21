import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Loading fuelConsumption.cvs
path="https://raw.githubusercontent.com/JameelMalas/Python-Models/main/FuelConsumption.csv?_sm_au_=iVV5tPQb47FP7jD7L321jK0f1JH33"
df=pd.read_csv(path)
df.head()


#subset that includes variables of interest
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


#Let's plot Emission values with respect to Engine size:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Creating train and test dataset
#Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()




#In reality, there are multiple variables that impact the co2emission. When more than one independent variable is present,
# the process is called multiple linear regression. An example of multiple linear regression is predicting co2emission using
# the features FUELCONSUMPTION_COMB, EngineSize and Cylinders of cars. The good thing here is that multiple linear
# regression model is the extension of the simple linear regression model.



from sklearn import linear_model
regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)



from sklearn.metrics import r2_score

test_x = np.asanyarray(train[['FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(train[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))

