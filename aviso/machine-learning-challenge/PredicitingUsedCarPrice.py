#!/usr/bin/env python
# coding: utf-8

# # Used Car Price Prediction
a. Problem 

○ Predict used car prices for the given dataset. 
○ Dataset is provided. (DataScienceCodingChallengeJunior-ML.csv) 
○ Choose the appropriate ML algorithm and explain why you have chosen that 
algorithm. 
○ Achieve the best possible result you can. Elegance of programming is not a 
criterion. 

b. Dataset 

○ Data Features 
■ Price - The dependent variable 
■ Make - Manufacturer and model details of the car 
■ Location - City where the car sale happened 
■ Age - Age of vehicle in years 
■ Odometer - Number of kilometers car has run 
■ FuelType - Fuel consumed by the car 
■ Transmission - Automatic gear or Stick-shift (manual gear change) 
■ OwnerType - How many times has the car changed hands 
■ Mileage - Mileage of the car measured as Km/L (for Petrol and Diesel) or 
Km/Kg (for LPG and CNG) 
■ EngineCC - Engine Capacity measured in CC 
■ PowerBhp - Engine’s power measured in BhP Units 
○ Data munging has been done to a significant extent but look out for Invalid 
values in a few cases. 

c. Question 

○ What are your observations on the data? 
○ What are the metrics you have chosen? And Why those metrics? 
○ What is the algorithm you have chosen? Why? 
○ What can be done to improve the model performance? 


d. Submission 
○ Python Program File (.py) which can be executed. 

# ## TODO Use MLFLow to caputre and track the hypertunning & diffirent run results

# ## Methodology 
# 
# 
# 1. Linear Regression: Quick to train and test as a baseline algorithm
# 
# 2. Gradient Boost: To account for non-linear relationships By splitting the data into 100 regions
# 
# 3. Random Forest : To account for the large number of features in the
#     Dataset and compare a bagging technique with the Gradient Boost method.
# 
# 4. XGBoost: To improve performance compared to standard Gradient
#     Boosting using regularization, second order gradients and added
#     support for parallel compute.
# 
# 5. KMeans + Linear Regression Ensemble: 
#     In order to capitalize on the linear regression results using ensemble learning,
# 
# 6. Light GBM: To improve performance compared to Gradient
#     Boosting with a leaf-wise-tree growth approach and improved
#     speed compared to XGBoos

# ## Metrics 
# 
# * Mean Squared Error(MSE)
# * Root-Mean-Squared-Error(RMSE).
# * Mean-Absolute-Error(MAE).
# * R² or Coefficient of Determination.
# * Adjusted R²
# 
# 
# ### I will be using RMSE & R2 for model comparisions. RMSE primarily because want to panalize if there is huge diffirence in predicted and actual price

# ## Data Prepocessing 

# In[400]:


import pandas as pd
import numpy as np
import requests
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import os
import seaborn as sns


# In[235]:


df = pd.read_csv("DataScienceCodingChallengeJunior-ML.csv")


# In[236]:


df.head()


# ## Check for Missing Values

# In[21]:


missing_values = df.isnull().sum()
missing_values


# ### No missing value hence we are good! No need to fill up anything

# In[237]:


df.describe()


# In[238]:


df.columns


# ### Mileage, EngineCC & PowerBhp need to be checked

# In[239]:


#df['Mileage']
np.issubdtype(df['Mileage'].dtype, np.number)


# In[240]:


np.issubdtype(df['EngineCC'].dtype, np.number)


# In[241]:


np.issubdtype(df['PowerBhp'].dtype, np.number)


# In[242]:


df.shape


# In[243]:


df[df.Mileage.apply(lambda x: x.isnumeric())].shape


# In[244]:


df[df.EngineCC.apply(lambda x: x.isnumeric())].shape


# In[245]:


df[df.PowerBhp.apply(lambda x: x.isnumeric())].shape


# ### +AC0-1 character is present in above feature hence we need to take care of this. 
# 
# Two options
# * Remove those data points/rows
# * Replace this by mean of the feature

# In[247]:


df.shape


# In[250]:


df = df[(df != '+AC0-1').all(1)]


# In[252]:


df.shape


# #### We just lost 19 data points hence removing this data points will be a better approach

# In[255]:


df[df.Mileage.apply(lambda x: x.isnumeric())]


# In[256]:


df.select_dtypes(include=[np.number])


# ## Data Exploration 

# In[257]:


df['OwnerType'].value_counts()


# In[261]:


df['OwnerType'].value_counts().plot(kind='bar')


# In[262]:


df['Make'].value_counts()


# In[264]:


df['Make'].value_counts()[:20].plot(kind='bar')


# ### TODO : Make can be used to identify the brand 
# 
# 
# Have collected list of car brand names

# In[297]:


car_brands = pd.read_csv("car_brands.csv")


# In[301]:


car_brands


# In[319]:


len(df['Make'].unique())


# In[328]:


df['Brand'] = df['Make'].str[:5]


# In[329]:


df['Brand'].unique()


# In[330]:


df['Brand'] = df['Make'].str[:3]


# In[331]:


df['Brand'].unique()


# In[265]:


df['Location'].value_counts()


# In[266]:


df['Location'].value_counts()[:20].plot(kind='bar')


# In[267]:


df['FuelType'].value_counts()


# In[268]:


df['FuelType'].value_counts()[:20].plot(kind='bar')


# In[269]:


df['Transmission'].value_counts()


# In[271]:


df['Transmission'].value_counts().plot(kind='bar')


# In[274]:


x = df.Price
plt.figure(figsize=(10,6))
sns.distplot(x).set_title('Frequency Distribution Plot of Prices')


# In[275]:


df.columns


# In[332]:


cat_val = ['Brand','Make', 'Location', 'FuelType','Transmission', 'OwnerType']

for col in cat_val:
    print ([col]," : ",df[col].unique())


# ## Visualization / Scatter-Matrix and Histogram of prices

# In[27]:


df["Price"].hist(bins = 50, log = True)


# In[278]:


plt.figure(figsize=(20,10))
sns.boxplot(y='Price', data=df, width=0.5)


# In[279]:


conda install wordcloud


# In[282]:


plt.figure(figsize=(20,12))
sns.regplot(x='Age', y='Price', data=df).set_title('Vehicle Age vs Price')


# #### Less age more price (3-5 years looks good)

# In[281]:


plt.figure(figsize=(20,12))
sns.regplot(x='Odometer', y='Price', data=df).set_title('Vehicle Odometer vs Price')


# In[285]:


plt.figure(figsize=(10,6))
sns.regplot(x='Odometer', y='Price', data=df).set_title('Odometer vs Price')


# ### Less Odometer range more the price

# In[288]:


df.info()


# In[289]:


df.Mileage = df.Mileage.astype(float)
df.EngineCC = df.EngineCC.astype(float)
df.PowerBhp = df.PowerBhp.astype(float)


# In[ ]:


### Correlation


# In[290]:


df.corr()


# In[292]:


plt.figure(figsize=(10,6))
sns.heatmap(corr, vmax=1, square=True)


# ### Train & Test Split Creation

# In[333]:


df.head()


# In[334]:


df.drop('Make',axis=1,inplace=True)


# In[335]:


df.head()


# In[355]:


# Seperation of Predcitors (Features) and the Labes (Targets)

y = df["Price"].copy()
df.drop("Price", axis=1,inplace=True)


# ### Got this on internet/stack-overflow

# In[356]:


# Since Scikit-Learn doesn't hanldes DataFrame, we build a class for it

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[357]:


# Setting categorical and numerical attributes

categorical_features = ['Brand','Location','FuelType','Transmission','OwnerType']

numerical_features = list(df.drop(categorical_features, axis=1))

# Building the Pipelines

numerical_pipeline = Pipeline([
    ("selector", DFSelector(numerical_features)),
    ("std_scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("selector", DFSelector(categorical_features)),
    ("encoder", OneHotEncoder(sparse=True))
])

full_pipeline = FeatureUnion(transformer_list =[
    ("numerical_pipeline", numerical_pipeline),
    ("categorical_pipeline", categorical_pipeline)
])


# In[358]:


numerical_features


# ### New features/columns added because of OneHotEncoder

# In[359]:


df.shape


# #### 5 Categorical encoded using OneHotEncoder
# 
# #### 5 Numerical

# In[360]:


len(df['Brand'].unique()) + len(df['Location'].unique()) + len(df['FuelType'].unique()) + len(df['Transmission'].unique()) + len(df['OwnerType'].unique())


# In[361]:


48 + 5


# In[362]:


len(df['Location'].unique())


# In[363]:


X = full_pipeline.fit_transform(df)


# In[365]:


X.shape


# In[366]:


y.shape


# In[367]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[370]:


X_train.shape


# In[372]:


y_train.shape


# In[371]:


X_test.shape


# In[373]:


y_test.shape


# ## Models

# In[380]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # Regression Model

# In[375]:


linreg = LinearRegression()
linreg.fit(X_train, y_train) 


# In[376]:


linreg.score(X_train, y_train)


# In[377]:


linreg.score(X_test, y_test)


# In[378]:


y_pred = linreg.predict(X_test)


# In[381]:


lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# #### Residual plots 
# 
# 
# Used to check the error between actual values and predicted values. If a linear model is appropriate, we expect to see the errors to be randomly spread and have a zero mean.

# In[379]:


plt.figure(figsize=(10,6))
sns.residplot(x=y_pred, y=y_test)


# The mean of the points might be close to zero but obviously they are not randomly spread. The spread is close to a U-shape which indicates a linear model might not be the best option for this task.

# ### DecisionTreeRegressor

# In[384]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)


# In[385]:


y_pred = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, y_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[420]:


tree_reg.score(X_test, y_test)


# #### R2 of DecisionTreeRegressor better than Linear Regression 

# ### RandomForestRegressor

# In[421]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42, n_jobs =-1, max_depth = 30 )
forest_reg.fit(X_train, y_train)


# In[387]:


y_pred = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, y_pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[427]:


forest_reg.score(X_test,y_test)


# ### GradientBoostingRegressor

# In[402]:


gbrt=GradientBoostingRegressor(n_estimators=100)


# In[403]:


gbrt.fit(X_train, y_train)


# In[404]:


y_pred=gbrt.predict(X_test)


# In[424]:


gbrt.score(X_test,y_test)


# In[425]:


np.sqrt(mean_squared_error(y_test,y_pred))


# ### KNeighborsRegressor

# In[408]:


from sklearn.neighbors import KNeighborsRegressor


# In[410]:


neigh = KNeighborsRegressor(n_neighbors=3)


# In[411]:


neigh.fit(X_train, y_train)


# In[412]:


y_pred=neigh.predict(X_test)


# In[445]:


np.sqrt(mean_squared_error(y_test,y_pred))


# #### R2 of RandomForestRegressor better than DecisionTreeRegressor and also RMSE is also less 

# In[388]:


from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[389]:


# Offline i used CV=10

scores = cross_val_score(lin_reg, X_test, y_test,
                         scoring="neg_mean_squared_error", cv=4)
lin_rmse_scores = np.sqrt(-scores)

display_scores(lin_rmse_scores)


# In[390]:


# Offline i used CV=10

scores = cross_val_score(tree_reg, X_test, y_test,
                         scoring="neg_mean_squared_error", cv=4)
tree_rmse_scores = np.sqrt(-scores)

display_scores(tree_rmse_scores)


# In[391]:


# Offline i used CV=8

from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest_reg, X_test, y_test,
                         scoring="neg_mean_squared_error", cv=2)
forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)


# In[429]:


# Offline i used CV=8

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gbrt, X_test, y_test,
                         scoring="neg_mean_squared_error", cv=2)
rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)


# In[430]:


# Offline i used CV=8

from sklearn.model_selection import cross_val_score

scores = cross_val_score(neigh, X_test, y_test,
                         scoring="neg_mean_squared_error", cv=2)
rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)


# In[434]:


conda install -c conda-forge xgboost


# ## XGBRegressor

# In[435]:


from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV


# In[438]:


xgb_reg = XGBRegressor(n_estimators=100,
                           eta=0.05,
                           learning_rate=0.02,
                           gamma=2,
                           max_depth=6,
                           min_child_weight=1,
                           colsample_bytree=0.8,
                           subsample=0.3,
                           reg_alpha=2,
                           base_score=9.99)


# In[439]:


xgb_reg.fit(X_train,y_train)


# In[440]:


y_pred = xgb_reg.predict(X_test)
xgb_mse = mean_squared_error(y_test, y_pred)
xgb_rmse = np.sqrt(forest_mse)
xgb_rmse


# In[442]:


xgb_reg.score(X_test,y_test)


# In[444]:


# Offline i used CV=8

from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb_reg, X_test, y_test,
                         scoring="neg_mean_squared_error", cv=2)
rmse_scores = np.sqrt(-scores)

display_scores(rmse_scores)


# ## RMSE comparision 
# * LR-RMSE 5.600700933422275
# * DT-RMSE 4.849837741272038
# * RF-RMSE 3.6618054376005156
# * GradientBoostingRegressor-RMSE 4.491727876721665
# * KNeighborsRegressor-RMSE 4.397893638120978
# * XGBRegressor-RMSE 3.6618054376005156

# ### RandomForest Looks winner here! RF and XGBRegressor are very close!

# ## Feature Importance

# In[448]:


feature_importances = forest_reg.feature_importances_
feature_importances


# In[449]:


cat_encoder = categorical_pipeline.named_steps["encoder"]
#cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = categorical_features #+ cat_encoder
sorted(zip(feature_importances, attributes), reverse=True)


# ## Final Prediction and conclusion

# In[450]:


final_model = forest_reg

from sklearn.metrics import mean_squared_error
final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)


# In[396]:


final_rmse


# ## TODO - Make it modular an call using main
def main(argv):
    ## Make it modular 
    
if __name__ == "__main__":
   main(sys.argv[1:])
# 
# ## TODO
# 
# * DNN for improving prediction & avoiding overfitting
# * Tune Decision-Tree parameters/XGBoost (like the number of trees, depth, etc)
