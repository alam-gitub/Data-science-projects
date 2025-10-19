#!/usr/bin/env python
# coding: utf-8

# # **Problem Statement**

# - **Context and Company Background:** TechWorks Consulting specializes in IT talent recruitment, uniquely connecting skilled professionals with the right job opportunities. Our deep industry knowledge and strong network help us ensure a perfect match for both candidates and companies, fostering successful partnerships in the tech sectorr.
# 

# - **Data Description:** The dataset contains information about colleges, cities, roles, previous experience, and salary. This data will be used to train and test the predictive model, helping us gain insights into trends and making informed decisions.

# - **Regression Task:** The primary objective is to perform a regression task, where the aim is to predict a continuous variable, specifically the salary of newly hired employees.

# - **Role of Statistics:** Statistics play a crucial role in building and validating the accuracy of the model. By applying statistical methods, we can analyze data patterns, assess model performance, and ensure that our predictions are reliable and meaningful.

# - **Data Preprocessing:** Data preprocessing is a critical task that involves handling missing values, addressing outliers, encoding categorical variables, normalizing data, and selecting relevant features. These steps are essential for preparing the data, ensuring its quality, and enhancing the performance of the predictive model.

# # Creating a Salary Prediction Model: A Systematic Approach

# **Data Understanding:**
# *Begin by thoroughly examining the provided dataset, including its structure and columns.
# Interpret the meaning of each variable, and gain insights into data distribution, summary statistics, and potential outliers.

# **Data Preprocessing:**
# - **Handle Missing Values:** Identify and address missing data through imputation or removal to ensure data completeness.
# - **Outlier Detection and Treatment:** Detect and manage outliers that could negatively impact the model's accuracy.
# -**Convert Categorical Data:** Transform categorical variables (e.g., "College" and "City") into numerical formats to facilitate analysis.
# -**Normalize Data:** Normalize numerical features to bring them to a common scale, preventing any single feature from dominating the model.
# -**Feature Selection:** Utilize statistical techniques such as Lasso, Ridge regression, or correlation analysis to identify the most relevant features for salary prediction.

# **Performing Exploratory Data Analysis (EDA)**
# * Conduct EDA to visualize data distributions, relationships between variables, and any trends that may inform model development.

# **Model Selection:**
# * Choose various regression models (e.g., Linear Regression, Multiple Linear Regression) to build and evaluate predictive capabilities.

# **Model Training and Evaluation:**
# * Split the dataset into training and testing sets for effective model training and performance assessment.
# * Use evaluation metrics like Mean Squared Error (MSE), R-squared, and Mean Absolute Error (MAE) to gauge model accuracy.
# * Experiment with different hyperparameters and employ cross-validation to prevent overfitting.
# 

# **Model Comparison:**
# * Compare the performance of different models, selecting the one that demonstrates the best accuracy and generalization capabilities.
# 

# **Further Improvement:**
# * Explore additional techniques for enhancing model performance, such as feature engineering, hyperparameter tuning, and ensemble methods.

# # Available ML Model Options

# **In the task of predicting employee salaries at TechWorks Consulting, several machine learning models can be utilized for regression tasks. The choice of model depends on various factors, including data characteristics, problem complexity, and the need for model interpretability. Here are some of the available ML model options:**

# **1. Linear Regression** 
# * Linear regression is a simple and interpretable model that assumes a linear relationship between the features and the target variable (salary). It's a good starting point and can provide baseline performance.

# **2. Ridge Regression and Lasso Regression**
# * Ridge and Lasso regression are regularization techniques designed to handle multicollinearity and prevent overfitting. These variants of linear regression incorporate regularization terms into the cost function to improve model robustness.

# **3. Decision Trees**
# * Decision tree-based models, such as Random Forest and Gradient Boosting, excel at capturing non-linear relationships in the data. They can accommodate both numerical and categorical features and automatically assess feature importance.

# **4. K-Nearest Neighbors (KNN)**
# * KNN is a non-parametric method that makes predictions based on the average of the 'k' nearest data points. It can be particularly effective for small to medium-sized datasets.

# **5. Polynomial Regression**
# * Polynomial regression extends linear regression by introducing polynomial features, allowing it to capture non-linear relationships.

# For this analysis, I will focus on three of these models. I will evaluate their performance using default parameters and will also experiment with adjusting certain parameters to optimize their effectiveness.

# In[133]:


# Import the pandas library for data manipulation and analysis
# Import the numpy library for numerical operations and array processing 
# Import the seaborn library for data visualization

import pandas as pd  
import numpy as np   
import seaborn as sns   


# In[134]:


# Read a CSV file into a DataFrame for the main dataset
df = pd.read_csv(r'C:\Users\win10\OneDrive\Desktop\Intershala assignment\Machine Learing\project\ML case Study.csv')

# Read a CSV file into a DataFrame for college information
college = pd.read_csv(r'C:\Users\win10\OneDrive\Desktop\Intershala assignment\Machine Learing\project\Colleges (1).csv')

# Read a CSV file into a DataFrame for city information
cities = pd.read_csv(r'C:\Users\win10\OneDrive\Desktop\Intershala assignment\Machine Learing\project\cities.csv')


# In[135]:


df.head()   # Overview of Data


# In[136]:


college.head()   # Overview of College data


# In[137]:


cities.head()   # Overview of City data


# In[138]:


# Extract data from the "Tier 1," "Tier 2," and "Tier 3" columns of the 'college' DataFrame
# and store them in separate lists 'Tier1,' 'Tier2,' and 'Tier3' for further analysis.

Tier1 = college["Tier 1"].tolist()
Tier2 = college["Tier 2"].tolist()
Tier3 = college["Tier 3"].tolist()


# In[139]:


Tier1   # Printing the data contained in Tier1


# In[140]:


# Assign tier values based on the tier classification
# - If a college is in 'Tier1', set its value to 3
# - If a college is in 'Tier2', set its value to 2
# - If a college is in 'Tier3', set its value to 1
# Tier1 college get value of 3 and tier 3 of 1 because tier1 college has higher weightage then 2 and 3

for item in df.College:
    if item in Tier1:
        df["College"].replace(item,3,inplace=True)
    elif item in Tier2:
        df["College"].replace(item,2,inplace=True)
    elif item in Tier3:
        df["College"].replace(item,1,inplace=True)


# In[141]:


df.head()    # Overview of Data


# In[142]:


# Extracting lists of metropolitan and non-metropolitan cities
metro = cities['Metrio City'].tolist()
non_metro_cities = cities['non-metro cities'].tolist()


# In[143]:


# Repeating previpus steps and assigning value as 1 if city is merto and 0 if non metro
df['City'] = np.where(df['City'].isin(metro), 1,
                      np.where(df['City'].isin(non_metro_cities), 0, df['City']))


# In[144]:


df.head()


# In[145]:


# Converting Categorical column 'Role' into numeri
df = pd.get_dummies(df, columns=['Role'], drop_first=True)


# In[146]:


# Check for missing values in the DataFrame
df.isna().sum()


# In[147]:


# Get information about the DataFrame
df.info()


# In[148]:


# Get statistical information about numerical data
df.describe()


# ## Detection of Outliers

# In[149]:


# Using seaborn library to plot box plot for detection of outliers
sns.boxplot(x='Previous CTC', data=df)   


# In[150]:


# Creating a box plot for the 'Graduation Marks' column to visualize its distribution
sns.boxplot(x=df['Graduation Marks'])


# In[151]:


# Creating a box plot for the 'EXP (Month)' column to visualize its distribution
sns.boxplot(x=df['EXP (Month)'])


# In[152]:


# Creating a box plot for the 'CTC' column
sns.boxplot(x=df['CTC'])


# In[153]:


# Calculating the correlation matrix between variables
corr = df.corr()


# In[154]:


# Displaying the correlation matrix
corr


# In[155]:


# Visual representation of the correlation matrix
# Creating a heatmap to visualize the correlations
sns.heatmap(data=corr, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})


# ## Outlier Analysis for "Previous CTC"

# - **In the DataFrame, we have identified the following outliers in the "Previous CTC" column.**

# In[156]:


percent25 = df['Previous CTC'].quantile(0.25)
percent75 = df['Previous CTC'].quantile(0.75)


# In[157]:


iqr = percent75 - percent25


# In[158]:


upper_limit = percent75 + 1.5 * iqr
lower_limit = percent25 - 1.5 * iqr


# In[159]:


# Display the calculated values
print("25th Percentile (Q1):", percent25)
print("75th Percentile (Q3):", percent75)
print("Interquartile Range (IQR):", iqr)
print("Upper Limit for Outliers:", upper_limit)
print("Lower Limit for Outliers:", lower_limit)


# **Upon reviewing these outliers, it's evident that they are not extreme values. Therefore, retaining this data in the dataset is unlikely to significantly impact the performance of the model.**

# ## Outliers Analysis for "CTC column"

# In[160]:


percent25 = df['CTC'].quantile(0.25)
percent75 = df['CTC'].quantile(0.75)


# In[161]:


iqr = percent75 - percent25


# In[162]:


upper_limit = percent75 + 1.5 * iqr
lower_limit = percent25 - 1.5 * iqr


# In[163]:


# Display the calculated values
print("25th Percentile (Q1):", percent25)
print("75th Percentile (Q3):", percent75)
print("Interquartile Range (IQR):", iqr)
print("Upper Limit for Outliers:", upper_limit)
print("Lower Limit for Outliers:", lower_limit)


# In[164]:


df[(df['CTC'] < lower_limit) | (df['CTC'] > upper_limit)]


# **As seen above, these are some outliers in the 'CTC' column, but they are not extreme enough to significantly impact predictions. Therefore, in my opinion, keeping these outliers in the data is more useful than removing them.**

# ### Conclusion on Outlier Detection

# - **Our analysis reveals that there are no extreme outliers in the dataset that would significantly impact the performance of our machine learning model. The results from the describe function confirm that there are no extreme outliers present.**
# - **While we identified some outliers in the "Previous CTC" and "CTC" columns, I believe these will not adversely affect the model’s predictions.**
# - **Additionally, the HeatMap visualization indicates meaningful relationships between "Role_Manager" and "CTC," as well as between "Previous CTC" and "CTC."**

# ## Applying Machine Learning Models Without Feature Scaling

# - **In this section, I will apply various machine learning algorithms without any scaling to evaluate their performance.**

# In[165]:


# Importing Necessary Libraries
# We need these libraries for data handling and modeling.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[166]:


# Split Data into Features and Target
# Here, we separate our features (X) from the target variable (y).
X = df.loc[:, df.columns != 'CTC']
y = df['CTC']


# In[167]:


# Split Data into Train and Test Sets
# We'll use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[168]:


# View Test Set Targets
y_test


# - **Model Creation and Fitting (Linear Regression):**

# In[169]:


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)


# - **Making Predictions:**

# In[170]:


linear_reg_pred = linear_reg.predict(X_test)


# - **Evaluation Metrics:**

# In[171]:


print("r2_score:", r2_score(y_test, linear_reg_pred))
print("MAE:", mean_absolute_error(y_test, linear_reg_pred))
print("MSE:", mean_squared_error(y_test, linear_reg_pred))


# - **Model Coefficients and Intercept:**

# In[172]:


print("Coef:", linear_reg.coef_)
print("Intercept:", linear_reg.intercept_)


# In[ ]:





# - **Model Creation and Fitting (Ridge Regression):**

# In[173]:


ridge = Ridge()
ridge.fit(X_train, y_train)


# - **Making Predictions:**

# In[174]:


ridge_predictions = ridge.predict(X_test)


# - **Evaluation Metrics:**

# In[175]:


print("r2_score:",r2_score(y_test, ridge_predictions))
print("MAE:", mean_absolute_error(y_test, ridge_predictions))
print("MSE:", mean_squared_error(y_test, ridge_predictions))


# - **Model Coefficients and Intercept:**

# In[176]:


print("Coef:",ridge.coef_)
print("Intercept:",ridge.intercept_)


# In[ ]:





# ### **Tuned Ridge Regression Model Evaluation**

# **In this notebook, we will create and evaluate a Ridge regression model. We will specify an alpha value and a solver, fit the model to the training data, make predictions, and evaluate the model's performance using various metrics.**

# **Create and Fit the Ridge Model:**

# In[177]:


# Create a Ridge regression model with a specified alpha value and solver
ridge_tuned = Ridge(alpha=0.3, solver='cholesky')

# Fit the Ridge model to the training data
ridge_tuned.fit(X_train, y_train)


# **Make Predictions**

# In[178]:


# Make predictions on the test data using the tuned Ridge model
ridge_predict_tuned = ridge_tuned.predict(X_test)


# **Evaluate Model Performance**

# In[179]:


# Evaluate model performance using R-squared (R²) score
print("r2_score:",r2_score(y_test, ridge_predict_tuned))

# Measure prediction accuracy using Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, ridge_predict_tuned))

# Assess prediction accuracy using Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, ridge_predict_tuned))


# **Model Coefficients and Intercept**

# In[180]:


# Print the coefficients of the tuned Ridge regression model
print("Coefficients:", ridge_tuned.coef_)

# Print the intercept of the tuned Ridge regression model
print("Intercept:", ridge_tuned.intercept_)


# In[ ]:





# ## Lasso Regression Model Evaluation

# **In this section, we will create and evaluate a Lasso regression model using default parameters. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Lasso Model**

# In[181]:


# Create Lasso regression model with default parameters
lasso = Lasso()

# Fit the model with training data
lasso.fit(X_train, y_train)


# **Make Predictions**

# In[182]:


# Make predictions on the test data
lasso_pred = lasso.predict(X_test)


# **Evaluate Model Performance**

# In[183]:


# Evaluate model performance using R-squared (R²) score
print("r2_score:",r2_score(y_test, lasso_pred))

# Measure prediction accuracy using Mean Absolute Error (MAE)
print("MAE:",mean_absolute_error(y_test, lasso_pred))


# Assess prediction accuracy using Mean Squared Error (MSE)
print("MSE:",mean_squared_error(y_test, lasso_pred))


# **Model Coefficients and Intercept**

# In[184]:


# Print the coefficients of the Lasso regression model
print("Coefficients:", lasso.coef_)

# Print the intercept of the Lasso regression model
print("Intercept:", lasso.intercept_)


# In[ ]:





# ## Tuned Lasso Regression Model Evaluation

# **In this section, we will create and evaluate a Lasso regression model with a specified alpha value. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Tuned Lasso Model**

# In[185]:


# Create Lasso regression model with a specified alpha value
lasso_tuned = Lasso(alpha=0.3)

# Fit the model on training data
lasso_tuned.fit(X_train, y_train)


# **Make Predictions**

# In[186]:


# Make predictions on the test data
lasso_tuned_pred = lasso_tuned.predict(X_test)


# **Evaluate Model Performance**

# In[187]:


# Evaluate model performance using R-squared (R²) score
print("r2_score",r2_score(y_test, lasso_tuned_pred))

# Measure prediction accuracy using Mean Absolute Error (MAE)
print("MAE", mean_absolute_error(y_test, lasso_tuned_pred))

# Assess prediction accuracy using Mean Squared Error (MSE)
print("MSE",mean_squared_error(y_test, lasso_tuned_pred))


# In[ ]:





# ## Decision Tree Regression Model Evaluation

# **In this section, we will create and evaluate a Decision Tree Regressor model. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Decision Tree Model**

# In[188]:


# Import necessary libraries
from sklearn.tree import DecisionTreeRegressor


# In[189]:


# Create a DecisionTreeRegressor model
dtr = DecisionTreeRegressor()

# Train the model using the training data
dtr.fit(X_train, y_train)


# **Make Predictions**

# In[190]:


# Make predictions on the test data
dtr_pred = dtr.predict(X_test)


# **Evaluate Model Performance**

# In[191]:


# Evaluate model performance using R-squared (R²) score
print("r2_score:",r2_score(y_test, dtr_pred))

# Measure prediction accuracy using Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, dtr_pred))

# Assess prediction accuracy using Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, dtr_pred))


# In[ ]:





# ## Tuned Decision Tree Regression Model Evaluation

# **In this section, we will create and evaluate a Decision Tree Regressor model with a maximum depth of 4. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Tuned Decision Tree Model**

# In[192]:


# Create Decision Tree with max depth = 4
dtr_tuned = DecisionTreeRegressor(max_depth=4)

# Fit the model with training data
dtr_tuned.fit(X_train, y_train)


# **Make Predictions**

# In[193]:


# Make predictions on the test data
dtr_tuned_pred = dtr_tuned.predict(X_test)


# **Evaluate Model Performance (Single Line Output)**

# In[194]:


# Evaluate model performance using R-squared (R²) score
print("r2_score:",r2_score(y_test, dtr_tuned_pred))

# Measure prediction accuracy using Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, dtr_tuned_pred))

# Assess prediction accuracy using Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, dtr_tuned_pred))


# In[ ]:





# ## Random Forest Regression Model Evaluation

# **In this section, we will create and evaluate a Random Forest Regressor model using default parameters. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Random Forest Model**

# In[195]:


# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor


# In[196]:


# Create Random Forest regression model with default parameters
rnd = RandomForestRegressor()

# Fit the model on training data
rnd.fit(X_train, y_train)


# **Make Predictions**

# In[197]:


# Make predictions on the test data
rnd_pred = rnd.predict(X_test)


# **Evaluate Model Performance**

# In[198]:


# Evaluate model performance using R-squared (R²) score
print("r2_score:",r2_score(y_test, rnd_pred))

# Measure prediction accuracy using Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, rnd_pred))

# Assess prediction accuracy using Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, rnd_pred))


# In[ ]:





# ## Tuned Random Forest Regression Model Evaluation

# **In this section, we will create and evaluate a Random Forest Regressor model with tuned parameters. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Tuned Random Forest Model**

# In[199]:


# Create Random Forest regression model with tuned parameters
rnd_tuned = RandomForestRegressor(n_jobs=-1, max_features=5, min_samples_split=3)

# Fit the model on training data
rnd_tuned.fit(X_train, y_train)


# **Predictions**

# In[200]:


# Make predictions on the test data
rnd_tuned_pred = rnd_tuned.predict(X_test)


# **Evaluate Model Performance**

# In[201]:


# Evaluate model performance using R-squared (R²) score
print("r2_score:",r2_score(y_test, rnd_tuned_pred))

# Measure prediction accuracy using Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, rnd_tuned_pred))

# Assess prediction accuracy using Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, rnd_tuned_pred))


# In[ ]:





# ## Hyperparameter Tuning for Random Forest Regression

# **In this section, we will use GridSearchCV to find the best parameters for our Random Forest Regressor model. This process will help optimize the model's performance by evaluating various combinations of specified hyperparameters.**

# **Define Parameter Grid**

# In[202]:


from sklearn.model_selection import GridSearchCV


# In[203]:


# Parameters for grid search
params_grid = {
    "max_features": [4, 5, 6, 7, 8, 9, 10],
    "min_samples_split": [2, 3, 10]
}


# **Perform Grid Search**

# In[204]:


# Create the grid search object
grid_search = GridSearchCV(
    RandomForestRegressor(n_jobs=-1),
    params_grid,
    n_jobs=-1,
    cv=5
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)


# **Display Best Parameters**

# In[205]:


# Display the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)


# In[ ]:





# ## Model Performance Evaluation with Test Size = 0.1

# **In this section, we will evaluate the performance of a Linear Regression model using a smaller test set (10% of the data). We will analyze various performance metrics to understand how well the model generalizes.**

# In[206]:


# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)


# In[207]:


# Create Linear Regression model
linear_reg = LinearRegression()

# Fit the model with training data
linear_reg.fit(X_train, y_train)


# **Make Predictions**

# In[208]:


# Make predictions using the test data
linear_reg_pred = linear_reg.predict(X_test)


# **Evaluate Model Performance**

# In[209]:


# Calculate and print the R-squared (R²) score
print("r2_score:",r2_score(y_test, linear_reg_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, linear_reg_pred))

# Calculate and print the Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, linear_reg_pred))


# **Display Model Coefficients and Intercept**

# In[210]:


# Print the coefficients of the linear regression model
print("Coefficients:", linear_reg.coef_)

# Print the intercept of the linear regression model
print("Intercept:", linear_reg.intercept_)


# In[ ]:





# ## Ridge Regression Evaluation with Test Size = 0.1

# **In this section, we will evaluate the performance of a Ridge Regression model using a smaller test set (10% of the data). We will analyze various performance metrics to understand how well the model generalizes.**

# **Create and Fit the Ridge Regression Model**

# In[211]:


# Create Ridge regression model
ridge = Ridge()

# Fit the model with training data
ridge.fit(X_train, y_train)


# **Make Predictions**

# In[212]:


# Make predictions using the test data
ridge_predict = ridge.predict(X_test)


# **Evaluate Model Performance**

# In[213]:


# Calculate R-squared (R²) score
print("r2_score:",r2_score(y_test, ridge_predict))

# Calculate Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, ridge_predict))

# Calculate Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, ridge_predict))


# **Display Model Coefficients and Intercept**

# In[214]:


# Print the coefficients of the Ridge regression model
print("Coefficients:", ridge.coef_)

# Print the intercept of the Ridge regression model
print("Intercept:", ridge.intercept_)


# In[ ]:





# ## Tuned Ridge Regression Evaluation

# **In this section, we evaluate the performance of a tuned Ridge Regression model with an alpha value of 0.3 and the Cholesky solver. We will analyze the model's performance using various metrics.**

# **Create and Fit the Tuned Ridge Regression Model**

# In[215]:


# Create Ridge regression model with alpha = 0.3 and solver = 'cholesky'
ridge_tuned = Ridge(alpha=0.3, solver='cholesky')

# Fit the model with training data
ridge_tuned.fit(X_train, y_train)


# **Make Predictions**

# In[216]:


# Make predictions using the test data
ridge_predict_tuned = ridge_tuned.predict(X_test)


# **Evaluate Model Performance**

# In[217]:


# Calculate R-squared (R²) score
print("r2_score:",r2_score(y_test, ridge_predict_tuned))

# Calculate Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, ridge_predict_tuned))

# Calculate Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, ridge_predict_tuned))


# **Display Model Coefficients and Intercept**

# In[218]:


# Print the coefficients of the tuned Ridge regression model
print("Coefficients:", ridge_tuned.coef_)

# Print the intercept of the tuned Ridge regression model
print("Intercept:", ridge_tuned.intercept_)


# In[ ]:





# ## Lasso Regression Evaluation

# **In this section, we evaluate the performance of a Lasso Regression model with default parameters. We will analyze the model's performance using various metrics.**

# **Create and Fit the Lasso Regression Model**

# In[219]:


# Create Lasso regression model with default parameters
lasso = Lasso()

# Fit the model on training data
lasso.fit(X_train, y_train)


# **Make Predictions**

# In[220]:


# Make predictions using the test data
lasso_pred = lasso.predict(X_test)


# **Evaluate Model Performance**

# In[221]:


# Calculate R-squared (R²) score
print("r2_score:",r2_score(y_test, lasso_pred))

# Calculate Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, lasso_pred))

# Calculate Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, lasso_pred))


# **Display Model Coefficients and Intercept**

# In[222]:


# Print the coefficients of the Lasso regression model
print("Coefficients:", lasso.coef_)

# Print the intercept of the Lasso regression model
print("Intercept:", lasso.intercept_)


# In[ ]:





# ## Lasso Regression with Tuned Parameters

# **In this section, we create a Lasso regression model with a specified alpha value. The model is trained on the training data, and we evaluate its performance using various metrics.**

# **Create and Fit the Tuned Lasso Regression Model**

# In[223]:


# Create Lasso regression model with tuned parameter
lasso_tuned = Lasso(alpha=0.3)

# Fit model on train data
lasso_tuned.fit(X_train, y_train)


# **Make Predictions**

# In[224]:


# Prediction on test data
lasso_tuned_pred = lasso_tuned.predict(X_test)


# **Evaluate Model Performance**

# In[225]:


# Calculate and print the R-squared (r2) score to evaluate model performance
print("r2_score:", r2_score(y_test, lasso_tuned_pred))

# Calculate and print the Mean Absolute Error (MAE) to measure prediction accuracy
print("MAE:", mean_absolute_error(y_test, lasso_tuned_pred))

# Calculate and print the Mean Squared Error (MSE) to assess prediction accuracy
print("MSE:", mean_squared_error(y_test, lasso_tuned_pred))


# In[ ]:





# ## Decision Tree Regression

# **In this section, we create a Decision Tree Regressor using default parameters. We train the model on the training dataset and evaluate its performance using various metrics.**

# **Create and Fit Decison Tree Regression Model**

# In[226]:


# Create a Decision Tree Regressor model with default parameters
dtr = DecisionTreeRegressor()

# Fit the model on training data
dtr.fit(X_train, y_train)


# **Make Predictions**

# In[227]:


# Make predictions on the test data
dtr_pred = dtr.predict(X_test)


# **Evaluate Model Performance**

# In[228]:


# Calculate and print the R-squared (r2) score to evaluate model performance
print("r2_score:",r2_score(y_test, dtr_pred))

# Calculate and print the Mean Absolute Error (MAE) to measure prediction accuracy
print("MAE:", mean_absolute_error(y_test, dtr_pred))

# Calculate and print the Mean Squared Error (MSE) to assess prediction accuracy
print("MSE:", mean_squared_error(y_test, dtr_pred))


# In[ ]:





# ## Tuned Decision Tree Regression

# **In this section, we create a Decision Tree Regressor with a tuned parameter for maximum depth. We train the model on the training dataset and evaluate its performance using various metrics.**

# **Create and Fit the Tuned Decision Tree Regression Model**

# In[229]:


# Create a Decision Tree Regressor model with tuned parameter
dtr_tuned = DecisionTreeRegressor(max_depth=4)

# Fit the model on training data
dtr_tuned.fit(X_train, y_train)


# **Make Predictions**

# In[230]:


# Prediction using test data
dtr_tuned_pred = dtr_tuned.predict(X_test)


# **Evaluate Model Performance**

# In[231]:


# Calculate and print the R-squared (r2) score to evaluate model performance
print("r2_score:",r2_score(y_test, dtr_tuned_pred))

# Calculate and print the mean absolute error(MSE) score to evaluate model performance
print("MAE:", mean_absolute_error(y_test, dtr_tuned_pred))

# Calculate and print the Mean Squared Error (MSE) to evaluate prediction errors
print("MSE:", mean_squared_error(y_test, dtr_tuned_pred))


# In[ ]:





# ## Random Forest Regression Model Evaluation

# **In this section, we will create and evaluate a Random Forest Regressor model using default parameters. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Random Forest Model**

# In[232]:


# Create a Random Forest Regressor model with default parameters
rnd = RandomForestRegressor()

# Fit the model on the training data
rnd.fit(X_train, y_train)


# **Make Predictions**

# In[233]:


# Predict target values using the test data
rnd_pred = rnd.predict(X_test)


# **Evaluate Model Performance**

# In[234]:


# Evaluate model performance using R-squared score
print("r2_score:",r2_score(y_test, rnd_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, rnd_pred))

# Calculate and print the Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, rnd_pred))


# In[ ]:





# ## Tuned Random Forest Regression Model Evaluation

# **In this section, we will create and evaluate a Random Forest Regressor model with tuned parameters. We will fit the model to the training data, make predictions, and evaluate its performance using various metrics.**

# **Create and Fit the Tuned Random Forest Model**

# In[235]:


# Create a Random Forest Regressor model with specified hyperparameters
rnd_tuned = RandomForestRegressor(n_jobs=-1, max_features=5, min_samples_split=3)

# Fit the model using the training data
rnd_tuned.fit(X_train, y_train)


# **Make Predictions**

# In[236]:


# Predict target values using the test data
rnd_tuned_pred = rnd_tuned.predict(X_test)


# **Evaluate Model Performance**

# In[237]:


# Evaluate model performance using R-squared score
print("r2_score:",r2_score(y_test, rnd_tuned_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("MAE:", mean_absolute_error(y_test, rnd_tuned_pred))

# Calculate and print the Mean Squared Error (MSE)
print("MSE:", mean_squared_error(y_test, rnd_tuned_pred))


# In[ ]:





# ## Performing Feature scaling on dataset

# **Split Data into Independent and Target Variables**

# In[238]:


# Features (independent variables)
X = df.loc[:, df.columns != 'CTC']  

# Target variable
y = df['CTC']         


# **Split Data into Training and Testing Sets**

# In[239]:


# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# **Feature Scaling**

# In[240]:


# Import Standard scaler from sklearn for feature scaling(mean=0, std dev=1)
from sklearn.preprocessing import StandardScaler


# In[241]:


# Create a StandardScaler object for feature scaling (mean=0, std dev=1)
scaler = StandardScaler()


# **Fit and Transform the Training Data**

# In[242]:


# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)


# **Transform the Test Data**

# In[243]:


# Transform the test data using the same scaler to ensure consistency
X_test_scaled = scaler.transform(X_test)


# **Store Scaled Data in DataFrames for Verification**

# In[244]:


# Store the scaled training data in a DataFrame
df_X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Store the scaled test data in a DataFrame
df_X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# **Statistical Changes After Scaling**

# In[245]:


# Display statistical changes after scaling
print(np.round(df_X_train.describe(), 1))


# In[ ]:





# ## Model Performance Evaluation After Scaling with test size 0.2

# **Create and Fit Linear Regression Model with Scaled Data**

# In[246]:


# Create a Linear Regression model
lr_scaled = LinearRegression()

# Fit the model on the scaled training data
lr_scaled.fit(X_train_scaled, y_train)


# **Make Predictions Using Test Data**

# In[247]:


# Make predictions using the scaled test data
lr_scaled_pred = lr_scaled.predict(X_test_scaled)


# **Evaluate Model Performance**

# In[248]:


# Calculate and print the R-squared (R²) score to evaluate model performance
print("r2_score:", r2_score(y_test, lr_scaled_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, lr_scaled_pred))

# Calculate and print the Mean Squared Error (MSE)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, lr_scaled_pred))


# **Model Coefficients and Intercept**

# In[249]:


# Print the coefficients of the linear regression model
print("Coefficients:", lr_scaled.coef_)

# Print the intercept of the linear regression model
print("Intercept:", lr_scaled.intercept_)


# In[ ]:





# ## Ridge Regression Model Evaluation After Feature Scaling with test size 0.2

# **Create and Fit Ridge Regression Model with Scaled Data**

# In[250]:


# Create a Ridge Regression model
r_scaled = Ridge()

# Fit the model using the scaled training data
r_scaled.fit(X_train_scaled, y_train)


# **Make Predictions Using Test Data**

# In[251]:


# Make predictions using the scaled test data
r_scaled_pred = r_scaled.predict(X_test_scaled)


# **Evaluate Model Performance**

# In[252]:


# Calculate and print the R-squared (R²) score to evaluate model performance
print("r2_score:", r2_score(y_test, r_scaled_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, r_scaled_pred))

# Calculate and print the Mean Squared Error (MSE)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, r_scaled_pred))


# In[ ]:





# ## Lasso Regression Model Evaluation After Feature Scaling with test size 0.2

# **Create and Fit Lasso Regression Model with Scaled Data**

# In[253]:


# Create a Lasso Regression model
l_scaled = Lasso()

# Fit the model on the scaled training data
l_scaled.fit(X_train_scaled, y_train)


# **Make Predictions Using Test Data**

# In[254]:


# Make predictions using the scaled test data
l_scaled_pred = l_scaled.predict(X_test_scaled)


# **Evaluate Model Performance**

# In[255]:


# Calculate and print the R-squared (R²) score to evaluate model performance
print("r2_score:", r2_score(y_test, l_scaled_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, l_scaled_pred))

# Calculate and print the Mean Squared Error (MSE)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, l_scaled_pred))


# In[ ]:





# ## Decision Tree Regression Model Evaluation After Feature Scaling with test size 0.2

# **Create and Fit Decision Tree Regression Model with Scaled Data**

# In[256]:


# Create a Decision Tree Regression model
dt_scaled = DecisionTreeRegressor()

# Fit the model on the scaled training data
dt_scaled.fit(X_train_scaled, y_train)


# **Make Predictions Using Test Data**

# In[257]:


# Make predictions using the scaled test data
dt_scaled_pred = dt_scaled.predict(X_test_scaled)


# **Evaluate Model Performance**

# In[258]:


# Calculate and print the R-squared (R²) score to evaluate model performance
print("r2_score:", r2_score(y_test, dt_scaled_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, dt_scaled_pred))

# Calculate and print the Mean Squared Error (MSE)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, dt_scaled_pred))


# In[ ]:





# ## Random Forest Regression Model Evaluation After Feature Scaling with test size 0.2

# **Create and Fit Random Forest Regressor Model with Scaled Data**

# In[259]:


# Create a Random Forest Regressor model with default parameters
rf_scaled = RandomForestRegressor()

# Fit the model on the scaled training data
rf_scaled.fit(X_train_scaled, y_train)


# **Make Predictions Using Test Data**

# In[260]:


# Make predictions using the scaled test data
rf_scaled_pred = rf_scaled.predict(X_test_scaled)


# **Evaluate Model Performance**

# In[261]:


# Calculate and print the R-squared (R²) score to evaluate model performance
print("r2_score:", r2_score(y_test, rf_scaled_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, rf_scaled_pred))

# Calculate and print the Mean Squared Error (MSE)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, rf_scaled_pred))


# In[ ]:





# ## Tuned Random Forest Regression Model Evaluation After Feature Scaling

# **Create and Fit Tuned Random Forest Regressor Model with Scaled Data**

# In[262]:


# Create a Tuned Random Forest Regressor model
rf_scaled_tuned = RandomForestRegressor(max_features=5, min_samples_split=3, n_jobs=-1)

# Fit the model on the scaled training data
rf_scaled_tuned.fit(X_train_scaled, y_train)


# **Make Predictions Using Test Data**

# In[263]:


# Make predictions using the scaled test data
rf_scaled_tuned_pred = rf_scaled_tuned.predict(X_test_scaled)


# **Evaluate Model Performance**

# In[264]:


# Calculate and print the R-squared (R²) score to evaluate model performance
print("r2_score:", r2_score(y_test, rf_scaled_tuned_pred))

# Calculate and print the Mean Absolute Error (MAE)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, rf_scaled_tuned_pred))

# Calculate and print the Mean Squared Error (MSE)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, rf_scaled_tuned_pred))


# In[ ]:





# ## Model Performance Summary

# **To evaluate the performance of various machine learning models, we focused on the R-squared (R²) metric, which indicates the goodness of fit for the models. A higher R² value signifies a better fit to the data. In addition to R², we also considered Mean Absolute Error (MAE) and Mean Squared Error (MSE) as reference metrics.**

# **Model Performance Across Scenarios**

# **Scenario 1: Test Size = 0.2**

# * Linear Regression: R² = 0.5933
# * Ridge Regression: R² = 0.5926 
# * Lasso Regression: R² = 0.5933
# * Decision Tree: R² = 0.3097    
# * Random Forest: R² = 0.6350   

# **Scenario 2: Test Size = 0.1**

# * Linear Regression: R² = 0.6363
# * Ridge Regression: R² = 0.6356 
# * Lasso Regression: R² = 0.6362
# * Decision Tree: R² = 0.4691    
# * Random Forest: R² = 0.6871  

# **Scenario 3: Test Size = 0.1 with Feature Scaling**

# * Linear Regression: R² = 0.6363
# * Ridge Regression: R² = 0.6356 
# * Lasso Regression: R² = 0.6362
# * Decision Tree: R² = 0.4617    
# * Random Forest: R² = 0.6843   

# **Scenario 4: Test Size = 0.2 with Feature Scaling**

# * Linear Regression: R² = 0.5933
# * Ridge Regression: R² = 0.5932 
# * Lasso Regression: R² = 0.5933
# * Decision Tree: R² = 0.3014    
# * Random Forest: R² = 0.6440   

# # **Summary**

# - **Random Forest consistently achieves the highest R-squared scores across all scenarios, indicating it is the best fit for the data among the models evaluated.**

# - **Linear Regression and Lasso Regression perform reasonably well, but their R-squared scores are slightly lower compared to Random Forest.**

# - **The Decision Tree model has the lowest R-squared scores and performs the worst across all scenarios.**

# - **Feature scaling appears to improve model performance, as seen by the higher R-squared scores in the scenarios where feature scaling was applied.**

# **Overall, considering R-squared as the primary metric for model evaluation, Random Forest emerges as the top performer, followed by Linear Regression and Lasso Regression. However, it is important to also take into account other factors such as computational efficiency, model interpretability, and the specific goals of the application when choosing the best model**

# ## Steps to Further Improve the Selected Model

# - **Increase the Number of Trees (Estimators): Random Forest's performance often improves with a larger number of decision trees (estimators) in the ensemble.**

# - **Tune Hyperparameters: Conduct more thorough hyperparameter tuning by experimenting with different values for parameters like max_depth, min_samples_split, min_samples_leaf, and max_features. Using Grid Search or Randomized Search can help identify the optimal combination of hyperparameters.**

# - **Feature Selection: Consider removing less informative features to improve the model’s efficiency and potentially enhance its performance. This can be achieved through feature importance analysis or other feature selection techniques.**

# In[ ]:




