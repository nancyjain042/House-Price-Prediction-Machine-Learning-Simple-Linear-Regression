# House Price Prediction

## Project Overview:
In this project, we have built a Simple Linear Regression model to study the linear relationship between the total area (in sqft) and the price of houses in Bengaluru. The goal is to predict house prices based on their total area.

## Linear Regression:
Linear Regression is a statistical technique used for finding the linear relationship between a dependent variable (in this case, house price) and one or more independent variables (in this case, total area). In Simple Linear Regression, there's only one independent variable.

## Ordinary Least Square Method:
To find the best-fitting line, we use the Ordinary Least Square (OLS) method, which minimizes the sum of the squares of the residuals (the vertical distance between data points and the regression line).

## Model Assumptions:
Linear Regression makes several assumptions:

## Linear Relationship: The relationship between the independent and dependent variables should be linear.
**Multivariate Normality**: All variables should follow a multivariate normal distribution.

**No or Little Multicollinearity**: The independent variables should not be highly correlated.

**No Autocorrelation**: The residuals (errors) should be independent of each other.

**Homoscedasticity**: The variance of the residuals should remain constant across the regression line.

## Software Information:

The project was developed in a Jupyter Notebook.
Python with the Anaconda distribution was used, which includes essential libraries.
### Python Libraries:

**NumPy**: For numerical array operations.
**Pandas**: For data manipulation and analysis.

**Scikit-Learn**: For machine learning.

**Matplotlib**: For creating plots and visualizations.

## Exploratory Data Analysis:

The dataset was imported into a Pandas DataFrame.

Data dimensions, summary, and descriptive statistics were examined.

Visual exploration of the relationship between total area and house price was performed.

## Independent and Dependent Variables:

Total area (in sqft) is the independent variable (X).

House price is the dependent variable (y).

## Data Preparation:

The data was split into training and test sets using Scikit-Learn's train_test_split function.
Reshaping of the data was done to ensure compatibility with Scikit-Learn.

## Model Building:

A Simple Linear Regression model was instantiated and trained using the training data.
Predictions were made on the test data.

## Model Slope and Intercept:

The model's slope and intercept terms were computed, which represent the relationship between total area and price.
Making Predictions:

House price predictions were made using the trained model.

## Regression Metrics for Model Performance:
Two key metrics were used to evaluate the model:

**Root Mean Square Error (RMSE)**: RMSE measures the standard deviation of the residuals, indicating how spread out the predictions are from the actual values. Lower RMSE values indicate a better fit to the data.
**R-Squared (R2) Score**: R2 score measures the goodness of fit, indicating the percentage of variance explained by the model. Values close to 1 indicate a good fit, while negative values suggest a poor model fit.

## Interpretation and Conclusion:

**RMSE value**: 74.5124, indicating a spread in predictions.
**R2 Score**: 0.5410, explaining 54.10% of the variance in house prices.

The R2 score suggests that the model is not good enough to deploy, as it doesn't provide a good fit to the data.

## Visualization:

A scatter plot and the regression line were plotted to visualize the relationship between area and price.
Checking for Overfitting and Underfitting:

Training set score: 0.5356
Test set score: 0.5410

Both scores are similar, indicating no overfitting or underfitting.

