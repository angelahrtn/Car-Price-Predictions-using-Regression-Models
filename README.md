# Car-Price-Prediction
# Project Overview
The goal of the project is to predict car prices using regression models based on various features of the cars. The analysis utilizes a dataset containing information about different cars, including features such as car age, original price, adjusted price, and more. By leveraging regression models, the project aims to provide accurate price predictions for cars based on these factors. The primary tool used for analysis is Python, along with libraries like Pandas and Scikit-learn for data preprocessing and model implementation.

The project emphasizes building robust regression models that can generalize well and provide practical insights into predicting car prices. This analysis demonstrates the use of machine learning models for regression tasks in a real-world dataset.

# Case Description
The primary problem addressed in this project is predicting the adjusted price of cars based on features such as:
  - Car Age: The age of the car in years.
  - Original Price: The initial price of the car.
  - Other Variables: Additional features present in the dataset (to be explored through feature importance).
Accurately predicting car prices is a complex task influenced by various factors. By using regression models, this project aims to estimate adjusted car prices based on these predictors and provide meaningful insights into the relationship between these variables.

# Objectives
- To build regression models that predict car prices based on car features.
- To evaluate the performance of the regression models using standard metrics such as Mean Absolute Error (MAE) and R-squared (R²).
- To identify the most significant features that influence car price predictions.
- To explore and visualize the correlations between features in the dataset.

# Project Steps and Features
1. Data Collection and Preprocessing
  - The dataset was preprocessed by handling missing values, normalizing features, and encoding any categorical variables.
  - Numerical features were scaled using techniques such as StandardScaler to ensure that each feature contributed equally to model training.

2. Exploratory Data Analysis (EDA)
  - The correlation between variables such as car age, original price, and adjusted price was examined using Pearson correlation heatmaps.
  - Visualizations like scatter plots and heatmaps helped in understanding the distribution and relationships between the key variables.

3. Regression Model Construction
- Linear Regression: The Linear Regression model was used to predict car prices. Metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) were assigned to evaluate the performance of the model both on the training and testing datasets.
- Decision Tree Regressor: Another model that was trained to predict car prices is the Decision Tree Regressor. This model was trained using the fit() method, and its performance was compared to that of the Linear Regression model based on the same metrics (MSE, MAE, RMSE, R²).

4. Model Evaluation
- The performance of both Linear Regression and Decision Tree Regressor was evaluated using the following metrics:
  - Mean Squared Error (MSE): To measure the average of the squares of errors (i.e., the difference between actual and predicted values).
  - Mean Absolute Error (MAE): To assess the average magnitude of prediction errors.
  - Root Mean Squared Error (RMSE): To measure the square root of the average squared differences between predicted and actual values.
  - R-squared (R²): To determine how well the models fit the data.

5. Results Interpretation
- The most important features impacting car price prediction were identified, such as car age and original price.
- Linear Regression performed reasonably well, while Ridge Regression provided better performance by reducing overfitting.

# Tools Used
- Python: For data processing and model development.
- Libraries:
  - Pandas: For data manipulation and cleaning.
  - Scikit-learn: For building regression models and evaluation metrics.
  - Matplotlib & Seaborn: For data visualization and EDA.
  - Statistics: For performing statistical computations.
  - Plotly Express: For interactive data visualization.

# Challenges
- Data Preprocessing: Handling missing values and scaling numerical features to ensure the model performs well.
- Model Selection: Choosing the most appropriate model for regression among multiple options such as Linear Regression, Ridge Regression, and Lasso Regression.
- Overfitting: Ensuring that the models generalize well to unseen data and do not overfit the training data, especially with more complex models like Polynomial Regression.

# Conclusion
This project successfully applied multiple regression models to predict car prices based on key car attributes. Among the models tested, Ridge Regression provided the best balance between bias and variance, making it the most reliable for price predictions. The project also demonstrated the importance of preprocessing steps like scaling and the effectiveness of regularization techniques such as Ridge and Lasso for improving model performance.

Further improvements could include more advanced feature engineering, using additional predictors, or applying more complex models like decision trees or gradient boosting. Nonetheless, this project provides a solid foundation for understanding how machine learning models can be applied to solve real-world regression problems.
