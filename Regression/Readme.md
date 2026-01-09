## 📂 Project Structure & Class Overview

| Class | Purpose / Description |
|-------|----------------------|
| **Dataset** | Encapsulates a dataset: feature matrix `X`, target vector `Y`, and the true weights used to generate `Y`. |
| **RegressionDataGenerator** | Generates random datasets for regression, with optional Gaussian noise. Includes methods for training and test data generation. |
| **LeastSquares** | Implements the least squares estimator: computes the optimal weights `w = (XᵀX)⁻¹ XᵀY` for linear regression. |
| **Matrix** | Provides utility methods for matrix operations: transpose, multiplication, and inversion (Gauss-Jordan method). |
| **PolynomialFeatures** | Expands features to polynomial terms up to a given degree for polynomial regression. |
| **CrossValidation** | Performs k-fold cross-validation to select the best polynomial degree and evaluate model performance. |
| **CVResult** | Data type that stores the results of cross-validation, including the best polynomial degree and corresponding MSE. |
| **Main** | Interactive program: asks user for dataset parameters, regression type, fits the model, predicts on new data, and computes metrics like MSE and R². |
| **RegressionPipeline** | A clean pipeline for solving regression problems using Cross-Validation with Polynomial Regression. |
| **HousePriceEstimator** | A class where the model is trained in order to predict house prices from a real world problem. Also Feature Engineering is applied. |



---

## Features

- **From scratch** implementation with no external ML libraries  
- Supports **linear and polynomial regression**  
- Generates synthetic datasets for testing  
- Computes **Mean Squared Error (MSE)** and **R²** for model evaluation  
- **RegressionPipeline.java** can be called for any regression problem. You just need to load and feature-engineer the data set properly.
- **TO DO:** Analysis of the results of HousePriceEstimator

---
