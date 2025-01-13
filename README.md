# Classical Machine Learning Pipeline (KNN, Logistic/Linear Regression)

This project demonstrates a traditional machine learning approach for both **classification** and **regression** tasks. Specifically, it explores:
- **Breed Identification (Classification)**: Uses Logistic Regression or KNN to predict categories.
- **Center Locating (Regression)**: Uses Linear Regression or KNN to predict numeric coordinates.

It includes **hyperparameter tuning** (e.g., varying \( k \), regularization lambda, or learning rate) and **performance metrics** (accuracy, macro-F1, MSE). By adjusting these parameters, the pipeline helps highlight how different algorithms and configurations can affect results.

---

## Table of Contents
- [Key Features](#key-features)
- [File Overview](#file-overview)
- [How to Run](#how-to-run)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

---

## Key Features

1. **Flexible Task Selection**  
   - **`center_locating`** for regression tasks (e.g., coordinate predictions).  
   - **`breed_identifying`** for classification tasks (e.g., identifying dog breeds).

2. **Multiple Methods**  
   - **DummyClassifier** (baseline)  
   - **KNN** (supports both classification & regression modes)  
   - **Logistic Regression** (classification only)  
   - **Linear Regression** (regression only)

3. **Hyperparameter Tuning & Visualization**  
   - Several scripts demonstrate how to iterate over different hyperparameters (like \( k \) in KNN or \(\lambda\) in Linear Regression) and measure performance (accuracy, macro-F1 for classification or MSE for regression).  
   - 3D visualization of results (e.g., learning rate vs. iterations vs. F1-score).

4. **Data Preprocessing**  
   - Loading and splitting data into training, test (and optional validation).  
   - Appending a bias term, normalizing features using training set mean & std.  
   - Handling different data types (e.g., extracted features vs. original images).

---

## File Overview

1. **`main.py`**  
   - The entry point for training and testing various ML methods.  
   - Parses command-line arguments, loads data, performs normalization, appends bias terms, initializes the chosen method, and runs evaluation.

2. **`testKNNClassification()`**  
   - Iterates over different values of \( k \) for classification tasks (breed_identifying).  
   - Tracks accuracy & macro-F1 to identify the best \( k \).

3. **`testKNNRegression()`**  
   - Iterates over different values of \( k \) for regression tasks (center_locating).  
   - Tracks MSE to identify the best \( k \).

4. **`testLogisticRegression()`**  
   - Explores different learning rates and max iterations for logistic regression.  
   - Uses 3D plotting to visualize macro-F1 across the hyperparameter search space.

5. **`testLinearRegression()`**  
   - Iterates over different \(\lambda\) values for ridge regression.  
   - Monitors train/test MSE to find the optimal regularization strength.

---

## How to Run

### Command-Line Arguments

Below are some key flags and their defaults. Run `python main.py --help` for the complete list.

- **`--task`**: `center_locating` (regression) or `breed_identifying` (classification).  
- **`--method`**: Choose among `dummy_classifier`, `knn`, `linear_regression`, `logistic_regression` (or `nn` for a placeholder).  
- **`--data_path`**: Path to your dataset (default: `"data"`).  
- **`--data_type`**: Either `"features"` or `"original"`. If you have a custom feature dataset, place it as `features.npz`.  
- **`--lmda`**: Lambda regularization term for linear regression (default: `10`).  
- **`--K`**: Number of neighbors (KNN).  
- **`--lr`**: Learning rate for methods that use gradient descent (e.g., logistic regression).  
- **`--max_iters`**: Maximum training iterations (for iterative methods).  
- **`--test`**: If included, the script trains on the full dataset and evaluates on the test set (no validation split).

### Examples

1. **Run KNN for Classification**  
   ```bash
   python main.py --task breed_identifying --method knn --K 5 --data_type features
