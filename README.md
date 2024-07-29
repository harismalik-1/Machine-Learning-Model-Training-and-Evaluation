
# Machine Learning Model Training and Evaluation

This project demonstrates the training and evaluation of various machine learning models using Python. The script can handle both classification and regression tasks using different methods. The dataset can be in the form of extracted features or original images.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
- [Tasks](#tasks)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Arguments](#arguments)

## Requirements

- Python 3.x
- NumPy
- argparse

## Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/ml-model-training.git
cd ml-model-training
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the following command:

```bash
python main.py --task <task> --method <method> --data_path <data_path> --data_type <data_type> [additional arguments]
```

Example:

```bash
python main.py --task breed_identifying --method logistic_regression --data_path data --data_type features
```

## Methods

The following methods are available:

- `dummy_classifier`: A simple classifier for testing purposes.
- `knn`: K-Nearest Neighbors for both classification and regression tasks.
- `linear_regression`: Linear regression for regression tasks.
- `logistic_regression`: Logistic regression for classification tasks.
- `nn`: Neural Networks (to be implemented in MS2).

## Tasks

The script can handle the following tasks:

- `center_locating`: A regression task.
- `breed_identifying`: A classification task.

## Data Preparation

Data can be in two forms:

1. Extracted features dataset (`features`)
2. Original image dataset (`original`)

Depending on the data type, the script will load and preprocess the data accordingly. You can further process the data by creating a validation set, normalizing, or adding a bias term.

## Training and Evaluation

The script trains the specified model on the training data and evaluates it on the test data. The evaluation metrics are:

- For regression tasks: Mean Squared Error (MSE)
- For classification tasks: Accuracy and Macro F1-score

## Arguments

- `--task`: The task to perform (`center_locating` or `breed_identifying`).
- `--method`: The method to use (`dummy_classifier`, `knn`, `linear_regression`, `logistic_regression`, `nn`).
- `--data_path`: Path to your dataset.
- `--data_type`: Type of data (`features` or `original`).
- `--lmda`: Lambda value for linear/ridge regression.
- `--K`: Number of neighboring data points used for KNN.
- `--lr`: Learning rate for methods with learning rate.
- `--max_iters`: Maximum iterations for iterative methods.
- `--test`: Use the test set for evaluation.
- `--nn_type`: Type of neural network to use (for MS2).
- `--nn_batch_size`: Batch size for neural network training (for MS2).

Feel free to add more arguments if needed!

## Contributing

If you want to contribute to this project, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
