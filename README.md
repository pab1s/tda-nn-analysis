# tda-nn-separability

>This project aims to explore the separability of data using Topological Data Analysis (TDA) and Neural Networks (NN).

*Author: Pablo Olivares Martínez*

## Introduction

In many machine learning tasks, the ability to separate data points into distinct classes is crucial. However, not all datasets are easily separable using traditional methods. Topological Data Analysis (TDA) provides a powerful framework for analyzing the shape and structure of data, allowing us to gain insights into its separability.

Neural Networks (NN) are a popular class of machine learning models that can learn complex patterns and relationships in data. By combining TDA with NN, we can leverage the strengths of both approaches to improve the separability of data.

## Project Structure

The project is organized as follows:

- `config/`: This directory contains configuration files for the project.
- `data/`: This directory contains the datasets used for the experiments.
- `datasets/`: This directory contains the datasets used for the experiments and some utilities related to the datasets.
- `logs/`: This directory contains log files generated during the experiments.
- `models/`: This directory contains the trained models for the experiments.
- `docs/`: This directory contains documentation for the project.
- `notebooks/`: This directory contains Jupyter notebooks that demonstrate the application of TDA and NN to the datasets and several tasks related to the project such as Exploratory Data Analysis (EDA).
- `outputs/`: This directory contains output files generated during the experiments.
- `tests/`: This directory contains unit tests for the project.
- `trainers/`: This directory contains the trainers for the models used in the experiments.
- `README.md`: This file provides an overview of the project.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/pab1s/tda-nn-separability.git`
2. Install the required dependencies: `make init`. To activate the virtual environment, run `conda activate tda-nn-separability`.
3. Explore the datasets in the `data/` directory.
4. To run the project main file, just use `python -m main`.
5. Run the Jupyter notebooks in the `notebooks/` directory to see examples of EDA, TDA and NN applied to the datasets.
6. Run the unit tests in the `tests/` directory to ensure everything is working correctly. You can use `make test` to run the tests.
7. After modifying the code, run `make export` to update the environment file.
