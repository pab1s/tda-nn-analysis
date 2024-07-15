# tda-nn-analysis

## Overview

This repository is dedicated to exploring the potential of Topological Data Analysis (TDA) to uncover unique properties of data processed by Convolutional Neural Networks (CNNs). By integrating TDA with advanced CNN models, this project aims to provide deeper insights into the evolving characteristics of data within these networks.

## Author

Pablo Olivares Martínez

## Project Introduction

### Purpose

While CNNs excel in tasks like image recognition, their complex inner workings often remain a black box, making it difficult to grasp the finer nuances of the data they process. This project uses TDA, specifically persistent homology, to probe these subtleties, revealing how data properties evolve through the layers of CNNs.

### Methodology

Our approach utilizes persistent homology to trace the topological changes in data as it progresses through different stages of learning within CNNs. This technique offers a fresh perspective, uncovering patterns that are not immediately apparent through traditional analysis methods.

### Key Findings

Our studies have involved applying TDA to several leading neural network architectures, including ResNet, EfficientNet, and DenseNet. We've observed significant shifts in topological complexity during training—initial simplifications help reduce noise, while subsequent increases foster the development of intricate data representations important for distinguishing between classes.

### Impact

By integrating topological regularizers into models like EfficientNet-B0 and DenseNet-121, we have achieved notable enhancements in model performance. These experiments demonstrate how TDA can not only reveal but also leverage distinctive data properties derivated from the network's transformations to optimize CNN operations effectively.

## Project Structure

The project is organized as follows:

- `config/`: This directory contains configuration files for the project.
- `data/`: This directory contains the datasets used for the experiments.
- `datasets/`: This directory contains the datasets used for the experiments and some utilities related to the datasets.
- `logs/`: This directory contains log files generated during the experiments.
- `callbacks/`: This directory contains the mecessary callbacks used during training.
- `factories/`: This directory contains the factory classes used to create different objects.
- `docs/`: This directory contains documentation for the project.
- `notebooks/`: This directory contains Jupyter notebooks that perform tasks related to the project such as Exploratory Data Analysis (EDA).
- `outputs/`: This directory contains output files generated during the experiments.
- `tests/`: This directory contains unit tests for the project.
- `trainers/`: This directory contains the trainers for the models used in the experiments.
- `README.md`: This file provides an overview of the project.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/pab1s/tda-nn-separability.git`
2. Install the required dependencies: `make init`. To activate the virtual environment, run `conda activate tda-nn-analysis`.
3. Explore the datasets in the `data/` directory.
4. To run the project main file, just use `python -m main`.
5. Run the Jupyter notebooks in the `notebooks/` directory to see examples of EDA, TDA and NN applied to the datasets.
6. Run the unit tests in the `tests/` directory to ensure everything is working correctly. You can use `make test` to run the tests.
7. After modifying the code, run `make export` to update the environment file.

### Keywords

- Convolutional Neural Networks (CNNs)
- Topological Data Analysis (TDA)
- Persistent Homology
- Data Properties
- Deep Learning Optimization
- Transfer Learning
