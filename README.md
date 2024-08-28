# TDA-NN Analysis

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository explores the potential of Topological Data Analysis (TDA) to uncover unique properties of data processed by Convolutional Neural Networks (CNNs). By integrating TDA with advanced CNN models, this project provides deeper insights into the evolving characteristics of data within these networks.

## Author

[Pablo Olivares Martínez](mailto:pablolivares1502@gmail.com)

## Project Introduction

### Purpose

While CNNs excel in tasks like image recognition, their complex inner workings often remain a black box. This project uses TDA, specifically persistent homology, to probe these subtleties, revealing how data properties evolve through the layers of CNNs.

### Methodology

Our approach utilizes persistent homology to trace topological changes in data as it progresses through different stages of learning within CNNs. This technique offers a fresh perspective, uncovering patterns not immediately apparent through traditional analysis methods.

### Key Findings

We've applied TDA to leading neural network architectures, including ResNet, EfficientNet, and DenseNet. Our observations show significant shifts in topological complexity during training—initial simplifications reduce noise, while subsequent increases foster the development of intricate data representations crucial for class distinction.

### Impact

By integrating topological regularizers into models like EfficientNet-B0 and DenseNet-121, we've achieved notable performance enhancements. These experiments demonstrate how TDA can reveal and leverage distinctive data properties to optimize CNN operations effectively.

## Project Structure

```
.
├── config/            # Configuration files
├── data/              # Datasets for experiments
├── datasets/          # Dataset utilities
├── logs/              # Experiment log files
├── callbacks/         # Training callbacks
├── factories/         # Factory classes
├── docs/              # Project documentation
├── notebooks/         # Jupyter notebooks (EDA, etc.)
├── outputs/           # Experiment output files
├── tests/             # Unit tests
├── trainers/          # Model trainers
└── README.md          # This file
```

## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/pab1s/tda-nn-analysis.git
   ```
2. Install dependencies:
   ```
   make init
   ```
3. Activate the virtual environment:
   ```
   conda activate tda-nn-analysis
   ```
4. Explore datasets in the `data/` directory.
5. Run experiment scripts:
   ```
   ./scripts/script_to_run.sh <args>
   ```
6. Explore Jupyter notebooks in `notebooks/` for EDA, TDA, and NN examples.
7. Run tests:
   ```
   make test
   ```
8. After modifications, update the environment:
   ```
   make export
   ```

## Related Thesis

This project is associated with a bachelor's thesis. The LaTeX source and additional information can be found in the [TFG repository](https://github.com/pab1s/TFG).

## Keywords

- Convolutional Neural Networks (CNNs)
- Topological Data Analysis (TDA)
- Persistent Homology
- Data Properties
- Deep Learning Optimization
- Transfer Learning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
