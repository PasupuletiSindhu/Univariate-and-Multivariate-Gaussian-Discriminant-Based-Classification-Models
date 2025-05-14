# Univariate-and-Multivariate-Gaussian-Discriminant-Based-Classification-Models

This repository contains a modular Python implementation of univariate and multivariate Gaussian discriminant-based classifiers. These models are used to perform classification based on statistical properties of different classes assuming normally distributed data.

## Project Structure

```
.
├── discriminants.py                         # Defines univariate and multivariate Gaussian discriminants
├── classifiers1.py                          # Implements classifiers that use the discriminants
└── HW2_Univariate_and_Multivariate_Discriminants.ipynb  # Demonstrates usage, data processing, and evaluation
```

## Overview

This project demonstrates how Gaussian-based discriminant functions can be used to build probabilistic classifiers. It supports both:

- Univariate Gaussian Discriminant (for single-dimensional data)
- Multivariate Gaussian Discriminant (for multi-dimensional data)

The classifier assigns a label to a new data point based on which class's discriminant function gives the highest value, effectively performing **maximum a posteriori (MAP)** estimation under Gaussian assumptions.

## Dataset

The dataset used in the notebook (`HW2_Univariate_and_Multivariate_Discriminants.ipynb`) is either synthetically generated or loaded using Pandas. It consists of numeric features and a categorical label column named `"Labels"`. Each row represents a sample belonging to one of multiple classes.

## Modules Description

### 1. `discriminants.py`

Defines the discriminant functions:

- `Discriminant`: Abstract base class for all discriminants.
- `GaussianDiscriminant`: Implements the univariate Gaussian discriminant:
  \
  g(x) = -log(σ) - ((x - μ)^2) / (2σ^2) + log(prior)
- `MultivariateGaussian`: Implements the multivariate Gaussian discriminant:
  \
  g(x) = -log|Σ| - 0.5 * (x - μ)^T Σ⁻¹ (x - μ) + log(prior)

### 2. `classifiers1.py`

Defines classifiers that utilize the discriminants:

- `Classifier`: Abstract base class.
- `DiscriminantClassifier`: 
  - Automatically fits one discriminant per class from a dataset.
  - Makes predictions by evaluating all class discriminants on a new sample and returning the class with the highest score.
  - Includes optional support for pooled covariance to implement linear discriminant analysis.

## Usage

### Basic Example

```python
from classifiers1 import DiscriminantClassifier

# Assume df is a Pandas DataFrame with features and a 'Labels' column
clf = DiscriminantClassifier()
clf.fit(df, label_key=["Labels"])

# Predict the class for a new sample
predicted_label = clf.predict(new_sample)
```

### Pooling Covariances (LDA)

```python
clf.pool_variances()  # Optional step to use pooled covariance
```

## Dependencies

- numpy
- pandas
- matplotlib (for visualization in the notebook)
- scikit-learn (optional, for evaluation)

Install dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## License

This project is intended for educational and experimental use.
