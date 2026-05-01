# machine_learning

Self-study experiments following CS231n → CS224n → nanoGPT.

## Contents

### MNIST
Handwritten digit classification, built from scratch with PyTorch.

| Model | Optimizer | Test Accuracy |
|-------|-----------|---------------|
| MLP (2-layer) | SGD | 91.2% |
| MLP (2-layer) | Adam | 96.8% |
| CNN (LeNet-style) | Adam | 99.1% |

Key finding: switching SGD → Adam on the same architecture gained 5.6pp.
Adding convolutions gained another 2.3pp with fewer parameters than
a wider MLP would need.

### Kaggle: House Prices
Tabular regression on the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

Pipeline:
- Feature engineering: log-transform skewed numerics, target encoding,
  missing value imputation
- Base models: Ridge, XGBoost, ElasticNet
- Meta-learner: stacking ensemble with 5-fold CV (Optuna for hyperparameter search)

## Environment

GPU:     RTX 2070

CUDA:    13.1

Python:  3.11 (conda env: ml)

PyTorch: latest stable

## Reference

[d2l-zh-pytorch](https://github.com/d2l-ai/d2l-zh) Release 2.0.0
