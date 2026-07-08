# Model Evaluation Results

## Overview
This document reports the held-out test metrics for the **Smart Credit Card Fraud Investigation** model. 
The core engine utilizes a gradient-boosted decision tree framework (`LightGBM`), integrated with `Isotonic Regression` for risk score calibration.

## Test Set Metrics
The model was evaluated against the final 15% unseen testing block, generating the following performance metrics based on a standard 0.5 (50%) confidence threshold:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **PR-AUC** | `0.9375` | **Area Under the Precision-Recall Curve** (Primary metric for highly imbalanced fraud datasets). |
| **Precision** | `0.7500` | Proportion of predicted frauds that were actually fraudulent. |
| **Recall** | `1.0000` | Proportion of actual fraudulent transactions successfully identified. |
| **F1-Score** | `0.8571` | The harmonic mean of Precision and Recall. |

## Technical Note: Class Imbalance Handling
During training, the LightGBM classifier leveraged a dynamic `scale_pos_weight` multiplier of **49.72** to shift the decision boundary and prioritize capturing complex fraud behaviors.
