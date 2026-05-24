# CL06_G03 - AI-Powered Obesity Level Classification System

## Project Overview

This project implements an MLOps pipeline for an AI-powered obesity level classification system. The system uses eating habit and physical condition data to classify a person's obesity level into one of several health categories.

This project extends the original Group Task 1 system blueprint by adding automated preprocessing, model training, model evaluation, monitoring, drift detection, DVC pipeline tracking, and GitHub Actions automation.

## Team Members

| Name | Student ID | Role |
|---|---:|---|
| Sahil | 105295665 | Project Manager / Scrum Master |
| Parav | 105321434 | Data Engineer |
| Seniru | 104786647 | ML Engineer |
| Nainika | 105321162 | MLOps / DevOps Lead |

## Main Features

- Automated preprocessing of the obesity dataset
- Random Forest and Decision Tree model training
- Model selection based on validation macro F1-score
- Model evaluation using accuracy, precision, recall, F1-score, classification report, and confusion matrix
- Drift monitoring using the Kolmogorov-Smirnov test
- GitHub Actions workflow for automated retraining on data or source code changes
- DVC pipeline for reproducible machine learning stages
- Centralised artifact storage for models, metrics, plots, processed data, and reports

## Repository Structure

```text
.github/workflows/     GitHub Actions workflow configuration
data/                  Training data and new incoming data
src/                   Preprocessing, model training, evaluation, and monitoring scripts
artifacts/             Generated model files, metrics, processed data, and plots
reports/               Evaluation and monitoring reports
dvc.yaml               DVC pipeline definition
requirements.txt       Python package dependencies
README.md              Project documentation
