# Breast Cancer Classification Analysis

## Project Overview
This project involves using various machine learning algorithms to classify tumors as either benign or malignant using the [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
->https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Algorithms Used
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

## File Structure
```
breast_cancer_classification/
├── data/
│   └── data.csv
├── notebooks/
│   └── breast_cancer_analysis.ipynb
└── visuals/
    ├── diagnosis_distribution.png
    ├── decision_tree.png
    └── confusion_matrix.png
```

## Results
| Model                | Accuracy               |
|----------------------|------------------------|
| KNN                  | 0.9590643274853801     |
| Logistic Regression  | 0.9824561403508771     |
| Decision Tree        | 0.9415204678362573     |
| Random Forest        | 0.9707602339181286     |


## How to Run
- Ensure you have Python and necessary libraries installed (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
- Run the notebook `breast_cancer_analysis.ipynb` step-by-step.

## Data Source
- [Breast Cancer Wisconsin Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

