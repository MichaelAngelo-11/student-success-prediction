# Predicting At-Risk Students with Machine Learning

**Mission:** Improve the quality of education in high school by changing the learning experience — through
different learning delivery/engagement strategies and a better student-teacher dynamic — to give students a
better overall experience and outcome.

## Project summary

This project builds an end-to-end, reproducible machine learning pipeline that predicts which students are
**at risk** of a poor academic outcome (failing or withdrawing) from demographic, registration, assessment and
learning-platform engagement data. The core idea generalises directly to a high-school setting: identifying at-risk
students early from behavioural/engagement signals lets teachers intervene with a different learning delivery
(more interactive content, different pacing, personalised support) and a healthier student-teacher dynamic
(early outreach, mentoring) before a student falls behind.

## Dataset

[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset) — a large, open
educational-data-mining dataset with demographics, registrations, assessment scores, and **over 10.6 million** raw
Virtual Learning Environment (VLE) interaction records (`studentVle.csv`), comfortably exceeding the 20,000-row
minimum.

**Note:** per assignment requirements, the raw dataset ZIP is **not committed to this repository**. It is hosted
in the Google Drive folder linked below, along with the saved trained models. This repository contains only code,
notebook, documentation, and lightweight output artifacts (figures, result CSVs).

## Repository structure

```
.
├── notebooks/
│   └── oulad_at_risk_classification.ipynb      # full, executed, end-to-end notebook
├── figures/                                    # EDA plots, training curves, confusion matrices, ROC curves
│   ├── class_balance.png
│   ├── confusion_matrices.png
│   ├── eda_boxplots.png
│   ├── nn_training_curves.png
│   └── roc_curves.png
├── reports/
│   ├── Student Success Prediction Report.docx.md  # full written report
│   ├── experiment_results.csv                     # 7 validation-set experiment runs
│   └── test_results.csv                           # final held-out test-set metrics
├── requirements.txt
├── README.md
└── README_UPLOAD_NOTES.md
```

The raw dataset and saved model binaries (`preprocessor.joblib`, `best_traditional_model.joblib`,
`best_nn_model.keras`) are not included in this repository (they are too large for GitHub) — see
`README_UPLOAD_NOTES.md` for where they are hosted instead.

## How to reproduce

1. Create an environment with the packages in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
2. Download the raw dataset ZIP from the Google Drive link below and unzip it into a `data/` folder you create
   in the project root (this folder is intentionally excluded from the repository).
3. Run the notebook top-to-bottom:
   ```
   jupyter nbconvert --to notebook --execute --inplace notebooks/oulad_at_risk_classification.ipynb
   ```
   or open it in Jupyter/VS Code and run all cells. Random seeds are fixed, so results are reproducible.

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting (best traditional) | 0.921 | 0.957 | 0.892 | 0.923 | 0.977 |
| Neural Network — larger layers / bigger batch (best deep learning) | 0.917 | 0.948 | 0.891 | 0.918 | 0.973 |

Seven systematically varied experiments (3 traditional ML baselines + 4 neural-network configurations) are
recorded with real validation-set metrics in `reports/experiment_results.csv`.

## Links

- GitHub repository: *[link](https://github.com/MichaelAngelo-11/student-success-prediction.git)*
- Google Drive folder : *[link](https://drive.google.com/drive/folders/1ojkiGGAU7tgQeX_CGRHMsx8FlqvSKIkI?usp=sharing)*
- Presentation video: *[link](https://drive.google.com/drive/folders/1TcCDP8o3NUV86_50BHTZ_5IPUCJ5wPbN?usp=sharing)*
