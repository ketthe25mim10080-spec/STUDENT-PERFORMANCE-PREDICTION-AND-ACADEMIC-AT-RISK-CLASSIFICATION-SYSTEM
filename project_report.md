# Technical Report: Student Academic Performance & Risk Prediction

## 1. Abstract

Early identification of academically at-risk students allows educational institutions to intervene proactively. This project implements a machine learning classification pipeline using a Random Forest algorithm to predict student pass/fail status based on daily study habits, past performance, and attendance records.

## 2. Problem Statement

Educational settings often struggle to detect academic decline before formal final examinations. The objective is to build a predictive binary classifier capable of assessing student academic risk levels based on key behavioral and historical metrics.

## 3. Dataset & Feature Engineering

The model operates on five primary features:

1. **StudyHours**: Average daily study hours ($1.0 - 10.0$).

2. **AttendanceRate**: Class attendance percentage ($50.0\% - 100.0\%$).

3. **PastScore**: Marks achieved in previous evaluation ($0.0 - 100.0$).

4. **Extracurricular**: Binary flag indicating active involvement in extracurricular activities ($0$ or $1$).

5. **SleepHours**: Average sleep duration ($4.0 - 9.0$ hours).

**Target Variable:** `Status` ($1$: Pass / Low Risk, $0$: At-Risk / Fail).

## 4. Methodology

1. **Preprocessing:** Standard Scaling ($Z$-score normalization) was applied across all continuous numeric features to eliminate scale disparity.

2. **Train-Test Split:** $80\%$ training set and $20\%$ test evaluation set.

3. **Model Selection:** A Random Forest Classifier ($n\_estimators=100$) was selected due to its robustness against overfitting and high interpretability for tabular data.

## 5. Model Evaluation

The pipeline automatically reports key performance indicators upon execution:

* **Accuracy:** Evaluates overall correctness across test samples.

* **Precision & Recall:** Tracks false positive and false negative rates to ensure at-risk students are reliably identified without excessive false alarms.

## 6. Conclusion & Future Scope

The Random Forest model effectively flags at-risk students using key performance indicators. Future enhancements include expanding feature dimensions (e.g., LMS engagement metrics) and deploying a web dashboard for interactive educator access.