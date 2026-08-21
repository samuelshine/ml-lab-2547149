# Machine Learning Lab

Coursework notebooks, one folder per lab. Each folder holds the notebook, the exported PDF, and any data it needs.

| Folder | Notebook | What it covers |
| --- | --- | --- |
| [lab-01-02](lab-01-02/) | [process.ipynb](lab-01-02/process.ipynb) | Data profiling, missing-value treatment, outlier detection and plots on air quality and crop production data |
| [lab-03](lab-03/) | [lab-3.ipynb](lab-03/lab-3.ipynb), [lab-3-extension.ipynb](lab-03/lab-3-extension.ipynb) | Simple linear regression, with an extension on the medical cost dataset |
| [lab-04](lab-04/) | [lab-4.ipynb](lab-04/lab-4.ipynb) | KNN classification and evaluation metrics for regression and classification |
| [lab-05](lab-05/) | [lab_5.ipynb](lab-05/lab_5.ipynb) | Linear regression by batch, mini-batch and stochastic gradient descent |
| [lab-06](lab-06/) | [lab_6.ipynb](lab-06/lab_6.ipynb) | Logistic regression against KNN on Breast Cancer Wisconsin |
| [lab-07](lab-07/) | [lab_7.ipynb](lab-07/lab_7.ipynb) | Decision trees on the Iris dataset |
| [lab-08](lab-08/) | [lab_8.ipynb](lab-08/lab_8.ipynb) | Categorical Naive Bayes on Play Tennis, compared with a decision tree and logistic regression |
| [lab-09](lab-09/) | [lab-9.ipynb](lab-09/lab-9.ipynb) | SVM classification with hyperparameter tuning, PCA, and PCA against LDA |
| [lab-10](lab-10/) | [lab-10.ipynb](lab-10/lab-10.ipynb) | XOR with an MLP built three ways: Keras, PyTorch, and low-level TensorFlow |
| [cia-1](cia-1/) | [cia-1.ipynb](cia-1/cia-1.ipynb) | CIA 1 submission |
| [dist-met](dist-met/) | [activity-2.ipynb](dist-met/activity-2.ipynb) | Data cleaning, complete case analysis, skew-based imputation, distance matrices |
| [ete](ete/) | [ete-1.ipynb](ete/ete-1.ipynb), [ete-2.ipynb](ete/ete-2.ipynb) | End Trimester Exam submission — see below |

## End Trimester Exam

The [ete](ete/) folder holds the two notebooks submitted for the End Trimester Exam, each with its exported PDF. Both work on the Iris dataset and follow the same skeleton: profile the raw data, check missing values, split before scaling, then build two variants of a model and compare them on cross-validated and held-out scores.

| Notebook | PDF | What it covers |
| --- | --- | --- |
| [ete-1.ipynb](ete/ete-1.ipynb) | [ete-1.pdf](ete/ete-1.pdf) | KNN on the 4 original features against KNN on 2 LDA discriminants. Class-wise histograms, IQR outlier check, correlation heatmap, `k` chosen by 5-fold stratified CV, decision boundaries drawn in both spaces, and a metric-by-metric comparison |
| [ete-2.ipynb](ete/ete-2.ipynb) | [ete-2.pdf](ete/ete-2.pdf) | An MLP at two capacities — one hidden layer of 5 neurons against two layers of 32 and 16. Repeated stratified 5-fold CV, train-versus-validation gap as an overfitting signal, loss curves, and a 10-seed rerun showing 45 test rows cannot separate the two |

Both notebooks keep the scaler (and LDA) inside a `Pipeline`, so every CV fold refits them on that fold's training part and no test statistics leak into the estimate.

## Working on this repo

Everything lands on `main`. GitHub only counts commits on the default branch towards the contribution graph, so work that stops on a side branch never shows up there.

For a new lab, either commit straight to `main`:

```bash
git checkout main && git pull
mkdir lab-11
# work, then
git add lab-11 && git commit -m "Add Lab 11" && git push
```

or branch and merge it back when the lab is done:

```bash
git checkout -b lab-11 main
# work, commit, push
git checkout main && git merge --no-ff lab-11 && git push
```

Either way the commits end up on `main`, which is what the graph counts.
