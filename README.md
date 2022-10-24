# EPFL Machine Learning Higgs

The first [project](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf) of EPFL Machine learning course (CS-433).

Code in Python using `numpy` **only**.

## Team Members

- Yiyang Feng: yiyang.feng@epfl.ch
- Naisong Zhou: naisong.zhou@epfl.ch
- Yuheng Lu: yuheng.lu@epfl.ch

## Aim :
The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles have mass. We are given a vector of features representing the decay signature of a collision event, and we want to predict whether this event was signal (a Higgs boson) or background (something else). To do this, we use different binary classification techniques and compare the results.

## Dataset:
Download dataset [here](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files). Extract all files and move into a new folder named `data`. It cannot be uploaded because of its size.

## Install requirements:

```shell
conda create -n mlproject python=3.9 jupyterlab=3.2 numpy=1.23.1 matplotlib=3.5.2 pytest=7.1.2 gitpython=3.1.18 black=22.6.0 pytest-mock=3.7.0
conda activate mlproject
```

## Performance:

|               | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| ------------- | ------------ | ------------- | ---------- | ------------ |
| Training      | 82.58        | 76.60         | 70.87      | 73.62        |
| Validation    | 82.62        | 76.34         | 71.09      | 73.62        |
| Test (AIcrowd) | 82.40        | -             | -          | 73.4         |

## Files
- `run.py` : produces nearly the same predictions (as a csv file) which we used in our best submission to the competition system.
- `implementations.py` : include 6 required functions for this project. cover all machine learning methods we have learned in class.
- `report.pdf` : our report for this project.
- `model_comparison.ipynb` : compare different models on the processed dataset using our feature engineering technique.
- `ablation_study.ipynb` : ablation study of different feature engineering techniques.
- `utils` : folder for some Python modules.
  - `preprocess.py` : functions for feature engineering.
  - `cross_validation.py` : functions for generating cross validation datasets and visualization.
  - `helpers.py` : functions for loading and writing csv files.
  - `prediction.py` : functions for predict linear regression or logistic regression.
- `data` : folder for datasets and submission files (not uploaded).
  - `submission_final.csv` : final submission file.
  - `train.csv` : training data.
  - `test.csv` : testing data.
