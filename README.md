# EPFL Machine Learning Higgs

The first [project](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf) of EPFL Machine learning course (CS-433).

Code in Python using `numpy` **only**.

## Team Members

- Yiyang Feng: yiyang.feng@epfl.ch
- Naisong Zhou: naisong.zhou@epfl.ch
- Yuheng Lu: yuheng.lu@epfl.ch

## Abstract
Machine learning has gain more application as new models are proposed. In this project, we applied machine learning techniques on CERN’s Higgs Boson Dataset. With noise distributed in data, we took different pre-processing methods for better performance and compared several models on this task.

## Dataset
Download dataset [here](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files). Extract all files and move into a new folder named `data`. It cannot be uploaded because of its size.

## Install requirements

```shell
conda create -n mlproject python=3.9 jupyterlab=3.2 numpy=1.23.1 matplotlib=3.5.2 pytest=7.1.2 gitpython=3.1.18 black=22.6.0 pytest-mock=3.7.0
conda activate mlproject
```

## Performance

|               | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| ------------- | ------------ | ------------- | ---------- | ------------ |
| Training      | 83.07        | 77.89         | 70.73      | 74.13        |
| Validation    | 83.00        | 77.38         | 70.88      | 73.99        |
| Test (AIcrowd) | 83.00        | -             | -          | 74.10         |

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
