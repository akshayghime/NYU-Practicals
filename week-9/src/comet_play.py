from comet_ml import Experiment
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def main():
    #create an experiment with your api key
    exp = Experiment(api_key="HDwtDordk4T6HcaKB3ZlxLTYn",
                     project_name="cancer_detection",
                     auto_param_logging=False)
    random_state = 50

    cancer = load_breast_cancer()
    print("cancer.keys(): {}".format(cancer.keys()))
    print("Shape of cancer data: {}\n".format(cancer.data.shape))
    print("Sample counts per class:\n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
    print("\nFeature names:\n{}".format(cancer.feature_names))

    return


if __name__ == "__main__":
    main()
