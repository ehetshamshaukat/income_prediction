from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.ensemble import BalancedRandomForestClassifier
import os
from src.utils import save_file_as_pickle
from dataclasses import dataclass


@dataclass
class ModelTrainingConfig:
    model_pickl_path = os.path.join("artifacts/pickle", "model.pkl")


class ModelTraining:
    def __init__(self):
        self.model_config = ModelTrainingConfig()

    def initiate_model_training(self, transform_train_dataset, transform_test_dataset):
        try:
            xtrain = transform_train_dataset[:, :-1]
            ytrain = transform_train_dataset[:, -1]

            xtest = transform_test_dataset[:, :-1]
            true_value = transform_test_dataset[:, -1]

            models = {
                "logisticsRegression": LogisticRegression(),
                "svm": SVC(),
                "DecisionTree": DecisionTreeClassifier(),
                "Randomforest": RandomForestClassifier(),
                "Adaboost": AdaBoostClassifier(),
                "Gradientboost": GradientBoostingClassifier(),
                "balanceRFC": BalancedRandomForestClassifier()
            }

            model_report = {}

            for model_name, model in models.items():
                model.fit(xtrain, ytrain)
                predicted_value = model.predict(xtest)
                accuracy = accuracy_score(true_value, predicted_value)
                precision = precision_score(true_value, predicted_value)
                recall = recall_score(true_value, predicted_value)
                model_report[model_name] = recall * 100

            best_model_name = max(model_report, key=model_report.get)
            save_file_as_pickle(self.model_config.model_pickl_path, best_model_name)

        except Exception as e:
            raise e
