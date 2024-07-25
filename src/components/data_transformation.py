import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
import os

from src.utils import save_file_as_pickle


@dataclass
class DataTransformationConfig:
    data_transformation_pickle_path = os.path.join("artifacts/pickle", "data_transformation.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformation(self):
        try:
            numerical_columns = ["age", "hours_per_week"]
            categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'race',
                                   'gender', 'native_country']

            numerical_column_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("standardization", StandardScaler())
            ])

            categorical_column_pipeline = Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder(sparse_output=False)),
                ("standardization", StandardScaler())
            ])

            preprocessing = ColumnTransformer([
                ("numerical_column_pipeline", numerical_column_pipeline, numerical_columns),
                ("categorical_columns_pipeline", categorical_column_pipeline, categorical_columns)
            ], remainder="passthrough")

            return preprocessing
        except Exception as e:
            raise e

    def initiate_data_transformation(self, train_dataset_path, test_dataset_path):
        try:
            train_dataset = pd.read_csv(train_dataset_path)
            test_dataset = pd.read_csv(test_dataset_path)

            transform = self.data_transformation()

            columns_to_drop = "income"
            target_feature = "income"

            xtrain = train_dataset.drop(columns=columns_to_drop, axis=1)
            ytrain = train_dataset[target_feature]

            xtest = test_dataset.drop(columns=columns_to_drop, axis=1)
            ytest = test_dataset[target_feature]

            # transforming depended features
            transform_xtrain = transform.fit_transform(xtrain)
            transform_xtest = transform.transform(xtest)

            # concatenating depended feature with independed feature
            transform_train = np.c_[transform_xtrain, np.array(ytrain)]
            transform_test = np.c_[transform_xtest, np.array(ytest)]

            # saving transformation config as pickle
            save_file_as_pickle(self.data_transformation_config.data_transformation_pickle_path, transform)

            # returning transform dataset for model training
            return transform_train, transform_test
        except Exception as e:
            raise e
