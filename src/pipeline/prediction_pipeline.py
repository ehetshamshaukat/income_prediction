import pandas as pd
import os

from src.utils import load_pickle_file


class Prediction:
    def __init__(self):
        pass

    def initiate_prediction(self, features):
        try:
            preprocessing_path = os.path.join("artifacts/pickle", "data_transformation.pkl")
            model_path = os.path.join("artifacts/pickle", "model.pkl")

            preprocessing = load_pickle_file(preprocessing_path)
            mdl = load_pickle_file(model_path)

            processed_data = preprocessing.transform(features)
            output = mdl.predict(processed_data)

            return output

        except Exception as e:
            raise


class Features:
    def __init__(self, age, workclass, education, marital_status, occupation, race,
                 gender, hours_per_week, native_country):
        self.age = age
        self.workclass = workclass
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.race = race
        self.gender = gender
        self.hours_per_week = hours_per_week
        self.native_country = native_country

    def to_dataframe(self):
        try:
            features_in_dict = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education": [self.education],
                "marital_status": [self.marital_status],
                "occupation": [self.occupation],
                "race": [self.race],
                "gender": [self.gender],
                "hours_per_week": [self.hours_per_week],
                "native_country": [self.native_country]
                }
            feature_to_df = pd.DataFrame(features_in_dict)
            return feature_to_df
        except Exception as e:
            raise e
