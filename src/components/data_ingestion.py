import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:
    train_dataset_path=os.path.join("artifacts/train_test_dataset","train.csv")
    test_dataset_path=os.path.join("artifacts/train_test_dataset","test.csv")

class DataIngestion:
    def __init__(self):
        self.dataset_path=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            path="dataset/income_prediction_dataset_processed.csv"
            df=pd.read_csv(path)

            # making directory to save train n test dataset
            os.makedirs(os.path.dirname(self.dataset_path.train_dataset_path),exist_ok=True)

            # splitting dataset into train n test
            train_dataset,test_dataset=train_test_split(df,test_size=0.33,random_state=69)

            # saving train n test dataset into directory
            train_dataset.to_csv(self.dataset_path.train_dataset_path)
            test_dataset.to_csv(self.dataset_path.test_dataset_path)

            # returning train n test path for further use
            return self.dataset_path.train_dataset_path,self.dataset_path.test_dataset_path

        except Exception as e:
            raise e