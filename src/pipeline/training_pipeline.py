import time
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

if __name__ == "__main__":
    print("*" * 148)
    di = DataIngestion()
    print("\t\t\t\t\t\t\tData Ingestion Starts")
    time.sleep(1)
    train_dataset_path, test_dataset_path = di.initiate_data_ingestion()
    print("\t\t\t\t\t\t\tData Ingestion Complete")

    print("*" * 148)
    time.sleep(1)
    dt = DataTransformation()
    print("\t\t\t\t\t\t\tData Transformation Starts")
    time.sleep(2)
    transformed_train_dataset, transformed_test_dataset = dt.initiate_data_transformation(train_dataset_path,
                                                                                          test_dataset_path)
    print("\t\t\t\t\t\t\tData Transformation Complete")
    print("*" * 148)


    time.sleep(1)
    print("\t\t\t\t\t\t\tModel Training Starts")
    mt = ModelTraining()
    output = mt.initiate_model_training(transformed_train_dataset, transformed_test_dataset)
    print("\t\t\t\t\t\t\tModel Training Complete")
    time.sleep(1)
    print("*" * 148)
