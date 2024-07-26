# Project: Income Prediction
## Problem statement
### 1. Goal
```
To predict the whether a person income is greater than 50000$ or not 
```
### 2. About Income prediction
```requirements
Income prediction is all about predicting of person income on different features like age,gender and work they do 
```
## Description
### 1. Dataset
```
1. Imabalance dataset, mean one category is way more than other category 
2. Dataset available on kaggle 
```

### 2. Features
``` 
Input features = [age, workclass, education, marital_status,occupation, race,gender, hours_per_week, native_country]
Target feature = [income]
```
### 3. Pipeline Structure
```requirements
Google define pipeline 
```
# Requirements
### 1. Language
```
Python 3.10
```
### 2. Libraries
```
1. numpy
2. pandas
3. scikit-learn
4. pickle
5. os 
6. streamlit 
7. imbalanced-learn
 ```
# code
### 1. Enviroment
```requirements
conda create -p venv python==3.10 -y 
```
### 2. Activate enviroment
```requirements
conda activate venv/
```
### 3. Setup
```
The setup.py is a Python script typically included with Python-written libraries or apps. Its objective is to ensure that the program is installed correctly. 
```
### 4. Components
- Data ingestion
```
reading data from different source and splitting data into train and test
```
- Data transformation
```
  reading train and test dataset and apply different transformation and save transformation setting in pickle format
```
- Model training
```requirements
transformed dataset and using different machine learning model and save the best model in pickle format
```
### 5. Pipeline
- Training pipeline
```
using components and creating pipeline for model training
```
- Prediction pipeline
```
taking data from user transform for model and predict 
```

## Run
#### 1. Download repository
```
git clone https://github.com/ehetshamshaukat/income_prediction.git
```
#### 2. Install dependences
```requirements
pip install -r requirements.txt
```
#### 3. Transformation and training
- data transformation and model training
  ```
  For model training, which will also save tranformation and model in pickle format
  python src/pipeline/training_pipeline.py
  ```
- Prediction
  ```
  For Prediction on new data
  python src/pipeline/prediction_pipeline.py
  ```
#### 4. Streamlit
```
streamlit run application.py
```
## Deployment
```
Deploy on AWS using Github actions which is CI CD technique
```
## Image
<img width="1484" alt="Screenshot 2024-07-25 at 3 34 42 PM" src="https://github.com/user-attachments/assets/7470f3ad-27b8-499a-b38e-dd647f5c2c17">
