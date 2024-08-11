import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from src.mlProject.exception import CustomException
from src.mlProject.utils import save_object
from sklearn.compose import ColumnTransformer

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

    def get_data_transformation_object(self):
        try:
            num_columns = ["area","stories","bedrooms","bathrooms","parking"]
            cat_columns = ["furnishingstatus","airconditioning","prefarea"]

            num_pipeline = Pipeline(steps=[
                                    ("Scaler",StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                                    ("OneHotEncoder",OneHotEncoder()),
                                    ("Scaler",StandardScaler(with_mean=False))
            ])
            print(f"categotrical columns {cat_columns}")
            print(f"numerical columns {num_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num-pipeline",num_pipeline,num_columns),
                    ("cat-pipeline",cat_pipeline,cat_columns)                    
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # print("####################################################3")
            # print(train_df.head(2))
            # print("####################################################3")
            # print(test_df.head(2))
            # print("####################################################3")

            print("Reading the dataset")

            preprocessing_obj = self.get_data_transformation_object()
            print(preprocessing_obj,"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            target_column_name = "price"
            
            # diving train data set
            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df = train_df[target_column_name]
            print("####################################################")
            print("input_features_train_df",input_features_train_df)
            print("####################################################")
            # diving test data set
            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df = test_df[target_column_name]

            print("applying preprocessing to train and test dataset")


            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            print("####################################################")
            print("len(input_feature_train_arr[0])",len(input_feature_train_arr[0]))
            print("####################################################")
            train_arr = np.c_[input_feature_train_arr,np.array(target_features_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_features_test_df)]
            print("####################################################")
            print("len(train_arr[0])",len(train_arr[0]))
            print("####################################################")
            print("Saved preprocessing")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                        )
            print('object saved sucessfully')
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
