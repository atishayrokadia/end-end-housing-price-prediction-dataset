from src.mlProject.exception import CustomException
import sys
from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from src.mlProject.utils import save_object,find_key_by_value
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_cofig = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            print("Splitting the data into test and train set")

            x_train,y_train,x_test,y_test = (
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )
            # print(x_test.columns,"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("x_train.shape",x_train.shape,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("x_test.shape",x_test.shape,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("y_train.shape",y_train.shape,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("y_test.shape",y_test.shape,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            models = {
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor()
            }

            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(x_train,y_train)
                y_train_pred = model.predict(x_train)

                y_test_pred = model.predict(x_test)

                train_model_score = mean_absolute_percentage_error(y_train, y_train_pred)

                test_model_score = mean_absolute_percentage_error(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            best_model_score = min(sorted(report.values()))
            print(find_key_by_value(report,best_model_score),best_model_score,"QQQQQQQQQQQQQQQQQQQQQq")
            print(report,"QQQQQQQQQQQQQQQQQQQQQQQ")
            best_model = models[find_key_by_value(report,best_model_score)]

            save_object(
                file_path=self.model_trainer_cofig.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            mape = mean_absolute_percentage_error(y_test,predicted)

            return mape  
            
        except Exception as e: 
            raise CustomException(e,sys) 