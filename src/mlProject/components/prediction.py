import pickle
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from mlProject.exception import CustomException



# Load the model from the pickle file
with open(Path("artifact\model.pkl"), 'rb') as file:
    model = pickle.load(file)

with open(Path("artifact\preprocessor.pkl"), 'rb') as file1:
    preprocessor = pickle.load(file1)

# Function to get user input and predict
def predict_value(user_input):
    # Convert the input string to a numpy array
    features = np.array(user_input).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Output the prediction
    print(f"Predicted Price: {prediction[0]}")
    return prediction[0]
class CustomData:
    def __init__(self,area:int,stories:int,bedrooms:int,bathrooms:int,airconditioning:str,parking:int,prefarea:str,furnishingstatus:str):
        self.area = area
        self.stories = stories
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.airconditioning = airconditioning
        self.parking = parking
        self.prefarea = prefarea
        self.furnishingstatus = furnishingstatus


    def custom_input_df(self):
        try:
            CustomData_input = {
                "area": [self.area],
                "stories" :[self.stories],
                "bedrooms": [self.bedrooms],
                "bathrooms":[self.bathrooms],
                "parking":[self.parking],
                "airconditioning":[self.airconditioning],
                "prefarea":[self.prefarea],
                "furnishingstatus":[self.furnishingstatus]
                }
        
            
        except Exception as e:
            raise CustomException(e,sys)
        return pd.DataFrame(CustomData_input)



# Run the prediction functionf.
if __name__ == "__main__":
    num_var = []
    cat_var = []
    # num_columns = ["area","stories","bedrooms","bathrooms","parking"]
    # cat_columns = ["airconditioning","prefarea","furnishingstatus"]
    # Get user inputs
    
    area = int(input("Enter area in sq ft: "))
    num_var.append(area)
    
    stories = int(input("Enter number of stories: "))
    num_var.append(stories)
    
    bedrooms = int(input("Enter number of bedrooms: "))
    num_var.append(bedrooms)
    
    bathrooms = int(input("Enter number of bathrooms: "))
    num_var.append(bathrooms)
    
    airconditioning = input('Airconditioning (yes/no): ')
    cat_var.append(airconditioning)
    
    parking = int(input("Enter number of parking spaces: "))
    num_var.append(parking)
    
    prefarea = input('Preferred area (yes/no): ')
    cat_var.append(prefarea)
    
    furnishingstatus = input('Furnishing status (unfurnished/furnished/semi-furnished): ')
    cat_var.append(furnishingstatus)
    

    
    customdata =  CustomData(area,stories,bedrooms,bathrooms,airconditioning,parking,prefarea,furnishingstatus)
    Custom_data_frame = customdata.custom_input_df()
    
    
    x = pd.DataFrame(Custom_data_frame)
    input_data = preprocessor.transform(x)


    # Make the prediction
    predict_value(input_data)
