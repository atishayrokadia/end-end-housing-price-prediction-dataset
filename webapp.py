from pathlib import Path
import pickle
from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.mlProject.components.prediction import CustomData, predict_value

with open(Path("artifact\model.pkl"), 'rb') as file:
    model = pickle.load(file)

with open(Path("artifact\preprocessor.pkl"), 'rb') as file1:
    preprocessor = pickle.load(file1)

app = Flask(__name__)   #initialise flask app

@app.route('/', methods=['GET'])
def homePage():                     # route to display the home page
    return render_template("index.html")


@app.route('/train',methods=['GET'])   #route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful"

@app.route('/predict',methods=['POST','GET'])
def index():
    if request.method=='POST':
        try:
            #reading the input
            area =int(request.form['area'])
            stories =int(request.form['stories'])
            bedrooms =int(request.form['bedrooms'])
            bathrooms =int(request.form['bathrooms'])
            airconditioning =str(request.form['airconditioning'])
            parking =int(request.form['parking'])
            prefarea =str(request.form['prefarea'])
            furnishingstatus =str(request.form['furnishingstatus'])
   

            customdata =  CustomData(area,stories,bedrooms,bathrooms,airconditioning,parking,prefarea,furnishingstatus)
            Custom_data_frame = customdata.custom_input_df()
    
    
            x = pd.DataFrame(Custom_data_frame)
            input_data = preprocessor.transform(x)
            predict = predict_value(input_data)
            return render_template('results.html', prediction = str(predict))    
        
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')                       



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080)


