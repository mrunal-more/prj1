from flask import Flask,render_template,request,jsonify,redirect,url_for
import numpy as np
import pickle
import json

with open('artifacts/dict_file.json','r') as file:
    Dict = json.load(file)

with open('artifacts/scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

with open('artifacts/marriage_age_model.pkl','rb') as file:
    Model = pickle.load(file)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods = ['POST','GET'])
def get_data():
    Data                = request.form
    gender              = Data['html_gender']
    religion            = Data['html_religion']
    caste               = Data['html_caste']
    mother_tongue       = Data['html_mother_tongue']
    country             = Data['html_country']
    height_cms          = Data['html_height_cms']
           
    user_data           = np.zeros(len(Dict['Column_Names']))
    user_data[0]        = Dict['gender'][gender]
    user_data[1]        = Dict['religion'][religion]
    user_data[2]        = Dict['caste'][caste]
    user_data[3]        = Dict['mother_tongue'][mother_tongue]
    user_data[4]        = Dict['country'][country]
    user_data[5]        = height_cms


    
    user_data_scale = Scaler.transform([user_data])
    result = Model.predict(user_data_scale)[0]
    print(result)
    return render_template('index.html',prediction=result)

if __name__ == "__main__":
    app.run(host = '0.0.0.0')