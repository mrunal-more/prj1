from flask import Flask,render_template,request,jsonify,redirect,url_for
import numpy as np
import json
import pickle

with open('artifacts/project_data.json','r') as file:
    project_data = json.load(file)


with open('artifacts/model.pkl','rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods = ['POST'])
def get_data():
    data = request.form
    # result = Charges_prediction(data)
    gender = data['html_gender']
    religion = data['html_religion']
    caste= data['html_caste']
    mother_tongue= data['html_mother_tongue']
    country= data['html_country']
    height_cms= data['html_height_cms']
           
    user_data = np.zeros(len(project_data['column_names']))
    user_data[0] = project_data['gender'][gender]
    user_data[1] = religion
    user_data[2] = caste
    user_data[3] = mother_tongue
    user_data[4] = country
    user_data[5] = height_cms

    result = model.predict([user_data])[0]
    print(result)
    return render_template('index.html',prediction =result)

if __name__ == "__main__":
    app.run(host = '0.0.0.0')