from flask import Flask, render_template, request, jsonify
import re
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('Homepage2.html')

@app.route('/Manualpage')

def Manualpage():
    return render_template('Manualpage.html')

@app.route('/AutomaticPage')

def Automaticpage():
    return render_template('AutomaticPage.html')


@app.route('/predict', methods = ['POST'])

def predict():
    data = {
        'temperature': request.form['temperature'],
        'humidity': request.form['humidity'],
        'ph': request.form['ph'],
        'soil_moisture': request.form['soil_moisture'],
        'co2': request.form['co2_concentration'],
        'nutrient_index': request.form['Nutrient_Index'],
        'soil_type': request.form['soil_type'],
        'growth_stage': request.form['growth_stage'],
        'crop_type': request.form['crop_type'],
    }

    #list the objects
    label_soil_names = ['Flowering', 'Seedling', 'Vegetative']
    label_growth_names = ['Clay', 'Loamy', 'Sandy'] 
    label_crop_names = ['label_apple', 'label_banana', 'label_blackgram', 'label_chickpea','label_coconut', 'label_coffee', 
                        'label_cotton', 'label_grapes', 'label_jute', 'label_kidneybeans', 'label_lentil', 'label_maize',
                          'label_mango', 'label_mothbeans', 'label_mungbean', 'label_muskmelon', 'label_orange', 'label_papaya', 
                          'label_pigeonpeas', 'label_pomegranate', 'label_rice','label_watermelon']

    
    # np.zero arrays for each label
    label_soil = np.zeros(3)
    label_growth_stage = np.zeros(3)
    label_crop_type = np.zeros(22)

    #the feature to be used for prediction
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    ph = float(data['ph'])
    soil_moisture = float(data['soil_moisture'])
    co2_concentration = float(data['co2'])
    Nutrient_Index = float(data['nutrient_index'])
    crop_type = data['crop_type']
    growth_stage = data['growth_stage']
    soil_type = data['soil_type']

    
    #using regex search through the objects for our object
    for label in label_crop_names:
        if re.search(f'{crop_type}$', label):
            crop_index = label_crop_names.index(label)
            label_crop_type[crop_index] = 1.
    for label in label_growth_names:
        if re.search(f'{growth_stage}$', label):
            growth_index = label_growth_names.index(label)
            label_growth_stage[growth_index] = 1.
    for label in label_soil_names:
        if re.search(f'{soil_type}$', label):
            soil_index = label_soil_names.index(label)
            label_soil[soil_index] = 1.
    

    prediction_list = [temperature, humidity, ph, soil_moisture, co2_concentration, Nutrient_Index] 
    prediction_list.extend(label_crop_type)
    prediction_list.extend(label_growth_stage)
    prediction_list.extend(label_soil)

    if len(prediction_list) == 34:
        prediction = model.predict([prediction_list])
        if prediction[0]==0:
            text = "Water crops max 3 times and don't forget to perform the eye test"
            return jsonify({'prediction' : text })
        elif prediction[0] == 1:
            text = "Water crops max 4 times and don't forget to perform the eye test"
            return jsonify({'prediction' : text})
        elif prediction[0] == 2:
            text = "Water crops max 6 times and don't forget to perform the eye test"
            return jsonify({'prediction' : text})    
    else:
        return jsonify({'error'})

    

if __name__ == '__main__':
    app.run(debug= True)