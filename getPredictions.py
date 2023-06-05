#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import joblib

app = Flask(__name__)

run_with_ngrok(app)

# Load the trained machine learning model
model_depress = joblib.load('model_depress.pkl')
model_anx = joblib.load('model_anx.pkl')
model_stress = joblib.load('model_stress.pkl')


# In[2]:


# Define a route that responds to GET requests for depression
@app.route('/predictDepressionLevel', methods=['GET'])
def predictDepressionLevel():
    # Get the input values from the request parameters
    input_values = request.args.get('input_values')
    print(input_values)

    # Convert the input values to a list of floats
    input_values = input_values.split(',')
    print(input_values)
    input_values = [float(value) for value in list(input_values)]

    # Use the machine learning model to predict a value based on the input values
    prediction = model_depress.predict([input_values])[0]

    # Return the predicted value as a JSON object
    return jsonify({'prediction': prediction})


# In[3]:


# Define a route that responds to GET requests for anxiety
@app.route('/predictAnxietyLevel', methods=['GET'])
def predictAnxietyLevel():
    # Get the input values from the request parameters
    input_values = request.args.get('input_values')
    print(input_values)

    # Convert the input values to a list of floats
    input_values = input_values.split(',')
    print(input_values)
    input_values = [float(value) for value in list(input_values)]

    # Use the machine learning model to predict a value based on the input values
    prediction = model_anx.predict([input_values])[0]

    # Return the predicted value as a JSON object
    return jsonify({'prediction': prediction})


# In[4]:


# Define a route that responds to GET requests for stress
@app.route('/predictStressLevel', methods=['GET'])
def predictStressLevel():
    # Get the input values from the request parameters
    input_values = request.args.get('input_values')
    print(input_values)

    # Convert the input values to a list of floats
    input_values = input_values.split(',')
    print(input_values)
    input_values = [float(value) for value in list(input_values)]

    # Use the machine learning model to predict a value based on the input values
    prediction = model_stress.predict([input_values])[0]

    # Return the predicted value as a JSON object
    return jsonify({'prediction': prediction})


# In[5]:


# Define a route that responds to GET requests for all
@app.route('/predictAll', methods=['GET'])
def predictAll():
    # Get the input values from the request parameters
    input_values = request.args.get('input_values')
    print(input_values)

    # Convert the input values to a list of floats
    input_values = input_values.split(',')
    print(input_values)
    input_values = [float(value) for value in list(input_values)]

    # Use the machine learning model to predict a value based on the input values
    prediction_depress = model_depress.predict([input_values])[0]
    prediction_anx = model_anx.predict([input_values])[0]
    prediction_stress = model_stress.predict([input_values])[0]

    # Return the predicted value as a JSON object
    return jsonify({
        'depression_level': prediction_depress,
        'anxiety_level': prediction_anx,
        'stress_level': prediction_stress
    })


# In[ ]:


# Start the application
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')


# In[ ]:




