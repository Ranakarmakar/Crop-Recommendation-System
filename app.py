from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)
port = int(os.getenv('PORT', 8000))
loaded_model = pickle.load(open('model.pkl', 'rb'))
class_names = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
               'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
               'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
               'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']


@app.route('/', methods=['GET'])
def hello_world():
    return render_template("home.html")


@app.route('/', methods=["POST"])
def predict():
    N = request.form["N"]
    P = request.form["P"]
    K = request.form["K"]
    Temperature = request.form["Temperature"]
    humidity = request.form["humidity"]
    ph = request.form["ph"]
    rainfall = request.form["rainfall"]
    print(loaded_model.summary())
    print(N, P, K, Temperature, humidity, ph, rainfall)
    prediction = loaded_model.predict([[N, P, K, Temperature, humidity, ph, rainfall]])
    result = "Recommended Crop is {} with a {:.2f}% Confidence. ".format(class_names[np.argmax(prediction)],
                                                                         100 * np.max(prediction))

    return render_template("home.html", prediction=result)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)
