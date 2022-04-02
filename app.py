from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from keras.models import load_model

app = Flask(__name__)
port = int(os.getenv('PORT', 8000))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# model = pickle.load(open('XGB_model.pkl', 'rb'))
class_names = ['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas',
               'Mothbeans', 'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate',
               'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple',
               'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee']

model = load_model('crop_ann_model_.h5')


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
    # print(model.summary())
    lst = [N, P, K, Temperature, humidity, ph, rainfall]
    print(lst)
    for i in range(0, len(lst)):
        lst[i] = int(lst[i])
    prediction = model.predict([lst])
    # prediction = model.predict([[90,	42,	43,	20.879744,	82.002744,	6.502985,	202.935536]])
    result = " {} with {:.2f}%  ".format(class_names[np.argmax(prediction)],
                                         100 * np.max(prediction))

    return render_template("home.html", prediction=result)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, debug=True)
    app.run(debug=True)
