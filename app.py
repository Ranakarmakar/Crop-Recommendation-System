from flask import Flask, render_template, request
import os
import numpy as np
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
port = int(os.getenv('PORT', 5000))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class_names = ['Rice', 'Maize', 'Chickpea', 'Kidney-beans', 'Pigeon-peas',
               'Moth beans', 'Mung-bean', 'Black gram', 'Lentil', 'Pomegranate',
               'Banana', 'Mango', 'Grapes', 'Watermelon', 'Muskmelon', 'Apple',
               'Orange', 'Papaya', 'Coconut', 'Cotton', 'Jute', 'Coffee']

model = load_model('crop_ann_model_.h5')
#model = pickle.load(open("gnb_model.pkl", 'rb'))


@app.route('/', methods=['GET'])
def hello_world():
    return render_template("home.html")


@app.route('/mai', methods=['GET', 'POST'])
def mai():
    if request.method == 'POST':
        N = request.form["N"]
        P = request.form["P"]
        K = request.form["K"]
        Temperature = request.form["Temperature"]
        humidity = request.form["humidity"]
        ph = request.form["ph"]
        rainfall = request.form["rainfall"]
        lst = [N, P, K, Temperature, humidity, ph, rainfall]
        for i in range(0, len(lst)):
            lst[i] = int(lst[i])
        prediction = model.predict([lst])
        result = "Recommended Crop is {} with {:.2f}% Confidence. ".format(class_names[np.argmax(prediction)],
                                                                           100 * np.max(prediction))
    else:
        result = " "

    return render_template("mai.html", prediction=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
    #app.run()
