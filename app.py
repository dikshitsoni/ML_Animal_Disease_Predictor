from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('rfc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("inner-page.html")

@app.route('/submit', methods=['POST'])
def submit():
    input_feature = [int(x) for x in request.form.values()]
    input_feature = [np.array(input_feature)]
    
    names = ['AnimalName', 'symptoms1', 'symptoms2',
             'symptoms3', 'symptoms4', 'symptoms5']
    
    data = pd.DataFrame(input_feature, columns=names)

    prediction = model.predict(data)
    prediction = int(prediction[0])
    
    if prediction == 0:
        result = "According to our study, we feel sad to inform this is a dangerous condition."
    else:
        result = "Your animal is in a normal condition. No danger detected."

    return render_template("output.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
