from flask import Flask, render_template, request, url_for

from sklearn import svm, model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression

import os
import pickle

app = Flask(__name__)

# this says inside static folder image folder, to set path
PEOPLE_FOLDER = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

formYield = [
    ["This", "Stuff"],
    ["That", "Another Stuff"],
    ["There", "Yet Another Stuff"]
]


@app.route('/')
def index():
    # return render_template('home.html', title = 'Home')
    # return render_to_response('yield_pred.html')
    # this is to get image file
    full_filename = os.path.join(
        "\\", app.config['UPLOAD_FOLDER'], 'back.jpg')
    # this to display
    return render_template("home.html", user_image=full_filename, formPairs=formYield, title='Home')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    yield2=None
    if request.method == 'POST':

        disname = request.form['disname']
        sea = request.form['sea']
        are = request.form['are']
        temp = request.form['temp']
        ph = request.form['ph']
        rn = request.form['rn']
        p = request.form['p']
        n = request.form['n']
        k = request.form['k']

        data = [[float(disname), float(sea), float(are), float(
            temp), float(ph), float(rn), float(p), float(n), float(k)]]

        crop_model = pickle.load(open('crop.pkl', 'rb'))
        
        prediction = crop_model.predict(data)[0]
        prediction=0
        harvest = 123
        sow=12
        year=1999
        data = [[float(disname), float(sea), float(are), float(temp), float(
            rn), float(sow), float(harvest), float(prediction), float(year)]]
       
        yield_model = pickle.load(open('yield.pkl', 'rb'))
        
        
        yield2=yield_model.predict(data)[0]
    return render_template('crop.html',yield2=yield2, prediction=prediction, title="Crop Prediction")


@app.route('/yield_predict', methods=['GET', 'POST'])
def yield_predict():
    prediction = None
    if request.method == 'POST':

        disname = request.form['disname']
        sea = request.form['sea']
        are = request.form['are']
        temp = request.form['temp']
        ph = request.form['ph']
        rn = request.form['rn']
        p = request.form['p']
        n = request.form['n']
        k = request.form['k']
        crop = request.form['crop']

        data = [[float(disname), float(sea), float(are), float(temp), float(
            rn), float(sow), float(harvest), float(crop), float(year)]]

        lr = pickle.load(open('yield.pkl', 'rb'))
        prediction = lr.predict(data)[0]

    return render_template('yield.html', prediction=prediction, title='Yield Estimation')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', title='About Us')
    # return render_to_response('yield_pred.html')


if __name__ == '__main__':
    app.run(debug=True)
