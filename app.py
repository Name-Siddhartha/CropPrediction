from flask import Flask, render_template, request
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
 
app = Flask(__name__)
 
 
@app.route('/')
def index():
    return render_template('home.html', title = 'Home')
    # return render_to_response('yield_pred.html')
 
 
@app.route('/predict',methods=['GET','POST'])
def predict():
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
 
        data =[[float(disname),float(sea),float(are),float(temp),float(ph),float(rn),float(p),float(n),float(k)]]
 
        lr = pickle.load(open('RandomForest.pkl', 'rb'))
        prediction = lr.predict(data)[0]
 
    return render_template('crop.html', prediction=prediction, title="Crop Prediction")
 

   
@app.route('/yield_predict', methods=['GET', 'POST'])
def yield_predict():
 
    if request.method == 'POST':
 
        disname = request.form['disname']
        sea = request.form['sea']
        are = request.form['are']
        temp = request.form['temp']
       
        rn = request.form['rn']
        sow = request.form['sow']
        harvest = request.form['harvest']
        crop = request.form['crop']
        year = request.form['year']

        
        
 
        data =[[float(disname),float(sea),float(are),float(temp),float(rn),float(sow),float(harvest),float(crop),float(year)]]
 
        lr = pickle.load(open('yield.pkl', 'rb'))
        prediction = lr.predict(data)[0]
 
    return render_template('yield.html', prediction=prediction, title = 'Yield Estimation')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', title = 'About Us')
    # return render_to_response('yield_pred.html')
 
if __name__ == '__main__':
    app.run()