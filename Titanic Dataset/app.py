import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
from sklearn import preprocessing
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_features = [x for x in request.form.values()]
    final_features = [np.array(float_features)]
    #final_features = pd.DataFrame(final_features)
    #scaler = preprocessing.MinMaxScaler()
    #X_tr = scaler.fit_transform(final_features)
    #X_tr = pd.DataFrame(X_tr)

    prediction = model.predict(final_features)
    
    #output = round(prediction[0],0)
    if (prediction == 1):
        a = "Survived"
    elif (prediction == 0):
        a = "Died"
    return render_template('index.html',prediction_text="Person {}".format(str(a)))
    


if (__name__ == "__main__"):
    
    app.run(debug=True)
