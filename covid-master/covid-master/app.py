import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    final_features = pd.DataFrame(final_features, columns=['age','gender','fever','coughing','d','s','t'])

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output<0.5:
        return render_template('index.html', prediction_text='May be, there is no possibility')
    else:
        return render_template('index.html', prediction_text='May be,there is a possibility')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)