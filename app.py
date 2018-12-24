import io

import flask
from flask import Flask, request, render_template
import dill as pickle
import pandas as pd
from sklearn.externals import joblib

## Choose model
#from models.pratos_base_model import PreProcessing
from models.pratos_reg_model import PreProcessing

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':
		# 1. Get data from request 
		data_file = request.files['dataset']
		data = data_file.read()
		dat = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")

		# 2. Get model
		model_file = request.files['model']
		
		#model = joblib.load(model_file)
		# Pratos pickle
		model = pickle.load(model_file)

		# 2. Predict and save results
		prediction = model.predict(dat)
		prediction_output = pd.DataFrame(prediction).reset_index(drop=False)
		prediction_output.columns = ["ID", "y_hat"]
		prediction_output.to_csv("data/pratos_flask_api/prediction_results.csv", index=False)
		print(prediction_output.head())

		# 3. Render results from prediction method
		return render_template('index.html', label="Prediction processed. Check folder for results.")


if __name__ == '__main__':
	
	#model = joblib.load('models/model.pkl')
	
	app.run(host='0.0.0.0', port=8000, debug=True)
