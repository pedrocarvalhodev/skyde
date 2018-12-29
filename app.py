import io
import argparse
import flask
from flask import Flask, request, render_template
import dill as pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib


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
		print(data_file)
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")

		# 2. Get model
		model_file = request.files['model']
		
		#model = joblib.load(model_file)
		# Pratos pickle
		model = pickle.load(model_file)

		# 2. Predict and save results
		prediction = model.predict(data)
		prediction_output = pd.DataFrame(prediction).reset_index(drop=False)
		prediction_output.columns = ["ID", "y_hat"]
		#output_path = f"data/{args.ml}/prediction_results.csv"
		output_path = f"data/gridCV/prediction_results.csv"
		
		prediction_output.to_csv(output_path, index=False)
		print(output_path, prediction_output.head())

		# 3. Render results from prediction method
		return render_template('index.html', label="Prediction processed. Check folder for results.")


@app.route('/var_importance', methods=['POST'])
def get_var_importance():
	if request.method=='POST':
		data_file = request.files['dataset']
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		data = data.head()

		return render_template('index.html',  tables=[data.to_html(classes='data')], titles=data.columns.values)


if __name__ == '__main__':
	from models.gridCV.model_pipeline import PreProcessing, FeatEngineering, FeatSelection
	
	#parser = argparse.ArgumentParser()
	#parser.add_argument("-ml")
	#args = parser.parse_args()
	
	app.run(host='0.0.0.0', port=8000, debug=True)
