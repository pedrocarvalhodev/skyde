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
		output_path = f"data/{args.ml}/prediction_results.csv"
		
		prediction_output.to_csv(output_path, index=False)
		print(output_path, prediction_output.head())

		# 3. Render results from prediction method
		return render_template('index.html', label="Prediction processed. Check folder for results.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-ml")
	args = parser.parse_args()
	#print(args.ml)

	#if args.ml == "random_forest_classifier":
	#	from models.random_forest_classifier import PreProcessing
	#	print("imported random_forest_classifier")
	#if args.ml == "random_forest_regressor":
	#	from models.random_forest_regressor import PreProcessing
	#	print("imported random_forest_regressor")
	#if args.ml == "gridCV":
	from models.gridCV.model_pipeline import PreProcessing, FeatEngineering, FeatSelection
	#print("imported gridCV")
	app.run(host='0.0.0.0', port=8000, debug=True)
