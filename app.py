import io
import base64

import argparse
import flask
from flask import Flask, request, render_template
import dill as pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("dark")

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	return flask.render_template('index.html')

@app.route('/about/')
def about():
    return flask.render_template('about.html')

@app.route('/report/')
def report():
    return flask.render_template('report.html')

@app.route('/train/')
def train():
    return flask.render_template('train.html')

@app.route('/evaluate/')
def evaluate():
    return flask.render_template('evaluate.html')

@app.route('/predict/')
def predict():
    return flask.render_template('predict.html')

@app.route('/viz_dataset/')
def viz_dataset():
    return flask.render_template('viz_dataset.html')

@app.route('/var_importance/')
def var_importance():
    return flask.render_template('var_importance.html')


@app.route('/train', methods=['POST'])
def run_model_train():
	if request.method=='POST':
		data_file = request.files['dataset']
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		#data = data.head()
		## model_pipeline (train.csv) -> download pickle model

		return render_template('train.html',  tables=[data.to_html(classes='data')], titles=data.columns.values)


@app.route('/evaluate', methods=['POST'])
def evaluate_model():
	if request.method=='POST':
		y_test = request.files['y_test']
		y_test = y_test.read()
		y_test = pd.read_csv(io.BytesIO(y_test), encoding='utf-8', sep=",")

		y_hat = request.files['y_hat']
		y_hat = y_hat.read()
		y_hat = pd.read_csv(io.BytesIO(y_hat), encoding='utf-8', sep=",")

		# y_test (train), y_hat (predicted results)
		y_test = y_test.reset_index(drop=False)
		y_test =y_test[["index","Survived"]].copy()

		res = y_test.merge(y_hat, left_on="index", right_on="ID", how="inner")
		res = res[["ID","Survived", "y_hat"]]

		res_table = res.groupby(["Survived", "y_hat"]).ID.count().reset_index(drop=False)
		res_table["perc"] = np.around(res_table.ID / res_table.ID.sum() * 100,1)

		return render_template('evaluate.html',  tables=[res_table.to_html(classes='res_table')], titles=res_table.columns.values)


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
		return render_template('predict.html', label="Prediction processed. Check folder for results.")


@app.route('/viz_dataset', methods=['POST'])
def get_viz_dataset():
	if request.method=='POST':
		data_file = request.files['dataset']
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		data = data.head()

		return render_template('viz_dataset.html',  tables=[data.to_html(classes='data')], titles=data.columns.values)


@app.route('/var_importance', methods=['POST'])
def get_var_importance():

	if request.method=='POST':

		# 1. Get and clean dataset
		data_file = request.files['dataset']
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		df = data.drop("PassengerId", axis=1)
		df = df.select_dtypes(include=[np.number]).copy()
		df = df.dropna().astype(float)

		# 2. Get y, X fields
		y = str(request.form['target_var'])
		print("target: ", y)
		X = [x for x in df.columns if x != y]
		df_X = df[X].copy()
		df_X['randomVar'] = np.random.randint(1, 6, df_X.shape[0])

		# 3. Run Variable Importance
		clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
		clf = clf.fit(df_X, df[y])
		features = pd.DataFrame()
		features['feature'] = df_X.columns
		features['importance'] = clf.feature_importances_
		features.sort_values(by=['importance'], ascending=True, inplace=True)
		features = features.sort_values(by="importance", ascending=True).reset_index(drop=False)
		features = features.head()
		
		# 4. Define viz
		plt.barh(list(features['feature'].values), list(features['importance'].values))
		plt.xlabel('Performance')
		plt.ylabel('Top Features \n Descending order')
		plt.title('How fast do you want to go today?')

		# 5. Save and render
		img = io.BytesIO()
		plt.savefig(img, format='png')
		img.seek(0)
		plot_url = base64.b64encode(img.getvalue()).decode()
		return '<img src="data:image/png;base64,{}">'.format(plot_url)


if __name__ == '__main__':
	from models.gridCV.model_pipeline import PreProcessing, FeatEngineering, FeatSelection
	
	app.run(host='0.0.0.0', port=8000, debug=True)
