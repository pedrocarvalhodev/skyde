import os
import io
import base64
import flask
import dill as pickle
import numpy as np
import pandas as pd
import models.gridCV.model_pipeline as model_gridCV

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("dark")

data_path = str(os.getcwd()) + "/data/"

app = flask.Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
	return flask.render_template('home.html')

@app.route('/about/')
def about():
    return flask.render_template('about.html')

@app.route('/report/')
def report():
    return flask.render_template('report.html')

@app.route('/train/')
def train():
	#ml_types = ['Regressor', 'Classifier']
	return flask.render_template('train.html')#, ml_types=ml_types)

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

@app.route('/features/')
def features():
    return flask.render_template('features.html')


@app.route('/evaluate', methods=['POST'])
def evaluate_model():
	if flask.request.method=='POST':
		data = flask.request.files['test']
		y_test = data.read()
		y_test = pd.read_csv(io.BytesIO(y_test), encoding='utf-8', sep=",")

		y_hat = flask.request.files['y_hat']
		y_hat = y_hat.read()
		y_hat = pd.read_csv(io.BytesIO(y_hat), encoding='utf-8', sep=",")

		# y_test (train), y_hat (predicted results)
		y = str(flask.request.form['target_var'])
		y_test = y_test.reset_index(drop=False)
		y_test =y_test[["index",y]].copy()

		# 3. Run Variable Importance
		ml_type = str(flask.request.form['ml_type'])
		if ml_type in ["Classifier", "LogisticRegression"]:
			# Merge test & predicted, return confusion matrix
			res = y_test.merge(y_hat, left_on="index", right_on="ID", how="inner")
			res = res[["ID",y, "y_hat"]]

			res_table = res.groupby([y, "y_hat"]).ID.count().reset_index(drop=False)
			res_table["perc"] = np.around(res_table.ID / res_table.ID.sum() * 100,1)
		
		if ml_type == "Regressor":
			res = y_test.merge(y_hat, left_on="index", right_on="ID", how="inner")
			res = res[["ID",y, "y_hat"]]
			rmse = ((res["y_hat"] - res[y]) ** 2).mean() ** .5
			res_table = pd.DataFrame({"rmse":rmse}, index=["evaluation result"])

		resp = flask.make_response(res_table.to_csv())
		resp.headers["Content-Disposition"] = "attachment; filename=model_evaluation.csv"
		resp.headers["Content-Type"] = "text/csv"
		return resp


@app.route('/predict', methods=['POST'])
def make_prediction():
	if flask.request.method=='POST':
		# 1. Get data from request 
		data_file = flask.request.files['dataset']
		print(data_file)
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")

		# 2. Get model
		model_file = flask.request.files['model']
		model = pickle.load(model_file)

		# 3. Predict and save results
		prediction = model.predict(data)
		prediction_output = pd.DataFrame(prediction).reset_index(drop=False)
		prediction_output.columns = ["ID", "y_hat"]
		
		resp = flask.make_response(prediction_output.to_csv())
		resp.headers["Content-Disposition"] = "attachment; filename=prediction_output.csv"
		resp.headers["Content-Type"] = "text/csv"
		return resp


@app.route('/viz_dataset', methods=['POST'])
def get_viz_dataset():
	if flask.request.method=='POST':
		data_file = flask.request.files['dataset']
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		data = data.head()

		return flask.render_template('viz_dataset.html',  tables=[data.to_html(classes='data')], titles=data.columns.values)


@app.route('/train', methods=['POST'])
def train_model():
	if flask.request.method=='POST':
		data_file = flask.request.files['train_dataset']
		data = data_file.read()
		train = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		y = str(flask.request.form['target_var'])
		ml_type = str(flask.request.form['ml_type'])

		model = model_gridCV.ml_pipeline(train=train, target=y, ml_type=ml_type)
		
		file_path_name = f"{data_path}{ml_type}_{y}.pk"
		with open(file_path_name, 'wb') as file:
			pickle.dump(model, file)

		resp = flask.Response(open(file_path_name, 'rb'))
		resp.headers["Content-Disposition"] = "attachment; filename=model.pk"
		return resp


@app.route('/features', methods=['POST'])
def features_model():
	if flask.request.method=='POST':
		data_file = flask.request.files['train_dataset']
		data = data_file.read()

		train = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		y = str(flask.request.form['target_var']) # Dropdown list

		df_feat = model_gridCV.ml_pipeline(train=train, target=y, ml_type="Features")
		
		resp = flask.make_response(df_feat.to_csv())
		resp.headers["Content-Disposition"] = "attachment; filename=df_features_transform.csv"
		resp.headers["Content-Type"] = "text/csv"
		return resp


@app.route('/var_importance', methods=['POST'])
def get_var_importance():

	if flask.request.method=='POST':

		# 1. Get and clean dataset
		data_file = flask.request.files['dataset']
		data = data_file.read()
		df = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		df = df.select_dtypes(include=[np.number]).copy()
		df = df.dropna().astype(float)

		# 2. Get y, X fields
		y = str(flask.request.form['target_var'])
		print("target: ", y)
		X = [x for x in df.columns if x != y]
		df_X = df[X].copy()
		df_X['Random'] = np.random.randint(1, 6, df_X.shape[0])

		for col in df_X.columns:
			if col in ["PassengerId", "index", "Unnamed: 0"]:
				df_X.drop(col, axis=1, inplace=True)

		# 3. Run Variable Importance
		ml_type = str(flask.request.form['ml_type'])
		if ml_type == "Classifier":
			clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
			clf = clf.fit(df_X, df[y])
		elif ml_type == "Regressor":
			clf = RandomForestRegressor(n_estimators=50, max_features='sqrt')
			clf = clf.fit(df_X, df[y])
		else:
			print("Warning. ML options error")
			return None
		features = pd.DataFrame()
		features['feature'] = df_X.columns
		features['importance'] = clf.feature_importances_
		features.sort_values(by=['importance'], ascending=True, inplace=True)
		features = features.sort_values(by="importance", ascending=True).reset_index(drop=False)
		features = features.head(10)
		
		# 4. Define viz
		plt.figure(figsize=(10,6))
		plt.barh(list(features['feature'].values), list(features['importance'].values))
		plt.xlabel('Relative Importance')
		plt.ylabel('Top Features \n Descending order')
		plt.title(f"Dataset features by importance \n Target: {y} \n ML method: {ml_type}")

		# 5. Save and render
		img = io.BytesIO()
		plt.savefig(img, format='png')
		img.seek(0)
		plot_url = base64.b64encode(img.getvalue()).decode()
		return '<center><img src="data:image/png;base64,{}"></center>'.format(plot_url)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
