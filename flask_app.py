import os
import io
import base64
import time
import datetime
import dill as pickle
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import models.gridCV.model_pipeline as model_gridCV
from pmdarima.arima import auto_arima

import flask
#import werkzeug
from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("dark")

main_path = str(os.getcwd())

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
	return flask.render_template('train.html')

@app.route('/evaluate/')
def evaluate():
    return flask.render_template('evaluate.html')

@app.route('/market_predictor/')
def market_predictor():
    return flask.render_template('market_predictor.html')

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

@app.route('/profiling/')
def profiling():
    return flask.render_template('profiling.html')

@app.route('/timeseries/')
def timeseries():
    return flask.render_template('timeseries.html')


@app.route('/profiling', methods=['POST'])
def get_profiling():
	if flask.request.method=='POST':
		import pandas_profiling
		data_file = flask.request.files['dataset']
		data = data_file.read()
		data = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		
		profile = pandas_profiling.ProfileReport(data)
		profile.to_file(outputfile=main_path+"/templates/"+"profiling_report.html")
		return flask.render_template("profiling_report.html")


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

@app.route('/market_predictor', methods=['POST'])
def get_market_predictor():
	if flask.request.method=='POST':
		stock_symbol = str(flask.request.form['StockSymbol'])
		#name = 'AMZN'
		end = datetime.datetime.today()
		start = end-datetime.timedelta(days=365)
		df = web.DataReader(stock_symbol,'iex',start=start,end=end)
		
		df.rename(index=str, columns={"close": "Close"}, inplace=True)
		df = df["Close"].to_frame()
		idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
		df.index = pd.DatetimeIndex(df.index)
		df = df.reindex(idx, fill_value=np.nan)
		df = df.interpolate(method='linear')
		df.dropna(inplace=True)

		stepwise_model = auto_arima(y=df, start_p=1, start_q=1,
                           max_p=7, max_q=7, max_d=7, max_order=14,
                           m=7, start_P=0, seasonal=True,
                           d=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

		stepwise_model.fit(df)

		n_periods=30
		future_forecast = stepwise_model.predict(n_periods=n_periods)
		future_forecast = pd.DataFrame(future_forecast,
		                               index = pd.date_range(start=df.index[-1], end=None, periods=n_periods, freq="D"),
		                               columns=['Prediction'])
		ds = pd.concat([df.tail(n_periods*4),future_forecast],axis=1)

		plt.figure(figsize=(10,6))
		plt.plot(list(ds.index), list(ds['Close'].values))
		plt.plot(list(ds.index), list(ds['Prediction'].values))
		plt.xlabel('Date')
		plt.ylabel('Stock Price')
		plt.title(f"Dataset features by importance \n Stock: {stock_symbol} \n 30 day prediction")
		# 5. Save and render
		img = io.BytesIO()
		plt.savefig(img, format='png')
		img.seek(0)
		plot_url = base64.b64encode(img.getvalue()).decode()
		return '<center><img src="data:image/png;base64,{}"></center>'.format(plot_url)


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
		
		file_path_name = f"{main_path}/data/{ml_type}_{y}.pk"
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


@app.route('/timeseries', methods=['POST'])
def get_timeseries():

	if flask.request.method=='POST':

		# 1. Get and clean dataset
		y = str(flask.request.form['target_var'])
		
		data_file = flask.request.files['dataset']
		data = data_file.read()
		df = pd.read_csv(io.BytesIO(data), encoding='utf-8', sep=",")
		df = df.select_dtypes(include=[np.number]).copy()
		df = df.dropna().astype(float)

		### 2. Get y, X fields
		#print("target: ", y)
		#X = [x for x in df.columns if x != y]
		#df_X = df[X].copy()
		#df_X['Random'] = np.random.randint(1, 6, df_X.shape[0])
		#for col in df_X.columns:
		#	if col in ["PassengerId", "index", "Unnamed: 0"]:
		#		df_X.drop(col, axis=1, inplace=True)

		#######------------------------------------------------ Vydia
		# https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/
		#df = pd.read_csv('/home/pedro/Downloads/train_timeseries.csv')

		#Subsetting the dataset
		#Index 11856 marks the end of year 2013
		from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
		import statsmodels.api as sm
		
		df = pd.read_csv('/home/pedro/Downloads/train_timeseries.csv', nrows = 11856)

		#Creating train and test set 
		#Index 10392 marks the end of October 2013 
		train=df[0:10392] 
		test=df[10392:]

		#Aggregating the dataset at daily level
		df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
		df.index = df.Timestamp 
		df = df.resample('D').mean()
		train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
		train.index = train.Timestamp 
		train = train.resample('D').mean() 
		test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
		test.index = test.Timestamp 
		test = test.resample('D').mean()


		#----
		y_hat_avg = test.copy()
		fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
		y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
		plt.figure(figsize=(16,8))
		plt.plot( train['Count'], label='Train')
		plt.plot(test['Count'], label='Test')
		plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
		plt.legend(loc='best')
		#######------------------------------------------------ Vydia


		
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


###################################
###################################

#----------------------
# for deployment, pass app.server (which is the actual flask app) to WSGI etc
dash_app1 = dash.Dash(__name__, server = app, url_base_pathname='/dashboard/' )

dash_app1.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])
#----------------------

#dash_app2 = dash.Dash(__name__, server = server, url_base_pathname='/reports/')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app2 = dash.Dash(__name__, server = app, url_base_pathname='/reports/', external_stylesheets=external_stylesheets)

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/' +
    '5d1ea79569ed194d432e56108a04d188/raw/' +
    'a9f9e8076b837d541398e999dcbac2b2826a81f8/'+
    'gdp-life-exp-2007.csv')


dash_app2.layout = html.Div([
    dcc.Graph(
        id='life-exp-vs-gdp',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['continent'] == i]['gdp per capita'],
                    y=df[df['continent'] == i]['life expectancy'],
                    text=df[df['continent'] == i]['country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.continent.unique()
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                yaxis={'title': 'Life Expectancy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

#---------------------------------------------
external_stylesheets_3 = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#dash_app3 = dash.Dash(__name__, external_stylesheets=external_stylesheets)
dash_app3 = dash.Dash(__name__, server = app, url_base_pathname='/reports_scatter/', external_stylesheets=external_stylesheets_3)

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/'
    'cb5392c35661370d95f300086accea51/raw/'
    '8e0768211f6b747c0db42a9ce9a0937dafcbd8b2/'
    'indicators.csv')

available_indicators = df['Indicator Name'].unique()

dash_app3.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Fertility rate, total (births per woman)'
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Life expectancy at birth, total (years)'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic'),

    dcc.Slider(
        id='year--slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].max(),
        marks={str(year): str(year) for year in df['Year'].unique()}
    )
])

@dash_app3.callback(
    dash.dependencies.Output('indicator-graphic', 'figure'),
    [dash.dependencies.Input('xaxis-column', 'value'),
     dash.dependencies.Input('yaxis-column', 'value'),
     dash.dependencies.Input('xaxis-type', 'value'),
     dash.dependencies.Input('yaxis-type', 'value'),
     dash.dependencies.Input('year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Year'] == year_value]

    return {
        'data': [go.Scatter(
            x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

#-------------------------------------------------------------

#dash_app2.layout = html.Div([html.H1('Hi there, I am app2 for reports')])
#dash_app1.layout = html.Div([html.H1('Hi there, I am app1 for dashboards')])

#@app.route('/timeseries/')
#def timeseries():
#    return flask.render_template('timeseries.html')

@app.route('/dashboard/')
def render_dashboard():
    return flask.redirect('/dash1')


@app.route('/reports/')
def render_reports():
    return flask.redirect('/dash2')

@app.route('/reports_scatter/')
def render_reports_scatter():
    return flask.redirect('/dash3')


application = DispatcherMiddleware(app, {
    '/dash1': dash_app1.server,
    '/dash2': dash_app2.server,
    '/dash3': dash_app3.server,
})

if __name__ == '__main__':
	#application.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
	run_simple('0.0.0.0', 8080, application, use_reloader=True, use_debugger=True)

