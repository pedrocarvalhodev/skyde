import flask
from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple

#from dash import Dash
import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import plotly.graph_objs as go

server = flask.Flask(__name__)


#----------------------
# for deployment, pass app.server (which is the actual flask app) to WSGI etc
dash_app1 = dash.Dash(__name__, server = server, url_base_pathname='/dashboard/' )

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
dash_app2 = dash.Dash(__name__, server = server, url_base_pathname='/reports/', external_stylesheets=external_stylesheets)

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

#dash_app2.layout = html.Div([html.H1('Hi there, I am app2 for reports')])
#dash_app1.layout = html.Div([html.H1('Hi there, I am app1 for dashboards')])

@server.route('/')
@server.route('/hello/')
def hello():
    return 'hello world!'

@server.route('/dashboard/')
def render_dashboard():
    return flask.redirect('/dash1')


@server.route('/reports/')
def render_reports():
    return flask.redirect('/dash2')

app = DispatcherMiddleware(server, {
    '/dash1': dash_app1.server,
    '/dash2': dash_app2.server
})

run_simple('0.0.0.0', 8080, app, use_reloader=True, use_debugger=True)