import flask
from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple

#from dash import Dash
import dash
import dash_core_components as dcc
import dash_html_components as html

server = flask.Flask(__name__)


#----------------------
# for deployment, pass app.server (which is the actual flask app) to WSGI etc
dash_app1 = dash.Dash(__name__, server = server, url_base_pathname='/dashboard' )

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

dash_app2 = dash.Dash(__name__, server = server, url_base_pathname='/reports')
dash_app2.layout = html.Div([html.H1('Hi there, I am app2 for reports')])
#dash_app1.layout = html.Div([html.H1('Hi there, I am app1 for dashboards')])

@server.route('/')
@server.route('/hello')
def hello():
    return 'hello world!'

@server.route('/dashboard')
def render_dashboard():
    return flask.redirect('/dash1')


@server.route('/reports')
def render_reports():
    return flask.redirect('/dash2')

app = DispatcherMiddleware(server, {
    '/dash1': dash_app1.server,
    '/dash2': dash_app2.server
})

run_simple('0.0.0.0', 8080, app, use_reloader=True, use_debugger=True)