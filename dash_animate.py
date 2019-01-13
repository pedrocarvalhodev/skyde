import dash
from dash.dependencies import Input, Output, Event, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime
from scipy import integrate
from itertools import cycle


app = dash.Dash()
interval_state = 1000
app.layout = html.Div([
    dcc.Markdown('''

Enter Number of trajectories (e.g. 1 or 2 or 3 or etc...):
'''),
    dcc.Input(
        id='n_traject_input',
        placeholder='Enter a value',
        type='text',
        value= '5'
    ),
    dcc.Interval(
        id='interval-element',
        interval=interval_state,
        n_intervals=0,
        max_intervals=10
    ),
    dcc.Markdown('''

Select frame speed:
'''),
    dcc.RadioItems(id='set-time',
        value=1000,
        options=[
            {'label': '1x', 'value': 1000},
            {'label': '2x', 'value': 500},
            {'label': '4x', 'value': 250},
            {'label': '10x', 'value': 50},
            {'label': 'Off', 'value': 60*60*1000} # or just every hour
        ]),
    dcc.Markdown('''

Epoch state:
'''),
    html.Div(id='display-time'),
    html.Button('Reset animation!', id='button'),

    html.Div([
    dcc.Graph(id='graph-with-slider', animate=True, animation_options = {'frame':{'duration': 0,'redraw': False},
                                                'layout': {'scene': {'xaxis': {'range': [-35, 35], 'autorange': False},
                                                                    'yaxis': {'range': [-40, 40], 'autorange': False},
                                                                    'zaxis': {'range': [-35, 75], 'autorange': False}}},
                                                'transition': {'duration': 0,'ease': 'cubic-in-out'}})
        ], style={'width': 800}),
    html.Div([
        dcc.Slider(
            id='time-slider',
            min=0,
            max=1000
            )
        ], style={'width': 700}),
])

@app.callback(Output('graph-with-slider', 'figure'), [Input('n_traject_input', 'value'),
                                                        Input('time-slider', 'value')],
                                                    [State('graph-with-slider', 'relayoutData')])
def display_data(n_trajectories, n_intervals, relayoutData):
    print(n_intervals)
    x_t, _ = calc_trajectories(int(n_trajectories))
    traces = {'data': [], 'name': str(n_intervals)}
    colors = ['#7FDBFF', '#39CCCC', '#3D9970', '#85144b', '#F012BE', '#B10DC9', 
    '#001f3f', '#0074D9', '#2ECC40', '#01FF70', '#FF4136', '#FFDC00 ','#FF851B', '#AAAAAA']
    for i in range(x_t.shape[0]):
        c = colors[(i + 1) % len(colors)]
        data_dict = {
            'x': x_t[i,:n_intervals+1,0],
            'y': x_t[i,:n_intervals+1,1],
            'z': x_t[i,:n_intervals+1,2],
            'mode': 'lines',
            'line': {'width': 3, 'color': c},
            'name': 'object-'+str(i),
            'type': 'scatter3d',
        }
        traces['data'].append(data_dict)
        data_dict = {
            'x': [x_t[i,n_intervals,0]],
            'y': [x_t[i,n_intervals,1]],
            'z': [x_t[i,n_intervals,2]],
            'mode': 'markers',
            'marker': {'size': 4, 'color': c, 'maxdisplayed': 1},
            'name': 'object-'+str(i),
            'type': 'scatter3d',
            'showlegend': False
        }
        traces['data'].append(data_dict)
    figure = {
        'data': traces,
        'layout': {'scene': {'xaxis': {'range': [-35, 35], 'autorange': False},
                        'yaxis': {'range': [-40, 40], 'autorange': False},
                        'zaxis': {'range': [-35, 75], 'autorange': False},
                        'aspectmode': "cube"
                        },
                'hovermode': 'closest',     
                    }
            }
    return figure

@app.callback(Output('display-time', 'children'), events=[Event('interval-element', 'interval')])
def display_time():
    return str(datetime.datetime.now())

@app.callback(Output('interval-element', 'interval'), [Input('set-time', 'value')])
def update_interval(value):
    return value
    
@app.callback(Output('time-slider', 'value'), [Input('interval-element', 'n_intervals')])
def update_slider_value(n_intervals):
    return n_intervals

@app.callback(Output('interval-element', 'n_intervals'), [Input('button', 'n_clicks')],
                                                events=[Event('n_traject_input', 'change')])
def reset_interval(n_clicks):
    return 0

def lorentz_deriv(data, t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    (x,y,z) = data
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def calc_trajectories(n_trajectories):
    """
    Choose random starting points, uniformly distributed from -15 to 15
    """
    np.random.seed(1)
    x0 = -15 + 30 * np.random.random((n_trajectories, 3))

    # Solve for the trajectories
    t = np.linspace(0, 4, 1000)
    x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                    for x0i in x0])

    return (x_t, t)

if __name__ == '__main__':
    app.run_server(debug=True)