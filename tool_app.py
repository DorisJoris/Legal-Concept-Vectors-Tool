# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:11:55 2022

@author: bejob
"""

#%% import
import pickle

import pandas as pd
import json

from dash import Dash, dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px

from visualization_data import get_input_visual_dfs




#%% Open pickled input visual_dfs
    
with open("databases/test_database.p", "rb") as pickle_file:
    test_database = pickle.load(pickle_file)      

#%% plotly
external_stylesheets = [dbc.themes.MINTY]
app = Dash(__name__, external_stylesheets=external_stylesheets)



# User input
tab_add_userinput = dbc.Card(dbc.CardBody([
    dcc.Textarea(
        id='user_input_text',
        value='Insert user text here',
        style={'width': '100%', 'height': 200},
    ),
    html.Button('Add', id='user_input_button', n_clicks=0)
    ]))


tab_graph = dbc.Card(dbc.CardBody([
    dcc.Graph(id="graph")
    ]))

tab_table = dbc.Card(dbc.CardBody([
    html.Div(id='table')
    ]))

tab_latex = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col(dbc.Label('Include columns:'),width=2),
        dbc.Col(dcc.Dropdown(id='latex_dropdown',multi=True))
        ]),
    dbc.Card(html.Div(id='latex'))
    ]))

travel_dist_table = dbc.Card(dbc.CardBody([
    html.H4(id='travel_dist_table_title'),
    dbc.Row([
        dbc.Col(html.Div(id='travel_dist_table')),
        dbc.Col(html.Div(id='word_table'))
        ])
    ]))


travel_dist_latex = dbc.Card(dbc.CardBody([
    html.H4(id='travel_dist_latex_title'),
    html.Div(id='travel_dist_latex')
    ]))

travel_dist = dbc.Card(dbc.CardBody([
    dbc.Row([
        dbc.Col(dbc.Label('Focus on:'),width=2),
        dbc.Col(dcc.Dropdown(id='travel_dist_dropdown',multi=False))
        ]),
    dbc.Row([
        dbc.Col(dbc.Label('Distance sorted rank:'),width=2),
        dbc.Col(dcc.Dropdown(options = [5,10,20,30,-30,-20,-10,-5],
                             value = 10,
                             id='travel_dist_range',multi=False))
        ]),
    dbc.Tabs([
        dbc.Tab(travel_dist_table, label='Table'),
        dbc.Tab(travel_dist_latex, label='Latex')
        ])
    ]))


tab_output = dbc.Card([
    dbc.CardHeader([
        dbc.Row([
            dbc.Col(dbc.Label('Select distance measure:'),width=2),
            dbc.Col(dcc.Dropdown(
                id="distance_type",
                options=['concept_vector',
                         'reverse_wmd_concept_bow',
                         'wmd_concept_bow'],
                value='concept_vector',
                multi=False
                )),
            ]),
        dbc.Row([
            dbc.Col(dbc.Label('Select neighbour type:'),width=2),
            dbc.Col(dcc.Dropdown(id='neighbour_dropdown',multi=True))
            ]),
        dbc.Row([
            dbc.Col(dbc.Label('Select point type:'),width=2),
            dbc.Col(dcc.Dropdown(id='document_dropdown',multi=True))
            ])
        ]),
    
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.RadioItems(id='text_selector'), width=6),
            dbc.Col(dbc.Tabs([
                    dbc.Tab(tab_graph, label='Graph'),
                    dbc.Tab(tab_table, label='Table'),
                    dbc.Tab(tab_latex, label='Latex table code'),
                    dbc.Tab(travel_dist, label='Travel distances')
                    ]), 
                width=6)
            ])
        
        ])
    ])


app.layout = dbc.Container([
    dcc.Store(id='visual_dfs'),
    dcc.Store(id='input_dict'),
    
    html.H1('Legal concept system'),
    
    dbc.Tabs([
        dbc.Tab(tab_add_userinput, label="Add user input text"),
        dbc.Tab(tab_output, label = "Output")
        ])
    
])

@app.callback(
    Output('visual_dfs', 'data'),
    Output('input_dict', 'data'),
    Input('user_input_button', 'n_clicks'),
    State('user_input_text', 'value')
)
def update_userdata(n_clicks, value):
    if n_clicks > 0:
        return get_input_visual_dfs(value, test_database)

@app.callback(
    Output('text_selector', 'options'),
    Output('text_selector', 'value'),
    Input('visual_dfs', 'data'),
    Input('input_dict', 'data')
    )
def update_text_selector(visual_dfs, input_dict):
    visual_dfs_list = json.loads(visual_dfs)
    idict = json.loads(input_dict)
    
    number_of_visual_dfs = len(visual_dfs_list)
    
    options = [{'label':'Full Text', 'value':0}]
    
    for i in range(1,number_of_visual_dfs):
        sentence_text = idict['sentence_dicts'][i-1]['full_text']
        options.append({'label':sentence_text,'value':i})
        
    value =0
    
    return options, value


# @app.callback(
#     Output('text_selector', 'value'),
#     Input('text_selector', 'value')
#     )    
# def single_select(value):
#     new_value = value[-1]
#     return [new_value]
    
            
@app.callback(
    Output('neighbour_dropdown','options'),
    Output('neighbour_dropdown','value'),
    Output('document_dropdown','options'),
    Output('document_dropdown','value'),
    Input("distance_type", "value"),
    Input('visual_dfs', 'data'),
    Input('text_selector', 'value'))
def init_multi_dropdowns(dist_option, visual_dfs, value):
    
    dfs = json.loads(visual_dfs)
    df = pd.read_json(dfs[value][dist_option], orient='split')
    neighbour_options = df['Neighbour type'].unique()
    neighbour_values = list()
    
    for option in neighbour_options:
        if 'Input' in option or 'closest' in option:
            neighbour_values.append(option)
    
    doc_options = df['Point type'].unique()
    doc_values = ['Text', 'Vector', 'Legal concept']
    
    return neighbour_options, neighbour_values, doc_options, doc_values

@app.callback(
    Output("table", "children"),
    Input("distance_type", "value"),
    Input('graph', 'clickData'),
    Input('neighbour_dropdown','value'),
    Input('document_dropdown','value'),
    Input('visual_dfs', 'data'),
    Input('text_selector', 'value'))
def update_table(dist_option, clickData, neighbour_options, doc_options, visual_dfs, value):
    dfs = json.loads(visual_dfs)
    df = pd.read_json(dfs[value][dist_option], orient='split')
    
    output_df = df
    output_df = output_df[output_df['Neighbour type'].isin(neighbour_options)]
    output_df = output_df[output_df['Point type'].isin(doc_options)]

    
    table_df = output_df[['Name','Neighbour type','Point type','Distance to input','BoW size']]
    
    if clickData == None:
        return dash_table.DataTable(table_df.to_dict('records'), 
                                  [{"name": i, "id": i} for i in table_df.columns],
                                  style_cell={'textAlign': 'left'},
                                  sort_action="native",
                                  sort_mode="multi"
                                      ) 
    else:
        filter_query = '{Name} = "' +clickData['points'][0]['text'] +'"'
        print(filter_query)
        return dash_table.DataTable(table_df.to_dict('records'), 
                                  [{"name": i, "id": i} for i in table_df.columns],
                                  style_table={'overflowX': 'auto'},
                                  style_cell={'textAlign': 'left'},
                                  sort_action="native",
                                  sort_mode="multi",
                                  style_data_conditional=[{
                                      'if': {
                                          'filter_query': filter_query,
                                          'column_id': 'Name'
                                      },
                                      'backgroundColor': 'tomato',
                                      'color': 'white'
                                  },
                                      {
                                      'if': {
                                          'filter_query': filter_query,
                                          'column_id': 'Neighbour type'
                                      },
                                      'backgroundColor': 'tomato',
                                      'color': 'white'
                                  },
                                      {
                                      'if': {
                                          'filter_query': filter_query,
                                          'column_id': 'Point type'
                                      },
                                      'backgroundColor': 'tomato',
                                      'color': 'white'
                                  },
                                      {
                                      'if': {
                                          'filter_query': filter_query,
                                          'column_id': 'Distance to input'
                                      },
                                      'backgroundColor': 'tomato',
                                      'color': 'white'
                                  }
                              ]
                                      ) 




@app.callback(
    Output("graph", "figure"),
    Input("distance_type", "value"),
    Input('neighbour_dropdown','value'),
    Input('document_dropdown','value'),
    Input('visual_dfs', 'data'),
    Input('text_selector', 'value'))
def update_bar_chart(dist_option, neighbour_options, doc_options, visual_dfs, value):
    dfs = json.loads(visual_dfs)
    df = pd.read_json(dfs[value][dist_option], orient='split')
    
    output_df = df
    output_df = output_df[output_df['Neighbour type'].isin(neighbour_options)]
    output_df = output_df[output_df['Point type'].isin(doc_options)]
    
    fig = px.scatter(output_df, 
                     x="X", y="Y", 
                     text="Name", 
                     hover_data=['Name','Distance to input'], 
                     color= 'Neighbour type',
                     color_discrete_sequence=px.colors.qualitative.Dark24,
                     symbol = 'Point type'
                     )
    fig.update_traces(textposition='top center')
    fig.update_layout(legend={'title_text':''})
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False,
                      xaxis_visible=False, xaxis_showticklabels=False)
    fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0.1)',
                    'legend_bgcolor': 'rgba(0, 0, 0, 0.1)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                        })
    
    
   
    return fig



@app.callback(
    Output('latex_dropdown','options'),
    Output('latex_dropdown','value'),
    Input("distance_type", "value"),
    Input('visual_dfs', 'data'),
    Input('text_selector', 'value'))
def init_latex_dropdown(dist_option, visual_dfs, value):
    dfs = json.loads(visual_dfs)
    df = pd.read_json(dfs[value][dist_option], orient='split')
    
    columns = list(df.columns)

    return columns, columns    

@app.callback(
    Output('latex', 'children'),
    Input("distance_type", "value"),
    Input('neighbour_dropdown','value'),
    Input('document_dropdown','value'),
    Input('latex_dropdown','value'),
    Input('visual_dfs', 'data'),
    Input('text_selector', 'value'))
def display_latex_code(dist_option, neighbour_options, doc_options, columns, visual_dfs, value):
    dfs = json.loads(visual_dfs)
    df = pd.read_json(dfs[value[-1]][dist_option], orient='split')
    
    output_df = df
    output_df = output_df[output_df['Neighbour type'].isin(neighbour_options)]
    output_df = output_df[output_df['Point type'].isin(doc_options)]
    
    output_df = output_df[[c for c in output_df.columns if c in columns]]
    
    latex_code = output_df.to_latex(index=False,
                                    caption=(f'{dist_option}'),
                                    label='tab:add label',
                                    float_format="%.4f")
    
    latex_list = list()
    for line in latex_code.split("\n"):
        latex_list.append(line)
        latex_list.append(html.Br())
    return latex_list


@app.callback(
    Output('travel_dist_dropdown','options'),
    Output('travel_dist_dropdown','value'),
    Input("distance_type", "value"),
    Input('visual_dfs', 'data'),
    Input('text_selector', 'value')
    )
def set_travel_dist_dropdown(dist_option, visual_dfs, value):
    wmd_options = ['reverse_wmd_concept_bow',
                   'wmd_concept_bow']
    
    dfs = json.loads(visual_dfs)
    
    if dist_option in wmd_options:
        output_df = pd.read_json(dfs[value][dist_option], orient='split')
        options = list(output_df[output_df['Point type'] == 'Legal concept'].loc[:,'Neighbour type'])
        
        return options, options[0]
        
    else:
        return ['Not a WMD'], 'Not a WMD'
    
@app.callback(
    Output('travel_dist_table', 'children'),
    Output('travel_dist_latex', 'children'),
    Output('travel_dist_table_title', 'children'),
    Output('travel_dist_latex_title', 'children'),
    Input("distance_type", "value"),
    Input('travel_dist_dropdown','value'),
    Input('travel_dist_range', 'value'),
    Input('input_dict', 'data')
    )

def update_travel_dist_tables(dist_option,neighbour_type, travel_dist_range, input_dict):
    if neighbour_type != 'Not a WMD':
        
        idict = json.loads(input_dict)
        
        travel_dist_list = idict['input_min_dist'][dist_option][neighbour_type][1]['travel_distance_pairs']
        
        new_travel_dist_list = list()
        
        for travel_dist in travel_dist_list:
            new_travel_dist_list.append(
                (travel_dist[0][0],
                travel_dist[1][0],
                travel_dist[2]
                )
                )
        
        name = f"{idict['input_min_dist'][dist_option][neighbour_type][0]}"
        
        new_travel_dist_list = sorted(new_travel_dist_list, key=lambda tup: tup[2])
        if travel_dist_range > 0:
            new_travel_dist_list = new_travel_dist_list[:travel_dist_range]    
        else:
            new_travel_dist_list = new_travel_dist_list[travel_dist_range:]
            
        travel_dist_df = pd.DataFrame(new_travel_dist_list, columns = ['Input word', 'Neighbour word','Travel distance'])
        
        
        latex_code = travel_dist_df.to_latex(index=False,
                                        caption=(f'{dist_option} + {neighbour_type} + {name} travel distance'),
                                        label='tab:add label',
                                        float_format="%.7f")
        
        latex_list = list()
        for line in latex_code.split("\n"):
            latex_list.append(line)
            latex_list.append(html.Br())
        
        
        table = dash_table.DataTable(travel_dist_df.to_dict('records'), 
                                      [{"name": i, "id": i} for i in travel_dist_df.columns],
                                      style_cell={'textAlign': 'left'},
                                      sort_action="native",
                                      sort_mode="multi"
                                          ) 
        
        
        
        return table, latex_list, name, name 


        
app.run_server(debug=False)
