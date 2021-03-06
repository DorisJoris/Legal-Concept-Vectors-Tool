# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:10:44 2022

@author: bejob
"""
#%% import
import pickle

import pandas as pd


from dash import Dash, dcc, html, Input, Output, dash_table, State
import dash_bootstrap_components as dbc
import plotly.express as px

from visualization_data import add_inputs_visual_dfs



#%% Open pickled input visual_dfs

with open("visualization_data/input_visual_dfs_new.p", "rb") as pickle_file:
    input_visual_dfs = pickle.load(pickle_file)     
    
with open("databases/test_database.p", "rb") as pickle_file:
    test_database = pickle.load(pickle_file)      

#%% plotly
external_stylesheets = [dbc.themes.MINTY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

user_input_str_list = list()

tab_graph = dbc.Card(dbc.CardBody([
    dcc.Graph(id="graph"),
    html.P(['Selected point:']),
    html.P(id='text-print', style= {"width": "90%",
                                    "height": "150px",
                                    "padding": "2px",
                                    "margin-left":"5%",
                                    "margin-right":"5%",
                                    "border": "1px solid grey",
                                    "border-radius": "5px",
                                    "overflow-wrap": "break-word",
                                    "overflow": "scroll"
                                    })
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

tab_add_userinput = dbc.Card(dbc.CardBody([
    dcc.Textarea(
        id='user_input_text',
        value='Insert user text here',
        style={'width': '100%', 'height': 200},
    ),
    html.Button('Add', id='user_input_button', n_clicks=0)
    ]))


app.layout = dbc.Container([
    dcc.Store(id='memory'),
    
    html.H1('Legal concept experiments'),
    dbc.Row([
        dbc.Col(dbc.Label('Select input:'),width=2),
        dbc.Col(dcc.Dropdown(
            id="input_select",
            options=list(input_visual_dfs.keys()),
            value=list(input_visual_dfs.keys())[0],
            multi=False
        )),
    ]),
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
    ]),
    dbc.Tabs(
        [dbc.Tab(tab_graph, label='Graph'),
         dbc.Tab(tab_table, label='Table'),
         dbc.Tab(tab_latex, label='Latex table code'),
         dbc.Tab(travel_dist, label='Travel distances'),
         dbc.Tab(tab_add_userinput, label="Add user input text")
            ]
        )
    
])

@app.callback(
    Output('neighbour_dropdown','options'),
    Output('neighbour_dropdown','value'),
    Output('document_dropdown','options'),
    Output('document_dropdown','value'),
    Input("input_select", "value"),
    Input("distance_type", "value"))
def init_multi_dropdowns(input_option, dist_option):
    
    neighbour_options = input_visual_dfs[input_option][0][dist_option]['Neighbour type'].unique()
    neighbour_values = list()
    
    for option in neighbour_options:
        if 'Input' in option or 'closest' in option:
            neighbour_values.append(option)
    
    doc_options = input_visual_dfs[input_option][0][dist_option]['Point type'].unique()
    doc_values = ['Text', 'Vector', 'Legal concept']
    
    return neighbour_options, neighbour_values, doc_options, doc_values

@app.callback(
    Output("table", "children"),
    Input("input_select", "value"),
    Input("distance_type", "value"),
    Input('graph', 'clickData'),
    Input('neighbour_dropdown','value'),
    Input('document_dropdown','value'))
def update_table(input_option, dist_option, clickData, neighbour_options, doc_options):
    
    output_df = input_visual_dfs[input_option][0][dist_option]
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
    Input("input_select", "value"),
    Input("distance_type", "value"),
    Input('neighbour_dropdown','value'),
    Input('document_dropdown','value'))
def update_bar_chart(input_option, dist_option, neighbour_options, doc_options):
    
    output_df = input_visual_dfs[input_option][0][dist_option]
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
    Output("graph", "clickData"),
    Input("input_select", "value"))
def rest_clickdata(input_option):
    return None


@app.callback(
    Output('text-print', 'children'),
    Input('graph', 'clickData'),
    Input("input_select", "value"),
    Input("distance_type", "value"))
def display_click_data(clickData,input_option, dist_option):
    
    output_df = input_visual_dfs[input_option][0][dist_option]
    
    if clickData == None:
        search_name = "Name: " + output_df.loc[0,'Name']
        search_type = "Neighbour type: " + output_df.loc[0,'Neighbour type']
        
        text = output_df.loc[0,'Text']
        
        output_text = [search_name,  html.Br(), search_type,  html.Br()] + [text]
    else:
        search_name = clickData['points'][0]['text']
        search_type = list(output_df.loc[output_df.index[output_df['Name'] == search_name], 'Neighbour type'])[0]
        text = list(output_df.loc[output_df['Name'] == search_name, 'Text'])[0]
        
        output_text = ["Name: " + search_name,  html.Br(),"Neighbour type: " + search_type,  html.Br()] + [text]
        
       
    return output_text

@app.callback(
    Output('latex_dropdown','options'),
    Output('latex_dropdown','value'),
    Input("input_select", "value"),
    Input("distance_type", "value"))

def init_latex_dropdown(input_option, dist_option):
    
    columns = list(input_visual_dfs[input_option][0][dist_option].columns)

    return columns, columns    

@app.callback(
    Output('latex', 'children'),
    Input("input_select", "value"),
    Input("distance_type", "value"),
    Input('neighbour_dropdown','value'),
    Input('document_dropdown','value'),
    Input('latex_dropdown','value'))
def display_latex_code(input_option, dist_option, neighbour_options, doc_options, columns):
    
    output_df = input_visual_dfs[input_option][0][dist_option]
    output_df = output_df[output_df['Neighbour type'].isin(neighbour_options)]
    output_df = output_df[output_df['Point type'].isin(doc_options)]
    
    output_df = output_df[[c for c in output_df.columns if c in columns]]
    
    latex_code = output_df.to_latex(index=False,
                                    caption=(f'{input_option} + {dist_option}'),
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
    Input("input_select", "value"),
    Input("distance_type", "value")
    )
def set_travel_dist_dropdown(input_option,dist_option):
    wmd_options = ['reverse_wmd_concept_bow',
                   'wmd_concept_bow']
    
    if dist_option in wmd_options:
        output_df = input_visual_dfs[input_option][0][dist_option]
        options = list(output_df[output_df['Point type'] == 'Legal concept'].loc[:,'Neighbour type'])
        
        return options, options[0]
        
    else:
        return ['Not a WMD'], 'Not a WMD'
    
@app.callback(
    Output('travel_dist_table', 'children'),
    Output('travel_dist_latex', 'children'),
    Output('travel_dist_table_title', 'children'),
    Output('travel_dist_latex_title', 'children'),
    Input("input_select", "value"),
    Input("distance_type", "value"),
    Input('travel_dist_dropdown','value'),
    Input('travel_dist_range', 'value')
    )

def update_travel_dist_tables(input_option,dist_option,neighbour_type, travel_dist_range):
    if neighbour_type != 'Not a WMD':
        travel_dist_list = input_visual_dfs[input_option][1]['input_min_dist'][dist_option][neighbour_type][1]['travel_distance_pairs']
        
        new_travel_dist_list = list()
        
        for travel_dist in travel_dist_list:
            new_travel_dist_list.append(
                (travel_dist[0][0],
                travel_dist[1][0],
                travel_dist[2]
                )
                )
        
        name = f"{input_visual_dfs[input_option][1]['input_min_dist'][dist_option][neighbour_type][0]}"
        
        new_travel_dist_list = sorted(new_travel_dist_list, key=lambda tup: tup[2])
        if travel_dist_range > 0:
            new_travel_dist_list = new_travel_dist_list[:travel_dist_range]    
        else:
            new_travel_dist_list = new_travel_dist_list[travel_dist_range:]
            
        travel_dist_df = pd.DataFrame(new_travel_dist_list, columns = ['Input word', 'Neighbour word','Travel distance'])
        
        
        latex_code = travel_dist_df.to_latex(index=False,
                                        caption=(f'{input_option} + {dist_option} + {neighbour_type} + {name} travel distance'),
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

# @app.callback(
#     Input('user_input_button', 'n_clicks'),
#     State('user_input_text', 'value')
# )
# def update_output(n_clicks, value):
#     if n_clicks > 0:
#         global user_input_str_list
#         global input_visual_dfs
#         global test_database
        
#         if value not in user_input_str_list:
#             user_input_str_list.append(value)
#             input_visual_dfs = add_inputs_visual_dfs(value, input_visual_dfs, test_database)
        
app.run_server(debug=False)
