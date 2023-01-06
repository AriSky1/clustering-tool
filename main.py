import pandas as pd
import numpy as np
import base64
import io
import dash
from dash import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash import ctx
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from flask import Flask
import psycopg2
from sqlalchemy import create_engine
from credentials import user, password, host, port, db,db2
from styles import style_title,style_text,style_upload,style_load,style_cats,style_cats2, style_btn_large,style_dropdown_rand, style_flex,style_input_col,style_input_save, style_dropdown_tbl,style_menu_div,style_tbl_div,style_current_tbl_name,style_menu_and_tbl_div,style_dbscan_btn,style_dbscan_btn_div,style_preview_txt,style_preview_tbl,style_input_titles,style_hyper_inputs,style_update_btn,style_inputs_and_update_btns,style_graphs_div

#test

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
server = Flask(__name__)





app = dash.Dash(__name__,
                server=server,
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True,
                prevent_initial_callbacks=True)

df = pd.DataFrame()
current_name=''


# -----------------------------------------------------
connection = psycopg2.connect(user=user,password=password,host=host,port=port,database=db)
connection.autocommit = True
cursor = connection.cursor()
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
s_tbls = pd.Series(cursor.fetchall(), dtype=object)
flat_list = [item for sublist in s_tbls for item in sublist]
cursor.close()

# ------------------------------------------------------------------
connection = psycopg2.connect(user=user,password=password,host=host,port=port,database=db2)
connection.autocommit = True
cursor = connection.cursor()
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
s_tbls = pd.Series(cursor.fetchall(), dtype=object)
flat_list_rand = [item for sublist in s_tbls for item in sublist]
cursor.close()


# LAYOUT -------------------------------------------------
app.layout = html.Div([
                html.Div('Clustering with DBSCAN ', style=style_title),
                html.Div('Create, upload and manage datasets. Try random datasets from public domain.',style=style_text),
                html.Div('Apply the density-based clustering algorithm to the data and download the output as a csv file.',style=style_text),
                html.Div('This version allows you to perform clustering on 3 numerical columns only.',style=style_text),
                html.Br(),
            html.Div([
                html.Div([
                    html.Plaintext('     Load data',style=style_cats),
                    html.Div([dcc.Upload(html.Div('UPLOAD'), id='upload_dcc',style=style_upload),
                        html.Button('Create new', id='create_new_btn', n_clicks=0,style=style_btn_large),]),
                            html.Div([
                                dcc.Dropdown(flat_list_rand, id="dropdown_tbl_rand", value='Random table.', style=style_dropdown_rand),
                                html.Button('Load', id='random_btn', n_clicks=0,style=style_btn_large),],
                            style=style_flex),
                                html.Br(),
                                html.Plaintext('     Modify & save',style=style_cats),
                                html.Button('Add Row', id='rows_btn', n_clicks=0, style=style_btn_large),
                                html.Div([html.Button('Add Column', id='cols_btn', n_clicks=0, style=style_btn_large),
                                dcc.Input(
                                        id='adding-rows-name',
                                        placeholder='Column name...',
                                        value='mycolumn',
                                        style=style_input_col
                                    ),], style={"display":"flex", "align-items":"flex-end"}),
                                    html.Div([html.Button('Save', id='save_btn', n_clicks=0, style=style_btn_large),
                                              dcc.Input(
                                                  id='user_tbl_name',
                                                  placeholder='Enter the table name...',
                                                  value='mytable',
                                                  style=style_input_save
                                              ), ], style=style_flex),
                                    html.Br(),
                                    html.Plaintext('     Manage database',style=style_cats),
                                    html.Div([
                                    dcc.Dropdown(flat_list, id="dropdown_tbl", value='Saved table.',style=style_dropdown_tbl),
                                    html.Button('Resfresh', id='refresh_btn', n_clicks=0,
                                                                  style=style_btn_large),
                                    ], style=style_cats2),
                                        html.Div([html.Button('Load', id='load_btn', n_clicks=0,
                                                                  style=style_btn_large),
                                        html.Button('Delete', id='delete_btn', n_clicks=0,
                                                    style=style_btn_large),
                                                  ], style=style_cats2),
                                        html.Div(id='placeholder_selection', children=[]),
                                        html.Div(id='placeholder_delete_one', children=[]),
                                        html.Div(id='placeholder_display', children=[]),
                                        html.Div(id='placeholder_save',children=[]),

                                                    ], style=style_menu_div),

                                html.Div([
                                        html.Div(id='placeholder_current_tbl_name', children=[], style=style_current_tbl_name),
                                        dash_table.DataTable(
                                                            id='mytable',
                                                            data=df.to_dict('records'),
                                                            columns=[{'name': i, 'id': i, 'deletable': True, 'renamable': True} for i in df.columns],
                                                            editable=True,
                                                            row_deletable=True,
                                                            sort_action="native",
                                                            sort_mode="single",
                                                            filter_action="native",
                                                            page_action='native',
                                                            page_current=0,
                                                            page_size=15,
                                                            style_table={'height': '600px', 'width':'700px',
                                                                         'maxWidth':'700px','overflowY': 'auto',
                                                                         'display':'flex', 'margin-left': '40px', },
                                                            style_data={'whiteSpace': 'normal',
                                                                        'height': 'auto',
                                                                        'width' : 'auto'},
                                                            fill_width=False,
                                                            style_cell={'textAlign': 'center',  'minWidth': '80px', 'width': '100px', 'maxWidth': '180px', 'font_family': 'Arial', 'color':'grey'}),
                                ],style=style_tbl_div
                                ),], style=style_menu_and_tbl_div),


                html.Div([html.Button('DBSCAN', id='use_dbscan', n_clicks=0, style=style_dbscan_btn),], style=style_dbscan_btn_div),
                    html.Br(),
                    html.Div(id="dbscan_part"),

])



# CALLBACKS


# CALLBACK UPATE TABLE --------------------------------------------
@app.callback(Output('mytable', 'data'),
              Output('mytable', 'columns'),
              Output('placeholder_current_tbl_name', 'children'),
              Input('upload_dcc', 'contents'),
              Input('create_new_btn', 'n_clicks'),
              Input('random_btn', 'n_clicks'),
              Input('load_btn', 'n_clicks'),
              Input('rows_btn', 'n_clicks'),
              Input('cols_btn', 'n_clicks'),
              State('mytable', 'data'),
              State('mytable', 'columns'),
              State('dropdown_tbl_rand', 'value'),
              State('dropdown_tbl', 'value'),
              State('upload_dcc', 'filename'),
              State('adding-rows-name', 'value'))

def update_table(contents, click_create_new, click_random, click_load, click_row, click_cols, data, columns,selection_rand, selection, filename, col_input):


    triggered_id = ctx.triggered_id
    if triggered_id == 'upload_dcc' and contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            columns = [{'name': i, 'id': i, 'deletable': True, 'renamable': True} for i in df.columns]
            data = df.to_dict(orient='records')
            current_name = filename
            return data, columns, current_name
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            columns = [{'name': i, 'id': i, 'deletable': True, 'renamable': True} for i in df.columns]
            data = df.to_dict(orient='records')
            current_name = filename
            return data, columns, current_name
        else:
            df=pd.DataFrame()

            columns = [{'name': i, 'id': i, 'deletable': True, 'renamable': True} for i in df.columns]
            current_name = 'No data'
            return data, columns, current_name
    if triggered_id == 'create_new_btn' and click_create_new > 0 :
        df = pd.DataFrame({'Column 1 ': [1, 2, 3],
                           'Column 2': [1, 2, 3],
                           'Column 3': [1, 2, 3]},
                          index=['Column 1', 'Column 2', 'Column 3'])
        columns = [{'name': col, 'id': col, 'deletable': True, 'renamable': True} for col in df.columns]
        data = df.to_dict(orient='records')
        current_name = ''
        return data, columns, current_name


    if triggered_id == 'random_btn' and click_random > 0 and selection_rand != 'Random table.':
        connection = psycopg2.connect(user=user, password=password, host=host, port=port, database=db2)
        cursor = connection.cursor()
        selection = str(selection_rand)
        fselection = str("public." + '"' + selection + '"')
        create_table_query = f'''SELECT * FROM {fselection} '''
        cursor.execute(create_table_query)
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        columns = [{'name': i, 'id': i, 'deletable': True, 'renamable': True} for i in df.columns]
        data = df.to_dict(orient='records')
        current_name = selection_rand
        cursor.close()

        return data, columns, current_name



    if triggered_id == 'load_btn' and click_load > 0 and selection != 'Select saved table.':
        connection = psycopg2.connect(user=user, password=password, host=host, port=port, database=db)
        cursor = connection.cursor()
        selection = str(selection)
        fselection = str("public." + '"' + selection + '"')
        create_table_query = f'''SELECT * FROM {fselection} '''
        cursor.execute(create_table_query)
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        columns = [{'name': i, 'id': i, 'deletable': True, 'renamable': True} for i in df.columns]
        data = df.to_dict(orient='records')
        current_name = selection
        cursor.close()
        return data, columns, current_name
    if triggered_id == 'rows_btn' and click_row > 0:
        data.append({c['id']: '' for c in columns})
        current_name=''
        return data, columns, current_name
    if triggered_id == 'cols_btn' and click_cols > 0 :
        columns.append({
            'name': col_input, 'id': col_input,
            'renamable': True, 'deletable': True})
        current_name = ''
        return data, columns, current_name
    else:
        df = pd.DataFrame()
        columns = [{'name': i, 'id': i, 'deletable': True, 'renamable': True} for i in df.columns]
        data = df.to_dict(orient='records')
        current_name = ''
        return data, columns, current_name

# END OF CALLBACK UPATE TABLE --------------------------------------------

#CALLBACK SAVE TABLE ---------------------------------------------------
@app.callback(
    Output('placeholder_save', 'children'),
    [Input('save_btn', 'n_clicks')],
    [State('user_tbl_name', 'value'),
    State('mytable', 'data'),
     State('mytable', 'columns')],
)
def save_to_db(n_clicks, value, data, columns):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db}')
    columns = [d['name'] for d in columns]
    output = html.Plaintext("")
    no_output = html.Plaintext("")
    if n_clicks > 0:
        table_name = value
        df = pd.DataFrame(data, index=None, columns=columns)
        df.to_sql(table_name.lower(), con=engine,
                  if_exists='replace',
                  schema = 'public',
                    index=False)
        engine.dispose()
        return output
    else:
        return no_output
# END OF CALLBACK SAVE TABLE ---------------------------------------------------


#CALLBACK REFRESH
@app.callback(
    Output('dropdown_tbl', 'options'),
    [Input('refresh_btn', 'n_clicks')],)

def refresh(n_clicks):

    if n_clicks > 0:
        connection = psycopg2.connect(user=user, password=password, host=host, port=port, database=db)
        cursor= connection.cursor()
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        cursor.execute(query)
        s_tbl_ids = pd.Series(cursor.fetchall())
        flat_list = [item for sublist in s_tbl_ids for item in sublist]
        cursor.close()
        return flat_list
# END OF CALLBACK REFRESH----------------

# CALLBACK DISPLAY SELECTION
@app.callback(
    Output('placeholder_selection', 'children'),
    Input('dropdown_tbl', 'value'),)

def display_selection(selection):

    if selection is not None:
            connection = psycopg2.connect(user=user, password=password, host=host, port=port, database=db)
            cursor = connection.cursor()
            fselection = str("public." + selection )
            create_table_query = f'''SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = '{selection}' '''
            cursor.execute(create_table_query)
            df_names = pd.Series(cursor.fetchall())
            df_names = df_names.values
            df_names = sum(df_names, ())
            create_table_query = '''SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' '''
            res=cursor.execute(create_table_query)
            s = pd.Series(cursor.fetchall())
            regular_list = s
            flat_list = [item for sublist in regular_list for item in sublist]
            cursor.close()
            if selection in flat_list:
                connection = psycopg2.connect(user=user, password=password, host=host, port=port, database=db)
                cursor = connection.cursor()
                create_table_query = f'''SELECT * FROM {fselection} '''
                cursor.execute(create_table_query)
                df = pd.DataFrame(cursor.fetchall(), columns=df_names)
                cursor.close()
                return html.Div([
                html.Div([html.Plaintext('Preview', style=style_preview_txt),], style={'padding-left':'120px'}),
                dash_table.DataTable(data=(df.head(2).to_dict('records')),
                                     columns=[{"name": i, "id": i} for i in df.columns],
                                     style_table=style_preview_tbl,),
        ], style={"width": '30px'})
            else:
                pass
# END OF CALLBACK DISPLAY SELECTION----------------



# CALLBACK DELETE ONE TABLE
@app.callback(
    Output('placeholder_delete_one', 'children'),
[Input('delete_btn', 'n_clicks'), Input('dropdown_tbl', 'value')],
    prevent_initial_call=True)

def delete_one(n_clicks, selection):

    if n_clicks > 0 and selection is not None:

        connection = psycopg2.connect(user=user, password=password, host=host, port=port, database=db)
        connection.autocommit = True
        cursor = connection.cursor()
        fselection = str('public.' + '"' + selection + '"')

        delete_table_query = f'''DROP TABLE IF EXISTS {fselection} CASCADE;'''
        cursor.execute(delete_table_query)
        cursor.close()

        return html.Plaintext("")
    else:
        pass


# END OF CALLBACK DELETE ONE TABLE----------------

# CALLBACK USE DBSCAN PART ----------------------------------------------------------
@app.callback(
    Output('dbscan_part', 'children'),
    [Input('use_dbscan', 'n_clicks')],
    [State('mytable', 'data'),
     State('mytable', 'columns')],
)


def display_dbscan_part(n_clicks, data, columns):

    if n_clicks > 0:
        pg = pd.DataFrame(data)
        return html.Div([
            html.Br(),
            html.Div([
            html.Div([html.P("Insert the label column", style=style_input_titles),
            dcc.Dropdown(id='xaxis-data',
                         options=[{'label': x, 'value': x} for x in pg.columns],
                         style={"width":"200px", 'background-color': '#F9F8F9'}),], style={'verticalAlign': 'top'}),
            html.Div([

                html.Div([html.P("Chose 3 columns to cluster"),],style=style_input_titles ),
            dcc.Dropdown(id='yaxis-data',
                         options=[{'label': x, 'value': x} for x in pg.columns], multi=True, style={"width": "200px", 'background-color': '#F9F8F9'}),],
                     style={'verticalAlign':'top'}),
                html.Div([html.Div([html.P("Set epsilon"),],style=style_input_titles),
                dcc.Input(id='input-epsilon',
                          type='text',
                          value="0.5",
                          min=0,
                          max=10000,
                          step=0.1,
                          debounce=True, style=style_hyper_inputs),], style={"verticalAlign": 'top'}),
                html.Div([html.Div([html.P("Set min_samples"), ],style=style_input_titles),
                dcc.Input(id='input-min_samples',
                          type='text',
                          min=1,
                          max=10000,
                          step=1,
                          value=7,
                          debounce=True, style=style_hyper_inputs),],style={"verticalAlign": 'top'} ),
                html.Div([html.Div([html.P("Set distance", style=style_input_titles), ]),
                dcc.Dropdown(id='input_distances',
                             options=['euclidean', 'manhattan', 'cosine', 'l1', 'l2', 'cityblock', 'braycurtis',
                                      'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
                                      'kulsinski',
                                       'minkowski', 'rogerstanimoto', 'russellrao',
                                      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'],
                             value='euclidean', style={'fontSize': 12,
                                                       'font-family': 'Arial',
                                                       'width': '150px',
                                                       'float': 'left', 'background-color': '#F9F8F9',
                                                       'padding-left': '2%',
                                                       }),], style={"verticalAlign": 'top'}),
                html.Button(id="submit-button", children="Update", style=style_update_btn),
            ],style=style_inputs_and_update_btns),

            html.Div([
                # distances plot ----------------------------------------------------

                html.Div(children=(dcc.Graph(id="output-div"))),
               html.Div([

                   html.Plaintext('The "elbow" method\n', style={"textAlign": 'center', 'fontSize': 15,}),

                   html.Plaintext('Zoom in with the cursor and pick the distance value \n that is located at the "elbow" of the curve. \n This value is your epsilon.',
                                  style={"textAlign": 'center'}),
                # DBSCAN graph ------------------------------------------------------
                html.Div(children=(
                    dcc.Graph(id="output-div2"))), ], style={'verticalAlign':'top'}),
            ], style=style_graphs_div),

            html.Br(),
            dcc.Store(id='stored-data', data=pg.to_dict('records')),

            html.Div([html.Button("Download CSV", id="download-btn"), dcc.Download(id="download",),],style={'padding-left':'5%'}),


            html.Br(),html.Br(),html.Br(),

        ])

# END OF CALLBACK USE DBSCAN PART ----------------------------------------------------------

# CALLBACK GENERATES GRAPH 1 --------------------------------
@app.callback(Output('output-div', 'figure'),
              Input('submit-button','n_clicks'),
              State('stored-data','data'),
              State('xaxis-data','value'),
              State('yaxis-data', 'value'),
              State('input-epsilon', 'value'),
              State('input-min_samples', 'value'),
              State('input_distances', 'value'),
              )
def make_graphs(n, data, x_data, y_data, eps, min_samples, dist):
    if n is None:
        return dash.no_update
    else:
        eps = eps
        min_sam = min_samples
        metric = dist
        df = pd.DataFrame(data)
        dfy = df[y_data]
        dfx = df[x_data]
        X = dfy.to_numpy(int)
        eps = float(eps)
        min_sam = int(min_sam)
        db = DBSCAN(eps=eps, min_samples=min_sam, metric=metric)
        y_pred = db.fit_predict(X)
        global df_cl
        df_cl = dfy.copy()
        df_cl['Clusters'] = y_pred
        df_cl['Clusters'] = df_cl['Clusters'].astype(str)
        df_cl['id'] = dfx

        fig = px.scatter_3d(df_cl, x=df_cl.iloc[:, 0], y=df_cl.iloc[:, 1], z=df_cl.iloc[:, 2],
                            color='Clusters',
                            opacity=0.6,

                            labels={'x': df_cl.columns[0], 'y': df_cl.columns[1], 'z': df_cl.columns[2]},
                            hover_name=dfx, height=600, width=600, )
        fig.update_scenes(xaxis_backgroundcolor='#000000',
                          yaxis_backgroundcolor='#000000',
                            zaxis_backgroundcolor='#000000',)

        return fig
# END OF CALLBACK GENERATES GRAPH 1 --------------------------------

# CALLBACK GENERATES GRAPH 2 --------------------------------

@app.callback(Output('output-div2', 'figure'),
              Input('submit-button','n_clicks'),
              State('stored-data','data'),
              State('xaxis-data','value'),
              State('yaxis-data', 'value'),
              )
def make_graphs(n, data, x_data, y_data):
    if n is None:
        return dash.no_update
    else:
        df = pd.DataFrame(data)
        dfy = df[y_data]
        dfx = df[x_data]
        X = dfy.to_numpy(int)
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1].astype(float)
        distances_fig = px.line(x=distances,
                                 width=500,
                                height=400,

                                labels={'y':'Data points ID', 'x':'Distance between the data point and its nearest neighbors'},
                                color_discrete_sequence=["#000000"])
        distances_fig.update_scenes(xaxis_backgroundcolor='#000000',
                          yaxis_backgroundcolor='#000000',
                           )
        distances_fig.update_layout(
            font_family="Arial",
            font_color="black",
            title_font_family="Arial",
            title_font_color="black",
            legend_title_font_color="black", plot_bgcolor="#EDDCED")
        return distances_fig

# END OF CALLBACK GENERATES GRAPH 2 --------------------------------

# CALLBACK BUTTON DOWNlOAD CLUSTERING OUTPUT --------------------------------
@app.callback(Output("download", "data"),
              Input("download-btn", "n_clicks"),
              Input('stored-data','data'),
              Input('xaxis-data', 'value'),
              Input('yaxis-data', 'value'),
              Input('input-epsilon', 'value'),
              Input('input-min_samples', 'value'),
              Input('input_distances', 'value'),
              )

def generate_csv(n_clicks, data, x_data, y_data, eps, min_samples, dist):
    df = pd.DataFrame(data)
    if n_clicks is None:
        return dash.no_update
    if n_clicks > 1:
        eps = eps
        min_sam = min_samples
        metric = dist
        dfy = df[y_data]
        dfx = df[x_data]
        X = dfy.to_numpy(int)
        eps = float(eps)
        min_sam = int(min_sam)
        db = DBSCAN(eps=eps, min_samples=min_sam, metric=metric)
        y_pred = db.fit_predict(X)
        global df_cl
        df_cl = dfy.copy()
        df_cl['Clusters'] = y_pred
        df_cl['Clusters'] = df_cl['Clusters'].astype(str)
        df_cl['id'] = dfx
    return dcc.send_data_frame(df_cl.to_csv, filename="DATA_DBSCAN.csv")

# END OF CALLBACK BUTTON DOWNlOAD CLUSTERING OUTPUT --------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
