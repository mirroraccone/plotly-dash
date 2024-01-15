import warnings
warnings.filterwarnings("ignore")


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import geopandas as gpd
from dash import dash_table


import pandas as pd

# Load data from Excel
df = pd.read_excel('Block_data.xlsx')
gdf = gpd.read_file('set_data/lat_long.shp')
df_district = pd.read_excel('set_data/district_data.xlsx')

df.fillna(0, inplace=True)
df_district.fillna(0, inplace=True)

df_district = df_district[['State','District','state_dis','Tap water connection Coverage',
                        'Reporting by Implementing Departments (Reporting rate)',
                        'Certification by Gram Sabhas (Certification Rate)',
                        'Reporting to Coverage Ratio',
                        'Certification to Coverage Ratio',
                        'Certification to Reporting Ratio', 'Year', 'lat_long']]

df_district.rename(columns={'State':'State1','District':'District1'},inplace=True)


df['State'] = df['State'].str.title()
df['District'] = df['District'].str.title()
df['Block'] = df['Block'].str.title()

df_district['State1'] = df_district['State1'].str.title()
df_district['District1'] = df_district['District1'].str.title()

gdf['STATE'] = gdf['STATE'].str.title()


original_states = ['Andaman & Nicobar Islands', 'Dadra & Nagar Haveli And Daman & Diu', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
                   'Jammu & Kashmir']

new_states = ['Andaman & Nicobar', 'Dadra & Nagar Haveli & Daman & Diu','Jammu And Kashmir']

state_mapping = dict(zip(original_states, new_states))

# Replace values in the "STATE" column
gdf['STATE'] = gdf['STATE'].replace(state_mapping)



gdf.rename(columns={"STATE":"State"},inplace=True)


gdf.merge(df_district,on='lat_long',how='left').groupby('State')['District'].agg(list).reset_index()


gdf = gdf.merge(df_district,on='lat_long',how='left')


gdf.rename(columns={"STATE":"State"},inplace=True)







external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

app.layout = html.Div(style={'backgroundColor': '#f2f2f2'}, children=[
    # First row with the first dropdown and second dropdown
    html.Div([
        html.Div([
            html.Label("State:"),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': state, 'value': state} for state in sorted(df['State'].unique())],
                value='Karnataka',
                className='dropdown',
                
            ),
        ], style={'width':'20%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("District:"),
            dcc.Dropdown(id='district-dropdown', className='dropdown'),
        ], style={ 'width':'15%','display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Block:"),
            dcc.Dropdown(id='block-dropdown', className='dropdown'),
        ], style={ 'width':'15%','display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("Metric:"),
            dcc.Dropdown(
                id='result-metric-dropdown',
                options=[
                        'Tap water connection Coverage',
                        'Reporting by Implementing Departments (Reporting rate)',
                        'Certification by Gram Sabhas (Certification Rate)',
                        'Reporting to Coverage Ratio',
                        'Certification to Coverage Ratio',
                        'Certification to Reporting Ratio',
                    
                ],
                multi=False,
                value='Tap water connection Coverage'
            )
        ], style={ 'width':'30%','display': 'inline-block','padding': '10px'}),

        html.Div([
            html.Label("Year:"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in sorted(df['Year'].unique())],
                value=df['Year'].unique()[1],
                className='dropdown'
            ),
        ], style={'width':'10%','display': 'inline-block', 'padding': '10px'}),
    ],style={'display': 'flex', 'flex-wrap':'wrap','padding': '10px','justify-content':'space-between'}),

    # Second row with the third dropdown and fourth dropdown
    #  html.Div([
    #     dash_table.DataTable(
    #         id='my-table',
    #         columns=[
    #             {'name': col, 'id': col} for col in df_table.columns
    #         ],
    #         data=df_table.to_dict('records'),
    #         style_table={'height': '300px', 'overflowY': 'auto', 'overflowX': 'auto'},
    #         style_cell={'whiteSpace': 'normal', 'textAlign': 'left'}
    #     )
    # ],
    # style={'width': '100%'}),
    # Fourth row with the choropleth map and the first bar chart
    html.Div([
        html.Div([
            dcc.Graph(id='choropleth-map-2021')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px','border-radius': '15px', 'border': '2px solid #ddd','background-color': '#f0f8ff'}),
        
        html.Div([
            dcc.Graph(id='choropleth-map')
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px','border-radius': '15px', 'border': '2px solid #ddd','background-color': '#f0f8ff'}),
        
    ], style={'width': '98%', 'margin':'auto','display': 'flex','justify-content':'space-around'}),

    html.Div([
                ## meticsss left ones
            html.Div([
                 html.Div([
                    dcc.Graph(id='all-metrics-bar-chart')
                ], className='row', style={'width': '100%', 
                                           'display': 'inline-block', 
                                           'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
                html.Div([
                    dcc.Graph(id='all-metrics-bar-chart-block')
                ], className='row', style={'width': '100%', 
                                           'display': 'inline-block', 
                                           'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
            ], style={'width': '48%', 'display': 'flex','flex-direction':'column','justify-content':'space-around'}),

            # Right side (continuous bar graph)
            html.Div([
                dcc.Graph(id='continuous-y-chart')
            ], className='row', style={'width': '48%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff','margin-left': '1%'}),   
            ],style={'display': 'flex','justify-content':'space-around','margin-bottom':'5px'}),
 # Third row with the second bar chart
    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart')
        ], style={'width': '98%','display': 'inline-block', 'padding': '10px','border-radius': '15px', 'border': '2px solid #ddd','background-color': '#f0f8ff'}),
    ]),
    html.Div([
            html.Div([
                dcc.Graph(id='multibargraph')
            ], className='row', style={'width': '50%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
            html.Div([
                dcc.Graph(id='multibargraph2')
            ], className='row', style={'width': '50%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
        ], style={'width': '98%', 'margin':'auto','display': 'flex','justify-content':'space-around'}),
    html.Div([
            html.Div([
                dcc.Graph(id='multibargraph-block')
            ], className='row', style={'width': '50%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
            html.Div([
                dcc.Graph(id='multibargraph2-block')
            ], className='row', style={'width': '50%', 
                                       'display': 'inline-block', 
                                       'padding': '10px', 'border-radius': '15px', 'border': '2px solid #ddd', 'background-color': '#f0f8ff'}),
        ], style={'width': '98%', 'margin':'auto','display': 'flex','justify-content':'space-around'}),
])
html.Div(
    children=[
        html.P("Copyright © Jal Jeevan Mission - Indian Institute of Management Banglore. All rights reserved."),
    ],
    style={
        'text-align': 'center',
        'background-color': '#f0f0f0',  # Optional: Set background color for the footer
        'padding': '10px',
        'position': 'flex',
        'bottom': '0',
        'width': '100%',
    }
)







# Define callback to update district dropdown based on selected state
@app.callback(
    Output('district-dropdown', 'options'),
    [Input('state-dropdown', 'value')]
)
def update_district_dropdown(selected_state):
    districts = df[df['State'] == selected_state]['District'].unique()
    return [{'label': district, 'value': district} for district in sorted(districts)]


@app.callback(
    [Output('block-dropdown', 'options'),
     Output('block-dropdown', 'value')],
    [Input('state-dropdown', 'value'),
     Input('district-dropdown', 'value')]
)
def update_block_dropdown(selected_state, selected_district):
    if selected_state is None:
        raise PreventUpdate
    
    if selected_district is None:
        # If district is not selected, return empty options for blocks
        return [], None
    else:
        # If both state and district are selected, return blocks for the selected district
        blocks = df[(df['State'] == selected_state) & (df['District'] == selected_district)]['Block'].unique()
        return [{'label': block, 'value': block} for block in sorted(blocks)], None



#map 2022
@app.callback(
    Output('choropleth-map-2021', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('result-metric-dropdown', 'value')]
)

def update_choropleth_map(selected_state, selected_metric):
    # Filter the GeoDataFrame based on the selected state
    filtered_gdf = gdf[(gdf['State'] == selected_state) & (gdf['Year'] == 2021)].to_crs('EPSG:4326')

    # Create the choropleth map
    fig = px.choropleth(
        filtered_gdf,
        geojson=filtered_gdf.geometry,
        locations=filtered_gdf.index,
        color=selected_metric,
        color_continuous_scale='blues',  # You can customize the color scale
        hover_name='District',
        labels={selected_metric: selected_metric},
        title=f'Choropleth Map for {selected_state} - {selected_metric} - 2021',
        range_color=(0, 100),
        color_continuous_midpoint=50
    )

    # Update layout
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            orientation='h', # Set orientation to 'h' for horizontal color bar
            # title=f'Choropleth Map for {selected_state} - {selected_metric} - 2021',
        )
    )

    return fig



#map 2023
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('result-metric-dropdown', 'value')]
)

def update_choropleth_map(selected_state, selected_metric):
    # Filter the GeoDataFrame based on the selected state
    filtered_gdf = gdf[(gdf['State'] == selected_state) & (gdf['Year'] == 2023)].to_crs('EPSG:4326')

    # Create the choropleth map
    fig = px.choropleth(
        filtered_gdf,
        geojson=filtered_gdf.geometry,
        locations=filtered_gdf.index,
        color=selected_metric,
        color_continuous_scale='blues',  # You can customize the color scale
        hover_name='District',
        labels={selected_metric: selected_metric},
        title=f'Choropleth Map for {selected_state} - {selected_metric} - 2023',
        range_color=(0, 100),
        color_continuous_midpoint=50
    )

    # Update layout
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            orientation='h', # Set orientation to 'h' for horizontal color bar
            # title=f'Choropleth Map for {selected_state} - {selected_metric} - 2023',
        )
    )

    return fig


## All metrics chart

@app.callback(
    Output('all-metrics-bar-chart', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_all_metrics_bar_chart(selected_state,selected_district, selected_year):
    if not selected_district:
        return px.bar()  # Empty bar chart

    # Filter data based on selected options
    filtered_data = df_district[
        (df_district['State1'] == selected_state) & 
        (df_district['District1'] == selected_district) &
        (df_district['Year'] == selected_year)
    ]

    # List of all metrics
    all_metrics = [
        'Tap water connection Coverage',
        'Reporting by Implementing Departments (Reporting rate)',
        'Certification by Gram Sabhas (Certification Rate)',
        'Reporting to Coverage Ratio',
        'Certification to Coverage Ratio',
        'Certification to Reporting Ratio'
    ][::-1]

    # Aggregate data for the entire district
    aggregated_data = filtered_data.groupby('Year')[all_metrics].mean().reset_index()

    # Melt the DataFrame to have a long format suitable for a bar chart
    melted_data = pd.melt(
        aggregated_data,
        id_vars=['Year'],
        value_vars=all_metrics,
        var_name='Metric',
        value_name='Value'
    )

    # Create and return the horizontal bar chart with metrics on y-axis and values on x-axis
    fig = px.bar(
        melted_data,
        y='Metric',
        x='Value',
        title=f"All Metrics in {selected_state} - {selected_district} ({selected_year})",
        labels={'Value': 'Metric Value'},
        orientation='h',
        height=500,
        text='Value',  # Enable labels on bars
        color_discrete_sequence=['green']  # Set the color to pink
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Format labels as numbers
    fig.update_xaxes(range=[0, 120])
    return fig




@app.callback(
    Output('all-metrics-bar-chart-block', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('district-dropdown', 'value'),
     Input('block-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)

def update_all_metrics_bar_chart_block(selected_state,selected_district, selected_block, selected_year):
    if  not selected_block or not selected_district or not selected_state:
        return px.bar()  # Empty bar chart

    # Filter data based on selected options
    filtered_data = df[
        (df['State'] == selected_state) & 
        (df['District'] == selected_district) &
        (df['Block'] == selected_block) &
        (df['Year'] == selected_year)
    ]

    # List of all metrics
    all_metrics = [
        'Tap water connection Coverage',
        'Reporting by Implementing Departments (Reporting rate)',
        'Certification by Gram Sabhas (Certification Rate)',
        'Reporting to Coverage Ratio',
        'Certification to Coverage Ratio',
        'Certification to Reporting Ratio'
    ][::-1]

    # Melting the DataFrame
    melted_data = pd.melt(
        filtered_data,
        id_vars=['Block'],
        value_vars=all_metrics,
        var_name='Metric',
        value_name='Metric Value'
    )

    # Create and return the horizontal bar chart
    fig = px.bar(
        melted_data,
        y='Metric',
        x='Metric Value',
        title=f"All Metrics in {selected_district} - {selected_block} ({selected_year})",
        labels={'Metric Value': 'Metric Value'},
        orientation='h',
        height=500,
        text='Metric Value',  # Enable labels on bars
        color_discrete_sequence=['green']  # Set the color to green
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Format labels as numbers

    fig.update_xaxes(range=[0, 120])


    return fig




#all graph in 1
@app.callback(
    Output('continuous-y-chart', 'figure'),
    [Input('result-metric-dropdown', 'value'),
     Input('state-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_continuous_y_chart(selected_metric, selected_state ,selected_year):
    if not selected_metric or not selected_year:
        return px.line()  # Empty line chart

    # Filter data based on selected options
    filtered_data = df_district[
        (df_district['State1'] == selected_state)&
        (df_district['Year'] == selected_year)

    ].sort_values(by=selected_metric, ascending=True)

    # Create and return the line chart
    fig = px.bar(
        filtered_data,
        x=selected_metric,
        y='District1',
        title=f"{selected_metric} in {selected_state} for ({selected_year})",
        labels={'District': 'District Name', selected_metric: selected_metric},
        height=900,
        text=selected_metric,  # Display values on top of bars
        color_discrete_sequence=['blue'],
    )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        margin=dict(l=15,),
    )
    fig.update_xaxes(range=[0, 120])
    return fig




# Define callback to update bar chart based on selected state, district, result metric, and year
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('district-dropdown', 'value'),
     Input('result-metric-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_bar_chart(selected_state, selected_district, selected_metric, selected_year):
    filtered_df = df[(df['State'] == selected_state) & (df['District'] == selected_district) & (df['Year'] == selected_year)]
    
    # Create a bar chart with formatted text values
    fig = go.Figure()
    fig.add_trace(go.Bar(x=filtered_df['Block'], y=filtered_df[selected_metric],
                        text=[f"{value:.2f}" if value != 0 else "0" for value in filtered_df[selected_metric]],
                        textposition='outside',
                        marker_color='blue',  # You can customize the color if needed
                        ))

    # Update layout
    fig.update_layout(
        title=f'Results for {selected_district}, {selected_state} - {selected_year}',
        xaxis=dict(title='Block'),
        yaxis=dict(title=selected_metric),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

# MULTI BAR GRAPH
@app.callback(
    Output('multibargraph', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def multibargraph(selected_state,selected_year):
    if not selected_state:
        return go.Figure()  # Empty figure
    
    # Filter data based on selected district
    filtered_data = df_district[(df_district['State1'] == selected_state)  & (df_district['Year'] == selected_year)]

    # Sort data by the first variable
    sorted_data = filtered_data.sort_values(by='Tap water connection Coverage', ascending=True)
    colors = {
        'Reporting by Implementing Departments (Reporting rate)': 'orange',
        'Certification by Gram Sabhas (Certification Rate)': 'green',
        'Tap water connection Coverage': 'blue'
    }
    # Create the multi-bar graph
    fig = go.Figure()

    # Add bars for each variable
    for variable in ['Certification by Gram Sabhas (Certification Rate)','Reporting by Implementing Departments (Reporting rate)', 'Tap water connection Coverage']:
        fig.add_trace(go.Bar(
            x=sorted_data[variable],
            y=sorted_data['District1'],
            name=variable,
            orientation='h',  # Set orientation to horizontal
            text=sorted_data[variable].apply(lambda x: f'{x:.1f}%'),  # Display data labels on the bars
            textposition='outside',
            marker=dict(line=dict(width=0),color=colors[variable]),
            # showlegend=False, 
        ))

    # Set x-axis limit from 0 to 100
    fig.update_xaxes(range=[0, 105])

    # Update layout to reduce bar gap and group gap
    fig.update_layout(
        # title=f'Tap Connection Statistics for {selected_district}',
        xaxis=dict(title='Percentage'),
        yaxis=dict(title='District Name'),
        barmode='group',
        bargap=0.2,        # Adjust the gap between bars
        bargroupgap=0.1,   # Adjust the gap between bar groups
        height=1600,        # Increase height
        legend=dict(x=0, y=-0.15, orientation='h')
    )

    return fig

@app.callback(
    Output('multibargraph2', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def multibargraph2(selected_state,selected_year):
    if not selected_state:
        return go.Figure()  # Empty figure
    
    # Filter data based on selected district
    filtered_data = df_district[(df_district['State1'] == selected_state) & (df_district['Year'] == selected_year)]



    # Sort data by the first variable
    sorted_data = filtered_data.sort_values(by='Tap water connection Coverage', ascending=True)
    colors = {'Certification to Reporting Ratio': 'maroon', 'Tap water connection Coverage': 'blue'}
    # Create the multi-bar graph
    fig = go.Figure()

    # Add bars for each variable
    for variable in ['Certification to Reporting Ratio','Tap water connection Coverage' ]:
        fig.add_trace(go.Bar(
            x=sorted_data[variable],
            y=sorted_data['District1'],
            name=variable,
            orientation='h',  # Set orientation to horizontal
            text=sorted_data[variable].apply(lambda x: f'{x:.1f}%'),  # Display data labels on the bars
            textposition='outside',
            marker=dict(line=dict(width=0),color=colors[variable]),
            # showlegend=False, 
        ))

    # Set x-axis limit from 0 to 100
    fig.update_xaxes(range=[0, 105])

    # Update layout to reduce bar gap and group gap
    fig.update_layout(
        # title=f'Tap Connection Statistics for {selected_district}',
        xaxis=dict(title='Percentage'),
        yaxis=dict(title='District Name'),
        barmode='group',
        bargap=0.2,        # Adjust the gap between bars
        bargroupgap=0.1,   # Adjust the gap between bar groups
        height=1600,        # Increase height
        legend=dict(x=0, y=-0.15, orientation='h')
    )

    return fig


# MULTI BAR GRAPH block
@app.callback(
    Output('multibargraph-block', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def multibargraph_block(selected_state,selected_district,selected_year):
    if not selected_state:
        return go.Figure()  # Empty figure
    
    # Filter data based on selected district
    filtered_data = df[(df['State'] == selected_state) & (df['District'] == selected_district)  & (df['Year'] == selected_year)]

    # Sort data by the first variable
    sorted_data = filtered_data.sort_values(by='Tap water connection Coverage', ascending=True)
    colors = {
        'Reporting by Implementing Departments (Reporting rate)': 'orange',
        'Certification by Gram Sabhas (Certification Rate)': 'green',
        'Tap water connection Coverage': 'blue'
    }
    # Create the multi-bar graph
    fig = go.Figure()

    # Add bars for each variable
    for variable in ['Certification by Gram Sabhas (Certification Rate)','Reporting by Implementing Departments (Reporting rate)','Tap water connection Coverage']:
        fig.add_trace(go.Bar(
            x=sorted_data[variable],
            y=sorted_data['Block'],
            name=variable,
            orientation='h',  # Set orientation to horizontal
            text=sorted_data[variable].apply(lambda x: f'{x:.1f}%'),  # Display data labels on the bars
            textposition='outside',
            marker=dict(line=dict(width=0),color=colors[variable]),
            # showlegend=False, 
        ))

    # Set x-axis limit from 0 to 100
    fig.update_xaxes(range=[0, 105])

    # Update layout to reduce bar gap and group gap
    fig.update_layout(
        # title=f'Tap Connection Statistics for {selected_district}',
        xaxis=dict(title='Percentage'),
        yaxis=dict(title='Block Name'),
        barmode='group',
        bargap=0.2,        # Adjust the gap between bars
        bargroupgap=0.1,   # Adjust the gap between bar groups
        height=1200,        # Increase height
        legend=dict(x=0, y=-0.15, orientation='h')
    )

    return fig

@app.callback(
    Output('multibargraph2-block', 'figure'),
    [Input('state-dropdown', 'value'),
     Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def multibargraph2_block(selected_state,selected_district,selected_year):
    if not selected_state:
        return go.Figure()  # Empty figure
    
    # Filter data based on selected district
    filtered_data = df[(df['State'] == selected_state) & (df['District'] == selected_district)  & (df['Year'] == selected_year)]



    # Sort data by the first variable
    sorted_data = filtered_data.sort_values(by='Tap water connection Coverage', ascending=True)
    colors = {'Certification to Reporting Ratio': 'maroon', 'Tap water connection Coverage': 'blue'}
    # Create the multi-bar graph
    fig = go.Figure()

    # Add bars for each variable
    for variable in ['Certification to Reporting Ratio','Tap water connection Coverage' ]:
        fig.add_trace(go.Bar(
            x=sorted_data[variable],
            y=sorted_data['Block'],
            name=variable,
            orientation='h',  # Set orientation to horizontal
            text=sorted_data[variable].apply(lambda x: f'{x:.1f}%'),  # Display data labels on the bars
            textposition='outside',
            marker=dict(line=dict(width=0),color=colors[variable]),
            # showlegend=False, 
        ))

    # Set x-axis limit from 0 to 100
    fig.update_xaxes(range=[0, 105])

    # Update layout to reduce bar gap and group gap
    fig.update_layout(
        # title=f'Tap Connection Statistics for {selected_district}',
        xaxis=dict(title='Percentage'),
        yaxis=dict(title='Block Name'),
        barmode='group',
        bargap=0.2,        # Adjust the gap between bars
        bargroupgap=0.1,   # Adjust the gap between bar groups
        height=1200,        # Increase height
        legend=dict(x=0, y=-0.15, orientation='h')
    )

    return fig



# Run the app
if __name__ == '__main__':
    app.run_server(debug=False,host='0.0.0.0')
