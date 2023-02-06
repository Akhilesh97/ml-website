# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:41:16 2023

@author: Akhilesh
"""

import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, dash_table, State
import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "4rem 2rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

labels = ['Red Meat and Poultry','Fruits','Vegetables','Carbonated Soft Drinks']
values = [271.2, 102.3, 50.8, 38]

fig_stats1 = go.Figure(data=[go.Pie(labels=labels, values=values,  hole=.3)])
fig_stats1.update_layout(title_text="Food Consumption habits in the U.S in 2019 in Lbs",)


labels = ['Men', 'Women']
values = [2700, 2200]

fig_stats2 = go.Figure(data=[go.Pie(labels=labels, values=values,  hole=.3)])
fig_stats2.update_layout(title_text="Average Calorie Intake in the U.S in 2021",)

fig_stats3 = go.Figure()
fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 1500,
    title = "Number of different foods in supermarkets",
    domain = {'row': 0, 'column': 0},
    ))

fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 33000,
    title = "Supermarkets in the U.S",
    domain = {'row': 0, 'column': 1}))

fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 91,
    title = "Percent population living in food deserts",
   
    domain = {'row': 1, 'column': 0}))

fig_stats3.add_trace(go.Indicator(
    mode = "number",
    value = 138000000000,
    title = "Food Imports by U.S - Fruits, Vegies, Wine, Snack",   
    domain = {'row': 1, 'column': 1}))


fig_stats3.update_layout(grid = {'rows': 2, 'columns': 2, 'pattern': "independent"})
sidebar = html.Div(
    [
        html.H2("Demographic Analysis", className="display-4"),
        html.Hr(),
        html.P(
            "This website shows detailed analysis of food consumption and availability", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Introduction", href="/", active="exact"),
                dbc.NavLink("Data_Prep-EDA", href="/data-prep", active="exact"),
                dbc.NavLink("Machine Learning Algorithms", href="/ml-algo", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

tab_introduction_content = html.Div([
        dbc.Row([
            dbc.Toast([
                html.P("""
                       Food consumption and availability in the U.S has changed dramatically over the past few decades. 
                       The American diet has shifted towards more processed, high-calorie, and convenient foods, which has led to an increase in obesity and related health problems. 
                       The fast food industry has also grown rapidly, contributing to this trend. 
                       At the same time, the rise of supermarkets and big box stores has made food more accessible and available in urban and rural areas.
                       People can now purchase a wide variety of food products from all over the world, leading to a more diverse food culture.
                       """),
                
                html.P("""
                       However, access to healthy, affordable food is still a challenge for many Americans. 
                       Food deserts, which are areas where access to fresh and nutritious food is limited, still exist in many urban and rural areas. 
                       Additionally, the COVID-19 pandemic has disrupted food systems, leading to shortages of certain products and increased food prices. 
                       Despite these challenges, the U.S remains one of the largest food producers in the world and continues to have a diverse and abundant food supply. 
                       However, there is ongoing effort to improve the food system and make healthy food more accessible to all Americans, through initiatives such as farm-to-table programs and urban agriculture.
                       """)
            ], header = "Food Consumption and Availability in the United States", style = {"width":"100%"})
        ]),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    dcc.Graph(figure = fig_stats3),
                    html.P("""
                           These statistics give a general idea of food availability in the U.S, but it's important to note that access to food can vary greatly by region and population, and that ongoing efforts are being made to improve access to healthy and affordable food for all Americans.
                           """)
                ], header = "Some Stats on Food Availability in the U.S", style = {"width":"100%"})  ,                
            ])   
            ]),
        html.Br(),
        html.Br(),                    
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    
                    dcc.Graph(figure = fig_stats1),
                    html.Ul([
                        html.Li("In 2019, the average American consumed approximately 271.2 pounds of red meat and poultry."),
                        html.Li("The average American consumed approximately 102.3 pounds of fruits in 2019."),
                        html.Li("In 2019, the average American consumed approximately 131.9 pounds of vegetables.")            ,            
                        html.Li("In 2019, the average American consumed approximately 50.8 gallons of beverages, including nearly 38 gallons of carbonated soft drinks."),                        
                    ])
                    
                ], header = "Some Stats on Food Consumption in the U.S", style = {"width":"100%"})    
            ]), 
            dbc.Col([
                dbc.Toast([
                        dcc.Graph(figure = fig_stats2),
                        html.Ul([
                          html.Li("In 2021, the average daily calorie intake for Americans was estimated to be around 2,700 calories for men and 2,200 calories for women."),
                          html.Li("Data from the USDA Economic Research Service shows that from 1970 to 2018, per capita calorie availability from added sugars increased by 39 percent.")  
                        ]),
                ], header = "Some Stats on Food Consumption in the U.S", style = {"width":"100%"}),                              
            ]),
             
        ])
    ])
tab_data_gathering = html.Div([
       
        dbc.Toast([
            html.P("""The main source of data was through the United States Department of Agriculture website.
                   Data was scraped from the website using selenium. The data consisted mainly of two broad categories.
                   """),
            html.Ul(id = "data-list", children = [html.Li("Food Availability"), html.Li("Food Consumption")])
        ], header = "Source of Data", style = {"width":"100%"}),
    
    
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    html.P("""The USDA's Economic Research Service's Food Availability Data System provides data on food and nutrient availability for consumption at the national level. 
                           The data includes food availability data, loss-adjusted food availability data, and nutrient availability data. 
                           The food availability data includes estimates for over 200 commodities including fruits, vegetables, grains, dairy products, meats, and more. 
                           However, the discontinuation of certain data collection methods has led to data limitations and the inability to calculate certain summary estimates past 2010 or for specific vegetables beyond 2019. 
                           The nutrient availability data calculates the daily food energy and 27 nutrients of the US food supply but has not been updated since 2010. 
                           The loss-adjusted food availability data is derived from food availability data but requires further improvement and is considered preliminary.                            
                           """),
                    html.Ul(id = "food-availabilty-list", children = [html.Li(html.A("Website used for scraping data",href="https://www.ers.usda.gov/data-products/food-availability-per-capita-data-system/", target="_blank")), 
                                                                      html.Li(html.A("Code for webscraping", href = "https://github.com/Akhilesh97", target="_blank"))])                           
                ], header = "Food Avaialability Data Set", style = {"width":"100%"}),    
            ]),
            dbc.Col([
                dbc.Toast([
                    html.P("""TERS provides 3 tables for food consumption and 2 tables for nutrient intake for the period 2015-2018. 
                           The data for 2015-2018 has updated documentation and differs from the 2007-2010 archived data. 
                           The 2015-2018 and 2007-2010 data should not be combined. An updated data file allowing users to view the data over time is coming soon.
                           """),
                    html.Ul(id = "food-availabilty-list", children = [html.Li(html.A("Website used for scraping data", href="https://www.ers.usda.gov/data-products/food-consumption-and-nutrient-intakes/", target="_blank")), 
                                                                      html.Li(html.A("Code for webscraping", href = "https://github.com/Akhilesh97",target="_blank"))]
                            )                           
                ], header = "Food Consumption Data Set", style = {"width":"100%"})
            ])
        ])
    ], style = {'background-image': 'static/images/images.png',
    'backgroundRepeat': 'no-repeat',
    'backgroundPosition': 'center top'})
                
tab_data_cleaning = html.Div([
        
        dbc.Toast([            
            dcc.RadioItems(
                options = [
                    {"label": "Food Availabilty", "value": "food-availability"},
                    {"label": "Food Consumption", "value": "food-consumption"},
                ],
                value = "food-availability",                    
                id = "data-cleaning-radio"
            )     
        ], style =  {"height": "100%"}, header = "Chose a dataset to view data cleaning process"),                
        html.Br(),
        dbc.Row([
            html.Div(id = "data-cleaning-content")    
        ])
        
    ])  

df_dairy = pd.read_csv("Food_Availability/cleaned_data/df_dairy.csv")
df_dairy.drop(columns = [i for i in df_dairy.columns if "Unnamed" in i], inplace = True)
cheeses = ['American Cheese', 'Other Cheeses', 'Cottage Cheese']
milk = ['Fluid Milk', 'Whole Milk/Evaporated_Condensed', 'Skim Milk/Evaporated_Condensed','Whole Milk/Dry', 'Nonfat Milk/Dry', 'Buttermilk/Dry']
frozen_products = [ 'Ice cream', 'Lowfat and non fat icecream', 'Sherbet', 'Frozen Yoghurt','Other Frozen Prodcuts']
creams = ['Fluid Cream', 'Sour Cream', 'Yoghurt', 'Butter']

tab_data_viz = html.Div([
        html.H3("Select from the below food categories"),
        dbc.Accordion([
            dbc.AccordionItem([

                dbc.Row([
                    html.H5("Year Wise Product Visualization"),
                    dcc.Dropdown(list(df_dairy["Year"]), list(df_dairy["Year"])[-1], id = "year-drop-down-dairy", placeholder = "Select a year"),
                    dcc.Graph(id = "year-wise-graph-all-prods")
                ]),
                dbc.Row([
                    html.H5("Product categories for a given year"),
                    dcc.Dropdown(["Cheese Products", "Milk Products", "Frozen Prdocuts", "Creams"], "Cheese Products", id = "prod-cat-dropdown", placeholder = "Select a category for the year"),    
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id = "year-wise-graph-one-prods")
                    ], width = 5),
                    dbc.Col([
                        dcc.Graph(id = "corr-graph-one-prods")
                    ], width = 5)
                ]),
                dbc.Row([
                    html.H5("Overall Time Series Analysis"),
                    dcc.Dropdown(list(df_dairy.columns), frozen_products, placeholder = "Select Multiple Products", id = "prod-overall-dropdown", multi = True),
                    dcc.Graph(id = "overall-graph-prod-wise")
                ])
            ], title = "Dairy Products")    
        ])    
    ])
tab_data_prep_content = html.Div([
        dbc.Tabs([
            dbc.Tab([
                html.Br(),
                tab_data_gathering,
                
            ], label = "Data Gathering"),
            dbc.Tab([
                html.Br(),
                tab_data_cleaning,
                
            ], label = "Data Cleaning"),
            dbc.Tab([
                html.Br(),
                tab_data_viz    
            ], label = "Data Visualisation")            
        ])
    ])

tab_machine_learning_algos = dbc.Card([
        dbc.Tabs([
            dbc.Tab([
                html.P("Linear Regression")
            ], label = "Linear Regression"),
            dbc.Tab([
                html.P("Logistic Regression")
            ], label = "Logistic Regression")
        ])
    ])


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
                dcc.Location(id="url"), sidebar, content    
            ])

def plot_table(table, width_ = 1100):
    return dash_table.DataTable(
        data = table.to_dict('records'),
        columns=[
            {'name': i, 'id': i} for i in table.columns
        ],
        style_table={
            'height': 500,
            'overflowY': 'scroll',
            'width': width_
        },
        export_format="csv",

    )

def prepare_data_report(data):
    data_types = pd.DataFrame(
        data.dtypes,
        columns=['Data Type']
    )

    missing_data = pd.DataFrame(
        data.isnull().sum(),
        columns=['Missing Values']
    )

    unique_values = pd.DataFrame(
        columns=['Unique Values']
    )
    for row in list(data.columns.values):
        unique_values.loc[row] = [data[row].nunique()]

    maximum_values = pd.DataFrame(
        columns=['Maximum Value']
    )
    for row in list(data.columns.values):
        maximum_values.loc[row] = [data[row].max()]
    dq_report = data_types.join(missing_data).join(unique_values).join(maximum_values)
    dq_report["Col_Name"] = dq_report.index
    dq_report.reset_index(drop = True, inplace=True)
    dq_report_cols = list(dq_report.columns)
    new_cols = [dq_report_cols[-1]] + dq_report_cols[0:-1]
    dq_report = dq_report[new_cols]
    return dq_report

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return tab_introduction_content
    elif pathname == "/data-prep":
        return tab_data_prep_content
    elif pathname == "/ml-algo":
        return tab_machine_learning_algos
    
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

@app.callback(Output(component_id="data-cleaning-content", component_property="children"),
              [Input(component_id="data-cleaning-radio", component_property="value")])
def render_data_cleaning(value):
    if value == "food-availability":
        df_raw = pd.read_excel("Food_Availability/raw_data/dyfluid.xlsx", sheet_name="FluidmilkPccLb")        
        df_trimmed = pd.read_csv("Food_Availability/trimmed_data/df_fluid_milk_trimmed.csv").drop(columns = ["Unnamed: 0"])
        df_cleaned = pd.read_csv("Food_Availability/cleaned_data/df_fluid_milk.csv")
        dq_report_trimmed = pd.read_csv("Food_Availability/dq_report_trimmed.csv").drop(columns = ["Unnamed: 0"])        
        dq_report_cleaned = pd.read_csv("Food_Availability/dq_report_cleaned.csv").drop(columns = ["Unnamed: 0"])                
        return html.Div([dbc.Accordion([
            dbc.AccordionItem([                
                dbc.Row([
                    dbc.Toast([
                        html.P("""The dataset had a lot of formatting issues when read as an Excel sheet given that most of the data was scrapped.
                               Hence, these formatting issues were addressed and the entire set was trimmed. 
                               """),
                       dbc.Button("View Raw Data", id="raw-df-button", n_clicks=0, className="me-1"),
                       dbc.Modal([
                               dbc.ModalHeader(dbc.ModalTitle("Raw Dataset - Fluid Milk")),
                               dbc.ModalBody(plot_table(df_raw))
                           ],
                         id="raw-df-modal",
                         size="xl",
                         is_open=False,
                       ),
                       dbc.Button("View Trimmed Data", id="trimmed-df-button", n_clicks=0, className="me-1"),
                       dbc.Modal([
                               dbc.ModalHeader(dbc.ModalTitle("Trimmed Dataset - Fluid Milk")),
                               dbc.ModalBody(plot_table(df_trimmed))
                           ],
                         id="trimmed-df-modal",
                         size="xl",
                         is_open=False,
                       ),
                       dbc.Button("View Cleaned Data", id="cleaned-df-button", n_clicks=0, className="me-1"),
                       dbc.Modal([
                               dbc.ModalHeader(dbc.ModalTitle("Cleaned Dataset - Fluid Milk")),
                               dbc.ModalBody(plot_table(df_cleaned))
                           ],
                         id="cleaned-df-modal",
                         size="xl",
                         is_open=False,
                       )
                    ], header = "From Raw data to Trimmed Data", style = {"width":"100%"}) 
                ]),
                html.Br(),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.Ul([
                                html.Li("In most cases in this dataset, if there are missing values, means that there was no data availabe during that year."),
                                html.Li("It is also observed that most of the missing values are appear contigously before or after a given time period. For example - For the category of dairy - Fluid Cream, we can see that there was no data recorded after 2006."),
                                html.Li("Hence, it would be fitting to replace these values with a 0 and ignore these values during visualization or model building.")                                
                            ]),
                            dbc.Button("View Sample Data Values", id="null-df-button", n_clicks=0, className="me-1"),
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Sample Dataset - Fluid Milk")),
                                    dbc.ModalBody(plot_table(df_trimmed))
                                ],
                              id="null-df-modal",
                              size="xl",
                              is_open=False,
                            )
                        ], header = "Handling Null Values", style = {"width":"100%"})
                    ]),                    
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Since we know that all the columns have floating point values, we can replace inconsistent values with the following logic
                                   """),
                            html.Ul([
                                html.Li("Try and return the type casted value of the record as a float,"),
                                html.Li("If an exception is raised return the value as NaN and print the value.")
                            ]),
                            html.P("""
                                   This way, if the type casted value is a floating point entered as a string, the value is returned correctly, else the inconsistent values is returned.
                                   """),
                           html.P("""
                                  The following steps can be performed for columns whose dtype is "object"
                                  """),
                           dbc.Button("View Sample Inconsistent Values", id="inconsistent-df-button", n_clicks=0, className="me-1"),
                           dbc.Modal([
                                   dbc.ModalHeader(dbc.ModalTitle("Inconsistent Dataset - Fluid Milk")),
                                   dbc.ModalBody(plot_table(df_trimmed))
                               ],
                             id="inconsistent-df-modal",
                             size="xl",
                             is_open=False,
                           )
                        ], header = "Handling Inconsistent Values", style = {"width":"100%"}),
                        
                    ])
                ])
            ], title = "Data Cleaning Process"),
            dbc.AccordionItem([
                
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown(list(dq_report_cleaned["Col_Name"]), list(dq_report_cleaned["Col_Name"])[0], id = "col-dropdown"),
                        ], header = "Select a column to view metrics")
                    ])    
                        
                ]),                
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.Div(id = "before-metrics"),
                            dbc.Button("View Entire Metrics", id="before-metrics-df-button", n_clicks=0, className="me-1"),
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Overall Metrics")),
                                    dbc.ModalBody(plot_table(dq_report_trimmed))
                                ],
                              id="before-metrics-df-modal",
                              size="xl",
                              is_open=False,
                            )                                            
                        ], header = "Data Metrics Before", style = {"width":"100%","height":"100%"})     
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.Div(id = "after-metrics"),
                            dbc.Button("View Entire Metrics", id="after-metrics-df-button", n_clicks=0, className="me-1"),
                            dbc.Modal([
                                    dbc.ModalHeader(dbc.ModalTitle("Overall Metrics")),
                                    dbc.ModalBody(plot_table(dq_report_cleaned))
                                ],
                              id="after-metrics-df-modal",
                              size="xl",
                              is_open=False,
                            )                                             
                        ], header = "Data Metrics Before", style = {"width":"100%","height":"100%"})     
                    ])                       
                ]),
            ], title = "View Before and After"),
            
        ],active_item = "item-0", always_open = True)])
                                  
@app.callback(
    Output('before-metrics', 'children'),
    Input('col-dropdown', 'value')
)
def update_output_before(value):
    df_trimmed = pd.read_csv("Food_Availability/trimmed_data/df_fluid_milk_trimmed.csv")
    dq_report_trimmed = prepare_data_report(df_trimmed)    
    vals = dq_report_trimmed[dq_report_trimmed["Col_Name"] == value].values.flatten()    
    keys = dq_report_trimmed[dq_report_trimmed["Col_Name"] == value].keys()    
    d = dict(zip(keys, vals))
    print(d)
    return html.Ul([
            html.Li("Column Name - %s"%d["Col_Name"]),
            html.Li("Data Type - %s"%d["Data Type"]),
            html.Li("Missing Values - %s"%d["Missing Values"]),
            html.Li("Unique Values - %s"%d["Unique Values"]),
            html.Li("Maximum Value - %s"%d["Maximum Value"])
        ])

@app.callback(
    Output('after-metrics', 'children'),
    Input('col-dropdown', 'value')
)
def update_output_after(value):
    df_cleaned = pd.read_csv("Food_Availability/cleaned_data/df_fluid_milk.csv")      
    dq_report_cleaned = prepare_data_report(df_cleaned)
    vals = dq_report_cleaned[dq_report_cleaned["Col_Name"] == value].values.flatten()    
    keys = dq_report_cleaned[dq_report_cleaned["Col_Name"] == value].keys()    
    d = dict(zip(keys, vals))
    return html.Ul([
            html.Li("Column Name - %s"%d["Col_Name"]),
            html.Li("Data Type - %s"%d["Data Type"]),
            html.Li("Missing Values - %s"%d["Missing Values"]),
            html.Li("Unique Values - %s"%d["Unique Values"]),
            html.Li("Maximum Value - %s"%d["Maximum Value"])
        ])
        
@app.callback(
    Output("year-wise-graph-all-prods", "figure"),
    Input("year-drop-down-dairy", "value")
)                                  
def update_year_wise_graph(value):
    df_to_plot = df_dairy[df_dairy["Year"] == value].drop(columns = ["Year", "All Dairy"])
    df_to_plot.drop(columns = [i for i in df_to_plot.columns if "total" in i.lower()], inplace = True)
    fig = go.Figure(go.Bar(
            y=list(df_to_plot.columns),
            x=list(df_to_plot.values.flatten()),            
            orientation='h'))
    fig.update_layout(
        title="Product availabilty for the year %d"%value,
        xaxis_title="Products",
        yaxis_title="Per Capita Per Pound",
        legend_title="Legend Title",
         
    )
    return fig

@app.callback(
    Output("year-wise-graph-one-prods", "figure"),
    Output("corr-graph-one-prods", "figure"),
    [Input("year-drop-down-dairy", "value"),
    Input("prod-cat-dropdown", "value")]
) 
def update_year_wise_prod_graph(year, prod):
    df_to_plot = df_dairy[df_dairy["Year"] == year].drop(columns = ["Year", "All Dairy"])
    df_to_plot.drop(columns = [i for i in df_to_plot.columns if "total" in i.lower()], inplace = True)
    if prod == "Cheese Products":
        cols = cheeses
    elif prod == "Milk Products":
        cols = milk
    elif prod == "Frozen Prdocuts":
        cols = frozen_products
    elif prod == "Creams":
        cols = creams
    df_to_plot_sub = df_to_plot[cols]
    fig = go.Figure(go.Bar(
            y=list(df_to_plot_sub.columns),
            x=list(df_to_plot_sub.values.flatten()),            
            orientation='h'))
    fig.update_layout(
        title="%s availabilty for the year %d"%(prod,year),
        xaxis_title="Products",
        yaxis_title="Per Capita Per Pound",
        legend_title="Legend Title",
        width = 500
    )
    fig2 = px.imshow(df_dairy[cols].corr())
    fig2.update_layout(
        title="Correlation between %s category products"%prod,
        xaxis_title="Products - %s"%prod,
        yaxis_title="Products - %s"%prod,
        legend_title="Legend Title",
        width = 500
    )
    return fig, fig2

@app.callback(
    Output("overall-graph-prod-wise", "figure"),
    Input("prod-overall-dropdown", "value")
) 
def update_overall_product_wise(products):
    fig = px.line(df_dairy, x='Year', y=products, title='Overall Time Series Analysis Prodcut Wise')    
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open


app.callback(
    Output("raw-df-modal", "is_open"),
    Input("raw-df-button", "n_clicks"),
    State("raw-df-modal", "is_open"),
)(toggle_modal)    
app.callback(
    Output("trimmed-df-modal", "is_open"),
    Input("trimmed-df-button", "n_clicks"),
    State("trimmed-df-modal", "is_open"),
)(toggle_modal)    
app.callback(
    Output("cleaned-df-modal", "is_open"),
    Input("cleaned-df-button", "n_clicks"),
    State("cleaned-df-modal", "is_open"),
)(toggle_modal) 
app.callback(
    Output("inconsistent-df-modal", "is_open"),
    Input("inconsistent-df-button", "n_clicks"),
    State("inconsistent-df-modal", "is_open"),
    
)(toggle_modal)  
app.callback(
    Output("null-df-modal", "is_open"),
    Input("null-df-button", "n_clicks"),
    State("null-df-modal", "is_open"),
    
)(toggle_modal)  
app.callback(
    Output("before-metrics-df-modal", "is_open"),
    Input("before-metrics-df-button", "n_clicks"),
    State("before-metrics-df-modal", "is_open"),
    
)(toggle_modal)   
app.callback(
    Output("after-metrics-df-modal", "is_open"),
    Input("after-metrics-df-button", "n_clicks"),
    State("after-metrics-df-modal", "is_open"),
    
)(toggle_modal)                                  

if __name__ == "__main__":
    app.run_server(debug=True)