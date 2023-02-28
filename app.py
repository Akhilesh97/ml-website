# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:41:16 2023

@author: Akhilesh
"""

import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, dash_table, State
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from plotly import tools
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
import scipy.cluster.hierarchy as sch

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True

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
                dbc.NavLink("Unsupervised Learning algo - Clustering", href="/clustering", active="exact"),
                dbc.NavLink("Unsupervised Learning algo - Assoc. Rules", href = "/arm", active = "exact"),
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
        html.Br(),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    dcc.Graph(figure = fig_stats3),
                    html.P("""
                           These statistics give a general idea of food availability in the U.S, but it's important to note that access to food can vary greatly by region and population, and that ongoing efforts are being made to improve access to healthy and affordable food for all Americans.
                           """)
                ], header = "Some Stats on Food Availability in the U.S", style = {"width":"100%"})  ,
            ])
            ])
        ]),
        html.Br(),
        html.Hr(),
        dbc.Row([
            dbc.Toast([
                html.P("The end product of this demographic analysis of food consumption habits is to predict the demand of food in the country. In other words, this can be formulated as demand forecasting problem"),
                html.Ol([
                    html.Li("What are the current trends in food consumption and demand?"),
                    html.Li("What is the target customer demographic for the food product?"),
                    html.Li("What are the purchasing habits of the target demographic?"),
                    html.Li("What is the current market competition for the food product?"),
                    html.Li("How does the food product differ from or compete with similar products?"),
                    html.Li("How does the price of the food product compare to similar products?"),
                    html.Li("What is the historical sales data for the food product and similar products?"),
                    html.Li("What is the distribution channel for the food product?"),
                    html.Li("What impact do promotions, discounts, and marketing campaigns have on demand?"),
                    html.Li("What are the economic and environmental factors affecting demand for the food product?"),
                ])
            ], header = "What are some question we can answer to predict the demand of certain food products available in the U.S?", style = {"width":"100%"})
        ])                           
    ])
tab_data_gathering = html.Div([

        dbc.Toast([
            html.P("""The main source of data was through the United States Department of Agriculture website.
                   Data was scraped from the website using selenium. The data consisted mainly of two broad categories.
                   """),
            html.Ul(id = "data-list", children = [html.Li("Food Availability"), html.Li("Food Consumption")])
        ], header = "Source of Data", style = {"width":"100%"}),

        html.Br(),
        html.Hr(),
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
                                                                      html.Li(html.A("Code for webscraping", href = "https://github.com/Akhilesh97/ml-website/blob/main/webscrap/Scrape_Food_Availabilty.ipynb", target="_blank"))])
                ], header = "Food Avaialability Data Set", style = {"width":"100%"}),
            ]),
            dbc.Col([
                dbc.Toast([
                    html.P("""TERS provides 3 tables for food consumption and 2 tables for nutrient intake for the period 2015-2018.
                           The data for 2015-2018 has updated documentation and differs from the 2007-2010 archived data.
                           The 2015-2018 and 2007-2010 data should not be combined. An updated data file allowing users to view the data over time is coming soon.
                           """),
                    html.Ul(id = "food-availabilty-list", children = [html.Li(html.A("Website used for scraping data", href="https://www.ers.usda.gov/data-products/food-consumption-and-nutrient-intakes/", target="_blank")),
                                                                      html.Li(html.A("Code for webscraping", href = "https://github.com/Akhilesh97/ml-website/blob/main/webscrap/Scrape_Food_Consumption.ipynb",target="_blank"))]
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
                    dcc.Graph(id = "year-wise-graph-all-prods"),
                    html.Br(),
                    html.Li(),
                    html.P("""
                        This chart shows a year wise availability of dairy products. This helps the user to understand the food availability trends for dairy products for a
                        given year. For example, it is observed that fluid milk was the highest consumed in the year 2021. Similar plots can be plotted for other product categories.
                        """)
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    html.H5("Product categories for a given year"),
                    dcc.Dropdown(["Cheese Products", "Milk Products", "Frozen Prdocuts", "Creams"], "Cheese Products", id = "prod-cat-dropdown", placeholder = "Select a category for the year"),
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id = "year-wise-graph-one-prods"),
                        html.Br(),
                        html.Hr(),
                        html.P("""
                        These set of bar graphs help us understand the food availabilty trends on a granular level. Under dairy products there are sub-categories such as -
                        Cheese Products, Milk Products, Frozen Products and Creams. The user may want to understand on a finer level the food availability stats for dairy products which
                        is served through these plots.
                        """)
                    ], width = 5),
                    dbc.Col([
                        dcc.Graph(id = "corr-graph-one-prods"),
                        html.Br(),
                        html.Hr(),
                        html.P("""
                        One may want to understand if the increase/decrease in the availability of one set of products plays a role in the certain other set of prodcut availability.
                        Hence, a correlation plot between 3 or more products under a category can help us answer this question.
                        """)
                    ], width = 5)
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    html.H5("Overall Time Series Analysis"),
                    dcc.Dropdown(list(df_dairy.columns), frozen_products, placeholder = "Select Multiple Products", id = "prod-overall-dropdown", multi = True),
                    dcc.Graph(id = "overall-graph-prod-wise"),
                    html.Br(),
                    html.Hr(),
                    html.P("""
                    Finally, a time-series plot serves as a one stop shop for a user to understand the overall trends in the food availability of a given set of products
                    """)
                ])
            ], title = "Dairy Products")
        ])
    ])

tab_clustering_overview = html.Div([
        dbc.Tabs([
             dbc.Tab([
                 html.Br(),
                 html.Hr(),
                 dbc.Row([
                     dbc.Toast([
                         html.P("""
                                Clustering is a type of unsupervised machine learning technique that involves grouping data points or objects into similar clusters based on their similarities or distances between them. In clustering, the data points within a cluster are more similar to each other than to the data points in other clusters.                       
                                The main objective of clustering is to find meaningful groups or patterns in the data, and it is often used to gain insights into the underlying structure of the data. Clustering can be used in a variety of applications, including customer segmentation, anomaly detection, image processing, and natural language processing.
                                 There are many different clustering algorithms, each with their own strengths and weaknesses. Some of the most popular clustering algorithms include:
                                """)
                     ], header = "What is Clustering?", style = {"width":"100%"})    
                 ]),
                 html.Br(),
                 html.Hr(),
                 dbc.Row([
                     dbc.Col([
                         dbc.Toast([
                             html.P("""
                                    This algorithm partitions the data into K clusters by minimizing the sum of the squared distances between the data points and the centroid of each cluster.
                                    """),
                             html.Img(src = "static/images/kmeans.png", style = {"width":"100%"})
                         ], header = "K-means Clustering", style = {"width":"100%"})    
                     ]),
                     dbc.Col([
                         dbc.Toast([
                             html.P("""
                                    This algorithm builds a hierarchy of clusters by recursively merging or splitting clusters based on their similarity or distance.
                                    """),
                             html.Img(src = "static/images/hclust.png", style = {"width":"100%"})
                         ], header = "Hierarchical Clustering", style = {"width":"100%"})    
                     ]) 
                 ]),
                 html.Br(),
                 html.Hr(),
                 dbc.Row([
                     dbc.Col([
                         dbc.Toast([
                             html.P("""
                                    This algorithm groups data points based on their density in the data space, and can identify clusters of varying shapes and sizes.
                                    """),
                             html.Img(src = "static/images/dbscan.png", style = {"width":"100%"})
                         ], header = "Density Based Clustering", style = {"width":"100%"})    
                     ]),
                     dbc.Col([
                         dbc.Toast([
                             html.P("""
                                    This algorithm uses graph theory to group data points based on the similarity of their connections in a network or graph.
                                    """),
                             html.Img(src = "static/images/spectral.png", style = {"width":"100%"})
                         ], header = "Spectral Clustering", style = {"width":"100%"})    
                     ]) 
                 ])
            ], label = "Clustering concepts"),
            dbc.Tab([
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Toast([
                        html.P("""
                               Clustering based on nutrient value of food can be useful in demand prediction because it allows companies to identify groups of customers who have similar dietary preferences and needs. By clustering customers based on the nutrient value of the food they consume, companies can better understand the demand patterns of different customer groups and tailor their marketing and sales efforts to meet the needs of each group.
                               Here are some ways clustering based on nutrient value can be helpful in demand prediction:
                               """),                        
                    ], header = "Overview", style = {"width":"100%"})    
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   By clustering customers based on the nutrient value of the food they consume, companies can identify which nutrients are most important to different customer segments. This information can be used to predict demand for foods that are high in specific nutrients, such as protein, fiber, or vitamins.
                                   """),
                            html.Img(src = "static/images/clust_app1.png")
                        ], header = "Identifying customer preferences", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   By understanding the nutrient preferences of different customer segments, companies can tailor their product offerings to meet the needs of each group. For example, a company could develop a line of products that are high in protein for customers who are looking to increase their protein intake.
                                   """),
                            html.Img(src = "static/images/clust_app2.jfif")
                        ], header = "Customizing product offerings", style = {"width":"100%"})
                    ])
                ]),
                html.Br(),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   Clustering based on nutrient value can help companies create targeted marketing campaigns that resonate with specific customer segments. For example, a company could create a campaign that emphasizes the nutritional benefits of its products to appeal to health-conscious customers.
                                   """),
                            html.Img(src = "static/images/clust_app3.jfif")
                        ], header = "Creating targeted marketing campaigns", style = {"width":"100%"})
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            html.P("""
                                   By analyzing historical demand patterns for different nutrient categories, companies can forecast future demand and adjust their production, inventory, and supply chain strategies accordingly.
                                   """),
                            html.Img(src = "static/images/clust_app4.jfif")
                        ], header = "Forecasting demand", style = {"width":"100%"})
                    ])
                ])
            ], label = "Use of Clustering in the Project")                                    
        ])
        
    ])
                                   
nut_val_df = pd.read_csv("Nutrient_Vals/nut_val_df.csv")   

macro_nutrients = ['WWEIA Category description','Protein (g)',
       'Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)',
       'Total Fat (g)']

vitamins = ['WWEIA Category description','Vitamin A, RAE (mcg_RAE)', 'Vitamin B-12 (mcg)',
       'Vitamin B-12, added\n(mcg)', 'Vitamin C (mg)',
       'Vitamin D (D2 + D3) (mcg)', 'Vitamin E (alpha-tocopherol) (mg)',
       'Vitamin E, added\n(mg)', 'Vitamin K (phylloquinone) (mcg)']

minerals = ['WWEIA Category description','Calcium (mg)', 'Phosphorus (mg)', 'Magnesium (mg)', 'Iron\n(mg)',
       'Zinc\n(mg)', 'Copper (mg)', 'Selenium (mcg)', 'Potassium (mg)']


fatty_acids = ['WWEIA Category description','Fatty acids, total saturated (g)',
       'Fatty acids, total monounsaturated (g)',
       'Fatty acids, total polyunsaturated (g)', 'Cholesterol (mg)',
       'Retinol (mcg)']

amino_acids = ['WWEIA Category description','Carotene, alpha (mcg)',
       'Carotene, beta (mcg)', 'Cryptoxanthin, beta (mcg)', 'Lycopene (mcg)',
       'Lutein + zeaxanthin (mcg)', 'Thiamin (mg)', 'Riboflavin (mg)',
       'Niacin (mg)', 'Vitamin B-6 (mg)', 'Folic acid (mcg)',
       'Folate, food (mcg)', 'Folate, DFE (mcg_DFE)', 'Folate, total (mcg)']

df_macros = nut_val_df[macro_nutrients]
df_fatty_acids = nut_val_df[fatty_acids]
df_vitamins = nut_val_df[vitamins]
df_minerals = nut_val_df[minerals]
df_aminos = nut_val_df[amino_acids]

def plot_nutrient_contributors():
    df_to_plot = pd.DataFrame(columns = ["Category", "Value"])      
    df_to_plot["Category"] = nut_val_df['WWEIA Category description'].value_counts().keys()[0:20]
    df_to_plot["Value"] = nut_val_df['WWEIA Category description'].value_counts().values[0:20]     
    fig = px.bar(df_to_plot, x="Value", y="Category", orientation='h')
    return fig

def plot_corr_matrix(df, columns):
    x = columns[1:]
    df = df.iloc[:,1:]
    heat = go.Heatmap(z = df.corr(),
                      x = x,
                      y = x,
                      xgap=1, ygap=1,
                      colorbar_thickness=20,
                      colorbar_ticklen=3,
                      hovertext = df.corr(),
                      hoverinfo='text'
                       )
 
    title = 'Correlation Matrix'               
 
    layout = go.Layout(title_text=title, title_x=0.5, 
                       width=500, height=500,
                       xaxis_showgrid=False,
                       yaxis_showgrid=False,
                       yaxis_autorange='reversed')
 
    fig=go.Figure(data=[heat], layout=layout)      
    return fig
    
def plot_wcss_elbow(df):
    X = df.iloc[:,1:].values

    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500, n_init = 10, random_state = 123)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig = go.Figure(data = go.Scatter(x = [1,2,3,4,5,6,7,8,9,10], y = wcss))


    fig.update_layout(title='WCSS vs. Cluster number',
                       xaxis_title='Clusters',
                       yaxis_title='WCSS')
    return fig
                     
tab_clustering_data_prep = html.Div([
        dbc.Row([
            
            dbc.Toast([
                html.P("""
                       The Food and Nutrient Database for Dietary Studies (FNDDS) is a database application developed to analyze dietary intake data from the National Health and Nutrition Examination Survey (NHANES) and What We Eat in America. It uses reported food and beverage portions to determine their nutrient values, converting them into gram amounts.
                       """),
                       
                html.P("""
                       NHANES is a nationally representative survey that aims to monitor the health and nutritional status of the civilian, noninstitutionalized U.S. population. Conducted by the National Center for Health Statistics, a division of the Centers for Disease Control and Prevention, NHANES is an ongoing survey that releases data every two years, with around 9,000 participants per cycle from sampled counties across the country.
                       """)                   ,
                html.P("""
                       The Food Surveys Research Group, which is part of the Beltsville Human Nutrition Research Center of the USDA's Agricultural Service, is responsible for the dietary data collection methodology and maintenance of the databases used to process the data. Trained interviewers use the USDA Automated Multiple-Pass Method (AMPM), a 5-step process, to collect dietary intake information.
                       """),
                html.P("""
                       We can see that the above dataset has a lot of features and using high diemensional data may not be suitable for the purpose of clustering. Hence, we categories the nutrient value of food as the follows -                      
                       """),
                html.Ul([
                    html.Li("Macro nutrients - cluster analysis based on 'Protein (g)','Carbohydrate (g)', 'Sugars, total\n(g)', 'Fiber, total dietary (g)'"),
                    html.Li("Vitamins - cluster analysis based on vitamins and minerals content of the foods"),
                    html.Li("Fatty acids - cluster analysis based on nutrients such as cholesterol and other saturated/unsaturated fatty acids."),
                    html.Li("Minerals"),
                    html.Li("Amino Acids")
                ]),
                dbc.Button("View Raw Data", id="raw-nut-val-df-button", n_clicks=0, className="me-1"),
                dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Raw Dataset - Fluid Milk")),
                        dbc.ModalBody(plot_table(nut_val_df))
                    ],
                  id="raw-nut-val-df-modal",
                  size="xl",
                  is_open=False,
                ),
            ], header = "About the Data", style = {"width":"100%"})
                                 
        ]),
        html.Br(),
        html.Hr(),
        dbc.Row([
            dbc.Toast([
                dcc.Graph(figure = plot_nutrient_contributors()),
                html.Br(),
                html.Hr(),
                html.P("""
                       Bar plots are a useful tool for exploring data in unsupervised learning because they allow us to visualize the distribution of categorical variables within a dataset. Unsupervised learning algorithms are used to identify patterns or clusters in data without using labeled data, and bar plots can help us understand the distribution of different categories or groups within the data.
                       """),
                html.P("""
                       From the above bar plot, we can view the top food contributors to the nutrients we are clustering.
                       """)
            ], header = "Top 20 Contributors", style = {"width":"100%"})
        ]),
        html.Br(),
        html.Hr(),
        html.H4("Correlation heatmaps for clustering analysis"),
        html.P("""
               In clustering, the goal is to group together data points or objects that are similar to each other. Correlation heatmaps can help us identify variables that are highly correlated with each other, indicating that they may be capturing similar or related information. This can be useful for identifying redundant variables that may not be necessary for clustering, or for understanding the underlying structure of the data.
               """),
        html.P("""
               Correlation heatmaps can also help us identify variables that are important for clustering. If we see a strong correlation between a certain variable and the clusters that are formed, it may indicate that this variable is particularly important for distinguishing between different groups of data points.

                Overall, correlation heatmaps can be a useful tool for exploring relationships between variables in a dataset and identifying patterns that may be useful for clustering analysis.
               """),
        html.Br(),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    dcc.Graph(figure = plot_corr_matrix(df_macros, macro_nutrients)),
                    html.Br(),
                    html.Hr(),
                    html.P("From the above plot, we can observe that there is a high correlation between Sugars and Carbohydrates. Hence, one of these can be eliminated during cluster visualization")
                ], header = "Macros",style = {"width":"100%"})
            ]),
            dbc.Col([
                dbc.Toast([
                    dcc.Graph(figure = plot_corr_matrix(df_fatty_acids, fatty_acids)),
                    html.Br(),
                    html.Hr(),
                    html.P("From the above plot, we can observe that there is a high correlation between Poly-unsaturated and mono-unsaturated fatty acids. Further, a high correlation between mono-unsaturated and total-saturated can also be observed. Hence, one of these can be eliminated during cluster visualization")
                ], header = "Fatty Acids",style = {"width":"100%"})
            ])
        ]),
        html.Br(),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    dcc.Graph(figure = plot_corr_matrix(df_minerals, minerals)),
                    html.Br(),
                    html.Hr(),
                    html.P("")
                ], header = "Minerals",style = {"width":"100%"})
            ]),
            dbc.Col([
                dbc.Toast([
                    dcc.Graph(figure = plot_corr_matrix(df_vitamins, vitamins)),
                    html.Br(),
                    html.Hr(),
                    html.P("From the above plot, we can observe a strong correlation between Vitamin B-12 and Vitamin A")
                ], header = "Vitamins",style = {"width":"100%"})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    dcc.Graph(figure = plot_corr_matrix(df_aminos, amino_acids)),
                    html.Br(),
                    html.Hr(),
                    html.P("From the above plot, a strong correlation between multiple amino acid groups is observed")
                ], header = "Amino Acids",style = {"width":"100%"})    
            ])
        ])
    ])                  
                       
tab_clustering_results = html.Div([
        html.Br(),
        
        dbc.Tabs([
             dbc.Tab([
                 html.Br(),
                 html.Hr(),
                 html.H6("Elbow Method For Clustering"),
                 html.Hr(),
                 html.P("""
                        The elbow method is a commonly used technique in clustering analysis to determine the optimal number of clusters to use for a given dataset. It is called the elbow method because the plot of the number of clusters against the within-cluster sum of squares (WCSS) creates a shape that resembles an elbow.
                        """),
                 html.P("""
                        The WCSS is a measure of the variation within each cluster, and the goal of clustering is to minimize this value. However, as the number of clusters increases, the WCSS tends to decrease. This is because each additional cluster allows for a more detailed grouping of data points, which reduces the within-cluster variation.
                        """),
                 html.P("""
                        The elbow method involves plotting the number of clusters against the WCSS and looking for the point of inflection or "elbow" in the plot. This point represents the number of clusters at which the addition of another cluster does not significantly decrease the WCSS. In other words, it represents the point at which the clustering algorithm is no longer capturing significantly more information by adding additional clusters.
                        """),
                 html.Hr(),
                 html.Br(),
                 dbc.Row([
                     dbc.Toast([
                         dcc.Graph(figure = plot_wcss_elbow(df_macros))
                     ], header = "Elbow method to determine the value of K", style = {"width":"100%"})
                 ]),                 
                 html.Br(),
                 html.H6("2-D Clustering"),
                 html.Hr(),
                 dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown([1,2,3,4,5], 2, id = "k-dropdown")
                        ], header = "Select the Value of K", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown(macro_nutrients, macro_nutrients[1], id = "k-features1")
                        ], header = "Select First Diemension Feature", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown(macro_nutrients, macro_nutrients[2], id = "k-features2")
                        ], header = "Select Second Diemension Feature", style = {"width":"100%"})    
                    ])
                    
                 ]),
                 html.Hr(),
                 html.Br(),
                 html.P("""
                        The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters in the dataset. It is a commonly used evaluation metric for clustering algorithms, and can help determine the optimal number of clusters to use for a given dataset.
                        The silhouette score ranges from -1 to 1, with a score of 1 indicating that the object is very similar to its own cluster and very dissimilar to other clusters, and a score of -1 indicating the opposite. A score close to 0 indicates that the object is close to the boundary between two clusters and could potentially belong to either one.
                        """),
                 html.Br(),
                 html.Hr(),
                 dbc.Row([
                     dcc.Graph(id = "sillhoutte-figure")    
                 ]),
                 html.Br(),
                 html.H6("3-D Clustering"),
                 html.Hr(),
                 dbc.Row([
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown([1,2,3,4,5], 2, id = "k-dropdown3d")
                        ], header = "Select the Value of K", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown(macro_nutrients, macro_nutrients[1], id = "k-features3d-1")
                        ], header = "Select First Diemension Feature", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown(macro_nutrients, macro_nutrients[2], id = "k-features3d-2")
                        ], header = "Select Second Diemension Feature", style = {"width":"100%"})    
                    ]),
                    dbc.Col([
                        dbc.Toast([
                            dcc.Dropdown(macro_nutrients, macro_nutrients[3], id = "k-features3d-3")
                        ], header = "Select Second Diemension Feature", style = {"width":"100%"})    
                    ])                    
                 ]),
                 dcc.Graph(id = "kmeans3d-figure")
             ], label = "K-Means Clustering"),
             dbc.Tab([
                 dbc.Row([
                     dbc.Toast([
                         html.P("""
                                For hierarchical clustering we base our results on different food categories as shown in the below bar chart.
                                The dataset is such that for different categories of food there are unique descriptions of food. 
                                To reduce the size of the dendrogram for cleaner visualization and better understanding, hierarchical clustering is performed
                                on these food categories and a dendrograms are displayed below.
                                """)
                     ], header = "Basis for Hierarchical Clustering", style = {"width":"100%"})    
                 ]),
                 
                 dbc.Row([
                     dbc.Toast([
                         html.Img(src = "static/images/Distribution_food.png", style = {"width":"100%"})    
                     ], header = "Distribution of Top 20 categories", style = {"width":"100%"})
                 ]),
                 
                 dbc.Row([
                     dbc.Toast([
                         html.Img(src = "static/images/veggies _dend .png", style = {"width":"100%"})    
                     ], header = "Dendrogram of Vegetables", style = {"width":"100%"})
                 ]),
                 
                 dbc.Row([
                     dbc.Toast([
                         html.Img(src = "static/images/cheese _dend .png", style = {"width":"100%"})    
                     ], header = "Dendrogram of Cheese", style = {"width":"100%"})
                 ]),
                 dbc.Row([
                     dbc.Toast([
                         html.Img(src = "static/images/liqour _dend .png", style = {"width":"100%"})    
                     ], header = "Dendrogram of Liquor", style = {"width":"100%"})
                 ]),                 
                 dbc.Row([
                     dbc.Toast([
                         html.Img(src = "static/images/dougnuts _dend .png", style = {"width":"100%"})    
                     ], header = "Dendrogram of Doughnuts", style = {"width":"100%"})
                 ]),
                 dbc.Row([
                     dbc.Toast([
                         html.Img(src = "static/images/fish _dend .png", style = {"width":"100%"})    
                     ], header = "Dendrogram of fishes", style = {"width":"100%"})
                 ]),                 
                 
             ], label = "Hierarchical Clustering")
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

tab_clustering = html.Div([
        dbc.Tabs([
            dbc.Tab([            
                html.Br(),
                html.Hr(),
                tab_clustering_overview
            ], label = "Overview"),
            
            dbc.Tab([
                html.Br(),
                html.Hr(),
                tab_clustering_data_prep
            ], label = "Data Preparation"),
            
            dbc.Tab([
               tab_clustering_results
            ], label = "Results and Conclusions")
                
        ])
    ])

arm_raw_df = pd.read_csv("ARM/Groceries_dataset.csv")
arm_processed_df = pd.read_csv("ARM/ItemList.csv")
tab_arm = html.Div([
        html.H4("Overview"),        
        html.Hr(),
        html.Br(),
        dbc.Row([            
            dbc.Toast([
                html.P("""
                       Association rule mining is a data mining technique used to discover interesting relationships, patterns, or associations between variables or items in large datasets. The goal of association rule mining is to identify rules that describe the co-occurrence of different items in a dataset.
                       """),
                html.P("""
                       The input data for association rule mining typically consists of a set of transactions or events, where each transaction includes a list of items. For example, in a grocery store, a transaction may be a customer's purchase, and the items may include bread, milk, and eggs.
                       """),
                html.P("""
                       The process of association rule mining involves finding patterns in the transactions to identify the relationships between different items. This is done by first identifying the frequent itemsets, which are the sets of items that occur together frequently in the transactions. The frequent itemsets can be found using algorithms such as Apriori, FP-Growth, or Eclat.
                       """),
                html.P("""
                       Once the frequent itemsets are identified, the next step is to generate the association rules, which describe the relationships between the items. An association rule is a statement of the form "if item A is present, then item B is also likely to be present." Each association rule consists of two parts: an antecedent and a consequent. The antecedent is the set of items that is used to predict the occurrence of the consequent.
                       """),
                html.P("""
                       The quality of an association rule is measured using metrics such as support, confidence, and lift. Support measures the frequency of occurrence of the itemset, confidence measures the conditional probability of the consequent given the antecedent, and lift measures the strength of the association between the antecedent and the consequent.
                       """)                       
            ], header = "What is Association Rule Mining?",style = {"width":"100%"})    
          
        ]),
        html.Br(),
        dbc.Row([
            dbc.Toast([
                html.Img(src = "static/images/supp_conf_lift.png", style = {"width":"60%"}),
                html.P("""
                       Support: Support is a measure of the frequency of occurrence of an itemset in the dataset. It is calculated as the proportion of transactions in the dataset that contain the itemset. For example, if we have a dataset of 1,000 transactions, and the itemset {milk, bread} appears in 200 transactions, then the support of {milk, bread} is 200/1,000 = 0.2.
                       """),
                html.P("""
                       Confidence: Confidence is a measure of the strength of the association between the antecedent and the consequent in a rule. It is calculated as the proportion of transactions that contain both the antecedent and the consequent, out of the transactions that contain the antecedent. For example, if we have a rule {milk} → {bread} with 100 transactions containing both milk and bread, and 200 transactions containing milk, then the confidence of the rule is 100/200 = 0.5.
                       """),
                html.P("""
                       Lift: Lift is a measure of the strength of the association between the antecedent and the consequent in a rule, compared to what would be expected if they were independent. A lift value of 1 indicates that the antecedent and consequent are independent, while a lift value greater than 1 indicates a positive association between them. Lift is calculated as the ratio of the support of the itemset containing both the antecedent and the consequent to the product of the supports of the antecedent and the consequent. For example, if we have a rule {milk} → {bread} with a support of 0.2, and the support of milk is 0.4 and the support of bread is 0.3, then the lift of the rule is (0.2) / (0.4 * 0.3) = 1.67. This indicates that the presence of milk is associated with 67% higher likelihood of the presence of bread, compared to what would be expected if milk and bread were independent.
                       """),
                
            ], header = ["Support, Confidence and Lift"],style = {"width":"100%"})    
        ]),
        html.Br(),
        html.Hr(),
        html.H4("Data Prep"),        
        html.Hr(),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    html.P("""
                           To perform association rule mining for food demand prediction, you will need a large dataset of food sales transactions. Each transaction should include information about the items purchased, the quantity of each item, and the date and time of the transaction. This data can be obtained from point of sale systems or online food ordering platforms.
                           """),
                    html.P("""
                           Groceries Dataset: This dataset contains over 9,000 transactions from a grocery store in Germany. It includes information about the items purchased, such as bread, milk, and vegetables. You can download the dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/groceries
                           """),
                    dbc.Button("View Raw Data", id="raw-arm-df-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Raw Dataset ARM")),
                            dbc.ModalBody(plot_table(arm_raw_df))
                        ],
                      id="raw-arm-df-modal",
                      size="xl",
                      is_open=False,
                    ),
                           
                ], header = ["Raw Data"],style = {"width":"100%"})    
            ]),
            dbc.Col([
                dbc.Toast([
                    html.P("""
                           This dataset consists of groceries purchased by customers on a given date. Hence, to convert them into transaction, a group by operation is performed by the 'date' column and the items purchased on that day are converted into a list.
                           Further, this dataset was read in as a transaction dataset in R.
                           """),
                    dbc.Button("View Processed Data", id="processed-arm-df-button", n_clicks=0, className="me-1"),
                    dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Processed Dataset ARM")),
                            dbc.ModalBody(plot_table(arm_processed_df))
                        ],
                      id="processed-arm-df-modal",
                      size="xl",
                      is_open=False,
                    ),
                ], header = ["Processed Data"],style = {"width":"100%"})
            ])
        ]),
        html.Br(),
        html.Hr(),
        html.H4("Results"),        
        html.Hr(),
        html.Br(),
        dbc.Row([
           
            dbc.Toast([
                html.Img(src = "static/images/freq_plot_arm.png", style = {"width":"100%"})
            ], header = "Frequency Plot", style = {"width":"100%"})
              
        ]),
        html.Hr(),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Toast([
                    html.Img(src = "static/images/scatter_arm.png", style = {"width":"100%"})
                ], header = "Scatter Plot of Rules", style = {"width":"100%"})
            ]), 
            dbc.Col([
                dbc.Toast([
                    html.Img(src = "static/images/items_lhs.png")
                ], header = "Items in LHS Group", style = {"width":"100%"})
            ]), 
        ]),
        html.Hr(),
        html.Br(),
        dbc.Row([
            
            dbc.Toast([
                html.Img(src = "static/images/arm_network.png")
            ], header = "Network Plot", style = {"width":"100%"})
              
        ]),
        html.Hr(),
        html.Br(),
        dbc.Row([
            
            dbc.Toast([
                html.Img(src = "static/images/arm_parallel.png")
            ], header = "Parallel Plot", style = {"width":"100%"})
              
        ]),
    ])

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
                dcc.Location(id="url"), sidebar, content    
            ])



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
    
    elif pathname == "/clustering":
        return tab_clustering
    elif pathname == "/arm":
        return tab_arm
    
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

def plot_sillhoutte(k, feat1, feat2):
    n_clusters = k
    X = df_macros.iloc[:,1:].values    
    fig = tools.make_subplots(rows=1, cols=2,
                                  print_grid=False,
                                  subplot_titles=('The silhouette plot for the various clusters.',
                                                  'The visualization of the clustered data.'),
                             )
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    fig['layout']['xaxis1'].update(title='The silhouette coefficient values',
                                   range=[-0.1, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    fig['layout']['yaxis1'].update(title='Cluster label',
                                   showticklabels=False,
                                   range=[0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        #colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #cmap = cm.get_cmap("nipy_spectral")
        #colors = cmap(cluster_labels.astype(float) / n_clusters)    
        filled_area = go.Scatter(y=np.arange(y_lower, y_upper),
                                 x=ith_cluster_silhouette_values,
                                 mode='lines',
                                 showlegend=False,
                                 line=dict(width=0.5),
                                  #color=colors),
                                 fill='tozerox')
        fig.append_trace(filled_area, 1, 1)
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    
    # The vertical line for average silhouette score of all the values
    axis_line = go.Scatter(x=[silhouette_avg],
                           y=[0, 10],
                           showlegend=False,
                           mode='lines',
                           line=dict(color="red", dash='dash',
                                     width =1) )
    
    fig.append_trace(axis_line, 1, 1)
    
    # 2nd Plot showing the actual clusters formed
    #colors = matplotlib.colors.colorConverter.to_rgb(cm.spectral(float(i) / n_clusters))
    #colors = 'rgb'+str(colors)
    feat1_index = df_macros.columns.get_loc(feat1)
    feat2_index = df_macros.columns.get_loc(feat2)
    clusters = go.Scatter(
                x=X[:,feat1_index],
                y=X[:,feat2_index],
                mode='markers',
                showlegend=False,
                marker=dict(
                    size=12,
                    color=cluster_labels,                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            )
    fig.append_trace(clusters, 1, 2)
    
    # Labeling the clusters
    centers_ = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    centers = go.Scatter(x=centers_[:, 0], 
                         y=centers_[:, 1],                    
                         showlegend=False,
                         mode='markers',
                         marker=dict(color='green', size=10,
                                     line=dict(color='black',
                                                             width=1))
                        )
    
    fig.append_trace(centers, 1, 2)
    
    fig['layout']['xaxis2'].update(title='Feature space for the 1st feature',
                                   zeroline=False)
    fig['layout']['yaxis2'].update(title='Feature space for the 2nd feature',
                                  zeroline=False)
    
    
    fig['layout'].update(title="Silhouette analysis for KMeans clustering on sample data "
                         "with n_clusters = %d" % n_clusters)
    return fig

def plot_clusters(n_clusters, X_, Y, Z):
    
    X = df_macros.iloc[:,1:].values    
    kmeans = KMeans(n_clusters = n_clusters, init="k-means++", max_iter = 500, n_init = 10, random_state = 123)
    identified_clusters = kmeans.fit_predict(X)
    data_with_clusters = df_macros.iloc[:,1:].copy()    
    data_with_clusters['Cluster'] = identified_clusters    
    fig = px.scatter_3d(data_with_clusters, x = X_, y=Y, z=Z,
                  color='Cluster', opacity = 0.8)
    fig.update_layout(width = 750, height = 750)
    return fig

@app.callback(
    Output("sillhoutte-figure", "figure"),
    [Input("k-dropdown","value"), Input("k-features1","value"), Input("k-features2","value")]    
)
def update_kmeans_graph(k, feat1, feat2):
    fig = plot_sillhoutte(k, feat1, feat2)
    return fig

@app.callback(
    Output("kmeans3d-figure", "figure"),
    [Input("k-dropdown3d","value"), 
     Input("k-features3d-1","value"), Input("k-features3d-2","value"),
     Input("k-features3d-3","value")]    
)
def update_kmeans_graph3d(k, feat1, feat2, feat3):
    fig = plot_clusters(k, feat1, feat2, feat3)
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
    Output("raw-nut-val-df-modal", "is_open"),
    Input("raw-nut-val-df-button", "n_clicks"),
    State("raw-nut-val-df-modal", "is_open"),
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

app.callback(
    Output("raw-arm-df-modal", "is_open"),
    Input("raw-arm-df-button", "n_clicks"),
    State("raw-arm-df-modal", "is_open"),
)(toggle_modal)
app.callback(
    Output("processed-arm-df-modal", "is_open"),
    Input("processed-arm-df-button", "n_clicks"),
    State("processed-arm-df-modal", "is_open"),
)(toggle_modal)                   

if __name__ == "__main__":
    app.run_server(debug=True)