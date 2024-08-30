import dash; import os; import sys; import csv; import numpy as np
from dash.dependencies import Input, Output
from dash import dcc, html
import chart_studio; 
import flask; import glob; 
import chart_studio.plotly as py
import plotly.graph_objs as go; 
import plotly.express as px; import pandas as pd; import base64
import Classifier as classify; 
import Preprocessor as UD; import os
from Preprocessor import wordcloudall, wordcloudneg, wordcloudneu, wordcloudpos
from Classifier import generate_mnb_pie_chart, generate_knn_pie_chart

from PIL import Image
pil_image = Image.open("C:\\Users\\Dell\\Documents\\Sentiment Analysis\\Sentiment Analysis Main\\YouTube-Emblem.png")

path = "C:\\Users\\Dell\Desktop\\Sentiment Analysis Main\\"

colorPalatte= {'positive':'#d90429', 'neutral': '#edeff4', 'negative':'#2b2d42'}
colorP = ['#d90429', '#2b2d42', '#edeff4']

mydict = {'label_mnb':'Multinomial Naive Bayes','label_knn': 'K - Nearest Neighbour'}
model_options = ['label_mnb', 'label_knn']

df = UD.df # labeled dataset
positive = UD.most_positive
negative = UD.most_negative
neutral = UD.most_neutral
vader_counts = UD.vader_counts
textblob_counts = UD.textblob_counts
top10 = UD.top10

mnb_ratios=classify.mnb_ratios
knn_ratois=classify.knn_ratios


df.rename(columns={'vader_comp_sentiment': 'sentiment score'}, inplace=True)

# Extract the first and second elements as x and y values
x = np.array([tup[0] for tup in top10])
y = np.array([tup[1] for tup in top10])

most_freq = go.Bar(
            x=x, 
            y=y, 
            name="Most Freq",
            marker=dict(color=colorPalatte["positive"])
        )
'''
plot_data = pd.DataFrame({
    'Sentiment': vader_counts.index.tolist() * 2,
    'Method': ['VADER'] * 3 + ['TextBlob'] * 3,
    'Count': list(vader_counts.values) + list(textblob_counts.values)
    })
# Plot the grouped bar chart
fig = px.bar(plot_data, x='Sentiment', y='Count', color='Method',
             labels={'Sentiment': 'Sentiment', 'Count': 'Number of Comments'},
             title='Sentiment Analysis Comparison (VADER vs TextBlob)',
             color_discrete_sequence = ['#d90429', '#2b2d42'])

updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
             dict(label='Positive',
                  method='update',
                  args=[{'visible': [True, False]},
                        {'title': 'Frequent Words'}]
                  ),
             dict(label='Comparison',
                  method='update',
                  args=[{'visible': [False, True]},
                        {'title': 'VADER vs. Textblob Sentiments Comparison'}]
                  )
         ]),
         pad={'r': 15, 't': 10},
         )
])

fig.update_traces=True'''

def generate_table(dataframe, max_rows=10):
    return html.Table(
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app = dash.Dash()

app.layout = html.Div([ 
    html.Div([
        html.Img(src=pil_image, style={'width': '100px', 
                                       'height': '50px'}), 
        # Header
        html.H1(children='YouTube Comment Analysis',
        style={'display': 'inline-block', 
            'margin': '10px',
            'padding': '5px 5px 0px 5px',
            'text-align': 'center',
            #'font-size': '40px',
            'font-family': 'Trade Gothic, sans-serif'
            }),
        html.H2(children='Charting Emotions', 
                style={'display': 'inline-block',
                       'padding': '0px',
                        'text-align': 'center',
                        'font-size': '20px',
                        'font-family': 'Trade Gothic, sans serif',
                        'font-style': 'italic'}
                     ),
         ],
    
      style={'display': 'flex', 
           'justify-content': 'space-between', 
           'align-items': 'center', 
           'background-color': '#f0f0f0', 
           'padding': '10px'}),

    # Main content
        # Pie Chart
        html.Div([ 
            dcc.Dropdown(
                id="MyModel",
                options=[{
                    'label': mydict.get(str(i)),
                    'value': i
                } for i in model_options],
                value='All Models',
                style={'font-family': 'Trade Gothic, sans-serif'}
        ),
            dcc.Graph(id='pie-graph')
        ],
            style={
                'float': 'left',
                'width': '40.00%',
                'padding': '0px 20px 0px 10px',
                'height': '100px'}
        ),
    # Bar Chart
    html.Div([
            dcc.Graph(
                    id='bar-graph',
                    figure={
                        'data': [most_freq],
                        'layout': go.Layout(title='Most Frequent Words',
                                            title_font=dict(family='Trade Gothic, sans-serif', size=20),
                        barmode='stack', showlegend=True,
                        )
                            },
                    style={
                    'float': 'right',
                    'width': '55.00%',
                    'padding': '5px 10px 10px 10px',
                    'height': 'auto'
                    }
                    )
                ]),
    
       html.Div([
           dcc.Dropdown( 
               id='wordcloud-dropdown', 
               options=[ 
                   {'label': 'All Words', 'value': 'all'},
                   {'label': 'Positive Words', 'value': 'positive'},
                   {'label': 'Negative Words', 'value': 'negative'},
                   {'label': 'Neutral Words', 'value': 'neutral'},
        ],
        value='all',
        style={'width': '70%'}
    ), 
    html.Img(id='wordcloud-image', style={'width': '60%', 'height': 'auto', 'padding': '20px 0px 0px'}),
    ], 
           style={'flex': '1'}),
    
    # Dropdown Table of Comments
    html.Div([
        dcc.Dropdown( 
            id='my-table-dropdown',
            options=[{'label': i, 'value': i}
            for i in ['All Comments', 'Positive', 'Neutral', 'Negative']
            ],
            #set to display al comments
            value='All Comments'),
        html.Div(id='table-container')
        ],
            style={'width': '45%',
                   'height': '400px', 'overflow-x': 'auto', 'overflow-y': 'auto', 'margin-top': '20px',
            'display': 'inline-block',
            'padding': '0px 5px 5px 10px',
            'font-family': 'Trade Gothic, sans-serif'}
            ),
])


# Callback to update the pie chart based on the selected model
@app.callback(
    Output('pie-graph', 'figure'),
    [Input('MyModel', 'value')]
)
def update_pie_chart(selected_model):
    if selected_model == 'label_mnb':
        return generate_mnb_pie_chart()
    elif selected_model == 'label_knn':
        return generate_knn_pie_chart()
    else:
        return generate_mnb_pie_chart()
    
# Callback to update wordcloud image based on dropdown selection
@app.callback(
    Output('wordcloud-image', 'src'),
    [Input('wordcloud-dropdown', 'value')]
)
def update_wordcloud(selected_wordcloud):
    if selected_wordcloud == 'all':
        return wordcloudall()
    elif selected_wordcloud == 'positive':
        return wordcloudpos()
    elif selected_wordcloud == 'negative':
        return wordcloudneg()
    elif selected_wordcloud == 'neutral':
        return wordcloudneu()
    else:
        return wordcloudall()

# table of comments
@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('my-table-dropdown', 'value')])
def table_update(value):
    simple_df = df[["sentiment score","comment"]]
    selected = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    if value != "All Comments":
        filtered_df = simple_df[simple_df["sentiment score"]==selected.get(value)]
    else:
         filtered_df = simple_df
    return generate_table(filtered_df)

if __name__ == '__main__':
    app.run_server(debug=False)
