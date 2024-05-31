import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from wordcloud import WordCloud
from io import BytesIO
import base64

# Prepare data - only 1st time
df = pd.read_csv('clean_data.csv')
df['Comment_Status'] = df['Comment'].apply(lambda x: 'No' if pd.isna(x) else 'Yes')
df['Sentiment'] = df['Positive_Score'].apply(lambda x: 'NaN' if pd.isna(x) else ('Positive' if x > 0.5 else 'Negative'))

### Reuse function when filtered dataframe
def df_yn_comment(df):
    no_df = df[df['Comment_Status']=='No'].iloc[:,:5]
    yes_df = df[df['Comment_Status']=='Yes']
    return no_df, yes_df

def comment2df(df):
    text_set = " ".join(df["Comment"].astype(str))
    check_list = [' ','.',',','ค่ะ','ครับ','เ','']
    tokens = word_tokenize(text_set, keep_whitespace=False)
    tokens = [token for token in tokens if token not in check_list and token not in thai_stopwords()]
    word_counts = {}
    for token in tokens:
        if token in word_counts:
            word_counts[token] += 1  # Increment count if word already exists
        else:
            word_counts[token] = 1  # Initialize count to 1 for new words
    df_words = pd.DataFrame(word_counts.items(), columns = ['word','count'])
    return df_words

def df_groupby(df):
    df_gb_gender = df.groupby('Gender').describe()
    df_gb_age = df.groupby('Age_Range').describe()
    df_gb_sentiment = df.groupby('Sentiment').describe().iloc[1:,:]
    df_gb_status = df.groupby('Comment_Status').describe()
    yes_df_gb_gender_sentiment = yes_df.groupby(['Gender','Sentiment']).describe()
    yes_df_gb_age_sentiment = yes_df.groupby(['Age_Range','Sentiment']).describe()
    return df_gb_gender, df_gb_age, df_gb_sentiment, df_gb_status, yes_df_gb_gender_sentiment, yes_df_gb_age_sentiment

def tranformed_df(df):
    count_surveys = df['ID'].count()
    count_yes = yes_df['ID'].count()
    df_sum = df.describe().loc[['max','min','mean'],:].reset_index()
    yes_df_sum = yes_df.describe().loc[['max','min','mean'],:].reset_index()
    df_point_age = df_gb_age['Mean_Point'].reset_index()
    df_point_gender = df_gb_gender['Mean_Point'].reset_index()
    df_count_status = df.groupby('Comment_Status').count().reset_index()
    df_count_sentiment = df_gb_sentiment['ID'].reset_index()
    yes_count_gender_sentiment = yes_df_gb_gender_sentiment['Mean_Point'].reset_index()
    yes_count_age_sentiment = yes_df_gb_age_sentiment['Mean_Point'].reset_index()
    table_compare = df_gb_status['Mean_Point'].reset_index()
    return count_surveys, count_yes, df_sum, yes_df_sum, df_point_age, df_point_gender, df_count_status, df_count_sentiment, yes_count_gender_sentiment, yes_count_age_sentiment, table_compare

no_df, yes_df = df_yn_comment(df)
df_gb_gender, df_gb_age, df_gb_sentiment, df_gb_status, yes_df_gb_gender_sentiment, yes_df_gb_age_sentiment = df_groupby(df)
count_surveys, count_yes, df_sum, yes_df_sum, df_point_age, df_point_gender, df_count_status, df_count_sentiment, yes_count_gender_sentiment, yes_count_age_sentiment, table_compare = tranformed_df(df)
df_words = comment2df(yes_df)
def top10(df):
    grouped_data_bar = df.groupby('word')['count'].sum()
    top_10_data = grouped_data_bar.sort_values(ascending=False).head(10)
    top_10_data = np.flip(top_10_data)
    return top_10_data
top10 = top10(df_words)
###

# Create charts
def create_bar_chart(data, x, y, title, x_label, y_label, color, color_map):
    fig = px.bar(
        data, x=x, y=y,
        barmode='group', title=title, color=color,
        color_discrete_map=color_map
    )
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig
def create_bar_chart_ng(data, x, y, title, x_label, y_label, color):
    fig = px.bar(
        data, x=x, y=y, barmode='group',title=title
    )
    fig.update_traces(marker_color=color)
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return fig

def create_hbar_chart(data, x, y, title, x_label, y_label, color):
    fig = px.bar(
        data, x=y, y=x,
        barmode='group', title=title,
        orientation='h'
    )
    fig.update_traces(marker_color=color)
    fig.update_layout(xaxis_title=y_label, yaxis_title=x_label)
    return fig

def create_pie_chart(data, names, values, title, color, color_map):
    fig = px.pie(data, names=names, values=values, title=title, color=color, color_discrete_map=color_map)
    return fig

def create_word_cloud(data):
    wt = list(data['word'])
    wordcloud = WordCloud(
                      font_path='tahoma.ttf', # font ที่เราต้องการใช้ในการแสดงผล เราเลือกใช้ tahoma 
                      stopwords=thai_stopwords(), # stop words ที่ใช้ซึ่งจะโดนตัดออกและไม่แสดงบน words cloud 
                      relative_scaling=0.3,
                      min_font_size=1,
                      background_color = "white",
                      width=800,
                      height=400,
                      max_words=500, # จำนวนคำที่เราต้องการจะแสดงใน Word Cloud
                      colormap='summer', 
                      scale=3,
                      font_step=4,
                      collocations=False,
                      regexp=r"[ก-๙a-zA-Z']+", # Regular expression to split the input text into token
                      margin=2
                      ).generate(' '.join(wt)) # input คำที่เราตัดเข้าไปจากตัวแปร wt ในรูปแบบ string
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Encode the plot in base64
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')  
    # Return the base64 encoded image
    return encoded_image

app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

# STYLE
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "17rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
DATEPICKER_STYLE = {
    "margin-left": "18rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
    # "fontSize": "0.5rem",
}
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
}

# Plotly Express - Global Font Style
global_font_style = { # Define your global font style
    'family': 'Tahoma, sans-serif',
    'size': 12,
    # 'color': '#333'  # Change color as needed
}
template = { # Define a customized template with global font properties
    'layout': {
        'font': global_font_style
    }
}
px.defaults.template = template # Update Plotly Express defaults with the customized template
# Plotly Express - Global Chart Color
minty_colors = {
    'primary': '#AAD9BB',
    'secondary': '#9DBC98',
    'accent': '#80BCBD',
    'positive': '#92C7CF',
    'negative' : '#F9F7C9',
    # Add more colors as needed
}

## COMPONENTS
content = html.Div(id="page-content", style=CONTENT_STYLE)

sidebar = html.Div(
    [
        html.P("COMMENT ANALYSIS", className="display-6"),
        html.P("APPLICATION", className="lead"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("D A T A", href="/", active="exact"),
                dbc.NavLink("C O M M E N T", href="/page-1", active="exact"),
                dbc.NavLink("W O R D S", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.Br(),
        html.P(
            "Analyze how comment sentiment correlates with satisfaction scores. Gain a richer understanding of customer experience and identify areas for improvement.", className='blockquote-footer'
        ),
    ],
    style=SIDEBAR_STYLE,
)

date_picker = html.Div(
    dcc.DatePickerRange(
        id='my-date-picker-range',  # ID to be used for callback
        calendar_orientation='horizontal',  # vertical or horizontal
        day_size=39,  # size of calendar image. Default is 39
        end_date_placeholder_text="Return",  # text that appears when no end date chosen
        with_portal=False,  # if True calendar will open in a full screen overlay portal
        first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
        reopen_calendar_on_clear=True,
        is_RTL=False,  # True or False for direction of calendar
        clearable=True,  # whether or not the user can clear the dropdown
        number_of_months_shown=1,  # number of months shown when calendar is open
        min_date_allowed=df['Date'].min(),  # minimum date allowed on the DatePickerRange component
        max_date_allowed=df['Date'].max(),  # maximum date allowed on the DatePickerRange component
        initial_visible_month=df['Date'].max(),  # the month initially presented when the user opens the calendar
        start_date=df['Date'].min(),
        end_date=df['Date'].max(),
        display_format='Do MMM YY',  # how selected dates are displayed in the DatePickerRange component.
        month_format='MMMM YYYY',  # how calendar headers are displayed when the calendar is opened.
        minimum_nights=2,  # minimum number of days between start and end date

        persistence=True,
        persisted_props=['start_date'],
        persistence_type='session',  # session, local, or memory. Default is 'local'

        updatemode='singledate'  # singledate or bothdates. Determines when callback is triggered
    ), style = DATEPICKER_STYLE
)

# Cards
img_source1 = 'https://t4.ftcdn.net/jpg/04/85/09/55/360_F_485095547_SEFnUTEscD7auyTTBqmhd7hfRA99sKPP.jpg'
img_source2 = 'https://assets-global.website-files.com/64022de562115a8189fe542a/642da315621cd3ff8894f9d6_Rating-Scales-For-Surveys-1024x577.jpeg'
op_card1 = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Br(),
                html.H2(count_surveys, className="display-3"),
                html.P("number of surveys", className="card-text"),
                html.P(
                    "This information stems from customer satisfaction surveys conducted among individuals utilizing our company's services.", className='blockquote-footer'
                ),
            ]
        ),
        dbc.CardImg(src=img_source1, bottom=True),
    ],
    style={"height": "400px"},
)
op_cards_container1 = dbc.Col(
    html.Div(
        [
            op_card1
        ]
    ),width=3
)

cp_card1 = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Br(),
                html.H2(count_yes, className="display-3"),
                html.P("number of comment surveys", className="card-text"),
                html.P(
                    "Data from customers utilizing our service.", className='blockquote-footer'
                ),
            ]
        ),
        dbc.CardImg(src=img_source2, bottom=True),
    ],
    style={"height": "400px"},
)
cp_cards_container1 = dbc.Col(
    html.Div(
        [
            cp_card1
        ]
    ),width=3
)

# PAGE COMPONENTS
overall_page = dbc.Row(
    [
        html.H2(
                "ABOUT DATA", className="lead"
        ),
        op_cards_container1,
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='op_count_age',
                    figure=create_bar_chart_ng(
                        df_point_age, x='Age_Range', y='count',
                        title='Count by age range',
                        x_label='Age range', y_label='# of surveys',
                        color=minty_colors['primary']
                    )
                ), className='border rounded'
            ), width=5
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='op_count_gender',
                    figure=create_bar_chart_ng(
                        df_point_gender, x='Gender', y='count',
                        title='Count by gender',
                        x_label='Gender', y_label='# of surveys',
                        color=minty_colors['secondary']
                    )
                ), className='border rounded'
            ), width=4
        ),
        html.Hr(),

        html.H2(
                "SATISFACTION POINTS", className="lead"
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='op_stat_point',
                    figure=create_bar_chart_ng(
                        df_sum, x='index', y='Mean_Point',
                        title='Statistics on satisfaction point',
                        x_label='Stat', y_label='Satisfaction point',
                        color=minty_colors['accent']
                    )
                ), className='border rounded'
            ), width=4
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='op_age_mean',
                    figure=create_bar_chart_ng(
                        df_point_age, x='Age_Range', y='mean',
                        title='Satisfaction point by age range',
                        x_label='Age range', y_label='Satisfaction point',
                        color=minty_colors['primary']
                    )
                ), className='border rounded'
            ), width=4
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='op_gender_mean',
                    figure=create_bar_chart_ng(
                        df_point_gender, x='Gender', y='mean',
                        title='Satisfaction point by gender',
                        x_label='Gender', y_label='Satisfaction point',
                        color=minty_colors['secondary']
                    )
                ), className='border rounded'
            ), width=4
        ),
                
    ],
    className="g-2",
)

comments_page = dbc.Row(
    [
        # def create_pie_chart(data, names, values, title, color_map)
        html.H2(
                "CUSTOMERS' COMMENTS", className="lead"
        ),
        cp_cards_container1,
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_have_count', 
                    figure=create_pie_chart(
                        df_count_status,
                        names='Comment_Status',
                        values='ID',
                        title='Comment status',
                        color='Comment_Status',
                        color_map={
                            'Yes': minty_colors['primary'],
                            'No': minty_colors['secondary']
                        }
                    )
                ), className='border rounded'
            ), width=5
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_pn_count', 
                    figure=create_pie_chart(
                        df_count_sentiment,
                        names='Sentiment',
                        values='count',
                        title='Comment Sentiment',
                        color='Sentiment',
                        color_map={
                            'Positive': minty_colors['positive'],
                            'Negative': minty_colors['negative']
                        }
                    )
                ), className='border rounded'
            ), width=4
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_pn_count_gender',
                    figure=create_bar_chart(
                        yes_count_gender_sentiment, x='Gender', y='count',
                        title='Count number of sentiments by gender',
                        x_label='Gender', y_label='# of surveys',
                        color='Sentiment',
                        color_map={
                            'Positive': minty_colors['positive'],
                            'Negative': minty_colors['negative']
                        }
                    )
                ), className='border rounded'
            ), width=6
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_pn_count_age',
                    figure=create_bar_chart(
                        yes_count_age_sentiment, x='Age_Range', y='count',
                        title='Count number of sentiments by age range',
                        x_label='Age range', y_label='# of surveys',
                        color='Sentiment',
                        color_map={
                            'Positive': minty_colors['positive'],
                            'Negative': minty_colors['negative']
                        }
                    )
                ), className='border rounded'
            ), width=6
        ),
        html.Hr(),
        html.H2(
                "STATISTICS ON SATISFACTION POINTS OF CUSTOMERS WHO HAVE PROVIDED COMMENTS", className="lead"
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_stat_point',
                    figure=create_bar_chart_ng(
                        yes_df_sum, x='index', y='Mean_Point',
                        title='Statistics on the satisfaction points',
                        x_label='Stat', y_label='Satisfaction point',
                        color=minty_colors['accent']
                    )
                ), className='border rounded'
            ), width=4
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_p_stat_point',
                    figure=create_bar_chart_ng(
                        yes_df_sum, x='index', y='Positive_Score',
                        title='Statistics on Positive Sentiment',
                        x_label='Stat', y_label='Satisfaction point',
                        color=minty_colors['primary']
                    )
                ), className='border rounded'
            ), width=4
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_n_stat_point',
                    figure=create_bar_chart_ng(
                        yes_df_sum, x='index', y='Negative_Score',
                        title='Statistics on Negative Sentiment',
                        x_label='Stat', y_label='Satisfaction point',
                        color=minty_colors['secondary']
                    )
                ), className='border rounded'
            ), width=4
        ),
        html.Hr(),
        html.H2(
                "COMPARSION FOR WHO HAVE AND HAVE NO PROVIDED COMMENTS", className="lead"
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_compare_avg_point',
                    figure=create_bar_chart_ng(
                        table_compare, x='Comment_Status', y='mean',
                        title='Average of satisfaction points comparison',
                        x_label='Status', y_label='Satisfaction point',
                        color=minty_colors['accent']
                    )
                ), className='border rounded'
            ), width=6
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='cp_compare_std_point',
                    figure=create_bar_chart_ng(
                        table_compare, x='Comment_Status', y='std',
                        title='SD of satisfaction points comparison',
                        x_label='Status', y_label='Satisfaction point',
                        color=minty_colors['secondary']
                    )
                ), className='border rounded'
            ), width=6
        ),                
    ],
    className="g-2",
)
words_page = dbc.Row(
    [
        html.H2(
                "WORDS FREQUENCY", className="lead"
        ),
        dbc.Col(
            html.Div(
                html.Img(src='data:image/png;base64,{}'.format(create_word_cloud(df_words)))
                # id='wp_word_cloud'
                # , className='border rounded'
            ), width=8
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(
                    id='wp_word_count',
                    figure=create_hbar_chart(
                        top10, x=top10.index, y=top10.values,
                        title='TOP 10 Words',
                        x_label='', y_label='count',
                        color=minty_colors['secondary']
                    )
                ), className='border rounded'
            ), width=4
        ),
    ],
    className="g-2",
)

# Layout of the app
app.layout = html.Div(
    [
        dcc.Location(id="url"), 
        sidebar, 
        # date_picker,
        content,
    ]
)

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return overall_page
    elif pathname == "/page-1":
        return comments_page
    elif pathname == "/page-2":
        return words_page
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == '__main__':
    app.run_server(debug=True)

'''
@app.callback(
    # [Output('wp_word_count', 'figure'),
    # Output('wp_word_cloud', 'children')],
    [Output('op_count_age', 'figure'),
     Output('op_count_gender', 'figure'),
     Output('wp_word_count', 'figure'),],
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date')]
)
def update_chart(start_date, end_date):
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    no_df, yes_df = df_yn_comment(filtered_df)
    df_gb_gender, df_gb_age, df_gb_sentiment, df_gb_status, yes_df_gb_gender_sentiment, yes_df_gb_age_sentiment = df_groupby(filtered_df)
    count_surveys, count_yes, df_sum, yes_df_sum, df_point_age, df_point_gender, df_count_status, df_count_sentiment, yes_count_gender_sentiment, yes_count_age_sentiment, table_compare = tranformed_df(filtered_df)
    df_words = comment2df(yes_df)
    top10 = top10(df_words)
    op_bar1 =create_bar_chart_ng(df_point_age, x='Age_Range', y='count',title='Count by age range',x_label='Age range', y_label='# of surveys',color=minty_colors['primary'])
    op_bar2=create_bar_chart_ng(df_point_gender, x='Gender', y='count',title='Count by gender',x_label='Gender', y_label='# of surveys',color=minty_colors['secondary'])
    bar = create_hbar_chart(top10, x=top10.index, y=top10.values,title='TOP 10 Words',x_label='', y_label='count',color=minty_colors['secondary'])
    # plt_obj = create_word_cloud(df_words)
    # buf = BytesIO()
    # plt_obj.savefig(buf, format="png")
    # wordcloud_img = html.Img(src=f"data:image/png;base64,{base64.b64encode(buf.getbuffer()).decode()}")
    return op_bar1, op_bar2, bar
    # return wordcloud_img, bar
# def update_wordcloud(_):
#     plt_obj = create_word_cloud(df_words)
#     buf = BytesIO()
#     plt_obj.savefig(buf, format="png")
#     data = base64.b64encode(buf.getbuffer()).decode("utf8")
#     plt_obj.close()
#     return html.Img(src=f"data:image/png;base64,{data}")
'''
