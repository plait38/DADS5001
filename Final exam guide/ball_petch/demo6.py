import dash
from dash import dcc, html, Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash import dash_table
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import os
import base64
import io

from wordcloud import WordCloud, STOPWORDS
import re
from pythainlp.corpus.common import thai_stopwords
import emoji
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


# Set up the YouTube API service
youTubeApiKey = "AIzaSyBc8WJTSS37uiN_TRnffRNZEs6qqwHlrP4"
youtube = build('youtube', 'v3', developerKey=youTubeApiKey)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Timothy1337/finetuning-sentiment-all_df")
model = AutoModelForSequenceClassification.from_pretrained("Timothy1337/finetuning-sentiment-all_df")

#Thai stop word variable.
thai_stopwords = list(thai_stopwords())

## Declare combined_comments_df as a global variable
combined_comments_df = None
VideoFromChannel_df =None
PositiveCloud = None
NegativeCloud = None

####################### function to search channel url from channel name ###########################
'''this function return channel_url, channel_title, channel_id ควรประกาศให้เป็น global variable จะได้เอาไปใช้ต่อเรื่อยๆ'''
def get_channel_url(channel_name,num_results=4):
    # Search for channels with the specified name
    search_response = youtube.search().list(
        q=channel_name,
        part='id,snippet',
        type='channel',
        maxResults=num_results
    ).execute()

    # Extract channel ID
    if 'items' in search_response and len(search_response['items']) > 0:
        channel_id = search_response['items'][0]['id']['channelId']
        channel_title = search_response['items'][0]['snippet']['title']
        # Construct and return channel URL and title
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        return channel_url, channel_title, channel_id
    else:
        # If no channel found, return None
        return None, None, None
####################### function to get video from channel_id ###########################
def get_video_from_chanel(channel_id,max_results=10):
    
    channelStats = youtube.channels().list(part = "snippet,contentDetails,statistics", id=channel_id).execute()
    #We can define any playlist we want, however here we are using all videos
    allUploadedVideosPlaylist =  channelStats["items"][0]['contentDetails']['relatedPlaylists']['uploads']

    next_page_token = None
    videos = [ ]
    while len(videos) < max_results:
        playlistData = youtube.playlistItems().list(playlistId=allUploadedVideosPlaylist,
                                                part='snippet',
                                                maxResults=max_results if next_page_token is None else None,
                                                pageToken=next_page_token).execute()
        videos += playlistData['items']
        next_page_token = playlistData.get('nextPageToken')

        if next_page_token is None:
            break

    video_ids=[]

    for i in range(len(videos)):
        video_ids.append(videos[i]["snippet"]["resourceId"]["videoId"])
        i+=1

    videoStatistics = []

    for i in range(len(video_ids)):
        videoData = youtube.videos().list(id=video_ids[i],part = "statistics").execute()
        videoStatistics.append(videoData["items"][0]["statistics"])
        i+=1

    VideoTitle=[ ]
    url=[ ]
    Published = [ ]
    Views=[ ]
    #LikeCount=[ ]
    #DislikeCount=[ ]
    Comments=[ ]

    for i in range(len(videos)):
        VideoTitle.append((videos[i])['snippet']['title'])
        url.append("https://www.youtube.com/watch?v="+(videos[i])['snippet']['resourceId']['videoId'])
        Published.append((videos[i])['snippet']['publishedAt'])
        Views.append(int((videoStatistics[i])['viewCount']))
        #LikeCount.append(int((videoStatistics[i])['likeCount']))
        if ("commentCount" in videoStatistics[i]):
            Comments.append(int((videoStatistics[i])['commentCount']))
        else:
            Comments.append(0)

    data={"Video Title" : VideoTitle, "Video url" : url, "Published" : Published, "Views" : Views,  "Comments" : Comments,"video id":video_ids} #"Like Count" : LikeCount,
    VideoFromChannel_df=pd.DataFrame(data)
    return VideoFromChannel_df

############################ function to get comments from video id################################################
def scrape_comments_for_videos(df_videos):
    counter = 1
    # Initialize an empty list to store the DataFrames
    all_comments_dfs = []

# Iterate through each row in the DataFrame
    for index, row in df_videos.iterrows():
        video_id = row['video id']
        video_title = row['Video Title']

        try:
            # Fetch comments for the video
            comments_data = []
            next_page_token = None

            while True:
                comments_response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    order='relevance',  # Order comments by relevance
                    pageToken=next_page_token
                ).execute()

                # Extract comments and append to the list
                for item in comments_response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
                    comments_data.append({
                        'video_id': video_id,
                        'video_title': video_title,
                        'comment': comment
                    })

                next_page_token = comments_response.get('nextPageToken')

                if not next_page_token:
                    break

            # Create a DataFrame to store comments for this video
            comments_df = pd.DataFrame(comments_data)
            
            # Append DataFrame to the list
            all_comments_dfs.append(comments_df)

        except HttpError as e:
            if e.resp.status == 403 and 'commentsDisabled' in str(e):
                print(f"Comments are disabled for the video with ID: {video_id}. Skipping.")
            else:
                print("An error occurred while retrieving comments:", e)

    # Concatenate all DataFrames into one DataFrame by row
    combined_comments_df = pd.concat(all_comments_dfs, ignore_index=True)

    return combined_comments_df

############################ get sentiment ################################################
def get_sentiment(comment):
    # Tokenize the comment
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)

    # Forward pass through the model
    outputs = model(**inputs)

    # Get predicted label (sentiment)
    predicted_label = torch.argmax(outputs.logits).item()

    return predicted_label


def sentiment_label(comment_dataframe):
    # input comments into get_sentiment function.
    # Apply the get_sentiment function to each comment and create a new column 'sentiment' in the DataFrame
    comment_dataframe['sentiment'] = comment_dataframe['comment'].apply(get_sentiment) #create new column called 'sentiment'
    comment_dataframe['sentiment'] = comment_dataframe['sentiment'].map({0: 'negative', 2: 'positive',1:'neutral'}) #mapping value
    return comment_dataframe

############ function to process comment in DataFrame #############################################
def text_process(text):
    if isinstance(text, float):
        text = str(text)
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ","#","%",")","(","&","+","-",",","1","2","3","4","5","6","7","8","9","0"," ","/","=","''","'",'""','"','""','"'))
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                     if word.lower not in thai_stopwords)
    return final

############ function to tokenize text #############################################
def text_tokens(dataFrame):
    dataFrame['text_tokens'] = dataFrame['comment'].apply(text_process)
    dataFrame['text_tokens'] = dataFrame['text_tokens'].apply(lambda s: emoji.replace_emoji(s,''))
    dataFrame['text_tokens'] = dataFrame['text_tokens'].apply(lambda x : x.replace(',',''))
    dataFrame = dataFrame.dropna(how='any')
    return dataFrame

############# Make positive cloud ################################################
def MakePositiveCloud(dataframe):
    df_pos = dataframe[dataframe['sentiment'] == 'positive']
    pos_word_all = " ".join(text for text in df_pos['text_tokens'])
    reg = r"[ก-๙a-zA-Z']+"
    fp = 'THSarabunNew.ttf'
    wordcloud = WordCloud(stopwords=thai_stopwords, background_color = 'white',
                          colormap= 'GnBu',
                          max_words=2000,#height = 300, width=500
                          font_path=fp, regexp=reg).generate(pos_word_all)
    plt.figure(figsize = (8,4))
    plt.imshow(wordcloud)
    plt.axis('off')
    # Convert plot to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes,format='png')
    img_bytes.seek(0)
    plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    return plot_base64
    
############# Make Negative cloud ################################################
def MakeNegativeCloud(dataframe):
    df_neg = dataframe[dataframe['sentiment'] == 'negative']
    neg_word_all = " ".join(text for text in df_neg['text_tokens'])
    reg = r"[ก-๙a-zA-Z']+"
    fp = 'THSarabunNew.ttf'
    wordcloud = WordCloud(stopwords=thai_stopwords, background_color = 'white', 
                          colormap='OrRd',
                          max_words=2000, #height = 300, width=500, 
                          font_path=fp, regexp=reg).generate(neg_word_all)
    plt.figure(figsize = (8,4))
    plt.imshow(wordcloud)
    plt.savefig('NegativeCloud.png')
    plt.axis('off')
    # Convert plot to base64
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes,format='png')
    img_bytes.seek(0)
    plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    return plot_base64


###################### Words Vectorizer ############################################
def Vectorizer(df_texttokens):
    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
    cvec.fit_transform(df_texttokens['text_tokens'])
    cvec.vocabulary_
    
    # Assuming cvec.vocabulary_ contains the word counts dictionary
    word_counts = cvec.vocabulary_
    # Convert dictionary to DataFrame
    df_word_counts = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Count'])
    return df_word_counts


# Initialize the Dash app
app = dash.Dash(__name__)

#app title
app.title = "Youtube Comments Sentiment Analysis"

#giving a flexibility to callback functions to exist even if the layout changes
app.config.suppress_callback_exceptions = True

videos_df = None

# Define the layout of the app
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

# Define the app layout with two tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Import data', value='tab1', children=[
            html.Div(children=[
                html.H1(children="YouTube Comments Sentiment Analysis"),
                html.Div(children='''
                    Enter channel name and number of videos.
                '''),

                dcc.Input(id='Channel_name', type='text', placeholder='Enter Channel Name...'),
                dcc.Input(id='number_of_videos', type='number', placeholder='Enter the number of videos you want from the channel...'),
                html.Button('Submit', id='import_comments_button', n_clicks=0),
                html.Div(children =[
                    dcc.Loading(
                    id = "loading-success-text",
                    type ='dot',
                    color = '#E74C3C',
                    children = [
                        html.Div(id="success_text")
                    ])
                 ], style={'margin-top': '20px','text-align': 'center','flex-direction': 'column'})
                
            ], style={'display': 'flex', 'flex-direction': 'column', 'padding': 10})
        ]),
        dcc.Tab(label='Data visualization', value='tab2', children=[
            dcc.Loading(
                id="loading-comments",
                color ='#E74C3C',
                type ='dot',
                children=[ #this is the second tab componants

                    html.Div(id='Channel_name_Text'),
                    html.Div(children=[
                            dcc.Graph(id='OverallSentimentPieChart'),
                            dcc.Graph(id='SentimentBarChart'),
                            html.Div(children=[
                                html.Div(children=[
                                    html.H3("Positive Word Cloud", style={'text-align': 'center'}),
                                    html.Img(id='positive_cloud', style={'padding': 10, 'display': 'flex', 'margin': '0 auto'})
                                ]),
                                html.Div(children=[
                                    html.H3("Negative Word Cloud", style={'text-align': 'center'}),
                                    html.Img(id='negative_cloud', style={'padding': 10, 'display': 'flex', 'margin': '0 auto'})
                                ]),
                
                            ], style={'display': 'flex', 'flexDirection': 'column', 'flex-wrap': 'wrap','align-items': 'center'})

                        ], style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap'}),

                    html.Div(children= [
                        html.H3("Select Video title to see Word cloud", style={'text-align': 'center'}),
                        html.Div(id = 'drop_down_video'),
                        dcc.Loading(
                            id='loading - cloud',
                            color ='#E74C3C',
                            type ='dot',
                            children= [
                                html.Div(children=[
                                    html.H3("Positive Word Cloud", style={'text-align': 'center'}),
                                    html.Img(id='Positive_ind', style={'padding': 10, 'display': 'flex', 'margin': '0 auto'})
                                ]),
                                html.Div(children=[
                                    html.H3("Negative Word Cloud", style={'text-align': 'center'}),
                                    html.Img(id='Negative_ind', style={'padding': 10, 'display': 'flex', 'margin': '0 auto'})
                                ]),
                
                            ], style={'display': 'flex', 'flexDirection': 'row', 'flex-wrap': 'wrap','align-items': 'center'})
                                
                            ],style={'display': 'flex', 'flex-direction': 'column','padding': 10})
                    
                ], style={'display': 'flex', 'flex-direction': 'column','padding': 10})
        ])
    ])

], style={'font-family': "Raleway, sans-serif"})


# Define callback to fetch videos from channel
@app.callback(
    [Output('Channel_name_Text', 'children'),
     Output('OverallSentimentPieChart', 'figure'),
     Output('SentimentBarChart', 'figure'),
     Output('positive_cloud', 'src'),
     Output('negative_cloud', 'src'),
     Output('success_text','children'),
     Output('drop_down_video','children')],
    [Input('import_comments_button', 'n_clicks')],
    [State('Channel_name', 'value'),
     State('number_of_videos', 'value')]
)
def get_data(n_clicks, Channel_name, number_of_videos):
    global VideoFromChannel_df
    global combined_comments_df
    global channel_url
    global channel_title
    global Channel_id
    global df
    

    if n_clicks > 0:
        channel_url, channel_title, Channel_id = get_channel_url(Channel_name)
        VideoFromChannel_df = get_video_from_chanel(Channel_id, number_of_videos)
        combined_comments_df = scrape_comments_for_videos(VideoFromChannel_df)

        Channel_name_text = html.Div([
            html.H2(f"Channel: {channel_title}"),
            html.A("Link to channel", href=channel_url)
        ])
        df = combined_comments_df.copy()
        df = sentiment_label(df)
        df = text_tokens(df)
        PositiveCloud = MakePositiveCloud(df)
        NegativeCloud = MakeNegativeCloud(df)


        success_text = html.Div(children=[
            html.H3('Import data successfully. Please go to Visualization tab.')
        ],style={'text-align': 'center'})

        sentiment_counts = df['sentiment'].value_counts()
        fig2 = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                      title='Overall Sentiment Ratio',
                      color_discrete_map={'negative': 'red', 'positive': '#9FE2BF', 'neutral': '#F9C488'})
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        

        # Define the maximum length of the video titles you want to display
        max_title_length = 15
        # Group the DataFrame by 'video_title' and 'sentiment', and sum the counts
        grouped_df = df.groupby(['video_title', 'sentiment']).size().reset_index(name='count')
        grouped_df['truncated_title'] = grouped_df['video_title'].apply(lambda x: x[:max_title_length] if len(x) > max_title_length else x)

        # Calculate total sentiment count for each video
        total_sentiment_counts = grouped_df.groupby('video_title')['count'].sum()

        # Merge total sentiment count back to the grouped DataFrame
        grouped_df = grouped_df.merge(total_sentiment_counts, on='video_title', suffixes=('', '_total'))

        # Calculate ratios
        grouped_df['negative_ratio'] = grouped_df['count'] / grouped_df['count_total']
        grouped_df['positive_ratio'] = grouped_df['count'] / grouped_df['count_total']

        grouped_df['truncated_title'] = grouped_df['video_title'].apply(lambda x: x[:max_title_length] if len(x) > max_title_length else x)
        # Define hover template with custom formatting including ratios
        hover_template = (
            '<b>%{customdata[0]}</b><br>'  # Video title
            'Count: %{y}<br>'  # Count
            'Negative Ratio: %{customdata[1]:.2%}<br>'  # Negative ratio
            'Positive Ratio: %{customdata[2]:.2%}'  # Positive ratio
            )
        fig1 = px.bar(grouped_df, x='truncated_title', y='count', color='sentiment',
                    title='Sentiment Analysis by Video',
                    labels={'count': 'Count', 'truncated_title': 'Video Title'},
                    hover_data={'video_title': True, 'negative_ratio': True, 'positive_ratio': True},  # Include ratio data in hover
                    custom_data=['video_title', 'negative_ratio', 'positive_ratio'],  # Pass ratio data as custom data
                    barmode='group',
                    color_discrete_map={'negative': 'red', 'positive': '#9FE2BF', 'neutral': '#F9C488'})

        # Update traces with hover template and adjust width if needed
        fig1.update_traces(textposition="outside", hovertemplate=hover_template)


        #dropdown 
        dropdown = html.Div([
            dcc.Dropdown(
                id='video-dropdown',
                options=[{'label': title, 'value': title} for title in df['video_title'].unique()],
                value=df['video_title'].iloc[0],  # Set default value to the first video title
                clearable=False,  # Disable the option to clear the dropdown
            ),
        ])

        return Channel_name_text, fig2, fig1, f"data:image/png;base64,{PositiveCloud}", f"data:image/png;base64,{NegativeCloud}",success_text,dropdown

# Define callback to generate word cloud for selected video
@app.callback(
    [Output('Positive_ind', 'src'),
     Output('Negative_ind','src')],
    [Input('video-dropdown', 'value')]
)
def generate_word_cloud(selected_video_title):
    # Filter comments for the selected video
    selected_video_comments = df[df['video_title'] == selected_video_title]
    
    PositiveCloud_ind = MakePositiveCloud(selected_video_comments)
    NegativeCloud_ind = MakeNegativeCloud(selected_video_comments)
    

    
    return f"data:image/png;base64,{PositiveCloud_ind}", f"data:image/png;base64,{NegativeCloud_ind}"



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)