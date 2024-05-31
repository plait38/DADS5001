import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc
import plotly.graph_objects as go
from dash import html, dash_table
from dash.dependencies import Input, Output, State
from transformers import pipeline, MarianMTModel, MarianTokenizer

app = dash.Dash(
   __name__,
   external_stylesheets = [
      dbc.themes.BOOTSTRAP,
      '/assets/styles.css'])

app.layout = dbc.Container([
    html.Div(children = 'EmoVerse Analyzer',
             className = 'Noto_regular_font header_text'),
    dbc.Row([
        dbc.Col(width= 8, children = [
                    html.Div("Unlocking the Soul of Songs",
                             className = 'Noto_regular_font body1_text'),
                    html.Div("Elevating Content Quality and Audience Connection. Dive deep into the heart of music with our app, analyzing song lyrics to reveal their true sentiment and meaning. Enhance audience comprehension and appreciation while boosting your business prospects effortlessly.",
                             className = 'Noto_regular_font body2_text'),
                             ]),
        dbc.Col(width = 4, style = {'height': '250px', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'overflow': 'hidden'},
                children = [
                    html.Img(src = '/assets/01.jpg', className = 'card-img-top rounded',
                             style = {'width': 'auto', 'max-height': '250px'})
                            ])
                        ], style = {'marginTop': '40px', 'marginBottom': '40px'}),

    dbc.Row([
        dbc.Col(width = 4, style = {'height': '500px', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'overflow': 'hidden'},
                children = [
                    html.Img(src = '/assets/02.jpg', className = 'card-img-top rounded',
                             style = {'Width': 'auto', 'max-height': '500px'})
                            ]),
        dbc.Col(width = 8, children = [
                html.Div("Start Now",
                        style = {'marginBottom': '10px'},
                        className = 'Noto_regular_font body1_text_center'),
                dcc.Textarea(id = "lyrics_input", placeholder = "Enter a song lyrics", 
                             style = {'width': '100%', 'height': '420px', 'padding': '15px', 'marginBottom': '10px', 'font-size': '12px', 'border-color': 'lightgray'}, 
                             className = 'Noto_regular_font rounded'),
                html.Div(
                        html.Button('GO', n_clicks = 0, id = 'lyrics_submit_buttom',
                                    className = 'Noto_regular_font button_style'),
                        style = {'display': 'flex', 'justify-content': 'center', 'margin-bottom': '25px'})
                                ])
                        ], style = {'marginTop': '40px'}),

    dbc.Row([
        dbc.Col(width = 5, children = [
            html.Div("Translated Lyrics",
                style = {'marginBottom': '10px', 'marginTop': '10px'},
                className = 'Noto_regular_font body1_text_center'),
           dbc.Card([
                html.Div(id = 'language_output', 
                        style = {'whiteSpace': 'pre-line', 'height': '30px', 'font-size': '12px', 'padding': '10px'}, 
                        className = 'Noto_regular_font'),
                html.Div(id = 'translate_output', 
                        style = {'whiteSpace': 'pre-line', 'height': '300px', 'font-size': '12px', 'padding': '10px'}, 
                        className = 'Noto_regular_font'),
                        ], style = {'height': '450px', 'border-color': 'lightgray', 'overflowX': 'auto', 'overflowY': 'auto'})
                    ], style = {'height': '500px'}),

        dbc.Col(width = 7, children = [
            html.Div("Classification Table",
                style = {'marginBottom': '10px', 'marginTop': '10px'},
                className = 'Noto_regular_font body1_text_center'),
            dbc.Card([
                dash_table.DataTable(
                    data = [],
                    style_table = {'overflowX': 'auto'},
                    id = 'data_table',
                    style_cell = {'textAlign': 'center', 'fontSize': '12px', 'fontFamily': 'Noto_regular_font'},
                    style_header = {'fontSize': '14px', 'textAlign': 'center', 'fontFamily': 'Noto_regular_font'},
                    style_cell_conditional = [{'if': {'column_id': 'Text'}, 'width': '400px', 'textAlign': 'left', 'whiteSpace': 'normal','overflow': 'wrap'}]
                            )
                        ], style = {'height': '450px', 'border-color': 'lightgray', 'overflowX': 'auto', 'overflowY': 'auto'})
                    ]),
                ], style = {'marginTop': '40px', 'height': '500px'}),

    dbc.Row([
        dbc.Col(width = 6, children = [
            dcc.Graph(figure = {}, id = 'sentiment_chart')
                            ]),
        dbc.Col(width = 6, children = [
            dcc.Graph(figure = {}, id = 'emotion_chart')
                            ]),
                        ], style = {'marginTop': '40px'})

                    ], style = {'margin': 'auto', 'padding': '40px', 'backgroundColor': '#fcfcfc'})

#Language translation model
@app.callback(
    [Output('language_output', 'children'),
     Output('translate_output', 'children')],
    [Input('lyrics_submit_buttom', 'n_clicks')],
    [State('lyrics_input', 'value')]
)
def lang_translation(n_clicks, lyrics):
  if n_clicks is not None and lyrics:
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model = model_ckpt)
    lang_class_result = pipe(lyrics, top_k = 1, truncation = True)[0]['label']

    if lang_class_result == 'th' or lang_class_result == 'zh' or lang_class_result == 'ja':
      model_name = "Helsinki-NLP/opus-mt-{}-en".format(lang_class_result)
      lang_name = {"th": "Thai", "zh": "Chinese", "ja": "Japanese"}[lang_class_result]
      model = MarianMTModel.from_pretrained(model_name)
      tokenizer = MarianTokenizer.from_pretrained(model_name)

      translated_text = ""
      for part in lyrics.split():
        inputs = tokenizer(part, return_tensors = "pt", padding = True)
        outputs = model.generate(**inputs)
        translated_text += tokenizer.decode(outputs[0], skip_special_tokens = True) + " "

      translated_text = translated_text.strip()
      if lang_class_result in ['th', 'zh', 'ja']:
        text_parts = translated_text.split('.')
      else:
        text_parts = translated_text.split('\n')

    else:
      return "The language of this song is: en", lyrics

    return "The language of this song is: {}".format(lang_name), '\n'.join(text_parts)

  else:
      return dash.no_update

#Text sentiment classification model
@app.callback(
    Output('data_table', 'data'),
    Input('translate_output', 'children')
)
def sentiment_classification(text):
    if text:
        if text.split()[0] in ['th', 'zh', 'ja']:
            words = [i.strip() for i in text.split('.')]
        else:
            words = [i.strip() for i in text.split('\n')]

        word_count = {}
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

        df = pd.DataFrame(list(word_count.items()), columns = ['Text', 'Count'])

        def sentiment_task(text):
            model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
            sentiment_task = pipeline("sentiment-analysis", model = model_path, tokenizer = model_path)
            return sentiment_task(text)

        def emotion_task(text):
          classifier = pipeline(task = "text-classification", model = "SamLowe/roberta-base-go_emotions", top_k = None)
          return classifier(text)

        df['Sentiment'] = df['Text'].apply(lambda x: sentiment_task(x)[0]['label'])
        df['Emotion'] = df['Text'].apply(lambda x: emotion_task(x)[0][0]['label'])
        df.drop(df[df['Text'] == ''].index, inplace = True)
        return df.to_dict('records')
    else:
        return []

#Sentiment graph
@app.callback(
    Output('sentiment_chart', 'figure'),
    Input('data_table', 'data')
)
def update_graph(data):
    if data:
        df = pd.DataFrame(data)
        sentiment_counts = df.groupby('Sentiment').sum().reset_index()
        sentiment_counts = sentiment_counts[sentiment_counts['Sentiment'] != 'neutral']
        fig = go.Figure(data = [
           go.Pie(
              labels = sentiment_counts['Sentiment'],
              values = sentiment_counts['Count'],
              textinfo = 'label + percent',
              name = 'Sentiment'),
              ])
        fig.update_traces(
           hole = 0.4,
           hoverinfo = " label + percent + name")
        fig.update_layout(
           title_text = "Sentiment Classified Donut Chart",
           annotations = [dict(text = 'Sentiment', x = 0.5, y = 0.5, font_size = 12, showarrow = False)])
        return fig
    else:
        return {}

#Emotions graph
@app.callback(
    Output('emotion_chart', 'figure'),
    Input('data_table', 'data')
)
def update_graph(data):
    if data:
        df = pd.DataFrame(data)
        emotion_counts = df.groupby('Emotion').sum().reset_index()
        emotion_counts = emotion_counts[emotion_counts['Emotion'] != 'neutral']
        fig = go.Figure(data = [
           go.Pie(
              labels = emotion_counts['Emotion'],
              values = emotion_counts['Count'],
              textinfo = 'label + percent',
              name = 'Emotion'),
              ])
        fig.update_traces(
           hole = 0.4,
           hoverinfo = " label + percent + name")
        fig.update_layout(
           title_text = "Emotional Classified Donut Chart",
           annotations = [dict(text = 'Emotional', x = 0.5, y = 0.5, font_size = 12, showarrow = False)])
        return fig
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True)
