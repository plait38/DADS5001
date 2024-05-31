# Import necessary libraries and modules
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import pdfminer  # Library for extracting text from PDF files
from pdfminer.high_level import extract_text  # Function to extract text from PDF files
from transformers import pipeline  # Hugging Face library for NLP pipelines
import base64  # Module for encoding and decoding binary data
import io  # Module for handling streams of binary data
from wordcloud import WordCloud  # Library for generating word clouds
import matplotlib.pyplot as plt  # Library for plotting
import numpy as np  # Library for numerical computing

# Initialize the summarization pipelines
# Initialize two summarization pipelines using Hugging Face transformers library
summarizer_falconsai = pipeline("summarization", model="Falconsai/medical_summarization")
summarizer_long_t5 = pipeline("summarization", model="pszemraj/long-t5-tglobal-base-16384-book-summary")

# Initialize the Dash app with Lux theme
# Create a Dash web application instance with Lux theme from dash_bootstrap_components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Define the layout of the app
# Create the layout structure using Dash HTML components from dash and dash_bootstrap_components

header = html.Div(children=[
    html.Div(children=[
    html.Span(children='Bryde', style={'font-size': '40px', 'color': '#6D5BFF', 'font-family': 'Londrina Solid, cursive', 'font-weight':'bold'}),
    html.Span(children='Summarizer', style={'font-size': '40px', 'color': '#52AEFF', 'font-family': 'Londrina Solid, cursive', 'font-weight':'bold'})
], style={'background-color': '#EEEEEE', 'padding': '5px',' display': 'inline-block'}),
    html.Div(children=[
        html.Span(children='PDF Summarization', style={'font-size': '50px','font-family': 'Londrina Solid, Monospace','text-align': 'center', 'font-weight':'bold', 'padding': '10px'}),
        html.P(children='''Bryde is an innovative text summarizer designed to facilitate \
               a comprehensive understanding of your chosen texts effortlessly. \
               With Bryde, you can quickly grasp the essence of news and research articles in PDF format, streamlining your reading experience. Whether you're keeping up with the latest developments or diving into scholarly research, Bryde simplifies complex information, making it more accessible and saving you valuable time. Bryde Summarizer, part of the DADS5001 at NIDA.''', style={'font-size': '16px','font-family': 'Londrina Solid, Monospace','width':'40%','margin':'auto','text-align': 'center'})
    ], style={'background-color': '#EEEEEE','padding': '20px', 'text-align': 'center'})
])

loading_spinner = html.Div(dbc.Spinner(color="primary", size="lg"), id="loading-spinner", style={"display": "none"})

content = dbc.Container([
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    html.Span(children='1. Choose Model', style={'font-size': '28px', 'font-family': 'Verdana, Sans-serif', 'color':'#DD761C' ,'font-weight':'bold','padding-top':'10px','padding-bottom':'10px'})
                ]),
                dbc.Row([
                    dcc.Dropdown(
                        id='model-selector',  # Unique identifier for dropdown component
                        options=[
                            {'label': 'Model 1 : Academic Article Summarizer', 'value': 'falconsai'},  # Option for Falconsai summarizer
                            {'label': 'Model 2 : News Summarizer', 'value': 'long_t5'}  # Option for Long T5 summarizer
                        ],
                        value='falconsai',  # Default value for dropdown
                        style={'width': '100%', 'height': '50px', 'font-size': '24px', 'color':'#6D5BFF','font-weight':'bold' ,'margin': '5px auto'}  # Style for dropdown component
                    )
                ]),
            ]),
            dbc.Col([
                dbc.Row([
                    dcc.Upload(
                        id='upload-data',  # Unique identifier for file upload component
                        children=html.Div([
                            '2. Drag and Drop or ',  # Instruction for file upload
                            html.A('Select a PDF File'),  # Link text for file selection
                        ]),
                        style={
                            'width': '100%',  
                            'height': '150px', 
                            'font-size': '26px', 
                            'font-family': 'Verdana, Sans-serif',
                            'font-weight':'bold',
                            'color':'#DD761C',
                            'lineHeight': '60px', 
                            'border' : 'black',
                            'borderWidth': '2px', 
                            'borderStyle': 'dashed',  
                            'borderRadius': '5px', 
                            'textAlign': 'center', 
                            'padding-top':'10px',
                            'margin': '10px auto'  
                        },
                        multiple=False  # Allow only single file upload
                    ),
                    #width=6  # Width of the column containing the file upload component
                ]),
                dbc.Row([
                    html.Div(id='uploaded-filename', style={'font-size': '22px','margin-top': '5px', 'textAlign': 'center'}),  # Uploaded filename
                    dbc.Button("Generate Summary", id="submit-button", n_clicks=0, color="success", className="mr-1", style={'height': '50px','margin': 'auto', 'width': '85%', 'height':'60px','font-size': '22px','text-align': 'center','margin-top': '10px','borderRadius': '5px'}),  # Submit button
                ]),
                
            ]),

        ]),
    ], style={'padding': '20px','textAlign': 'center'})  # Center-align the entire layout
], fluid=True)  # Create a fluid container for responsive layout

result = dbc.Container([
    html.Div([
        dbc.Row([
            dbc.Col(
                html.Div(id='output-text', style={'marginTop': '20px', 'textAlign': 'left'}),
                width=6  # Width of the column containing the output text
            ),
            dbc.Col(
                html.Div(html.Img(id='wordcloud'), style={'marginTop': '20px'}),
                width=6  # Width of the column containing the word cloud image
            )
        ])
    ], style={'padding': '20px','textAlign': 'center'})  # Center-align the entire layout
], fluid=True)  # Create a fluid container for responsive layout


spinner_style = {
    'border': '0px solid rgba(0, 0, 0, 0.1)',
    'borderLeftColor': '#7986cb',
    'borderRadius': '5px',
    'width': '100px',
    'height': '100px',
    'animation': 'spin 1s linear infinite',
    # 'padding' : '70px'
}

keyframes_style = {
    '@keyframes spin': {
        'to': {
            'transform': 'rotate(360deg)'
        }
    }
}

spinner_css = {
    **spinner_style,
    **keyframes_style,
    'width': '400px',  
    'height': '400px',  
    'borderRadius': '5px',
    'margin': 'auto', 
    'position': 'fixed',  
    'top': '100%', 
    'left': '50%',  
    'transform': 'translate(-50%, -50%)', 
}

# ใส่ spinner_css ลงใน layout ของ Dash app ของคุณ
spinner = dcc.Loading(
    id="loading-spinner",
    type="default",
    children=html.Div(id="loading-output"),
    style=spinner_css  # นำ spinner_css มาใช้เป็น style ของ Loading component
)

app.layout = html.Div(children=[
    header,
    content,
    spinner,
    result
])

# File upload, text extraction, and summarization
@app.callback(
    [Output("output-text", "children"), Output("wordcloud", "src"), Output("loading-spinner", "style")],
    [Input("submit-button", "n_clicks")],
    [State("upload-data", "contents"), State("model-selector", "value"), State("upload-data", "filename")],
)
def update_output(n_clicks, contents, selected_model, filename):
    if n_clicks is None:
        raise PreventUpdate  # ป้องกันการทำงานของ callback หากปุ่มยังไม่ได้คลิก

    # แสดง Spinner ขณะที่โมเดลกำลังทำงาน
    loading_style = {"display": "block"}

    if contents is None:
        raise PreventUpdate  # ป้องกันการทำงานของ callback หากไม่มีไฟล์ที่อัปโหลด

    try:
        # Your processing logic here
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        extracted_text = extract_text_from_pdf(io.BytesIO(decoded))

        if selected_model == 'falconsai':
            summarizer = summarizer_falconsai
        elif selected_model == 'long_t5':
            summarizer = summarizer_long_t5

        summarized_text = summarize_text(summarizer, extracted_text)
        wordcloud_data = generate_wordcloud(extracted_text)

        # ปิด Spinner เมื่อการประมวลผลเสร็จสมบูรณ์
        loading_style = {"display": "none"}

        return (
            html.Div([
                html.H3('Summary:'),
                html.Div(summarized_text)
            ]),
            wordcloud_data,
            loading_style
        )
    except Exception as e:
        print(e)
        raise PreventUpdate


#Show Filename first
@app.callback(
    Output('uploaded-filename', 'children'),
    [Input('upload-data', 'filename')]
)
def update_uploaded_filename(filename):
    if filename is not None:
        return html.Div([
            html.Span('Uploaded File: '),
            html.Span(filename, style={'font-weight': 'bold'})
        ])
    else:
        return ""


def extract_text_from_pdf(file):
    text = extract_text(file)  # Extract text from PDF file
    return text

def summarize_text(summarizer, text):
    summary = summarizer(text, max_length=500, min_length=20, do_sample=False)  # Summarize the input text
    return summary[0]['summary_text']  # Return the summarized text

def generate_wordcloud(text):
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white', contour_color='steelblue', contour_width=2).generate(text)  # Generate word cloud from text
    else:
        wordcloud = WordCloud(width=800, height=400, background_color='white', contour_color='steelblue', contour_width=2).generate('')  # Generate empty word cloud
    
    # Convert the word cloud to an image
    wordcloud_image = wordcloud.to_image()

    # Convert the image to base64 string
    img_byte_array = io.BytesIO()
    wordcloud_image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    wordcloud_base64 = "data:image/png;base64," + base64.b64encode(img_byte_array).decode()  # Convert image to base64 string
    return wordcloud_base64  # Return the base64-encoded image data



if __name__ == '__main__':
    app.run_server(debug=True)  # Run the Dash app in debug mode