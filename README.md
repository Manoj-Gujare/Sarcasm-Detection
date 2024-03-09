# Sarcasm Detection using LSTM

## Description
This project involves building a model to detect sarcasm in textual data using LSTM (Long Short-Term Memory) neural networks. The model is trained on a dataset consisting of sarcastic and non-sarcastic headlines. The dataset is obtained from two JSON files: `Sarcasm_Headlines_Dataset.json` and `Sarcasm_Headlines_Dataset_v2.json`, which are merged to form the complete dataset.

## Tools & Libraries
- Python (Version 3.x)
- Numpy
- Pandas
- TensorFlow (Version 2.x)
- Matplotlib
- NLTK
- Streamlit

## Get the Data
The dataset is obtained from two JSON files, `Sarcasm_Headlines_Dataset.json` and `Sarcasm_Headlines_Dataset_v2.json`. Both files are read using the Pandas library, and then concatenated to form the complete dataset.

## Clean the Data
The textual data is preprocessed by performing the following steps:
- Convert text to lowercase
- Remove URLs
- Remove Twitter handles
- Remove emojis
- Expand contractions
- Remove punctuation
- Tokenize the text
- Remove stop words

## Visualize the Most Occurring Words
A word cloud is generated to visualize the most occurring words in the sarcastic headlines.

## Train-Test-Split
The dataset is split into training and testing sets. The headlines are tokenized and padded to ensure uniform length sequences.

## Load GloVe Model
A pre-trained GloVe (Global Vectors for Word Representation) model is loaded to obtain word embeddings.<br>
Required GloVe model file - https://drive.google.com/file/d/12Utwrbl-z2GmsGXf3_i-BRyNRo-SD9MN/view?usp=sharing

## Build the Embedding Layer
An embedding layer is built using the loaded GloVe model to convert words into fixed-size vectors.

## Build the LSTM Model
The LSTM model is built using Sequential API from TensorFlow. It consists of an embedding layer, an LSTM layer with dropout, and a dense layer with sigmoid activation.

## Training
The model is trained using the training data with binary cross-entropy loss and Adam optimizer. Training is performed for 25 epochs.

## Visualize the Learning
Training and validation accuracy, as well as training and validation loss, are plotted to visualize the learning process.

## Function for Detecting Sarcasm
A function `predict_sarcasm()` is defined to predict whether a given text is sarcastic or not. The function takes a text input, preprocesses it, and then predicts sarcasm using the trained LSTM model.

## Model Persistence
The trained LSTM model is saved using the `pickle` library and stored in the file `sarcasm_detection.pkl` for future use.

## Streamlit Web Application
A Streamlit web application is created to allow users to input sentences and predict sarcasm using the trained model. The application provides an interactive interface for users to perform sarcasm detection on their text data.

## Usage
To use the trained model for sarcasm detection:
1. Call the `predict_sarcasm()` function with a text input.
2. The function will return whether the text is sarcastic or not.
3. Alternatively, run the Streamlit web application using `streamlit run <filename.py>`, where `<filename.py>` is the name of the Python script containing the Streamlit code.
