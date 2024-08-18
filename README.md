
# IMDB Movie Reviews Sentiment Analysis

This project is a Sentiment Analysis system built using LSTM (Long Short-Term Memory) to classify the sentiment of IMDB movie reviews as either positive or negative. The dataset contains 50,000 movie reviews labeled as positive or negative, with an equal distribution of sentiments.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)


## Overview

This project implements a sentiment analysis model to predict the sentiment (positive/negative) of movie reviews from the IMDB dataset. The model is built using TensorFlow and Keras, employing an LSTM layer to capture the sequential dependencies in the text data.

## Dataset

The dataset used in this project is the IMDB Dataset of 50K Movie Reviews, which can be downloaded from Kaggle.

- **Link:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 movie reviews
- **Labels:** Positive (1) and Negative (0)

## Installation

To run this project, you'll need to install the necessary Python packages. You can do so by running:

```bash
pip install -r requirements.txt
```

Make sure to also configure your Kaggle API credentials to download the dataset:

1. Download your Kaggle API token (`kaggle.json`) from your Kaggle account.
2. Place the `kaggle.json` file in the root directory of your project.

## Data Preprocessing

- **Tokenization:** The text data is tokenized using Keras' `Tokenizer`.
- **Padding:** The tokenized sequences are padded to a fixed length to ensure uniform input size for the model.

## Model

The sentiment analysis model is built using a Sequential model in Keras. It consists of:

- An Embedding layer to convert words into dense vectors.
- An LSTM layer to capture sequential dependencies in the text.
- A Dense layer with a sigmoid activation function for binary classification.

### Model Summary

```plaintext
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 200, 128)          640000    
                                                                 
 lstm (LSTM)                 (None, 128)               131584    
                                                                 
 dense (Dense)               (None, 1)                 129       
                                                                 
=================================================================
Total params: 771713 (2.94 MB)
Trainable params: 771713 (2.94 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

## Training

The model was trained for 5 epochs with a batch size of 64, using the Adam optimizer and binary crossentropy loss.

```python
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)
```

## Evaluation

The model achieved the following performance on the test data:

- **Test Loss:** 0.3139
- **Test Accuracy:** 88.3%

## Usage

To predict the sentiment of a new movie review, use the `predict_sentiment` function:

```python
def predict_sentiment(review):
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
```

## Results

The model successfully classifies the sentiment of movie reviews with an accuracy of 88.3%. Example results:

- **Review:** "This movie was fantastic. I loved it."  
  **Sentiment:** Positive
- **Review:** "This movie was not that good."  
  **Sentiment:** Negative

