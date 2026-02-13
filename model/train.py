import pandas as pd
import numpy as np
import pickle, re
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from myTokenize import SyllableTokenizer
import os

df = pd.read_csv('dataset.csv')
tokenizer = SyllableTokenizer()
df['processed_text'] = df['Text-MM'].apply(tokenizer.tokenize)

print(df['Sentiment'].value_counts())

# os._exit(0)
df_pos = df[df['Sentiment'] == 'Positive']
df_neg = df[df['Sentiment'] == 'Negative']
df_neu = df[df['Sentiment'] == 'Neutral']

target = 800 
df_neg_upsampled = resample(df_neg, replace=True, n_samples=target, random_state=42)
df_neu_upsampled = resample(df_neu, replace=True, n_samples=target, random_state=42)
df_pos_upsampled = resample(df_pos, replace=True, n_samples=target, random_state=42)

df_balanced = pd.concat([df_pos_upsampled, df_neg_upsampled, df_neu_upsampled])
df_balanced = shuffle(df_balanced)

encoder = LabelEncoder()
df_balanced['label_id'] = encoder.fit_transform(df_balanced['Sentiment'])

tokenizer = Tokenizer(num_words=5000, split=' ') 
tokenizer.fit_on_texts(df_balanced['processed_text'].values)

with open('tokenizer.pkl', 'wb') as f: pickle.dump(tokenizer, f)
with open('encoder.pkl', 'wb') as f: pickle.dump(encoder, f)

X = tokenizer.texts_to_sequences(df_balanced['processed_text'].values)
X = pad_sequences(X, maxlen=100)
Y = pd.get_dummies(df_balanced['label_id']).values

model = Sequential([
    Embedding(5000, 512, input_length=100),
    SpatialDropout1D(0.4),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=15, batch_size=32, validation_split=0.2)
model.save('burmese_sentiment_lstm.keras')