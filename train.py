import pandas as pd
import numpy as np
import pickle, os
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from myTokenize import SyllableTokenizer

burmese_syll_tokenizer = SyllableTokenizer()
df = pd.read_csv('datasets/cleaned_dataset.csv')

print("Original Class Distribution:")
print(df['Sentiment'].value_counts())

df['processed_text'] = df['Text-MM'].apply(burmese_syll_tokenizer.tokenize)

encoder = LabelEncoder()
df['label_id'] = encoder.fit_transform(df['Sentiment'])

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['label_id']),
    y=df['label_id']
)
class_weights_dict = dict(enumerate(weights))

max_features = 5000
max_len = 80      

keras_tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False)
keras_tokenizer.fit_on_texts(df['processed_text'].values)

X = keras_tokenizer.texts_to_sequences(df['processed_text'].values)
X = pad_sequences(X, maxlen=max_len)
Y = pd.get_dummies(df['label_id']).values

model = Sequential([
    Embedding(max_features, 100, input_length=max_len), 
    SpatialDropout1D(0.4),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2), 
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Starting Training...")
model.fit(
    X, Y, 
    epochs=20, 
    batch_size=32,
    validation_split=0.2, 
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)


os.makedirs('trained_models', exist_ok=True)
model.save('trained_models/burmese_sentiment_lstm.keras')
with open('trained_models/tokenizer.pkl', 'wb') as f: pickle.dump(keras_tokenizer, f)
with open('trained_models/encoder.pkl', 'wb') as f: pickle.dump(encoder, f)

print("Process Complete. Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))