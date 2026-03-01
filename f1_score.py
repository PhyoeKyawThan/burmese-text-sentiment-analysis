import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, f1_score
from myTokenize import SyllableTokenizer

df = pd.read_csv('datasets/dataset.csv')
burmese_syll_tokenizer = SyllableTokenizer()
df['processed_text'] = df['Text-MM'].apply(burmese_syll_tokenizer.tokenize)

model = load_model('trained_models/burmese_sentiment_lstm.keras')

with open('trained_models/tokenizer.pkl', 'rb') as f:
    keras_tokenizer = pickle.load(f)

with open('trained_models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
df['label_id'] = encoder.transform(df['Sentiment']) 

max_len = 80
X_test_seq = keras_tokenizer.texts_to_sequences(df['processed_text'].values)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

y_true = df['label_id'].values

predictions = model.predict(X_test_padded)
y_pred = np.argmax(predictions, axis=1)

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Macro F1-Score: {f1_macro:.4f}")
print(f"Weighted F1-Score: {f1_weighted:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=encoder.classes_))