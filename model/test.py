import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from myTokenize import SyllableTokenizer 

model = load_model('burmese_sentiment_lstm.keras')
with open('tokenizer.pkl', 'rb') as f:
    keras_tokenizer = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

syl_tokenizer = SyllableTokenizer()

def test_sentence(text):
    syllables = syl_tokenizer.tokenize(text)
    
    seq = keras_tokenizer.texts_to_sequences([syllables])
    
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded, verbose=0)
    
    print(f"\nInput: {text}")
    print(f"Syllables: {syllables}")
    print("-" * 30)
    for i, label in enumerate(encoder.classes_):
        print(f"{label}: {pred[0][i]*100:.2f}%")
    
    result = encoder.classes_[np.argmax(pred)]
    print(f"\nFinal Prediction: **{result}**")

test_sentence("ဒီဝန်ဆောင်မှုက တကယ်ကို စိတ်ပျက်ဖို့ကောင်းတယ်") 
test_sentence("ရန်ကုန်မြို့တွင် ယနေ့ မိုးရွာသွန်းနိုင်ပါသည်")
test_sentence("ဘာမှမဟုတ်တော့ပါဘူး ")
test_sentence("မင်းကြောင့် ရင်တွေပူနေတယ်")
test_sentence("ဗိုက် ပြည့် ပါပြီ။ ဒီထက် ပို မစား ချင်ဘူး။")
test_sentence("ဒီနေရာက တော်တော် မကောင်းတာပဲ")
test_sentence("ဒီနေ့တော့ တကယ်ကို စိတ်ချမ်းသာဖို့ကောင်းတဲ့ နေ့လေးပါပဲ")